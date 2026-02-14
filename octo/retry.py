"""Auto-retry and auto-remediation for graph invocation errors."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Max retries and exponential backoff delays (seconds)
_MAX_RETRIES = 3
_BACKOFF_TIMEOUT = [3, 9, 27]       # 3s, 9s, 27s  (×3 exponential)
_BACKOFF_RATE_LIMIT = [5, 15, 45]   # 5s, 15s, 45s (×3 exponential)


def _classify_error(error: BaseException) -> str | None:
    """Classify an error into a remediation category.

    Returns:
        "timeout", "rate_limit", "context_overflow", "connection_closed",
        "orphaned_tools", or None (unknown).
    """
    msg = str(error).lower()
    if "read timeout" in msg or "connect timeout" in msg or "timed out" in msg:
        return "timeout"
    if "rate limit" in msg or "too many requests" in msg or "throttling" in msg:
        return "rate_limit"
    if "too long" in msg or "context length" in msg or "input is too long" in msg:
        return "context_overflow"
    if "serviceunav" in msg or "service unavailable" in msg or "503" in msg:
        return "timeout"  # treat same as timeout — transient
    if "expected toolresult" in msg or "expected tool_result" in msg:
        return "orphaned_tools"
    if "connection was closed" in msg or "connection reset" in msg or "broken pipe" in msg:
        return "connection_closed"
    # Bedrock cross-region inference profiles can return "model identifier
    # is invalid" for oversized payloads or transient issues. Try compact first.
    if "model identifier is invalid" in msg:
        return "model_invalid"
    return None


async def auto_compact(app: Any, config: dict) -> bool:
    """Run automatic context compaction. Returns True if successful.

    Two-stage: tries LLM summarization first, falls back to crude
    message dropping if the summary model also fails (e.g. Bedrock
    returning 'model identifier is invalid' for both main and low-tier).
    """
    try:
        from langchain_core.messages import RemoveMessage, SystemMessage

        state = await app.aget_state(config)
        messages = state.values.get("messages", [])
        if len(messages) < 6:
            return False

        # Keep last ~25% of messages (more aggressive than manual /compact)
        keep_count = max(4, len(messages) // 4)
        old_msgs = messages[:-keep_count]
        removable = [m for m in old_msgs if getattr(m, "id", None)]
        if not removable:
            return False

        # Stage 1: try LLM summarization
        summary_msg = None
        try:
            from octo.models import make_model
            summary_model = make_model(tier="low")
            summary_lines = []
            for m in old_msgs:
                role = getattr(m, "type", "unknown")
                content = m.content if isinstance(m.content, str) else str(m.content)
                if content.strip():
                    summary_lines.append(f"[{role}]: {content[:300]}")

            summary_prompt = (
                "Summarize this conversation concisely in 2-3 paragraphs. "
                "Focus on: decisions made, tasks completed, and current objectives.\n\n"
                + "\n".join(summary_lines[-80:])
            )
            summary_response = await summary_model.ainvoke(summary_prompt)
            summary_msg = SystemMessage(
                content=(
                    "[Conversation summary — earlier messages were auto-compacted]\n\n"
                    + summary_response.content
                )
            )
        except Exception:
            # Stage 2: fallback — crude drop without summarization
            logger.warning(
                "Summary model failed, falling back to crude compaction",
                exc_info=True,
            )
            summary_msg = SystemMessage(
                content=(
                    "[Earlier conversation was auto-compacted (summary unavailable). "
                    f"{len(removable)} messages removed to reduce context size.]"
                )
            )

        remove_ops = [RemoveMessage(id=m.id) for m in removable]
        await app.aupdate_state(config, {"messages": remove_ops + [summary_msg]})

        before = len(messages)
        after = len(messages) - len(removable) + 1
        logger.info("Auto-compacted: %d -> %d messages", before, after)
        return True
    except Exception:
        logger.warning("Auto-compact failed", exc_info=True)
        return False


async def auto_repair_orphaned_tools(app: Any, config: dict) -> bool:
    """Repair orphaned tool_use blocks in the checkpoint.

    When Bedrock drops a connection mid-response (e.g. content policy), the
    checkpoint may contain an AIMessage with tool_calls but no corresponding
    ToolMessage.  Bedrock's Converse API rejects this on the next call:
    "Expected toolResult blocks at messages.X.content for the following Ids: ..."

    Fix: inject synthetic ToolMessages for all orphaned tool call IDs.
    Returns True if repairs were made.
    """
    try:
        from langchain_core.messages import AIMessage, ToolMessage

        state = await app.aget_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            return False

        # Collect all tool_call_ids that have responses
        responded_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None):
                responded_ids.add(msg.tool_call_id)

        # Find AIMessages with orphaned tool calls
        repairs: list[ToolMessage] = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                if tc_id and tc_id not in responded_ids:
                    repairs.append(ToolMessage(
                        content="[Tool call interrupted — connection was lost before execution]",
                        tool_call_id=tc_id,
                        name=tc.get("name", "unknown"),
                    ))

        if not repairs:
            return False

        # Inject synthetic tool results into the state
        await app.aupdate_state(config, {"messages": repairs})
        logger.info(
            "Repaired %d orphaned tool call(s) in checkpoint",
            len(repairs),
        )
        return True
    except Exception:
        logger.warning("Auto-repair orphaned tools failed", exc_info=True)
        return False


async def auto_clean_corrupted(app: Any, config: dict) -> bool:
    """Remove corrupted messages from the checkpoint.

    When ``auto_repair_orphaned_tools`` is not enough (e.g. the AIMessage
    itself is malformed or contains content-policy-violating data), strip
    the offending tail messages so the conversation can continue.

    Strategy: walk backwards from the end and remove any AIMessage that has
    tool_calls without matching ToolMessages, plus any dangling ToolMessages
    that reference non-existent tool_call_ids.
    Returns True if any messages were removed.
    """
    try:
        from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

        state = await app.aget_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            return False

        # Build tool_call_id → responded mapping
        responded_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None):
                responded_ids.add(msg.tool_call_id)

        # Collect IDs to remove: orphaned AI messages + dangling tool messages
        to_remove: list[str] = []
        ai_call_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", None) or []:
                    ai_call_ids.add(tc.get("id", ""))

        for msg in reversed(messages):
            msg_id = getattr(msg, "id", None)
            if not msg_id:
                continue
            if isinstance(msg, AIMessage):
                tcs = getattr(msg, "tool_calls", None) or []
                orphans = [tc for tc in tcs if tc.get("id", "") not in responded_ids]
                if orphans:
                    to_remove.append(msg_id)
            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id not in ai_call_ids:
                    to_remove.append(msg_id)

        if not to_remove:
            return False

        remove_ops = [RemoveMessage(id=mid) for mid in to_remove]
        await app.aupdate_state(config, {"messages": remove_ops})
        logger.info("Cleaned %d corrupted message(s) from checkpoint", len(to_remove))
        return True
    except Exception:
        logger.warning("Auto-clean corrupted messages failed", exc_info=True)
        return False


async def invoke_with_retry(
    app: Any,
    input_data: dict,
    config: dict,
    on_retry: Callable[[str, int], Awaitable[None]] | None = None,
) -> Any:
    """Invoke the graph with automatic retry for transient errors.

    Args:
        app: Compiled LangGraph app.
        input_data: {"messages": [...]}.
        config: LangGraph config with thread_id etc.
        on_retry: Optional async callback(message, attempt) for UI feedback.

    Returns:
        The graph invocation result.

    Raises:
        The original exception if all retries are exhausted.
    """
    last_error: BaseException | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await app.ainvoke(input_data, config=config)
        except Exception as e:
            category = _classify_error(e)
            last_error = e

            if category == "orphaned_tools":
                # Repair checkpoint and retry once — don't count as a retry attempt
                if on_retry:
                    await on_retry("Repairing orphaned tool calls in checkpoint...", attempt + 1)
                repaired = await auto_repair_orphaned_tools(app, config)
                if repaired:
                    try:
                        return await app.ainvoke(input_data, config=config)
                    except Exception as e2:
                        # If repair didn't fix it, try cleaning the bad messages
                        category2 = _classify_error(e2)
                        if category2 == "orphaned_tools":
                            # Still broken — remove the offending AI message entirely
                            if on_retry:
                                await on_retry("Removing corrupted messages from checkpoint...", attempt + 1)
                            cleaned = await auto_clean_corrupted(app, config)
                            if cleaned:
                                return await app.ainvoke(input_data, config=config)
                        raise e2
                raise  # repair found nothing, re-raise original

            if category == "context_overflow" and attempt == 0:
                if on_retry:
                    await on_retry("Auto-compacting context...", attempt + 1)
                compacted = await auto_compact(app, config)
                if compacted:
                    try:
                        return await app.ainvoke(input_data, config=config)
                    except Exception as e2:
                        raise e2
                raise  # compact failed, give up

            if category == "model_invalid" and attempt < _MAX_RETRIES:
                # Bedrock inference profiles return "model identifier is
                # invalid" for oversized payloads or transient issues.
                # First try: auto-compact to reduce context size.
                if attempt == 0:
                    if on_retry:
                        await on_retry("Model error — auto-compacting context...", attempt + 1)
                    compacted = await auto_compact(app, config)
                    if compacted:
                        try:
                            return await app.ainvoke(input_data, config=config)
                        except Exception as e2:
                            last_error = e2
                            # Fall through to retry with client reset
                # Subsequent tries: reset client + backoff
                try:
                    from octo.models import reset_bedrock_client
                    reset_bedrock_client()
                except ImportError:
                    pass
                delay = _BACKOFF_TIMEOUT[min(attempt, len(_BACKOFF_TIMEOUT) - 1)]
                if on_retry:
                    await on_retry(f"Model error, retrying in {delay}s ({attempt + 1}/{_MAX_RETRIES})...", attempt + 1)
                await asyncio.sleep(delay)
                continue

            if category == "connection_closed" and attempt < _MAX_RETRIES:
                # Connection dropped — reset the cached Bedrock client so
                # the next attempt creates a fresh connection.
                try:
                    from octo.models import reset_bedrock_client
                    reset_bedrock_client()
                except ImportError:
                    pass
                delay = _BACKOFF_TIMEOUT[min(attempt, len(_BACKOFF_TIMEOUT) - 1)]
                if on_retry:
                    await on_retry(f"Connection lost, reconnecting in {delay}s ({attempt + 1}/{_MAX_RETRIES})...", attempt + 1)
                await asyncio.sleep(delay)
                continue

            if category == "timeout" and attempt < _MAX_RETRIES:
                delay = _BACKOFF_TIMEOUT[min(attempt, len(_BACKOFF_TIMEOUT) - 1)]
                # On second timeout failure, try compacting — likely context-size induced
                if attempt == 1:
                    if on_retry:
                        await on_retry("Repeated timeouts — auto-compacting context...", attempt + 1)
                    compacted = await auto_compact(app, config)
                    if compacted:
                        logger.info("Auto-compacted after repeated timeouts, retrying...")
                    else:
                        if on_retry:
                            await on_retry(f"Timed out, retrying in {delay}s ({attempt + 1}/{_MAX_RETRIES})...", attempt + 1)
                        await asyncio.sleep(delay)
                else:
                    if on_retry:
                        await on_retry(f"Timed out, retrying in {delay}s ({attempt + 1}/{_MAX_RETRIES})...", attempt + 1)
                    await asyncio.sleep(delay)
                continue

            if category == "rate_limit" and attempt < _MAX_RETRIES:
                delay = _BACKOFF_RATE_LIMIT[min(attempt, len(_BACKOFF_RATE_LIMIT) - 1)]
                if on_retry:
                    await on_retry(f"Rate limited, waiting {delay}s ({attempt + 1}/{_MAX_RETRIES})...", attempt + 1)
                await asyncio.sleep(delay)
                continue

            # Unknown error or retries exhausted
            raise

    # Should not reach here, but just in case
    if last_error:
        raise last_error
