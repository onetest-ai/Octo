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
        "timeout", "rate_limit", "context_overflow", or None (unknown).
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
    return None


async def auto_compact(app: Any, config: dict) -> bool:
    """Run automatic context compaction. Returns True if successful."""
    try:
        from langchain_core.messages import RemoveMessage, SystemMessage
        from langchain_core.messages.utils import count_tokens_approximately
        from octo.models import make_model

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

        # Summarize old messages
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

        remove_ops = [RemoveMessage(id=m.id) for m in removable]
        await app.aupdate_state(config, {"messages": remove_ops + [summary_msg]})

        before = len(messages)
        after = len(messages) - len(removable) + 1
        logger.info("Auto-compacted: %d -> %d messages", before, after)
        return True
    except Exception:
        logger.warning("Auto-compact failed", exc_info=True)
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
