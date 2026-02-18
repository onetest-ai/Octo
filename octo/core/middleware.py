"""Custom middleware and error handling for Octo agents.

Provides:
- ``explain_error()`` — reusable async helper that asks a cheap LLM to explain
  any error with context and suggest remediation steps.
- ``ToolErrorMiddleware``  — catches tool execution errors and explains them
  via a low-tier LLM instead of crashing the agent.
- ``ToolResultLimitMiddleware`` — truncates oversized tool results before they
  enter the agent state, advising the model to paginate.
- ``build_summarization_middleware()`` — factory that returns the official
  LangChain ``SummarizationMiddleware`` pre-configured for Octo.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reusable error explainer — asks a cheap LLM to diagnose any error
# ---------------------------------------------------------------------------

_ERROR_EXPLAINER_PROMPT = """\
You are a concise technical troubleshooter for the Octo CLI assistant.

An error occurred during: {context}

Error type: {error_type}
Error message:
{error_message}

Full traceback (if available):
{traceback}

Configuration / environment details:
{details}

Instructions:
1. Explain what went wrong in 1-2 sentences (be specific, not generic).
2. List 1-3 concrete steps the user can take to fix it.
3. If this looks like a transient issue, say so.
Keep it under 100 words total. No markdown headers. Plain text only."""

# Lazily initialised shared model for error explanations
_error_model = None


def _get_error_model():
    global _error_model
    if _error_model is None:
        from octo.models import make_model
        _error_model = make_model(tier="low")
    return _error_model


async def explain_error(
    error: BaseException,
    *,
    context: str = "unknown operation",
    details: str = "",
) -> str:
    """Ask a cheap LLM to explain an error and suggest fixes.

    Parameters
    ----------
    error:
        The exception that was caught.
    context:
        Human-readable description of what was happening when the error
        occurred (e.g. "connecting to MCP server 'outlook'").
    details:
        Extra context like config snippets, env vars, or command args
        that help the LLM diagnose the issue.

    Returns a short plain-text explanation suitable for printing to
    the user.  Falls back to the raw error string if the LLM call
    itself fails.
    """
    import traceback as tb_mod

    tb_str = "".join(tb_mod.format_exception(type(error), error, error.__traceback__))
    # Truncate long tracebacks / details to avoid blowing context
    tb_str = tb_str[-2000:] if len(tb_str) > 2000 else tb_str
    details = details[-1000:] if len(details) > 1000 else details

    prompt = _ERROR_EXPLAINER_PROMPT.format(
        context=context,
        error_type=type(error).__name__,
        error_message=str(error)[:500],
        traceback=tb_str or "(not available)",
        details=details or "(none)",
    )
    try:
        model = _get_error_model()
        response = await model.ainvoke(prompt)
        return response.content
    except Exception:
        logger.debug("Error explainer LLM failed, returning raw error", exc_info=True)
        return f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Tool error middleware
# ---------------------------------------------------------------------------

# Prompt template for the error-explaining LLM
ERROR_EXPLANATION_PROMPT = """\
A tool call failed during agent execution. Analyse the error and explain \
what went wrong in 1-2 concise sentences. Be specific about the root cause \
and, if possible, suggest how to fix it.

Tool name: {tool_name}
Tool input: {tool_input}
Error type: {error_type}
Error message: {error_message}
"""


class ToolErrorMiddleware(AgentMiddleware):
    """Catch tool execution errors and return an LLM-generated explanation.

    Instead of letting a ToolException propagate and crash the agent loop,
    this middleware catches the error, asks a cheap low-tier model to explain
    what happened, and returns the explanation as a regular ToolMessage so
    the agent can recover or inform the user.

    Implements both sync ``wrap_tool_call`` and async ``awrap_tool_call``
    so that errors from async tools (e.g. MCP/Playwright) are also caught.

    Usage::

        from octo.middleware import ToolErrorMiddleware

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[ToolErrorMiddleware()],
        )
    """

    def __init__(self, model=None):
        self._model = model

    # -- lazy model init (avoids import-time side effects) --

    def _get_model(self):
        if self._model is None:
            from octo.models import make_model
            self._model = make_model(tier="low")
        return self._model

    # -- sync middleware hook --

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap every sync tool call with error handling."""
        try:
            return handler(request)
        except Exception as e:
            return self._handle_error(request, e)

    # -- async middleware hook --

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Wrap every async tool call with error handling."""
        try:
            return await handler(request)
        except Exception as e:
            return self._handle_error(request, e)

    # -- shared error handling --

    def _handle_error(self, request: ToolCallRequest, e: Exception) -> ToolMessage:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})
        logger.warning("Tool %s failed: %s: %s", tool_name, type(e).__name__, e)

        # Fast path: detect common Bedrock truncation issue where content
        # parameter gets cut off.  No LLM call needed — return actionable hint.
        fast = self._fast_diagnose(tool_name, tool_args, e)
        if fast:
            return ToolMessage(
                content=fast,
                tool_call_id=request.tool_call["id"],
            )

        explanation = self._explain(tool_name, tool_args, e)

        return ToolMessage(
            content=explanation,
            tool_call_id=request.tool_call["id"],
        )

    @staticmethod
    def _fast_diagnose(tool_name: str, tool_args: dict, e: Exception) -> str | None:
        """Return a canned fix for well-known error patterns (no LLM needed)."""
        # write_file with missing content — Bedrock output truncation
        if tool_name == "write_file" and "content" in str(e) and "Field required" in str(e):
            return (
                f"[Tool error] write_file failed: the 'content' parameter is missing "
                f"— this usually happens when the model output was truncated "
                f"(common with Bedrock). "
                f"Fix: write the content in SMALLER CHUNKS. "
                f"Split into multiple write_file calls of ~2000 chars each. "
                f"Use 'edit_file' to append subsequent chunks."
            )

        return None

    # -- error explanation via low-tier LLM --

    def _explain(self, tool_name: str, tool_input, error: Exception) -> str:
        """Ask a low-tier model to explain the tool error."""
        prompt = ERROR_EXPLANATION_PROMPT.format(
            tool_name=tool_name,
            tool_input=str(tool_input)[:500],
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        )
        try:
            model = self._get_model()
            response = model.invoke(prompt)
            return f"[Tool error] {response.content}"
        except Exception:
            # If the explanation model itself fails, return the raw error
            return f"[Tool error] {tool_name} failed: {type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Tool result size limiter — truncates oversized results before state entry
# ---------------------------------------------------------------------------

TRUNCATION_NOTICE = (
    "\n\n... [Result truncated from {original:,} to {limit:,} chars. "
    "Use pagination, more specific queries, or request smaller chunks "
    "to get the remaining data.]"
)


class ToolResultLimitMiddleware(AgentMiddleware):
    """Truncate tool results that exceed a character limit.

    Applied *after* the tool executes but *before* the result enters agent
    state.  When a result is too large, it is cut to ``max_chars`` and a
    notice is appended telling the model to paginate or narrow its query.

    This prevents a single huge tool result (e.g. search_issues returning
    300K+ chars) from blowing the context window.  The ``pre_model_hook``
    in the supervisor acts as a second safety-net, but catching the problem
    here — at the middleware level — is cheaper and preserves more useful
    context for the agent.

    The default limit is read from ``TOOL_RESULT_LIMIT`` in config (env var),
    and can be overridden per-instance via the ``max_chars`` constructor arg.

    Usage::

        from octo.middleware import ToolResultLimitMiddleware

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[ToolResultLimitMiddleware()],
        )
    """

    # Tool name substrings that signal "structured data — don't blindly truncate".
    # Matched against the resolved tool name (after MCP proxy unwrap).
    # These are substrings, not exact names, so they work regardless of MCP
    # server naming (e.g., "browser_snapshot", "playwright__browser_snapshot").
    STRUCTURED_TOOL_PATTERNS: tuple[str, ...] = (
        "screenshot", "snapshot", "console_messages", "network_requests",
        "page_content", "dom_", "element",
    )

    # Content markers that indicate structured/reference data that loses
    # meaning when truncated (e.g., element refs, base64 images, JSON-RPC).
    STRUCTURED_CONTENT_MARKERS: tuple[str, ...] = (
        '"ref":', '"selector":', '"base64":', '"data:image/',
        '"accessibilityName":', '"role":', '"children":',
    )

    def __init__(self, max_chars: int | None = None):
        if max_chars is None:
            from octo.config import TOOL_RESULT_LIMIT
            max_chars = TOOL_RESULT_LIMIT
        self.max_chars = max_chars

    def _resolve_tool_name(self, request: ToolCallRequest) -> str:
        """Get the actual tool name, unwrapping MCP proxy if needed."""
        tool_name = request.tool_call.get("name", "")
        if tool_name == "call_mcp_tool":
            return request.tool_call.get("args", {}).get("tool_name", "")
        return tool_name

    def _is_structured_tool(self, tool_name: str) -> bool:
        """Check if the tool name matches known structured-data patterns."""
        lower = tool_name.lower()
        return any(p in lower for p in self.STRUCTURED_TOOL_PATTERNS)

    def _has_structured_content(self, content: Any) -> bool:
        """Detect structured content by checking for reference markers.

        A quick heuristic: if the first 2000 chars contain JSON-like
        structural markers (element refs, selectors, base64), treat
        the entire result as structured data.
        """
        if isinstance(content, str):
            sample = content[:2000]
            return any(m in sample for m in self.STRUCTURED_CONTENT_MARKERS)
        if isinstance(content, list):
            sample = str(content[:3])[:2000]
            return any(m in sample for m in self.STRUCTURED_CONTENT_MARKERS)
        return False

    def _should_skip(self, request: ToolCallRequest, result: Any = None) -> bool:
        """Decide if this tool result should bypass truncation.

        Checks both the tool name (pattern match) and the content itself
        (structural markers). This way new MCP servers with structured
        output are handled automatically without hardcoding names.
        """
        tool_name = self._resolve_tool_name(request)
        if self._is_structured_tool(tool_name):
            return True
        # Check content for structural markers
        if result is not None:
            content = getattr(result, "content", None) if hasattr(result, "content") else result
            if content and self._has_structured_content(content):
                return True
        return False

    # -- sync --

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if self._should_skip(request, result):
            return result
        return self._maybe_truncate(result)

    # -- async --

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if self._should_skip(request, result):
            return result
        return self._maybe_truncate(result)

    # -- truncation logic --

    def _truncate_content(self, content) -> tuple[Any, bool]:
        """Truncate content if oversized. Returns (content, was_truncated)."""
        if isinstance(content, str) and len(content) > self.max_chars:
            notice = TRUNCATION_NOTICE.format(
                original=len(content), limit=self.max_chars,
            )
            return content[: self.max_chars] + notice, True

        if isinstance(content, list):
            total = sum(len(str(p)) for p in content)
            if total > self.max_chars:
                truncated_parts = []
                running = 0
                for part in content:
                    part_len = len(str(part))
                    if running + part_len > self.max_chars:
                        break
                    truncated_parts.append(part)
                    running += part_len
                notice = TRUNCATION_NOTICE.format(
                    original=total, limit=self.max_chars,
                )
                truncated_parts.append(notice)
                return truncated_parts, True

        return content, False

    def _truncate_command_messages(self, cmd: Command) -> Command:
        """Truncate oversized messages inside a Command (worker transfer).

        When a worker agent transfers back to the supervisor via
        transfer_to_octo, the Command contains ALL worker messages.
        Without truncation, huge tool results (175K+ chars) blow the
        supervisor's context.
        """
        update = cmd.update
        if not update or "messages" not in update:
            return cmd

        messages = update["messages"]
        if not messages:
            return cmd

        modified = False
        new_messages = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if content is not None:
                truncated, changed = self._truncate_content(content)
                if changed:
                    msg = msg.model_copy(update={"content": truncated})
                    modified = True
            new_messages.append(msg)

        if not modified:
            return cmd

        total_before = sum(len(str(getattr(m, "content", ""))) for m in messages)
        total_after = sum(len(str(getattr(m, "content", ""))) for m in new_messages)
        logger.info(
            "Truncated transfer messages: %d -> %d chars (%d messages)",
            total_before, total_after, len(new_messages),
        )

        new_update = dict(update)
        new_update["messages"] = new_messages
        return Command(graph=cmd.graph, update=new_update)

    def _maybe_truncate(self, result: ToolMessage | Command) -> ToolMessage | Command:
        if isinstance(result, Command):
            return self._truncate_command_messages(result)

        content = result.content
        truncated, changed = self._truncate_content(content)
        if changed:
            logger.info(
                "Truncated tool result: %d -> %d chars",
                len(str(content)), len(str(truncated)),
            )
            return ToolMessage(
                content=truncated,
                tool_call_id=result.tool_call_id,
            )

        return result


# ---------------------------------------------------------------------------
# Summarization middleware — wraps the official LangChain implementation
# ---------------------------------------------------------------------------

def build_summarization_middleware(
    trigger_tokens: int | None = None,
    keep_tokens: int | None = None,
):
    """Create a ``SummarizationMiddleware`` configured for Octo.

    Uses a low-tier model for cheap summarization.  All thresholds are
    token-based (not message-count) since individual messages can be huge.

    Args:
        trigger_tokens: Override token threshold for triggering compaction.
            Defaults to ``SUMMARIZATION_TRIGGER_TOKENS`` from config (40K).
        keep_tokens: Override how many tokens to keep after compaction.
            Defaults to ``SUMMARIZATION_KEEP_TOKENS`` from config (8K).

    Returns:
        A ``SummarizationMiddleware`` instance ready to be passed to
        ``create_agent(middleware=[...])``.
    """
    from langchain.agents.middleware import SummarizationMiddleware

    from octo.config import (
        SUMMARIZATION_KEEP_TOKENS,
        SUMMARIZATION_TRIGGER_TOKENS,
    )
    from octo.models import make_model

    return SummarizationMiddleware(
        model=make_model(tier="low"),
        trigger=("tokens", trigger_tokens or SUMMARIZATION_TRIGGER_TOKENS),
        keep=("tokens", keep_tokens or SUMMARIZATION_KEEP_TOKENS),
    )


# ---------------------------------------------------------------------------
# Prompt caching — Bedrock (cachePoint)
# ---------------------------------------------------------------------------

class BedrockCachingMiddleware(AgentMiddleware):
    """Add Bedrock prompt caching (cachePoint) to system messages.

    Bedrock Converse API supports prompt caching via ``cachePoint`` blocks
    in the system parameter.  ``_lc_content_to_bedrock()`` passes through
    dict blocks without a top-level ``"type"`` key as-is, so
    ``{"cachePoint": {"type": "default"}}`` flows through to the API.

    For non-Bedrock models, this middleware is a no-op.
    """

    def _apply_caching(self, request):
        """Return modified request if Bedrock, else None."""
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError:
            return None
        if not isinstance(request.model, ChatBedrockConverse):
            return None
        sys_msg = request.system_message
        if sys_msg and isinstance(sys_msg.content, str):
            new_content = [
                {"type": "text", "text": sys_msg.content},
                {"cachePoint": {"type": "default"}},
            ]
            new_sys = sys_msg.model_copy(update={"content": new_content})
            return request.override(system_message=new_sys)
        return None

    def wrap_model_call(self, request, handler):
        updated = self._apply_caching(request)
        return handler(updated if updated else request)

    async def awrap_model_call(self, request, handler):
        updated = self._apply_caching(request)
        return await handler(updated if updated else request)
