"""Custom middleware for Octo agents.

Provides:
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

        explanation = self._explain(tool_name, tool_args, e)

        return ToolMessage(
            content=explanation,
            tool_call_id=request.tool_call["id"],
        )

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

    def __init__(self, max_chars: int | None = None):
        if max_chars is None:
            from octo.config import TOOL_RESULT_LIMIT
            max_chars = TOOL_RESULT_LIMIT
        self.max_chars = max_chars

    # -- sync --

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        return self._maybe_truncate(result)

    # -- async --

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        return self._maybe_truncate(result)

    # -- truncation logic --

    def _maybe_truncate(self, result: ToolMessage | Command) -> ToolMessage | Command:
        if isinstance(result, Command):
            return result

        content = result.content

        if isinstance(content, str) and len(content) > self.max_chars:
            notice = TRUNCATION_NOTICE.format(
                original=len(content), limit=self.max_chars,
            )
            truncated = content[: self.max_chars] + notice
            logger.info(
                "Truncated tool result: %d -> %d chars",
                len(content), len(truncated),
            )
            return ToolMessage(
                content=truncated,
                tool_call_id=result.tool_call_id,
            )

        if isinstance(content, list):
            total = sum(len(str(p)) for p in content)
            if total > self.max_chars:
                # Truncate list-of-parts (structured content)
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
                logger.info(
                    "Truncated tool result (list): %d -> ~%d chars",
                    total, running,
                )
                return ToolMessage(
                    content=truncated_parts,
                    tool_call_id=result.tool_call_id,
                )

        return result


# ---------------------------------------------------------------------------
# Summarization middleware — wraps the official LangChain implementation
# ---------------------------------------------------------------------------

def build_summarization_middleware():
    """Create a ``SummarizationMiddleware`` configured for Octo.

    Uses a low-tier model for cheap summarization.  All thresholds are
    token-based (not message-count) since individual messages can be huge.

    Config (env vars via ``octo.config``):

    - ``SUMMARIZATION_TRIGGER_FRACTION`` (default 0.7) — context window %
    - ``SUMMARIZATION_TRIGGER_TOKENS``   (default 100000) — absolute token count
    - ``SUMMARIZATION_KEEP_TOKENS``      (default 20000) — tokens to keep after compaction

    Whichever trigger fires first activates summarization.

    Returns:
        A ``SummarizationMiddleware`` instance ready to be passed to
        ``create_agent(middleware=[...])``.
    """
    from langchain.agents.middleware import SummarizationMiddleware

    from octo.config import (
        SUMMARIZATION_KEEP_TOKENS,
        SUMMARIZATION_TRIGGER_FRACTION,
        SUMMARIZATION_TRIGGER_TOKENS,
    )
    from octo.models import make_model

    return SummarizationMiddleware(
        model=make_model(tier="low"),
        trigger=[
            ("fraction", SUMMARIZATION_TRIGGER_FRACTION),
            ("tokens", SUMMARIZATION_TRIGGER_TOKENS),
        ],
        keep=("tokens", SUMMARIZATION_KEEP_TOKENS),
    )
