"""OctoEngine — embeddable agent engine.

Build once per project/configuration, invoke per message.
This is the main entry point for embedding Octo in other services.

Usage::

    from octo.core import OctoEngine, OctoConfig
    from octo.core.storage import FilesystemStorage

    config = OctoConfig(
        llm_provider="anthropic",
        llm_credentials={"api_key": "sk-..."},
        storage=FilesystemStorage(root="/path/to/.octo"),
    )
    engine = OctoEngine(config)
    response = await engine.invoke("Hello!", thread_id="conv-123")

Thread Safety:
    OctoEngine is NOT thread-safe. Each OctoEngine instance mutates global
    module state during _ensure_built() (see _builder.py). Do NOT share
    engine instances across threads, and do NOT create multiple engines
    with different configs in the same process simultaneously.

    For multi-tenant scenarios, use one engine per process/worker,
    or wait for the builder refactor that eliminates global state mutation.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class OctoResponse:
    """Response from an engine invocation."""
    content: str
    thread_id: str
    context_tokens_used: int = 0
    context_tokens_limit: int = 200_000
    agent_name: str = ""  # which agent produced the final response
    error: str | None = None  # non-None if invocation failed
    error_traceback: str | None = None  # full traceback for debugging


class OctoEngineError(Exception):
    """Raised when OctoEngine encounters a non-recoverable error."""


class OctoEngine:
    """Embeddable Octo agent engine.

    Build once per project/config. Each invoke() call loads the checkpoint
    for the given thread, runs the LangGraph supervisor, and saves state.

    Args:
        config: OctoConfig with all engine parameters.
        validate: If True (default), validate config on construction.
    """

    def __init__(self, config: Any, *, validate: bool = True) -> None:
        from octo.core.config import OctoConfig
        if not isinstance(config, OctoConfig):
            raise TypeError(f"Expected OctoConfig, got {type(config).__name__}")

        if validate:
            config.validate()

        self.config = config
        self._app: Any = None  # compiled LangGraph app
        self._checkpointer: Any = None
        self._built = False

    async def _ensure_built(self) -> None:
        """Lazy-build the graph on first invocation.

        Raises OctoEngineError if the graph fails to build.
        """
        if self._built:
            return

        try:
            from octo.core._builder import build_engine_graph
            self._app, self._checkpointer = await build_engine_graph(self.config)
            self._built = True
        except Exception as e:
            raise OctoEngineError(f"Failed to build engine graph: {e}") from e

    async def invoke(
        self,
        message: str,
        *,
        thread_id: str = "default",
        user_id: str | None = None,
    ) -> OctoResponse:
        """Process one message. Loads checkpoint, runs graph, saves checkpoint.

        Args:
            message: The user's message text.
            thread_id: Conversation identifier (maps to checkpoint thread).
            user_id: Optional user identifier for multi-user scenarios.

        Returns:
            OctoResponse with the assistant's reply.
            If an error occurs, response.error will be set (content may be empty).
        """
        try:
            await self._ensure_built()
        except OctoEngineError as e:
            return OctoResponse(
                content="",
                thread_id=thread_id,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )

        try:
            from langchain_core.messages import HumanMessage

            config = {"configurable": {"thread_id": thread_id}}
            input_data = {"messages": [HumanMessage(content=message)]}

            result = await self._app.ainvoke(input_data, config=config)

            # Extract the last AI message
            messages = result.get("messages", [])
            last_ai = ""
            agent_name = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
                    last_ai = msg.content if isinstance(msg.content, str) else str(msg.content)
                    agent_name = getattr(msg, "name", "") or ""
                    break

            return OctoResponse(
                content=last_ai,
                thread_id=thread_id,
                agent_name=agent_name,
            )
        except Exception as e:
            logger.error("OctoEngine.invoke() failed: %s", e, exc_info=True)
            return OctoResponse(
                content="",
                thread_id=thread_id,
                error=f"Invocation failed: {e}",
                error_traceback=traceback.format_exc(),
            )

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
    ) -> AsyncIterator[dict]:
        """Stream response events.

        Yields dicts with event data (tokens, tool calls, agent switches).
        Raises OctoEngineError if the graph fails to build.
        """
        await self._ensure_built()

        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"messages": [HumanMessage(content=message)]}

        async for event in self._app.astream_events(input_data, config=config, version="v2"):
            yield event

    @property
    def is_built(self) -> bool:
        """Whether the engine graph has been built."""
        return self._built

    async def close(self) -> None:
        """Clean up resources (close DB connections, etc.)."""
        if self._checkpointer and hasattr(self._checkpointer, "conn"):
            try:
                await self._checkpointer.conn.close()
            except Exception:
                pass
        self._built = False
        self._app = None
        self._checkpointer = None

    def __repr__(self) -> str:
        return (
            f"OctoEngine(provider={self.config.llm_provider!r}, "
            f"model={self.config.default_model!r}, "
            f"built={self._built})"
        )
