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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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


class OctoEngine:
    """Embeddable Octo agent engine.

    Build once per project/config. Each invoke() call loads the checkpoint
    for the given thread, runs the LangGraph supervisor, and saves state.

    Args:
        config: OctoConfig with all engine parameters.
    """

    def __init__(self, config: Any) -> None:  # Any to avoid circular; expects OctoConfig
        from octo.core.config import OctoConfig
        if not isinstance(config, OctoConfig):
            raise TypeError(f"Expected OctoConfig, got {type(config).__name__}")

        self.config = config
        self._app: Any = None  # compiled LangGraph app
        self._checkpointer: Any = None
        self._built = False

    async def _ensure_built(self) -> None:
        """Lazy-build the graph on first invocation."""
        if self._built:
            return

        from octo.core._builder import build_engine_graph
        self._app, self._checkpointer = await build_engine_graph(self.config)
        self._built = True

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
        """
        await self._ensure_built()

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

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
    ) -> AsyncIterator[dict]:
        """Stream response events.

        Yields dicts with event data (tokens, tool calls, agent switches).
        """
        await self._ensure_built()

        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"messages": [HumanMessage(content=message)]}

        async for event in self._app.astream_events(input_data, config=config, version="v2"):
            yield event

    async def close(self) -> None:
        """Clean up resources (close DB connections, etc.)."""
        if self._checkpointer and hasattr(self._checkpointer, "conn"):
            try:
                await self._checkpointer.conn.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"OctoEngine(provider={self.config.llm_provider!r}, "
            f"model={self.config.default_model!r}, "
            f"built={self._built})"
        )
