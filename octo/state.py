"""LangGraph state definitions."""
from __future__ import annotations

from langgraph.graph import MessagesState


class OctoState(MessagesState):
    """Shared state for the Octi supervisor graph.

    Middleware (SummarizationMiddleware, TodoListMiddleware) manage
    summary and todos internally — we only extend with minimal fields.
    """
    active_agent: str = ""
