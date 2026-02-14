"""Planning tools — todo list and project state management.

Extracted from graph.py for reuse across engine configurations.

Two usage modes:

1. **CLI mode** (default): Module-level ``PLANNING_TOOLS`` list uses
   ``octo.config`` paths and sync filesystem I/O.  No extra setup.

2. **Engine mode**: Call ``make_planning_tools(storage)`` with a
   ``StorageBackend`` to get async tools that work with any backend.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from octo.core.storage.base import StorageBackend

_MAX_TODOS = 20  # auto-prune completed items beyond this


def _prune_todos(todos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Auto-prune completed items to prevent infinite growth."""
    if len(todos) <= _MAX_TODOS:
        return todos
    pending = [t for t in todos if t.get("status") != "completed"]
    completed = [t for t in todos if t.get("status") == "completed"]
    keep_completed = max(0, _MAX_TODOS - len(pending))
    if keep_completed:
        return completed[-keep_completed:] + pending
    return pending[-_MAX_TODOS:]


# ---------------------------------------------------------------------------
# Factory: create tools backed by a StorageBackend (engine mode)
# ---------------------------------------------------------------------------

def make_planning_tools(storage: StorageBackend) -> list:
    """Create planning tools that use the given storage backend.

    Returns a list of LangChain tools: [write_todos, read_todos,
    update_state_md].
    """
    _todos: list[dict[str, str]] = []
    _loaded = False

    async def _ensure_loaded() -> None:
        nonlocal _todos, _loaded
        if _loaded:
            return
        # Load most recent plan file
        plan_files = sorted(await storage.glob("plans/plan_*.json"), reverse=True)
        for pf in plan_files:
            try:
                data = json.loads(await storage.read(pf))
                if isinstance(data, list):
                    _todos = data
                    break
            except (json.JSONDecodeError, TypeError, FileNotFoundError):
                continue
        _loaded = True

    @tool
    async def write_todos(todos: list[dict[str, str]]) -> str:
        """Write or update the task plan. Each todo is {task, status}.
        Status can be: pending, in_progress, completed.
        Use this to break complex tasks into steps before executing.

        IMPORTANT: When starting a NEW task, write only the new plan items.
        Do NOT carry over completed items from previous tasks — they are
        auto-archived. Only include items relevant to the current task."""
        nonlocal _todos
        await _ensure_loaded()

        todos = _prune_todos(todos)
        _todos = todos

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        await storage.write(
            f"plans/plan_{ts}.json",
            json.dumps(todos, indent=2, ensure_ascii=False) + "\n",
        )
        return f"Updated plan with {len(todos)} tasks."

    @tool
    async def read_todos() -> str:
        """Read the current task plan."""
        await _ensure_loaded()

        if not _todos:
            return "No active plan."

        completed = sum(1 for t in _todos if t.get("status") == "completed")
        active = len(_todos) - completed
        result = json.dumps(_todos, indent=2, ensure_ascii=False)

        if active == 0 and completed > 0:
            result += (
                "\n\n[All tasks completed. For the next task, call write_todos "
                "with ONLY the new plan items — do not carry these over.]"
            )
        return result

    @tool
    async def update_state_md(
        current_position: str = "",
        active_plan: str = "",
        recent_decisions: str = "",
        session_continuity: str = "",
    ) -> str:
        """Update .octo/STATE.md with current project state.
        Call this after completing significant work or before ending a session.

        Args:
            current_position: Brief description of where we are (phase, status).
            active_plan: Current active plan or goals.
            recent_decisions: Key decisions made in this session.
            session_continuity: Notes for resuming (stopped at, next steps).
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sections = [f"# Project State\n\n_Last updated: {now}_"]

        if current_position:
            sections.append(f"## Current Position\n{current_position}")
        if active_plan:
            sections.append(f"## Active Plan\n{active_plan}")
        if recent_decisions:
            sections.append(f"## Recent Decisions\n{recent_decisions}")
        if session_continuity:
            sections.append(f"## Session Continuity\n{session_continuity}")

        await storage.write("STATE.md", "\n\n".join(sections) + "\n")
        return f"STATE.md updated at {now}"

    return [write_todos, read_todos, update_state_md]


# ---------------------------------------------------------------------------
# Module-level tools: CLI mode (backward compat, uses octo.config paths)
# ---------------------------------------------------------------------------

def _load_todos_from_disk() -> list[dict[str, str]]:
    """Load the most recent plan from .octo/plans/plan_*.json."""
    from octo.config import OCTO_DIR, PLANS_DIR

    plan_files = sorted(PLANS_DIR.glob("plan_*.json"), reverse=True)
    for pf in plan_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    # Migration: check legacy .octo/plan.json
    legacy = OCTO_DIR / "plan.json"
    if legacy.is_file():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _save_todos_to_disk(todos: list[dict[str, str]]) -> None:
    """Persist todos to .octo/plans/plan_<datetime>.json."""
    from octo.config import PLANS_DIR

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = PLANS_DIR / f"plan_{ts}.json"
    path.write_text(json.dumps(todos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_todos: list[dict[str, str]] = _load_todos_from_disk()


@tool
def write_todos(todos: list[dict[str, str]]) -> str:
    """Write or update the task plan. Each todo is {task, status}.
    Status can be: pending, in_progress, completed.
    Use this to break complex tasks into steps before executing.

    IMPORTANT: When starting a NEW task, write only the new plan items.
    Do NOT carry over completed items from previous tasks — they are
    auto-archived. Only include items relevant to the current task."""
    global _todos

    todos = _prune_todos(todos)
    _todos = todos
    _save_todos_to_disk(todos)
    return f"Updated plan with {len(todos)} tasks."


@tool
def read_todos() -> str:
    """Read the current task plan."""
    if not _todos:
        return "No active plan."

    completed = sum(1 for t in _todos if t.get("status") == "completed")
    active = len(_todos) - completed
    result = json.dumps(_todos, indent=2, ensure_ascii=False)

    if active == 0 and completed > 0:
        result += (
            "\n\n[All tasks completed. For the next task, call write_todos "
            "with ONLY the new plan items — do not carry these over.]"
        )
    return result


@tool
def update_state_md(
    current_position: str = "",
    active_plan: str = "",
    recent_decisions: str = "",
    session_continuity: str = "",
) -> str:
    """Update .octo/STATE.md with current project state.
    Call this after completing significant work or before ending a session.

    Args:
        current_position: Brief description of where we are (phase, status).
        active_plan: Current active plan or goals.
        recent_decisions: Key decisions made in this session.
        session_continuity: Notes for resuming (stopped at, next steps).
    """
    from octo.config import STATE_PATH

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [f"# Project State\n\n_Last updated: {now}_"]

    if current_position:
        sections.append(f"## Current Position\n{current_position}")
    if active_plan:
        sections.append(f"## Active Plan\n{active_plan}")
    if recent_decisions:
        sections.append(f"## Recent Decisions\n{recent_decisions}")
    if session_continuity:
        sections.append(f"## Session Continuity\n{session_continuity}")

    STATE_PATH.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return f"STATE.md updated at {now}"


PLANNING_TOOLS = [write_todos, read_todos, update_state_md]
