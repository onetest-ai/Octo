"""Planning tools — todo list and project state management.

Extracted from graph.py for reuse across engine configurations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.tools import tool


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
_MAX_TODOS = 20  # auto-prune completed items beyond this


@tool
def write_todos(todos: list[dict[str, str]]) -> str:
    """Write or update the task plan. Each todo is {task, status}.
    Status can be: pending, in_progress, completed.
    Use this to break complex tasks into steps before executing.

    IMPORTANT: When starting a NEW task, write only the new plan items.
    Do NOT carry over completed items from previous tasks — they are
    auto-archived. Only include items relevant to the current task."""
    global _todos

    # Auto-prune: drop old completed items to prevent infinite growth.
    # Keep all pending/in_progress + most recent completed up to limit.
    if len(todos) > _MAX_TODOS:
        pending = [t for t in todos if t.get("status") != "completed"]
        completed = [t for t in todos if t.get("status") == "completed"]
        keep_completed = max(0, _MAX_TODOS - len(pending))
        todos = completed[-keep_completed:] + pending if keep_completed else pending[-_MAX_TODOS:]

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

    # Hint: if all items are completed, the plan is done — start fresh next time
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
