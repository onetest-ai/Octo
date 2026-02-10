"""Session registry — persists thread IDs for easy recovery.

Stored at .octo/sessions.json as a JSON array of session records.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from octo.config import OCTO_DIR

SESSIONS_PATH = OCTO_DIR / "sessions.json"


def _load_all() -> list[dict]:
    if not SESSIONS_PATH.exists():
        return []
    try:
        return json.loads(SESSIONS_PATH.read_text())
    except (json.JSONDecodeError, TypeError):
        return []


def _save_all(sessions: list[dict]) -> None:
    SESSIONS_PATH.write_text(json.dumps(sessions, indent=2) + "\n")


def save_session(
    thread_id: str,
    *,
    preview: str = "",
    model: str = "",
) -> None:
    """Create or update a session record."""
    sessions = _load_all()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    existing = next((s for s in sessions if s["thread_id"] == thread_id), None)
    if existing:
        existing["updated_at"] = now
        if preview:
            existing["preview"] = preview[:120]
        if model:
            existing["model"] = model
    else:
        sessions.append({
            "thread_id": thread_id,
            "created_at": now,
            "updated_at": now,
            "preview": preview[:120],
            "model": model,
        })

    _save_all(sessions)


def get_last_session() -> dict | None:
    """Return the most recently updated session, or None."""
    sessions = _load_all()
    if not sessions:
        return None
    return max(sessions, key=lambda s: s.get("updated_at", ""))


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent sessions, newest first."""
    sessions = _load_all()
    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions[:limit]
