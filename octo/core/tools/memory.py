"""Memory tools — daily logs and long-term memory.

Extracted from graph.py for reuse across engine configurations.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from langchain_core.tools import tool


@tool
def write_memory(content: str) -> str:
    """Append an entry to today's daily memory file (.octo/memory/YYYY-MM-DD.md).

    Use this whenever you learn something worth remembering:
    - User preferences or facts ("remember this")
    - Decisions made during a session
    - Lessons learned from mistakes
    - Important events or outcomes

    Each call appends a timestamped bullet to the daily file.

    Args:
        content: The memory entry to record (one line or a few lines).
    """
    from octo.config import MEMORY_DIR

    today = date.today().isoformat()
    path = MEMORY_DIR / f"{today}.md"
    now = datetime.now(timezone.utc).strftime("%H:%M")

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = f"# {today}\n"

    # Append timestamped entry
    entry = f"\n- [{now}] {content}"
    path.write_text(existing + entry + "\n", encoding="utf-8")
    return f"Memory saved to {today}.md"


@tool
def read_memories(days: int = 3) -> str:
    """Read recent daily memory files.

    Returns the contents of the last N days of memory files,
    most recent first. Useful for recalling recent context.

    Args:
        days: Number of days to look back (default 3).
    """
    from octo.config import MEMORY_DIR

    if not MEMORY_DIR.is_dir():
        return "No memories found."

    today = date.today()
    entries = []

    for i in range(days):
        d = today - timedelta(days=i)
        path = MEMORY_DIR / f"{d.isoformat()}.md"
        if path.is_file():
            entries.append(path.read_text(encoding="utf-8").strip())

    if not entries:
        return "No recent memories found."
    return "\n\n---\n\n".join(entries)


@tool
def update_long_term_memory(content: str) -> str:
    """Replace the long-term memory file (.octo/persona/MEMORY.md).

    This is your curated, persistent memory — distilled insights, not raw logs.
    Call this to update what you remember about the user, yourself, projects,
    preferences, and lessons learned.

    Pass the FULL new content (this replaces the file entirely).
    Read the current content first with the Read tool, then modify and save.

    Args:
        content: The complete new content for MEMORY.md.
    """
    from octo.config import PERSONA_DIR

    path = PERSONA_DIR / "MEMORY.md"
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")

    # Count approximate size for feedback
    lines = content.strip().split("\n")
    return f"MEMORY.md updated ({len(lines)} lines)"


MEMORY_TOOLS = [write_memory, read_memories, update_long_term_memory]
