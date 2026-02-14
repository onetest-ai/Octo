"""Memory tools — daily logs and long-term memory.

Extracted from graph.py for reuse across engine configurations.

Two usage modes:

1. **CLI mode** (default): Module-level ``MEMORY_TOOLS`` list uses
   ``octo.config`` paths and sync filesystem I/O.  No extra setup.

2. **Engine mode**: Call ``make_memory_tools(storage)`` with a
   ``StorageBackend`` to get async tools that work with any backend
   (local filesystem, S3, etc.).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from octo.core.storage.base import StorageBackend


# ---------------------------------------------------------------------------
# Factory: create tools backed by a StorageBackend (engine mode)
# ---------------------------------------------------------------------------

def make_memory_tools(storage: StorageBackend) -> list:
    """Create memory tools that use the given storage backend.

    Returns a list of LangChain tools: [write_memory, read_memories,
    update_long_term_memory].
    """

    @tool
    async def write_memory(content: str) -> str:
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
        today = date.today().isoformat()
        path = f"memory/{today}.md"
        now = datetime.now(timezone.utc).strftime("%H:%M")

        if await storage.exists(path):
            existing = await storage.read(path)
        else:
            existing = f"# {today}\n"

        entry = f"\n- [{now}] {content}"
        await storage.write(path, existing + entry + "\n")
        return f"Memory saved to {today}.md"

    @tool
    async def read_memories(days: int = 3) -> str:
        """Read recent daily memory files.

        Returns the contents of the last N days of memory files,
        most recent first. Useful for recalling recent context.

        Args:
            days: Number of days to look back (default 3).
        """
        today = date.today()
        entries = []

        for i in range(days):
            d = today - timedelta(days=i)
            path = f"memory/{d.isoformat()}.md"
            if await storage.exists(path):
                text = await storage.read(path)
                entries.append(text.strip())

        if not entries:
            return "No recent memories found."
        return "\n\n---\n\n".join(entries)

    @tool
    async def update_long_term_memory(content: str) -> str:
        """Replace the long-term memory file (.octo/persona/MEMORY.md).

        This is your curated, persistent memory — distilled insights, not raw logs.
        Call this to update what you remember about the user, yourself, projects,
        preferences, and lessons learned.

        Pass the FULL new content (this replaces the file entirely).
        Read the current content first with the Read tool, then modify and save.

        Args:
            content: The complete new content for MEMORY.md.
        """
        await storage.write("persona/MEMORY.md", content + "\n")
        lines = content.strip().split("\n")
        return f"MEMORY.md updated ({len(lines)} lines)"

    return [write_memory, read_memories, update_long_term_memory]


# ---------------------------------------------------------------------------
# Module-level tools: CLI mode (backward compat, uses octo.config paths)
# ---------------------------------------------------------------------------

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

    lines = content.strip().split("\n")
    return f"MEMORY.md updated ({len(lines)} lines)"


MEMORY_TOOLS = [write_memory, read_memories, update_long_term_memory]
