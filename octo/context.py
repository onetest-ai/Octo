"""System prompt composer — reads persona files and memory."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from octo.config import PERSONA_DIR, MEMORY_DIR, STATE_PATH


def _read_if_exists(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def build_system_prompt() -> str:
    """Compose the supervisor system prompt from .octo/persona/ files."""
    parts: list[str] = []

    # Current date and time — helps with search relevance and time-aware responses
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    parts.append(
        f"# Current Date & Time\n\n"
        f"- **Today**: {now.strftime('%A, %B %d, %Y')}\n"
        f"- **UTC**: {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"- **Local**: {local_now.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Use the current year ({now.year}) when searching for recent information."
    )

    # Core identity
    for name in ("SOUL.md", "IDENTITY.md", "USER.md", "AGENTS.md"):
        text = _read_if_exists(PERSONA_DIR / name)
        if text:
            parts.append(text)

    # Today's memory
    today = date.today().isoformat()
    mem = _read_if_exists(MEMORY_DIR / f"{today}.md")
    if mem:
        parts.append(f"# Today's Memory ({today})\n\n{mem}")

    # Long-term memory
    ltm = _read_if_exists(PERSONA_DIR / "MEMORY.md")
    if ltm:
        parts.append(f"# Long-Term Memory\n\n{ltm}")

    # Project state (persists across sessions)
    state = _read_if_exists(STATE_PATH)
    if state:
        parts.append(f"# Current Project State\n\n{state}")

    return "\n\n---\n\n".join(parts)
