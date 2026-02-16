"""System prompt composer — reads persona files and memory."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from octo.config import PERSONA_DIR, MEMORY_DIR, STATE_PATH, SYSTEM_PROMPT_BUDGET


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

    # Today's memory (capped to keep system prompt lean)
    today = date.today().isoformat()
    mem = _read_if_exists(MEMORY_DIR / f"{today}.md")
    if mem:
        _MEM_CAP = 3000
        if len(mem) > _MEM_CAP:
            mem = "[... earlier entries omitted ...]\n" + mem[-_MEM_CAP:]
        parts.append(f"# Today's Memory ({today})\n\n{mem}")

    # Long-term memory
    ltm = _read_if_exists(PERSONA_DIR / "MEMORY.md")
    if ltm:
        parts.append(f"# Long-Term Memory\n\n{ltm}")

    # Project state (persists across sessions)
    state = _read_if_exists(STATE_PATH)
    if state:
        parts.append(f"# Current Project State\n\n{state}")

    # Enforce budget: truncate lowest-priority sections first.
    # Parts order: [0]=date, [1..4]=identity files, [5+]=memory/state (lower priority)
    _SEP = "\n\n---\n\n"
    total = sum(len(p) for p in parts) + len(_SEP) * max(0, len(parts) - 1)
    if total > SYSTEM_PROMPT_BUDGET:
        overflow = total - SYSTEM_PROMPT_BUDGET
        # Truncate from end (lowest priority first); protect indices 0..4
        protected = min(5, len(parts))
        for idx in range(len(parts) - 1, protected - 1, -1):
            if overflow <= 0:
                break
            if len(parts[idx]) > 500:
                cut = min(overflow, len(parts[idx]) - 500)
                parts[idx] = parts[idx][:len(parts[idx]) - cut] + (
                    "\n[... truncated to fit prompt budget ...]"
                )
                overflow -= cut

    return _SEP.join(parts)
