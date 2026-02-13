"""Message cache — dedup processed messages across poll cycles."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Prune entries older than this many days
_PRUNE_DAYS = 7


class MessageCache:
    """JSON-backed set of processed message IDs with TTL pruning.

    Each entry is ``{key: "<source>:<chat_id>:<message_id>", ts: "<ISO>"}``.
    On load, entries older than ``_PRUNE_DAYS`` are silently dropped.
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._entries: dict[str, str] = {}  # key → ISO timestamp
        self._load()

    def is_processed(self, source: str, chat_id: str, message_id: str) -> bool:
        key = f"{source}:{chat_id}:{message_id}"
        return key in self._entries

    def mark_processed(self, source: str, chat_id: str, message_id: str) -> None:
        key = f"{source}:{chat_id}:{message_id}"
        self._entries[key] = datetime.now(timezone.utc).isoformat()
        self._save()

    def mark_batch(self, keys: list[str]) -> None:
        """Mark multiple keys at once (each key is ``source:chat_id:message_id``)."""
        now = datetime.now(timezone.utc).isoformat()
        for k in keys:
            self._entries[k] = now
        self._save()

    @property
    def size(self) -> int:
        return len(self._entries)

    # --- Internal ---

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            cutoff = datetime.now(timezone.utc) - timedelta(days=_PRUNE_DAYS)
            cutoff_iso = cutoff.isoformat()
            if isinstance(raw, list):
                # Legacy format: [{key, ts}]
                for entry in raw:
                    if isinstance(entry, dict):
                        k = entry.get("key", "")
                        ts = entry.get("ts", "")
                        if k and ts >= cutoff_iso:
                            self._entries[k] = ts
            elif isinstance(raw, dict):
                for k, ts in raw.items():
                    if isinstance(ts, str) and ts >= cutoff_iso:
                        self._entries[k] = ts
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load VP message cache: %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Prune on save
        cutoff = datetime.now(timezone.utc) - timedelta(days=_PRUNE_DAYS)
        cutoff_iso = cutoff.isoformat()
        self._entries = {k: ts for k, ts in self._entries.items() if ts >= cutoff_iso}
        self._path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
