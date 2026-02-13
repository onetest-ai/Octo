"""VP statistics — audit log writer + usage stats aggregator."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VPStats:
    """Manages audit.jsonl (append-only decision log) and stats.json (counters).

    Audit entries are JSONL (one JSON object per line, append-only).
    Stats are simple counters updated on each record() call.
    """

    def __init__(self, stats_path: Path, audit_path: Path) -> None:
        self._stats_path = stats_path
        self._audit_path = audit_path

    def record(self, entry: dict[str, Any]) -> None:
        """Append an audit entry and update counters."""
        # Ensure timestamp
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Append to audit.jsonl
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Update counters
        stats = self._load_stats()
        stats["total"] = stats.get("total", 0) + 1
        decision = entry.get("decision", "")
        if decision in ("respond", "disclaim", "escalate", "skip", "monitor"):
            stats[decision] = stats.get(decision, 0) + 1

        # Per-user counters
        user = entry.get("user_email", "unknown")
        by_user = stats.setdefault("by_user", {})
        by_user[user] = by_user.get(user, 0) + 1

        self._save_stats(stats)

    def get_stats(self, days: int = 7) -> dict[str, Any]:
        """Aggregate stats from audit log for the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        totals: dict[str, int] = {
            "total": 0,
            "respond": 0,
            "disclaim": 0,
            "escalate": 0,
            "skip": 0,
            "monitor": 0,
        }
        by_user: dict[str, int] = {}
        by_category: dict[str, int] = {}
        confidence_sum = 0.0
        confidence_count = 0

        for entry in self._iter_audit():
            if entry.get("timestamp", "") < cutoff_iso:
                continue
            totals["total"] += 1
            decision = entry.get("decision", "")
            if decision in totals:
                totals[decision] += 1

            user = entry.get("user_email", "unknown")
            by_user[user] = by_user.get(user, 0) + 1

            cat = entry.get("category", "")
            if cat:
                by_category[cat] = by_category.get(cat, 0) + 1

            conf = entry.get("confidence")
            if conf is not None:
                confidence_sum += float(conf)
                confidence_count += 1

        return {
            **totals,
            "avg_confidence": (
                round(confidence_sum / confidence_count, 1) if confidence_count else 0
            ),
            "escalation_rate": (
                round(totals["escalate"] / totals["total"], 2)
                if totals["total"]
                else 0
            ),
            "by_user": dict(sorted(by_user.items(), key=lambda x: -x[1])[:10]),
            "by_category": by_category,
            "period_days": days,
        }

    def get_audit_log(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the last N audit entries."""
        entries = list(self._iter_audit())
        return entries[-n:]

    # --- Internal ---

    def _load_stats(self) -> dict[str, Any]:
        if self._stats_path.is_file():
            try:
                return json.loads(self._stats_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _save_stats(self, stats: dict[str, Any]) -> None:
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _iter_audit(self):
        """Iterate over audit.jsonl entries."""
        if not self._audit_path.is_file():
            return
        try:
            for line in self._audit_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Failed to read VP audit log: %s", e)
