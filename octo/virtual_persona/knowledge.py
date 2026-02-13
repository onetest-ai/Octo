"""Conversation knowledge cache — classified thread topics and summaries."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only re-classify a thread if at least this many new messages since last sync
_RECLASSIFY_THRESHOLD = 5

# Max messages to store per thread in the cache
_MAX_CACHED_MESSAGES = 100

_CLASSIFY_PROMPT = """\
Classify this conversation thread. Respond with ONLY valid JSON, no other text.

{{
  "topic": "<short topic label, 3-8 words>",
  "summary": "<2-3 sentence summary of what was discussed>",
  "key_points": ["<point 1>", "<point 2>", ...]
}}

Messages (most recent last):
{messages}"""


class ConversationKnowledge:
    """Local classified cache of Teams/Telegram conversations.

    Maintains a thread index at ``knowledge/threads.json`` mapping
    chat_id to topic, summary, key points, participants, and metadata.
    """

    def __init__(self, knowledge_dir: Path) -> None:
        self._dir = knowledge_dir
        self._index_path = knowledge_dir / "threads.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    def get_thread_context(self, chat_id: str) -> dict[str, Any] | None:
        """Return classified context for a thread, or None if unknown."""
        return self._index.get(chat_id)

    async def sync_thread(
        self,
        chat_id: str,
        messages: list[dict[str, str]],
        participants: list[str] | None = None,
        chat_meta: dict[str, str] | None = None,
        self_emails: set[str] | None = None,
        classify: bool = False,
    ) -> dict[str, Any] | None:
        """Index a thread's metadata and optionally classify via LLM.

        Always saves basic metadata (participants, message count, timestamps,
        preview, engagement stats).  Only runs LLM classification when
        *classify* is True and the message count increased by
        ``_RECLASSIFY_THRESHOLD``.

        Args:
            chat_id: Unique thread/chat identifier.
            messages: List of ``{role, content, sender_name, timestamp}`` dicts.
            participants: Optional list of participant emails.
            chat_meta: Optional dict with chat_type, topic, created.
            self_emails: Artem's email addresses — used to calculate engagement.
            classify: If True, run LLM topic classification.

        Returns:
            The thread context dict.
        """
        now = datetime.now(timezone.utc).isoformat()
        existing = self._index.get(chat_id, {})
        meta = chat_meta or {}

        # Engagement stats — how active was the user in this thread
        self_count = 0
        if self_emails:
            for m in messages:
                role = m.get("role", "")
                email = m.get("sender_email", "").lower()
                if role == "assistant" or email in self_emails:
                    self_count += 1
        total = len(messages) or 1
        engagement = round(self_count / total, 3)

        # Always save basic metadata — no AI needed
        context: dict[str, Any] = {
            "topic": meta.get("topic") or existing.get("topic", ""),
            "chat_type": meta.get("chat_type") or existing.get("chat_type", ""),
            "summary": existing.get("summary", ""),
            "key_points": existing.get("key_points", []),
            "participants": participants or existing.get("participants", []),
            "message_count": len(messages),
            "self_message_count": self_count,
            "engagement": engagement,
            "last_updated": now,
        }

        # Build a preview from recent messages
        if messages:
            last_msg = messages[-1]
            sender = last_msg.get("sender_name", "")
            content = last_msg.get("content", "")[:200]
            context["preview"] = f"{sender}: {content}" if sender else content
            # Timestamp of most recent message
            ts = last_msg.get("timestamp", "")
            if ts:
                context["last_message_at"] = ts

        # Auto-generate topic from first few messages if no topic yet
        if not context["topic"] and messages:
            context["topic"] = self._extract_topic_simple(messages)

        # LLM classification (optional, for richer context)
        if classify:
            prev_count = existing.get("message_count", 0)
            if len(messages) - prev_count >= _RECLASSIFY_THRESHOLD or not existing:
                classified = await self._classify(messages)
                if classified:
                    context.update(classified)

        self._index[chat_id] = context
        self._save()

        # Cache messages on disk (last N only)
        self._save_messages(chat_id, messages[-_MAX_CACHED_MESSAGES:])

        return context

    @staticmethod
    def _extract_topic_simple(messages: list[dict[str, str]]) -> str:
        """Extract a rough topic from the first few messages — no LLM needed."""
        # Grab content from first few messages
        snippets = []
        for m in messages[:5]:
            text = m.get("content", "").strip()
            if text and len(text) > 3:
                snippets.append(text[:80])
        if not snippets:
            return "Chat"
        # Use the longest snippet as a rough topic
        longest = max(snippets, key=len)
        # Trim to first sentence or 60 chars
        for sep in (".", "!", "?", "\n"):
            idx = longest.find(sep)
            if 5 < idx < 60:
                return longest[:idx + 1]
        return longest[:60]

    def search(self, query: str) -> list[dict[str, Any]]:
        """Keyword search across thread summaries and topics."""
        q = query.lower()
        results = []
        for chat_id, ctx in self._index.items():
            haystack = " ".join([
                ctx.get("topic", ""),
                ctx.get("summary", ""),
                " ".join(ctx.get("key_points", [])),
            ]).lower()
            if q in haystack:
                results.append({"chat_id": chat_id, **ctx})
        return results

    def list_threads(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the N most recently updated threads."""
        items = [
            {"chat_id": k, **v} for k, v in self._index.items()
        ]
        items.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        return items[:n]

    def get_cached_messages(self, chat_id: str) -> list[dict[str, str]]:
        """Return cached messages for a thread, or empty list."""
        path = self._messages_dir / self._chat_hash(chat_id)
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def get_active_threads(
        self,
        min_engagement: float = 0.0,
        min_messages: int = 3,
    ) -> list[dict[str, Any]]:
        """Return threads where user was active, sorted by engagement.

        Args:
            min_engagement: Minimum engagement ratio (0.0 - 1.0).
            min_messages: Minimum total messages in thread.

        Returns:
            List of thread dicts with ``chat_id`` key, sorted by engagement desc.
        """
        results: list[dict[str, Any]] = []
        for chat_id, ctx in self._index.items():
            if ctx.get("message_count", 0) < min_messages:
                continue
            if ctx.get("engagement", 0) < min_engagement:
                continue
            results.append({"chat_id": chat_id, **ctx})
        results.sort(key=lambda x: x.get("engagement", 0), reverse=True)
        return results

    @property
    def thread_count(self) -> int:
        return len(self._index)

    @property
    def _messages_dir(self) -> Path:
        return self._dir / "messages"

    @staticmethod
    def _chat_hash(chat_id: str) -> str:
        """Short filename-safe hash of a chat ID."""
        return hashlib.sha256(chat_id.encode()).hexdigest()[:16] + ".json"

    # --- Internal ---

    async def _classify(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        """Run topic classification via low-tier LLM."""
        try:
            from octo.models import make_model
        except ImportError:
            logger.warning("Cannot import make_model for knowledge classification")
            return None

        # Format last 20 messages for the prompt
        recent = messages[-20:]
        formatted = "\n".join(
            f"[{m.get('role', '?')}]: {m.get('content', '')[:300]}"
            for m in recent
        )
        prompt = _CLASSIFY_PROMPT.format(messages=formatted)

        try:
            model = make_model(tier="low")
            response = await model.ainvoke(prompt)
            text = response.content.strip()
            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return {
                "topic": data.get("topic", "Unknown"),
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []),
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Thread classification failed: %s", e)
            return None

    def _load(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._index_path.is_file():
            try:
                data = json.loads(
                    self._index_path.read_text(encoding="utf-8")
                )
                if isinstance(data, dict):
                    self._index = data
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to load VP knowledge index: %s", e)

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _save_messages(self, chat_id: str, messages: list[dict[str, str]]) -> None:
        """Cache messages to disk for offline access."""
        self._messages_dir.mkdir(parents=True, exist_ok=True)
        path = self._messages_dir / self._chat_hash(chat_id)
        path.write_text(
            json.dumps(messages, ensure_ascii=False) + "\n", encoding="utf-8"
        )
