"""VP Poller — HeartbeatRunner-style async loop for Teams message polling."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Skip messages older than this many minutes
_MAX_MESSAGE_AGE_MINUTES = 30

# Artem's email patterns (messages from these are skipped)
_SELF_EMAILS: set[str] = set()


def set_self_emails(emails: list[str]) -> None:
    """Register Artem's own email addresses so VP skips own messages."""
    _SELF_EMAILS.update(e.lower() for e in emails)


class VPPoller:
    """Background async loop that polls Teams for new messages and routes them through the VP graph.

    Follows the HeartbeatRunner pattern:
    - start() → asyncio.Task running _loop()
    - stop() → sets _stop_event
    - _tick() → main poll cycle
    """

    def __init__(
        self,
        vp_graph: Any,
        octo_app: Any,
        graph_lock: asyncio.Lock,
        interval: int,
        active_start: time,
        active_end: time,
        on_escalation: Callable[[str, str, str], Awaitable[None]] | None = None,
        octo_config: dict[str, Any] | None = None,
        callbacks: list | None = None,
    ) -> None:
        self._vp_graph = vp_graph
        self._octo_app = octo_app
        self._lock = graph_lock
        self._interval = interval
        self._active_start = active_start
        self._active_end = active_end
        self._on_escalation = on_escalation
        self._octo_config = octo_config or {}
        self._callbacks = callbacks or []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._running = False
        self._last_poll: datetime | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_poll(self) -> datetime | None:
        return self._last_poll

    def _is_active_hours(self) -> bool:
        now = datetime.now().time()
        if self._active_start <= self._active_end:
            return self._active_start <= now <= self._active_end
        return now >= self._active_start or now <= self._active_end

    # --- Main poll cycle ---

    async def _tick(self) -> None:
        """Execute one poll cycle: fetch chats → filter messages → invoke VP graph → send responses."""
        from octo.config import VP_DIR
        from octo.virtual_persona.access_control import AccessControl
        from octo.virtual_persona.cache import MessageCache

        ac = AccessControl(VP_DIR / "access-control.yaml")
        if not ac.is_enabled():
            logger.debug("VP poll skipped: disabled")
            return

        if not self._is_active_hours():
            logger.debug("VP poll skipped: outside active hours")
            return

        prev_poll = self._last_poll
        self._last_poll = datetime.now(timezone.utc)

        cache = MessageCache(VP_DIR / "message-cache.json")

        # Fetch Teams chats
        chats, _ = await self._fetch_chats()
        if not chats:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=_MAX_MESSAGE_AGE_MINUTES)

        for chat in chats:
            chat_id = chat.get("id", "")
            if not chat_id:
                continue

            # Skip delegated threads (Artem is handling it)
            if ac.is_delegated(chat_id):
                continue

            chat_ignored = ac.is_ignored(chat_id)

            # Skip chats with no new messages since last poll —
            # avoids fetching message history for inactive conversations.
            last_msg = chat.get("lastMessage") or {}
            last_msg_ts = last_msg.get("createdDateTime", "")
            if prev_poll and last_msg_ts:
                try:
                    from datetime import datetime as _dt
                    msg_dt = _dt.fromisoformat(last_msg_ts.replace("Z", "+00:00"))
                    if msg_dt < prev_poll:
                        continue
                except (ValueError, TypeError):
                    pass  # Can't parse — fetch to be safe

            try:
                messages = await self._fetch_messages(chat_id)
            except _ChatNotAccessible:
                continue  # Normal: legacy/bot chat, silently skip
            if not messages:
                continue

            # --- Aggregate new messages per chat before processing ---
            # People often send 3-7 consecutive messages. We collect all
            # unprocessed messages, merge them into one query, and run the
            # VP graph once per chat — not once per message.

            from octo.virtual_persona.content_filter import (
                is_safe_for_llm,
                sanitize,
            )


            pending: list[dict[str, Any]] = []  # collected new messages
            chat_type = chat.get("chatType", "")

            # Find Artem's last message timestamp — anything before it
            # is already "handled" (Artem saw and replied).
            chronological = list(reversed(messages))
            last_self_idx = -1
            for idx, msg in enumerate(chronological):
                sender_email, _ = _extract_sender(msg)
                if sender_email in _SELF_EMAILS:
                    last_self_idx = idx

            for msg_idx, msg in enumerate(chronological):
                msg_id = msg.get("id", "")
                if not msg_id:
                    continue

                if cache.is_processed("teams", chat_id, msg_id):
                    continue

                sender_email, sender_name = _extract_sender(msg)

                # Resolve missing email from chat members (Graph API messages
                # only include displayName, not email — we need the members list)
                if not sender_email and sender_name:
                    sender_email = await self._resolve_email(
                        chat_id, sender_name
                    )

                if sender_email in _SELF_EMAILS:
                    cache.mark_processed("teams", chat_id, msg_id)
                    continue

                # Ignored chats: skip unless sender is a priority user
                if chat_ignored and not ac.is_priority_user(sender_email):
                    cache.mark_processed("teams", chat_id, msg_id)
                    continue

                # Skip messages that came before Artem's last reply —
                # he already saw and responded, no need to answer again.
                if last_self_idx >= 0 and msg_idx <= last_self_idx:
                    cache.mark_processed("teams", chat_id, msg_id)
                    continue

                if chat_type in ("group", "meeting"):
                    if not _mentions_self(msg):
                        cache.mark_processed("teams", chat_id, msg_id)
                        continue

                created = msg.get("createdDateTime", "")
                if created and created < cutoff.isoformat():
                    cache.mark_processed("teams", chat_id, msg_id)
                    continue

                content, content_type = _extract_body(msg)
                if content_type == "html":
                    content = _strip_html(content)
                content = content.strip()
                has_attachments = bool(msg.get("attachments"))
                if not content and not has_attachments:
                    cache.mark_processed("teams", chat_id, msg_id)
                    continue

                # Sanitize content (clean but never reject individual messages —
                # short messages like "ок" or "да" are fine when aggregated)
                if content:
                    content, _fa = sanitize(content)

                attachment_refs = await self._extract_attachments(
                    msg, chat_id, msg_id
                )
                if attachment_refs:
                    refs_text = "\n".join(
                        f"[Attached file: {r['name']} \u2192 {r['path']}]"
                        for r in attachment_refs
                    )
                    content = f"{content}\n\n{refs_text}"

                pending.append({
                    "msg_id": msg_id,
                    "content": content,
                    "sender_email": sender_email,
                    "sender_name": sender_name,
                })

            if not pending:
                continue

            # Use the last (most recent) message as the primary sender
            last = pending[-1]
            last_msg_id = last["msg_id"]

            # Aggregate: if multiple messages, join them chronologically
            if len(pending) == 1:
                aggregated_query = last["content"]
            else:
                parts = []
                for p in pending:
                    name = p["sender_name"] or p["sender_email"]
                    parts.append(f"{name}: {p['content']}")
                aggregated_query = "\n\n".join(parts)

            # Safety check on the aggregated query (not per-message)
            safe, unsafe_reason = is_safe_for_llm(aggregated_query)
            if not safe:
                logger.info(
                    "VP: skipping chat %s — aggregated query unsafe (%s)",
                    chat_id[:20], unsafe_reason,
                )
                for p in pending:
                    cache.mark_processed("teams", chat_id, p["msg_id"])
                continue

            # Build thread context from all messages in chat
            context_msgs = _build_context(messages, last_msg_id)

            # Invoke VP graph once for the entire batch
            await self._process_message(
                query=aggregated_query,
                context=context_msgs,
                user_email=last["sender_email"],
                user_name=last["sender_name"],
                chat_id=chat_id,
                message_id=last_msg_id,
            )

            # Mark all collected messages as processed
            for p in pending:
                cache.mark_processed("teams", chat_id, p["msg_id"])

            # Incremental knowledge sync + profile enrichment (once per chat)
            await self._sync_thread_knowledge(chat_id, messages, chat=chat)
            await self._enrich_profile(last["sender_email"], chat_id)

    async def _process_message(
        self,
        query: str,
        context: list[dict[str, str]],
        user_email: str,
        user_name: str,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Invoke VP graph for a single message and handle the result."""
        from octo.virtual_persona.state import VPState

        state: dict[str, Any] = {
            "query": query,
            "context": context,
            "user_email": user_email,
            "user_name": user_name,
            "chat_id": chat_id,
            "message_id": message_id,
            "source": "teams",
            "_octo_app": self._octo_app,
            "_octo_config": self._octo_config,
        }

        # Acquire graph lock
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("VP poll: graph lock busy, skipping message %s", message_id)
            return

        try:
            result = await self._vp_graph.ainvoke(state)
        except Exception as exc:
            logger.error("VP graph invocation failed: %s", exc)
            return
        finally:
            self._lock.release()

        decision = result.get("decision", "skip")
        confidence = result.get("confidence", 0)
        category = result.get("category", "")
        response = result.get("response", "")

        # Always log confidence decisions (visible even without --debug)
        _decision_emoji = {
            "respond": "\u2705", "disclaim": "\u26a0\ufe0f",
            "escalate": "\U0001f6a8", "monitor": "\U0001f441",
            "skip": "\u23ed\ufe0f",
        }
        logger.info(
            "VP %s %s  %s  %.0f%%  %s  from=%s",
            _decision_emoji.get(decision, "?"), decision.upper(),
            category, confidence, chat_id[:20], user_name or user_email,
        )

        if decision == "skip":
            return

        # Respond / disclaim: send formatted response to sender via Teams
        if decision in ("respond", "disclaim") and response:
            await self._send_response(chat_id, message_id, response)

        # Monitor / escalate: silently notify Artem (invisible to sender)
        if decision in ("monitor", "escalate") and self._on_escalation:
            notification = self._format_notification(result)
            try:
                await self._on_escalation(notification, chat_id, message_id)
            except Exception as exc:
                logger.error("VP notification delivery failed: %s", exc)

    # --- Teams MCP tool wrappers ---

    async def _fetch_chats(
        self, limit: int = 50, slim: bool = False,
    ) -> tuple[list[dict], bool]:
        """Fetch one page of Teams chats via MCP (max 50).

        Args:
            limit: Max chats to return (capped at 50 per call to keep
                   MCP STDIO responses small).
            slim: If True, request minimal fields only (id, chatType, topic).

        Returns:
            (chats, has_more) — list of chat dicts and whether more pages exist.
        """
        from octo.graph import get_mcp_tool

        tool = get_mcp_tool("list-chats")
        if tool is None:
            logger.warning("VP: msteams list-chats tool not available")
            return [], False

        per_call = min(limit, 50)
        try:
            params: dict[str, Any] = {"limit": per_call}
            if slim:
                params["slim"] = True
            result = await tool.ainvoke(params)
            data = self._parse_tool_result(result)
            if data is None:
                return [], False
            if isinstance(data, list):
                chats = data
                has_more = False
            elif isinstance(data, dict):
                if "error" in data:
                    logger.warning("VP: list-chats error: %s", data["error"])
                    return [], False
                chats = data.get("chats", data.get("value", []))
                has_more = bool(data.get("hasMore", False))
            else:
                logger.warning("VP: unexpected list-chats type: %s", type(data).__name__)
                return [], False
            logger.info("VP: fetched %d chats (page, hasMore=%s)", len(chats), has_more)
            return chats, has_more
        except Exception as exc:
            logger.warning("VP: failed to fetch Teams chats: %s", exc)
            return [], False

    @staticmethod
    def _parse_tool_result(result: Any) -> Any:
        """Parse an MCP tool result to a Python object.

        MCP tools via langchain-mcp-adapters return content blocks:
        ``[{"type": "text", "text": "<json>"}]``.  This method extracts the
        text, parses it as JSON, and returns the resulting dict/list.
        """
        # MCP content-block list: [{"type": "text", "text": "..."}]
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and first.get("type") == "text":
                result = first.get("text", "")

        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                logger.warning("VP: tool result not valid JSON (len=%d)", len(result))
                return None
        return result

    async def _fetch_messages(self, chat_id: str, limit: int = 30) -> list[dict]:
        """Fetch recent messages from a Teams chat.

        Returns list of messages, or empty list on error.
        Raises ``_ChatNotAccessible`` for 404/403 (normal for legacy/bot chats).
        """
        from octo.graph import get_mcp_tool

        tool = get_mcp_tool("list-chat-messages")
        if tool is None:
            return []

        try:
            result = await tool.ainvoke({"chatId": chat_id, "limit": limit})
            data = self._parse_tool_result(result)
            if data is None:
                return []
            # Check for Graph API error response
            if isinstance(data, dict) and "error" in data:
                err = data.get("error", "")
                err_str = err if isinstance(err, str) else str(err)
                if "Not found" in err_str or "NotFound" in str(data.get("code", "")):
                    raise _ChatNotAccessible(chat_id, "not_found")
                if "403" in err_str or "Permission" in err_str:
                    raise _ChatNotAccessible(chat_id, "forbidden")
                logger.warning("VP: messages error for %s: %s", chat_id[:20], err_str[:100])
                return []
            return data if isinstance(data, list) else data.get("value", data.get("messages", []))
        except _ChatNotAccessible:
            raise
        except Exception as exc:
            err_str = str(exc)
            if "404" in err_str or "403" in err_str or "Not Found" in err_str:
                raise _ChatNotAccessible(chat_id, "http_error") from exc
            logger.warning("VP: failed to fetch messages for chat %s: %s", chat_id[:20], exc)
            return []

    async def _send_response(self, chat_id: str, message_id: str, text: str) -> None:
        """Send a response via Teams MCP.

        Uses send-chat-message (not reply-to) because the Graph API
        /replies endpoint returns 405 for many chat types.
        """
        from octo.graph import get_mcp_tool

        tool = get_mcp_tool("send-chat-message")
        if tool:
            try:
                await tool.ainvoke({"chatId": chat_id, "content": text})
            except Exception as exc:
                logger.error("VP: failed to send response to chat %s: %s", chat_id, exc)
        else:
            logger.warning("VP: send-chat-message tool not available")

    async def _fetch_chat_members(self, chat_id: str) -> list[dict]:
        """Fetch chat members for profile enrichment."""
        from octo.graph import get_mcp_tool

        tool = get_mcp_tool("list-chat-members")
        if tool is None:
            return []

        try:
            result = await tool.ainvoke({"chatId": chat_id})
            data = self._parse_tool_result(result)
            if data is None:
                return []
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "error" in data:
                    return []
                # MCP server returns {"members": [...], "count": N}
                return data.get("members", data.get("value", []))
            return []
        except Exception as exc:
            logger.warning("VP: failed to fetch chat members: %s", exc)
            return []

    async def _resolve_email(self, chat_id: str, display_name: str) -> str:
        """Resolve a sender's email by fetching chat members.

        Graph API messages only include displayName, not email.
        Fetching members populates the contacts cache as a side effect.
        """
        members = await self._fetch_chat_members(chat_id)
        name_lower = display_name.lower()
        for m in members:
            dn = m.get("displayName") or ""
            if dn.lower() == name_lower:
                return (m.get("email", "") or "").lower()
        return ""

    # --- Attachment handling ---

    async def _extract_attachments(
        self,
        msg: dict,
        chat_id: str,
        msg_id: str,
    ) -> list[dict[str, str]]:
        """Extract attachments from a Teams message, download to workspace.

        Returns list of {name, path, content_type} for each saved file.
        Files are saved to .octo/workspace/<date>/vp-attachments/<chat_id>/.
        The returned paths can be passed as references to Octo — agents will
        read the files on-demand using their filesystem tools.
        """
        attachments = msg.get("attachments") or []
        if not attachments:
            return []

        from octo.config import RESEARCH_WORKSPACE

        today = datetime.now().strftime("%Y-%m-%d")
        attach_dir = RESEARCH_WORKSPACE / today / "vp-attachments" / chat_id[:20]
        attach_dir.mkdir(parents=True, exist_ok=True)

        refs: list[dict[str, str]] = []
        for att in attachments:
            content_type = att.get("contentType", "")
            name = att.get("name", "")
            content_url = att.get("contentUrl", "")

            if not name:
                name = f"attachment-{msg_id[:8]}"

            # Teams inline images: hosted content in message body
            # These have contentType like "reference" and a contentUrl
            if content_url and content_type == "reference":
                # Save URL reference — Octo can fetch if needed
                ref_path = attach_dir / f"{name}.url"
                ref_path.write_text(content_url, encoding="utf-8")
                refs.append({
                    "name": name,
                    "path": str(ref_path),
                    "content_type": "reference_url",
                })
                continue

            # File attachments with inline content (rare but possible)
            content = att.get("content", "")
            if content and isinstance(content, str):
                safe_name = "".join(
                    c if c.isalnum() or c in ".-_" else "_" for c in name
                )
                file_path = attach_dir / safe_name
                file_path.write_text(content[:50000], encoding="utf-8")
                refs.append({
                    "name": name,
                    "path": str(file_path),
                    "content_type": content_type or "text/plain",
                })
                continue

            # Adaptive card or other structured attachment
            att_content = att.get("content", "")
            if isinstance(att_content, dict):
                safe_name = "".join(
                    c if c.isalnum() or c in ".-_" else "_" for c in name
                ) + ".json"
                file_path = attach_dir / safe_name
                file_path.write_text(
                    json.dumps(att_content, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                refs.append({
                    "name": name,
                    "path": str(file_path),
                    "content_type": content_type or "application/json",
                })
                continue

            # Fallback: just log the attachment metadata as a reference
            if content_url:
                refs.append({
                    "name": name,
                    "path": content_url,
                    "content_type": content_type or "unknown",
                })

        if refs:
            logger.info("VP: extracted %d attachment(s) from message %s", len(refs), msg_id)

        return refs

    # --- Knowledge and profile helpers ---

    async def _sync_thread_knowledge(
        self, chat_id: str, messages: list[dict], chat: dict | None = None
    ) -> None:
        """Incrementally sync thread knowledge for this chat."""
        from octo.config import VP_DIR
        from octo.virtual_persona.knowledge import ConversationKnowledge

        knowledge = ConversationKnowledge(VP_DIR / "knowledge")

        # Normalize raw messages → clean {role, content, sender_name, timestamp}
        clean_msgs: list[dict[str, str]] = []
        participants: dict[str, str] = {}  # email → name
        for m in messages:
            nm = normalize_message(m)
            if nm["type"] != "message":
                continue  # skip system events
            if nm["content"]:
                clean_msgs.append({
                    "role": nm["role"],
                    "content": nm["content"][:500],
                    "sender_name": nm["sender_name"],
                    "sender_email": nm["sender_email"],
                    "timestamp": nm["timestamp"],
                })
            if nm["sender_email"]:
                participants[nm["sender_email"]] = nm["sender_name"]

        # Normalize chat metadata
        chat_meta = normalize_chat(chat) if chat else None

        if clean_msgs:
            await knowledge.sync_thread(
                chat_id,
                clean_msgs,
                participants=list(participants.keys()),
                chat_meta=chat_meta,
                self_emails=_SELF_EMAILS,
            )

    async def _enrich_profile(self, email: str, chat_id: str) -> None:
        """Auto-enrich a new contact's profile from Teams member info."""
        if not email:
            return

        from octo.config import VP_DIR
        from octo.virtual_persona.profiles import PeopleProfiles

        profiles = PeopleProfiles(VP_DIR / "profiles.json")
        existing = profiles.get_profile(email)

        # Only enrich if profile is new (no name set yet)
        if existing and existing.get("name"):
            return

        members = await self._fetch_chat_members(chat_id)
        if members:
            await profiles.enrich_from_teams(email, members)

    async def full_sync(
        self,
        on_progress: Callable[[int, int, int, int], Any] | None = None,
        max_pages: int = 10,
    ) -> dict[str, int]:
        """Full re-sync of all thread knowledge (for /vp sync command).

        Fetches chats page by page (50 per page) via MCP, processing
        each page's chats before fetching the next.  This keeps each MCP
        response small and ensures partial progress is saved.

        Args:
            on_progress: Optional callback(synced, skipped, errors, total)
                called after each chat is processed.
            max_pages: Max number of chat pages to fetch (50 chats each).

        Returns dict with keys: synced, skipped, ignored, errors, total.
        """
        from octo.config import VP_DIR
        from octo.virtual_persona.access_control import AccessControl

        ac = AccessControl(VP_DIR / "access-control.yaml")

        synced = 0
        skipped = 0
        ignored = 0
        errors = 0
        total = 0
        seen_ids: set[str] = set()

        # Page through chats 50 at a time (slim=True: only need id/chatType/topic)
        has_more = True
        for page_num in range(max_pages):
            if not has_more:
                break

            if page_num == 0:
                chats, has_more = await self._fetch_chats(limit=50, slim=True)
            else:
                chats, has_more = await self._fetch_chats_next()

            if not chats:
                logger.info("VP sync: page %d returned 0 chats, stopping", page_num + 1)
                break

            logger.info("VP sync: page %d → %d chats (hasMore=%s)", page_num + 1, len(chats), has_more)

            # Process each chat in this page immediately
            for chat in chats:
                chat_id = chat.get("id", "")
                if not chat_id or chat_id in seen_ids:
                    continue
                seen_ids.add(chat_id)
                total += 1

                if ac.is_ignored(chat_id):
                    ignored += 1
                    continue

                # Blanket try/except — no single chat can break the sync
                try:
                    messages = await self._fetch_messages(chat_id, limit=50)
                except _ChatNotAccessible:
                    skipped += 1
                    if on_progress:
                        on_progress(synced, skipped, errors, total)
                    continue
                except Exception:
                    errors += 1
                    if on_progress:
                        on_progress(synced, skipped, errors, total)
                    continue

                if not messages:
                    skipped += 1
                    if on_progress:
                        on_progress(synced, skipped, errors, total)
                    continue

                try:
                    await self._sync_thread_knowledge(chat_id, messages, chat=chat)
                    synced += 1
                except Exception as exc:
                    logger.debug("VP sync: knowledge sync failed for %s: %s", chat_id[:20], exc)
                    errors += 1

                if on_progress:
                    on_progress(synced, skipped, errors, total)

        logger.info("VP sync done: synced=%d skipped=%d ignored=%d errors=%d total=%d (pages=%d)",
                     synced, skipped, ignored, errors, total, min(page_num + 1, max_pages))
        return {"synced": synced, "skipped": skipped, "ignored": ignored, "errors": errors, "total": total}

    async def _fetch_chats_next(self) -> tuple[list[dict], bool]:
        """Fetch the next page of chats (follows pagination from _fetch_chats).

        Returns:
            (chats, has_more) — list of chat dicts and whether more pages exist.
        """
        from octo.graph import get_mcp_tool

        tool = get_mcp_tool("list-chats-next")
        if tool is None:
            return [], False

        try:
            result = await tool.ainvoke({"slim": True})
            data = self._parse_tool_result(result)
            if data is None:
                return [], False
            if isinstance(data, dict) and "error" in data:
                logger.debug("VP: list-chats-next error: %s", data.get("error", ""))
                return [], False
            if isinstance(data, dict):
                chats = data.get("chats", [])
                has_more = bool(data.get("hasMore", False))
                logger.info("VP: fetched %d chats (next page, hasMore=%s)", len(chats), has_more)
                return chats, has_more
            chats = data if isinstance(data, list) else []
            return chats, False
        except Exception as exc:
            logger.debug("VP: fetch next chats failed: %s", exc)
            return [], False

    # --- Notification formatting ---

    @staticmethod
    def _format_notification(result: dict[str, Any]) -> str:
        """Format a private notification for Artem (Telegram/console).

        Uses emojis for visual categorization. Plain text — no HTML tags
        (Telegram markdown-to-HTML converter would escape them).
        """
        decision = result.get("decision", "unknown")
        urgent = "urgent" in (result.get("escalation_flags") or [])
        confidence = result.get("confidence", 0)

        # Header with emoji categorization
        if urgent:
            header = "\U0001f6a8 URGENT ESCALATION"
        elif decision == "escalate":
            header = "\u26a0\ufe0f Escalation"
        else:
            header = "\U0001f441 Monitor"

        # Confidence bar
        filled = round(confidence / 10)
        bar = "\u2588" * filled + "\u2591" * (10 - filled)

        # Category emoji
        cat = result.get("category", "")
        cat_emoji = {
            "technical_ai_ml": "\U0001f916",
            "adjacent_technical": "\U0001f4bb",
            "general_knowledge": "\U0001f4da",
            "personal_decision": "\U0001f464",
            "sensitive_confidential": "\U0001f512",
            "outside_expertise": "\U0001f30d",
            "realtime_context": "\u23f0",
            "social_relationship": "\U0001f91d",
            "hr_legal": "\u2696\ufe0f",
            "financial": "\U0001f4b0",
            "urgent_emergency": "\U0001f6a8",
        }.get(cat, "\u2753")

        name = result.get("user_name", "")
        email = result.get("user_email", "")
        sender = f"{name} ({email})" if name else email

        parts = [
            header,
            "",
            f"\U0001f464 {sender}",
            f"{cat_emoji} {cat}  |  {bar} {confidence:.0f}%",
        ]

        flags = result.get("escalation_flags", [])
        if flags:
            parts.append(f"\U0001f6a9 {', '.join(flags)}")

        query = result.get("query", "")
        if query:
            parts.append(f"\n\U0001f4ac {query[:500]}")

        reasoning = result.get("classification_reasoning", "")
        if reasoning:
            parts.append(f"\n\U0001f9e0 {reasoning}")

        # Include suggested answer from Octo (if available)
        raw_answer = result.get("raw_answer", "")
        if raw_answer and not raw_answer.startswith("[VP Error"):
            parts.append(f"\n\u2705 Suggested answer:\n{raw_answer[:1000]}")

        if decision == "escalate":
            parts.append(f"\n\U0001f512 Thread locked \u2014 /vp release to unlock")

        parts.append(f"\n\u21a9\ufe0f Reply to respond  |  \"ignore\" to mute")

        return "\n".join(parts)

    # --- Async loop ---

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await self._tick()
            except Exception:
                logger.exception("VP poll tick failed")

    def start(self) -> None:
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("VP poller started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("VP poller stopped")


class _ChatNotAccessible(Exception):
    """Raised when a chat returns 404/403 — normal for legacy/bot chats."""

    def __init__(self, chat_id: str, reason: str = "") -> None:
        self.chat_id = chat_id
        self.reason = reason
        super().__init__(f"Chat {chat_id} not accessible: {reason}")


# --- Utilities ---


def _mentions_self(msg: dict) -> bool:
    """Check if a message @mentions Artem (by email or userId).

    Works with our MCP server's ``mentions`` field:
    ``[{"displayName": "Artem Rozumenko", "userId": "..."}]``
    """
    mentions = msg.get("mentions")
    if not mentions or not isinstance(mentions, list):
        return False
    for m in mentions:
        if not isinstance(m, dict):
            continue
        # Match by userId if we have self user IDs in the future,
        # for now match by email from contacts or displayName heuristic
        name = (m.get("displayName") or "").lower()
        user_id = m.get("userId", "")
        # Check if any self email's local part matches the display name
        for se in _SELF_EMAILS:
            local = se.split("@")[0].replace("_", " ").replace(".", " ").lower()
            if local and local in name:
                return True
        # Check userId match (if we track self user IDs in the future)
        # For now, displayName match is sufficient
    return False


def _extract_sender(msg: dict) -> tuple[str, str]:
    """Extract (email, display_name) from a Teams message.

    Handles three ``from`` formats:
    1. Our MCP server's flat dict: ``{"displayName": ..., "email": ..., "userId": ...}``
    2. Raw Graph API nested: ``{"user": {"displayName": ..., "email": ...}}``
    3. Legacy string (displayName only): ``"Display Name"``
    """
    sender = msg.get("from")
    if sender is None:
        return "", ""
    if isinstance(sender, str):
        return "", sender
    if not isinstance(sender, dict):
        return "", ""

    # Format 1: our MCP server's flat dict (displayName + email + userId)
    if "displayName" in sender and "user" not in sender:
        email = (sender.get("email", "") or "").lower()
        name = sender.get("displayName", "")
        return email, name

    # Format 2: raw Graph API nested {user: {displayName, email, ...}}
    user = sender.get("user", {})
    if isinstance(user, dict):
        email = (
            user.get("email", "")
            or user.get("userPrincipalName", "")
            or ""
        ).lower()
        name = user.get("displayName", "")
        if email or name:
            return email, name

    return "", ""


def _extract_body(msg: dict) -> tuple[str, str]:
    """Extract (content, content_type) from a Teams message.

    Handles both nested {"body": {"content": ..., "contentType": ...}} and
    flattened {"body": "content string", "contentType": "html"}.
    """
    body = msg.get("body", "")
    if isinstance(body, dict):
        return body.get("content", ""), body.get("contentType", "text").lower()
    if isinstance(body, str):
        ct = msg.get("contentType", "text").lower()
        return body, ct
    return "", "text"


def _strip_html(html: str) -> str:
    """Minimal HTML tag stripper."""
    import re
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    return text.strip()


def normalize_message(msg: dict) -> dict[str, str]:
    """Normalize a raw Teams message to a clean dict.

    Input can be Graph API format or custom server format.
    Output is always::

        {
            "id": "msg-id",
            "sender_email": "user@example.com",
            "sender_name": "Display Name",
            "role": "user" | "assistant",
            "content": "plain text body",
            "timestamp": "2026-02-13T14:00:00Z",
            "type": "message",
            "has_attachments": "true" | "false",
        }
    """
    email, name = _extract_sender(msg)
    body_text, content_type = _extract_body(msg)
    if content_type == "html" and body_text:
        body_text = _strip_html(body_text)
    body_text = (body_text or "").strip()

    # Determine role
    role = "assistant" if email in _SELF_EMAILS else "user"

    # Message type: skip system events
    msg_type = msg.get("messageType", msg.get("type", "message")) or "message"

    return {
        "id": msg.get("id", ""),
        "sender_email": email,
        "sender_name": name,
        "role": role,
        "content": body_text[:2000],
        "timestamp": msg.get("createdDateTime", ""),
        "type": msg_type,
        "has_attachments": str(bool(msg.get("attachments"))).lower(),
    }


def normalize_chat(chat: dict) -> dict[str, str]:
    """Normalize a raw Teams chat to a clean dict.

    Output::

        {
            "chat_id": "...",
            "chat_type": "oneOnOne" | "group" | "meeting",
            "topic": "Chat topic or empty",
            "created": "2026-01-15T10:00:00Z",
        }
    """
    return {
        "chat_id": chat.get("id", ""),
        "chat_type": chat.get("chatType", ""),
        "topic": chat.get("topic") or "",
        "created": chat.get("createdDateTime", ""),
    }


def _build_context(
    messages: list[dict], current_msg_id: str, max_context: int = 10
) -> list[dict[str, str]]:
    """Build recent context messages for the VP graph (excludes current message)."""
    context: list[dict[str, str]] = []
    for m in messages:
        if m.get("id") == current_msg_id:
            continue
        nm = normalize_message(m)
        if nm["content"] and nm["type"] == "message":
            context.append({"role": nm["role"], "content": nm["content"][:300]})
    return context[-max_context:]
