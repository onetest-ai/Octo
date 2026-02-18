"""Telegram bot transport — shared thread with console."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from octo.config import TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_ID, OCTO_DIR

logger = logging.getLogger(__name__)

AUTH_FILE = OCTO_DIR / "authorized_users.json"


def _load_authorized_users() -> dict[str, str]:
    """Load authorized user IDs → names from JSON file."""
    if not AUTH_FILE.is_file():
        return {}
    try:
        return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_authorized_users(users: dict[str, str]) -> None:
    """Save authorized user IDs → names to JSON file."""
    AUTH_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _is_authorized(user_id: str) -> bool:
    """Check if a Telegram user is authorized."""
    if TELEGRAM_OWNER_ID and user_id == TELEGRAM_OWNER_ID:
        return True
    users = _load_authorized_users()
    return user_id in users


def _wrap_markdown_tables(text: str) -> str:
    """Wrap pipe-delimited markdown tables in code fences.

    CommonMark doesn't support GFM tables, so we pre-wrap them in
    code fences so they render as ``<pre>`` blocks in the HTML output.
    """
    lines = text.split("\n")
    result = []
    in_table = False
    table_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        is_separator = is_table_line and set(stripped.replace("|", "").replace("-", "").strip()) <= {" ", ":"}

        if is_table_line or (in_table and is_separator):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                result.append("```")
                result.extend(table_lines)
                result.append("```")
                in_table = False
                table_lines = []
            result.append(line)

    if in_table:
        result.append("```")
        result.extend(table_lines)
        result.append("```")

    return "\n".join(result)


def _markdown_to_telegram_html(text: str) -> str:
    """Convert Markdown to Telegram-compatible HTML.

    Uses markdown-it-py for parsing, then post-processes to Telegram's
    supported HTML subset: <b>, <i>, <s>, <u>, <code>, <pre>, <a>, <blockquote>.
    """
    try:
        from markdown_it import MarkdownIt

        # Pre-process: wrap markdown tables in code fences so they render as <pre>
        text = _wrap_markdown_tables(text)

        md = MarkdownIt("commonmark", {"html": False})
        html = md.render(text)
    except ImportError:
        logger.debug("markdown-it-py not available, using regex fallback")
        html = _regex_markdown_to_html(text)
        return html

    # --- Post-process standard HTML → Telegram subset ---

    # Replace unsupported tags with Telegram equivalents
    html = html.replace("<strong>", "<b>").replace("</strong>", "</b>")
    html = html.replace("<em>", "<i>").replace("</em>", "</i>")
    html = html.replace("<del>", "<s>").replace("</del>", "</s>")

    # Headers → bold text (Telegram has no header tags)
    html = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"<b>\1</b>", html, flags=re.DOTALL)

    # Paragraphs → strip tags, separate with double newline
    html = re.sub(r"<p>(.*?)</p>", r"\1\n", html, flags=re.DOTALL)

    # Horizontal rules
    html = html.replace("<hr>", "———").replace("<hr />", "———").replace("<hr/>", "———")

    # Lists → plain text bullets
    # First handle ordered list items with counters
    _li_counter = [0]

    def _replace_ol_li(match: re.Match) -> str:
        _li_counter[0] += 1
        return f"{_li_counter[0]}. {match.group(1).strip()}\n"

    # Process <ol> blocks: number the items
    def _process_ol(match: re.Match) -> str:
        _li_counter[0] = 0
        content = match.group(1)
        content = re.sub(r"<li>(.*?)</li>", _replace_ol_li, content, flags=re.DOTALL)
        return content

    html = re.sub(r"<ol[^>]*>(.*?)</ol>", _process_ol, html, flags=re.DOTALL)

    # Unordered lists → bullet points
    html = re.sub(r"<li>(.*?)</li>", r"• \1\n", html, flags=re.DOTALL)
    html = re.sub(r"</?[ou]l[^>]*>", "", html)

    # Tables → preformatted text (Telegram doesn't support tables)
    def _table_to_pre(match: re.Match) -> str:
        table_html = match.group(0)
        # Strip all HTML tags from table content
        text_content = re.sub(r"<[^>]+>", " ", table_html)
        text_content = re.sub(r" +", " ", text_content).strip()
        return f"<pre>{text_content}</pre>"

    html = re.sub(r"<table.*?</table>", _table_to_pre, html, flags=re.DOTALL)

    # Strip images (Telegram doesn't render inline images in text)
    html = re.sub(r"<img[^>]*>", "", html)

    # Strip any remaining unsupported tags but keep their content
    _SUPPORTED = {"b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-emoji"}

    def _strip_unsupported(match: re.Match) -> str:
        tag = match.group(1).split()[0].strip("/").lower()
        if tag in _SUPPORTED:
            return match.group(0)
        return ""

    html = re.sub(r"<(/?\w[^>]*)>", _strip_unsupported, html)

    # Clean up excessive newlines
    html = re.sub(r"\n{3,}", "\n\n", html).strip()

    return html


def _regex_markdown_to_html(text: str) -> str:
    """Minimal regex-based Markdown→HTML fallback (no external deps)."""
    # Escape HTML special chars first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Code blocks (``` ... ```)
    text = re.sub(
        r"```\w*\n(.*?)```",
        r"<pre>\1</pre>",
        text,
        flags=re.DOTALL,
    )

    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Bold (**text** or __text__)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # Italic (*text* or _text_) — avoid matching list bullets
    text = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

    # Strikethrough
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Headers → bold
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # Horizontal rules
    text = re.sub(r"^---+$", "———", text, flags=re.MULTILINE)

    return text


class TelegramTransport:
    """Async Telegram bot that feeds into the same LangGraph graph."""

    def __init__(
        self,
        graph_app: Any,
        thread_id: str,
        on_message: Callable[[str], None] | None = None,
        on_response: Callable[[str], None] | None = None,
        callbacks: list | None = None,
        graph_lock: asyncio.Lock | None = None,
        on_command: Callable[[str, str], Any] | None = None,
        # Swarm group-mode params
        swarm_mode: bool = False,
        swarm_role: str = "worker",
        swarm_name: str = "",
        group_chat_id: int | None = None,
    ) -> None:
        self.graph_app = graph_app
        self.thread_id = thread_id
        self.on_message = on_message    # called when user message arrives
        self.on_response = on_response  # called when AI response is ready
        self.callbacks = callbacks or [] # LangChain callbacks (e.g. CLI tool panels)
        self.graph_lock = graph_lock     # shared lock to prevent races with heartbeat/cron
        self.on_command = on_command     # async (cmd, args) -> str | None for slash commands
        self._app: Application | None = None
        # Generic reply router: telegram_msg_id → async handler(update, text)
        # Used by VP notifications, background tasks, and any future reply-routed feature.
        self._reply_handlers: dict[int, Callable[[Update, str], Awaitable[None]]] = {}
        # Swarm group mode — multiple bots in one Telegram group
        self._swarm_mode = swarm_mode
        self._swarm_role = swarm_role     # "supervisor" catches unaddressed msgs
        self._swarm_name = swarm_name     # for name-based mention detection
        self._group_chat_id = group_chat_id
        self._bot_username: str = ""      # resolved at start() via get_me()
        self._bot_id: int = 0             # resolved at start() via get_me()

    def _is_group_chat(self, update: Update) -> bool:
        """Check if the message comes from a group/supergroup chat."""
        if not update.message:
            return False
        return update.message.chat.type in ("group", "supergroup")

    def _should_respond(self, update: Update) -> bool:
        """In swarm group mode, decide if this Octo should respond.

        Rules:
        1. Direct @mention of this bot → respond
        2. Name mention (SWARM_NAME in text) → respond
        3. Reply to this bot's previous message → respond
        4. No bot mentioned at all AND role is supervisor → respond
        5. Otherwise → silently ignore (listen but don't react)
        """
        msg = update.message
        if not msg or not msg.text:
            return False

        text = msg.text

        # 1. Check for @bot_username mention in entities
        if self._bot_username and msg.entities:
            for entity in msg.entities:
                if entity.type == "mention":
                    mention = text[entity.offset:entity.offset + entity.length]
                    if mention.lstrip("@").lower() == self._bot_username.lower():
                        return True

        # 2. Check for SWARM_NAME in text (e.g. "atlas, check the logs")
        if self._swarm_name and self._swarm_name.lower() in text.lower():
            return True

        # 3. Reply to this bot's message → this bot should handle it
        if msg.reply_to_message and msg.reply_to_message.from_user:
            if msg.reply_to_message.from_user.id == self._bot_id:
                return True

        # 4. Supervisor catches unaddressed messages
        if self._swarm_role == "supervisor":
            # But if another bot IS mentioned, let that bot handle it
            if msg.entities:
                for entity in msg.entities:
                    if entity.type == "mention":
                        mention = text[entity.offset:entity.offset + entity.length]
                        # A mention that isn't this bot → another bot is addressed
                        if self._bot_username and mention.lstrip("@").lower() != self._bot_username.lower():
                            return False
            return True

        return False

    def _should_respond_media(self, update: Update) -> bool:
        """Decide if this Octo should handle a media message (document/photo/voice).

        In private chat or non-swarm: always respond.
        In group mode: respond if reply-to this bot, or if supervisor and no reply target.
        """
        if not self._swarm_mode or not self._is_group_chat(update):
            return True

        msg = update.message
        if not msg:
            return False

        # Reply to this bot's message
        if msg.reply_to_message and msg.reply_to_message.from_user:
            if msg.reply_to_message.from_user.id == self._bot_id:
                return True

        # Caption with @mention or name
        caption = msg.caption or ""
        if self._bot_username and f"@{self._bot_username}" in caption:
            return True
        if self._swarm_name and self._swarm_name.lower() in caption.lower():
            return True

        # Supervisor catches unaddressed media
        if self._swarm_role == "supervisor":
            if msg.reply_to_message and msg.reply_to_message.from_user:
                # Replying to another bot's message — let them handle it
                if msg.reply_to_message.from_user.id != self._bot_id:
                    return False
            return True

        return False

    async def send_to_peer(self, peer_bot_username: str, message: str) -> None:
        """Send a message in the group chat mentioning a peer Octo bot.

        Used for visible inter-Octo delegation (human stays in the loop).
        """
        chat_id = self._group_chat_id
        if not chat_id or not self._app:
            logger.warning("send_to_peer: no group chat ID or app not started")
            return

        text = f"@{peer_bot_username} {message}"
        html = _markdown_to_telegram_html(text)
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=html, parse_mode="HTML")
        except Exception:
            try:
                await self._app.bot.send_message(chat_id=chat_id, text=text)
            except Exception:
                logger.exception("Failed to send peer message to group")

    async def _send_typing(self, chat_id: int, stop_event: asyncio.Event) -> None:
        """Keep sending 'typing' action until stop_event is set."""
        try:
            while not stop_event.is_set():
                await self._app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                # Typing indicator lasts ~5s on Telegram, refresh every 4s
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=4)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass  # silently stop if chat action fails

    async def _trim_history_if_needed(self, config: dict) -> None:
        """Trim checkpoint history if it exceeds TELEGRAM_HISTORY_LIMIT.

        Lighter-weight than auto_compact — removes old messages without LLM
        summarization. Runs inside graph lock before every Telegram invocation.
        """
        from octo.config import TELEGRAM_HISTORY_LIMIT
        if not TELEGRAM_HISTORY_LIMIT:
            return

        try:
            state = await self.graph_app.aget_state(config)
            messages = state.values.get("messages", [])

            if len(messages) <= TELEGRAM_HISTORY_LIMIT:
                return

            from langchain_core.messages import RemoveMessage, SystemMessage
            from octo.retry import _sanitize_compact_boundary

            split_idx = _sanitize_compact_boundary(
                messages, len(messages) - TELEGRAM_HISTORY_LIMIT,
            )
            removable = [m for m in messages[:split_idx] if getattr(m, "id", None)]
            if not removable:
                return

            marker = SystemMessage(
                content=(
                    f"[Telegram history trimmed — {len(removable)} older "
                    f"messages removed to stay within {TELEGRAM_HISTORY_LIMIT} "
                    f"message limit.]"
                )
            )
            remove_ops = [RemoveMessage(id=m.id) for m in removable]
            await self.graph_app.aupdate_state(
                config, {"messages": remove_ops + [marker]},
            )
            logger.info(
                "Telegram history trimmed: %d -> %d messages",
                len(messages), len(messages) - len(removable) + 1,
            )
        except Exception:
            logger.debug("Telegram history trim failed (non-blocking)", exc_info=True)

    async def _invoke_graph(self, chat_id: int, user_content: str | list, sender_name: str = "") -> str:
        """Send user content through the graph with typing indicator.

        Args:
            chat_id: Telegram chat ID (for typing indicator).
            user_content: String or list of content blocks (multimodal).
            sender_name: Display name for channel tagging.
        """
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._send_typing(chat_id, stop_typing))

        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            if self.callbacks:
                config["callbacks"] = self.callbacks

            # Tag message with channel metadata so supervisor can adapt formatting
            if self._swarm_mode:
                channel_tag = f"[Channel: Telegram Group | User: {sender_name} | Bot: {self._swarm_name}]"
            else:
                channel_tag = f"[Channel: Telegram | User: {sender_name}]" if sender_name else ""
            if isinstance(user_content, str):
                tagged = f"{channel_tag}\n\n{user_content}" if channel_tag else user_content
            else:
                # Multimodal — prepend tag to the first text block
                tagged = list(user_content)
                if tagged and tagged[0].get("type") == "text" and channel_tag:
                    tagged[0] = dict(tagged[0])
                    tagged[0]["text"] = f"{channel_tag}\n\n{tagged[0]['text']}"

            from octo.retry import invoke_with_retry

            if self.graph_lock:
                await self.graph_lock.acquire()
            try:
                await self._trim_history_if_needed(config)
                result = await invoke_with_retry(
                    self.graph_app,
                    {"messages": [HumanMessage(content=tagged)]},
                    config,
                )
            finally:
                if self.graph_lock:
                    self.graph_lock.release()
            # Extract the last substantive AI message, skipping handoff boilerplate
            _HANDOFF_PHRASES = {"transferring back to supervisor", "transferring to supervisor"}
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    # content can be str or list of blocks (Bedrock multi-block)
                    content = msg.content
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        ).strip()
                    if not content:
                        continue
                    if content.strip().lower() in _HANDOFF_PHRASES:
                        continue
                    return content
            return ""
        finally:
            stop_typing.set()
            await typing_task

    async def _reply(self, update: Update, response_text: str, voice_reply: bool = False) -> None:
        """Send response as formatted HTML (always) and optionally as voice."""
        if not response_text:
            return

        # Send as Telegram HTML with plain-text fallback
        for chunk in _split_message(response_text):
            html_chunk = _markdown_to_telegram_html(chunk)
            try:
                await update.message.reply_text(html_chunk, parse_mode="HTML")
            except Exception:
                # Fallback to plain text if HTML parsing fails
                logger.debug("HTML send failed, falling back to plain text")
                await update.message.reply_text(chunk)

        # Send voice reply if requested and ElevenLabs is available
        if voice_reply:
            try:
                from octo.voice import synthesize
                audio = await synthesize(response_text)
                if audio:
                    await update.message.reply_voice(voice=audio)
            except Exception:
                logger.exception("Failed to send voice reply")

        # Echo response to CLI console
        if self.on_response:
            self.on_response(response_text)

    def _sender_name(self, update: Update) -> str:
        """Get display name for the Telegram sender."""
        user = update.message.from_user if update.message else None
        if user:
            return user.full_name or user.username or str(user.id)
        return "Unknown"

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming document/photo attachments."""
        if not update.message:
            return

        # Swarm group mode — only handle if addressed to this bot
        if self._swarm_mode and self._is_group_chat(update):
            if not self._should_respond_media(update):
                return

        sender_id = str(update.message.from_user.id) if update.message.from_user else ""
        if not _is_authorized(sender_id):
            await update.message.reply_text("Not authorized. Ask the owner to run /authorize.")
            return

        sender = self._sender_name(update)
        caption = update.message.caption or ""

        # Determine file to download
        if update.message.document:
            file_obj = update.message.document
            file_name = file_obj.file_name or "document"
        elif update.message.photo:
            # Photos come as a list of sizes — pick the largest
            file_obj = update.message.photo[-1]
            file_name = "photo.jpg"
        else:
            return

        logger.info("Telegram file from %s: %s", sender, file_name)

        if self.on_message:
            self.on_message(f"[{sender}] (file) {file_name}: {caption or '(no caption)'}")

        try:
            status_msg = await update.message.reply_text("Processing file...")

            # Download to uploads
            from octo.attachments import copy_to_uploads, UPLOADS_DIR, _IMAGE_EXTENSIONS, _BINARY_EXTENSIONS
            import tempfile

            tg_file = await context.bot.get_file(file_obj.file_id)
            ext = Path(file_name).suffix.lower()

            # Reject binaries
            if ext in _BINARY_EXTENSIONS:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text(f"Binary files ({ext}) are not supported.")
                return

            # Download to temp, then copy to uploads
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
            await tg_file.download_to_drive(tmp_path)
            dest = copy_to_uploads(tmp_path, filename=file_name)
            os.unlink(tmp_path)

            # Build message with attachment reference
            if ext in _IMAGE_EXTENSIONS:
                import base64 as _b64
                import mimetypes as _mt
                with open(dest, "rb") as f:
                    data = _b64.b64encode(f.read()).decode("utf-8")
                mime = _mt.guess_type(dest)[0] or "image/jpeg"
                user_content = [
                    {"type": "text", "text": caption or "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ]
            else:
                user_content = (
                    f"{caption or 'Process the attached file.'}\n\n"
                    f"[Attached: `{dest}` ({file_name}) — "
                    f"use appropriate tools or skills to read this file]"
                )

            try:
                await status_msg.edit_text("Processing...")
            except Exception:
                pass

            response_text = await self._invoke_graph(
                update.message.chat_id, user_content, sender_name=sender,
            )
            try:
                await status_msg.delete()
            except Exception:
                pass
            await self._reply(update, response_text)

        except Exception:
            logger.exception("Error handling Telegram document")
            await update.message.reply_text("Something went wrong. Check the console.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return

        sender_id = str(update.message.from_user.id) if update.message.from_user else ""
        if not _is_authorized(sender_id):
            # In group mode, silently ignore unauthorized instead of replying
            if self._swarm_mode and self._is_group_chat(update):
                return
            await update.message.reply_text("Not authorized. Ask the owner to run /authorize.")
            return

        # Swarm group mode — only respond if addressed to this bot
        if self._swarm_mode and self._is_group_chat(update):
            if not self._should_respond(update):
                return

        user_text = update.message.text
        sender = self._sender_name(update)
        logger.info("Telegram message from %s: %s", sender, user_text[:100])

        # Strip @bot_username from the text so the graph doesn't see it
        if self._swarm_mode and self._bot_username:
            user_text = re.sub(
                rf"@{re.escape(self._bot_username)}\b", "", user_text,
            ).strip()

        # --- Reply routing: reply to a tracked message → dispatch to handler ---
        reply_to = update.message.reply_to_message
        if reply_to and reply_to.message_id in self._reply_handlers:
            handler = self._reply_handlers.pop(reply_to.message_id)
            await handler(update, user_text)
            return

        # --- Slash command routing → CLI command handler ---
        if user_text.startswith("/") and self.on_command:
            parts = user_text.split(maxsplit=1)
            cmd = parts[0][1:]  # strip leading /
            args = parts[1] if len(parts) > 1 else ""
            if self.on_message:
                self.on_message(f"[{sender}] {user_text}")
            try:
                result = await self.on_command(cmd, args)
                if result is not None:
                    await self._reply(update, result)
                    return
                # result is None → command not recognized, fall through to graph
            except Exception as e:
                logger.exception("Telegram command /%s failed", cmd)
                await update.message.reply_text(f"Command failed: {e}")
                return

        # Show in CLI console immediately
        if self.on_message:
            self.on_message(f"[{sender}] {user_text}")

        try:
            # Send a status message so the user knows we're working on it
            status_msg = await update.message.reply_text("Processing...")
            response_text = await self._invoke_graph(update.message.chat_id, user_text, sender_name=sender)
            # Delete the status message before sending the real response
            try:
                await status_msg.delete()
            except Exception:
                pass
            await self._reply(update, response_text)
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logger.warning("Telegram message handling timed out: %s", e)
                await update.message.reply_text("Request timed out. Please try again.")
            elif "rate limit" in error_str or "throttling" in error_str:
                logger.warning("Rate limited during Telegram message: %s", e)
                await update.message.reply_text("Rate limited. Please wait a moment and try again.")
            elif "too long" in error_str or "context length" in error_str:
                logger.warning("Context overflow during Telegram message: %s", e)
                await update.message.reply_text("Conversation too long. Send /clear to reset, or ask me to /compact.")
            else:
                logger.exception("Error handling Telegram message")
                await update.message.reply_text("Something went wrong. Check the console.")

    async def _handle_vp_reply(
        self, update: Update, text: str,
        teams_chat_id: str = "", teams_message_id: str = "",
    ) -> None:
        """Handle a reply to a VP escalation notification.

        The reply is an instruction TO Octo, not the literal text to send.
        For example: "tell her we'll meet next week" or "yes, proceed with option B".

        Flow:
        - Reply "ignore" → mute Teams chat + release delegation.
        - Any other text → Octo processes as instruction → persona formats → sends to Teams.
        """
        reply_to = update.message.reply_to_message

        if not teams_chat_id:
            await update.message.reply_text("Could not resolve Teams chat.")
            return

        from octo.config import VP_DIR
        from octo.virtual_persona.access_control import AccessControl

        ac = AccessControl(VP_DIR / "access-control.yaml")
        cmd = text.strip().lower()

        if cmd == "ignore":
            ac.ignore_chat(teams_chat_id)
            ac.release_thread(teams_chat_id)
            await update.message.reply_text("Chat ignored and delegation released.")
            logger.info("VP reply: ignored chat %s", teams_chat_id[:30])
            return

        # The reply is an instruction to Octo — process it through the
        # full pipeline: Octo supervisor → persona formatting → send to Teams.
        status_msg = await update.message.reply_text("Processing...")

        # Extract the original notification text for context
        original_notification = reply_to.text or reply_to.caption or ""

        try:
            formatted = await self._process_vp_instruction(
                instruction=text,
                teams_chat_id=teams_chat_id,
                notification_context=original_notification,
            )
        except Exception as exc:
            logger.error("VP reply processing failed: %s", exc)
            await status_msg.edit_text(f"Processing failed: {exc}")
            return

        if not formatted:
            await status_msg.edit_text("Could not generate a response.")
            return

        # Send the formatted response to Teams
        from octo.graph import get_mcp_tool
        from octo.virtual_persona.poller import VPPoller

        tool = get_mcp_tool("send-chat-message")
        if not tool:
            await status_msg.edit_text("No Teams MCP tools available.")
            return

        try:
            result = await tool.ainvoke({
                "chatId": teams_chat_id,
                "content": formatted,
            })
            data = VPPoller._parse_tool_result(result)
            if isinstance(data, dict) and "error" in data:
                await status_msg.edit_text(f"Failed to send: {data['error']}")
                return
        except Exception as exc:
            await status_msg.edit_text(f"Failed to send: {exc}")
            return

        ac.release_thread(teams_chat_id)
        # Show what was sent
        preview = formatted[:200] + ("..." if len(formatted) > 200 else "")
        await status_msg.edit_text(f"Sent to Teams. Delegation released.\n\n{preview}")
        logger.info("VP reply: sent to Teams chat %s", teams_chat_id[:30])

    async def _process_vp_instruction(
        self,
        instruction: str,
        teams_chat_id: str,
        notification_context: str,
    ) -> str:
        """Process a VP reply instruction through Octo + persona formatting.

        Takes the user's instruction (e.g. "tell her we'll meet next week"),
        feeds it through Octo supervisor for substance, then persona-formats
        the result into the user's voice.

        Returns the formatted response text, or empty string on failure.
        """
        if not self.graph_app:
            # Fallback: no Octo app available, just persona-format the instruction directly
            return await self._persona_format_only(instruction)

        octo_thread_id = f"vp:{teams_chat_id}"

        # Build instruction with context from the original notification
        prompt = (
            "You are helping the user respond to a Teams message. "
            "The user has read the escalation notification and replied with an instruction. "
            "Generate the actual response to send to the colleague based on the user's instruction.\n\n"
        )
        if notification_context:
            prompt += f"Original notification:\n{notification_context[:1500]}\n\n"
        prompt += f"User's instruction: {instruction}\n\n"
        prompt += (
            "Generate a natural, complete response to send to the colleague. "
            "Do NOT include any meta-commentary — just the response text itself."
        )

        config = {"configurable": {"thread_id": octo_thread_id}}

        try:
            if self.graph_lock:
                await self.graph_lock.acquire()
            try:
                result = await self.graph_app.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config=config,
                )
            finally:
                if self.graph_lock:
                    self.graph_lock.release()

            # Extract last AI message
            messages = result.get("messages", [])
            raw = ""
            for msg in reversed(messages):
                content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
                if content and hasattr(msg, "type") and msg.type == "ai":
                    raw = content
                    break
            if not raw and messages:
                last = messages[-1]
                raw = getattr(last, "content", str(last))

            if not raw:
                return ""

            # Persona format the raw answer
            return await self._persona_format_only(raw)
        except Exception as exc:
            logger.error("VP instruction processing via Octo failed: %s", exc)
            # Fallback: persona-format the instruction directly
            return await self._persona_format_only(instruction)

    async def _persona_format_only(self, text: str) -> str:
        """Quick persona formatting without full Octo pipeline."""
        try:
            from octo.virtual_persona.graph import _load_persona_prompt
            from octo.models import make_model

            persona_prompt = _load_persona_prompt()
            format_prompt = (
                f"{persona_prompt}\n\n---\n"
                f"Rewrite the following in the user's style. Keep it concise and natural. "
                f"Do NOT add any preamble — just the reformatted text.\n\n"
                f"Text:\n{text[:3000]}"
            )
            model = make_model(tier="low")
            response = await model.ainvoke(format_prompt)
            result = response.content.strip()
            if "🤖" not in result:
                result += " 🤖"
            return result
        except Exception as exc:
            logger.warning("VP persona format fallback failed: %s", exc)
            return text + " 🤖" if "🤖" not in text else text

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming voice messages: transcribe → graph → voice reply."""
        if not update.message or not update.message.voice:
            return

        # Swarm group mode — only handle if addressed to this bot
        if self._swarm_mode and self._is_group_chat(update):
            if not self._should_respond_media(update):
                return

        sender_id = str(update.message.from_user.id) if update.message.from_user else ""
        if not _is_authorized(sender_id):
            if self._swarm_mode and self._is_group_chat(update):
                return
            await update.message.reply_text("Not authorized. Ask the owner to run /authorize.")
            return

        sender = self._sender_name(update)
        logger.info("Telegram voice from %s (%ds)", sender, update.message.voice.duration or 0)

        # Show in CLI console immediately
        if self.on_message:
            self.on_message(f"[{sender}] (voice message)")

        try:
            # Send status so user knows we're working
            status_msg = await update.message.reply_text("Transcribing...")

            # Download voice file
            file = await context.bot.get_file(update.message.voice.file_id)
            audio_bytes = bytes(await file.download_as_bytearray())

            # Transcribe
            from octo.voice import transcribe
            user_text = await transcribe(audio_bytes)
            if not user_text:
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await update.message.reply_text("Could not transcribe the voice message.")
                return

            logger.info("Transcribed voice from %s: %s", sender, user_text[:100])

            # Show transcription in CLI
            if self.on_message:
                self.on_message(f"[{sender}] (transcribed) {user_text}")

            # Update status and process
            try:
                await status_msg.edit_text("Processing...")
            except Exception:
                pass
            response_text = await self._invoke_graph(update.message.chat_id, user_text, sender_name=sender)
            try:
                await status_msg.delete()
            except Exception:
                pass
            await self._reply(update, response_text, voice_reply=True)

        except Exception:
            logger.exception("Error handling Telegram voice message")
            await update.message.reply_text("Something went wrong. Check the console.")

    async def _handle_authorize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /authorize command — owner-only, adds a user by reply or ID."""
        if not update.message or not update.message.from_user:
            return

        sender_id = str(update.message.from_user.id)
        # Only owner can authorize
        if not TELEGRAM_OWNER_ID or sender_id != TELEGRAM_OWNER_ID:
            await update.message.reply_text("Only the owner can authorize users.")
            return

        users = _load_authorized_users()

        # If replying to someone's message, authorize that user
        if update.message.reply_to_message and update.message.reply_to_message.from_user:
            target = update.message.reply_to_message.from_user
            target_id = str(target.id)
            target_name = target.full_name or target.username or target_id
            users[target_id] = target_name
            _save_authorized_users(users)
            await update.message.reply_text(f"Authorized: {target_name} ({target_id})")
            return

        # Otherwise, expect /authorize <user_id> [name]
        args = context.args or []
        if not args:
            # List current authorized users
            if not users:
                await update.message.reply_text("No authorized users. Use /authorize <user_id> or reply to a message.")
                return
            lines = [f"  {uid}: {name}" for uid, name in users.items()]
            await update.message.reply_text("Authorized users:\n" + "\n".join(lines))
            return

        target_id = args[0]
        target_name = " ".join(args[1:]) if len(args) > 1 else target_id
        users[target_id] = target_name
        _save_authorized_users(users)
        await update.message.reply_text(f"Authorized: {target_name} ({target_id})")

    async def _handle_revoke(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /revoke command — owner-only, removes a user."""
        if not update.message or not update.message.from_user:
            return

        sender_id = str(update.message.from_user.id)
        if not TELEGRAM_OWNER_ID or sender_id != TELEGRAM_OWNER_ID:
            await update.message.reply_text("Only the owner can revoke access.")
            return

        args = context.args or []
        if not args:
            await update.message.reply_text("Usage: /revoke <user_id>")
            return

        target_id = args[0]
        users = _load_authorized_users()
        removed = users.pop(target_id, None)
        if removed:
            _save_authorized_users(users)
            await update.message.reply_text(f"Revoked: {removed} ({target_id})")
        else:
            await update.message.reply_text(f"User {target_id} not found.")

    def _target_chat_id(self) -> int | None:
        """Resolve the chat ID for outbound messages.

        In swarm group mode, target the shared group chat.
        Otherwise, target the owner's private chat.
        """
        if self._swarm_mode and self._group_chat_id:
            return self._group_chat_id
        if TELEGRAM_OWNER_ID:
            return int(TELEGRAM_OWNER_ID)
        return None

    async def send_proactive(self, text: str, source: str = "") -> None:
        """Send a proactive message to the owner or swarm group."""
        if not self._app:
            return

        chat_id = self._target_chat_id()
        if not chat_id:
            return

        if source:
            text = f"**[{source}]**\n\n{text}"

        for chunk in _split_message(text):
            html_chunk = _markdown_to_telegram_html(chunk)
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id, text=html_chunk, parse_mode="HTML",
                )
            except Exception:
                logger.debug("Proactive HTML send failed, falling back to plain text")
                try:
                    await self._app.bot.send_message(chat_id=chat_id, text=chunk)
                except Exception:
                    logger.exception("Failed to send proactive message to Telegram")

    async def send_tracked_message(
        self,
        text: str,
        on_reply: Callable[[Update, str], Awaitable[None]],
    ) -> None:
        """Send a message to the owner and register a reply handler.

        When the user replies to this Telegram message, ``on_reply(update, text)``
        is called.  This is the generic mechanism used by VP notifications,
        background tasks, and any future reply-routed feature.
        """
        if not self._app:
            return

        chat_id = self._target_chat_id()
        if not chat_id:
            return
        sent_msg = None
        for chunk in _split_message(text):
            html_chunk = _markdown_to_telegram_html(chunk)
            try:
                sent_msg = await self._app.bot.send_message(
                    chat_id=chat_id, text=html_chunk, parse_mode="HTML",
                )
            except Exception:
                try:
                    sent_msg = await self._app.bot.send_message(
                        chat_id=chat_id, text=chunk,
                    )
                except Exception:
                    logger.exception("Failed to send tracked message")

        if sent_msg:
            self._reply_handlers[sent_msg.message_id] = on_reply
            # Cap map size to prevent unbounded growth
            if len(self._reply_handlers) > 200:
                oldest = list(self._reply_handlers)[:100]
                for k in oldest:
                    del self._reply_handlers[k]

    async def send_bg_notification(self, text: str, task_id: str) -> None:
        """Send a background task notification, tracked for reply routing.

        Reply routing behavior depends on task status:
        - **paused**: resumes the task with the reply as the answer
        - **completed/failed**: spawns a follow-up task with original context
        """

        async def _on_reply(update: Update, reply_text: str) -> None:
            from octo.background import get_worker_pool
            pool = get_worker_pool()
            if not pool:
                await update.message.reply_text("Background worker pool not initialized.")
                return

            # Try resume first (for paused tasks)
            ok = await pool.resume_task(task_id, reply_text)
            if ok:
                await update.message.reply_text(f"Task {task_id} resumed with your answer.")
                return

            # Not paused — spawn a follow-up task with original context
            new_task = await pool.follow_up(task_id, reply_text)
            if new_task:
                await update.message.reply_text(
                    f"Follow-up task {new_task.id} dispatched "
                    f"(based on {task_id} results)."
                )
            else:
                await update.message.reply_text(f"Task {task_id} not found.")

        await self.send_tracked_message(text, on_reply=_on_reply)

    async def send_vp_notification(
        self, text: str, teams_chat_id: str, teams_message_id: str,
    ) -> None:
        """Send a VP notification, tracked for reply routing to Teams."""

        async def _on_reply(update: Update, reply_text: str) -> None:
            await self._handle_vp_reply(update, reply_text, teams_chat_id, teams_message_id)

        await self.send_tracked_message(text, on_reply=_on_reply)

    async def send_document(self, file_path: str, caption: str = "", chat_id: int | None = None) -> bool:
        """Send a file as a Telegram document.

        Args:
            file_path: Absolute path to the file to send.
            caption: Optional caption (max 1024 chars, HTML supported).
            chat_id: Target chat. Defaults to TELEGRAM_OWNER_ID.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._app:
            return False
        target = chat_id or (int(TELEGRAM_OWNER_ID) if TELEGRAM_OWNER_ID else None)
        if not target:
            return False

        try:
            with open(file_path, "rb") as f:
                html_caption = _markdown_to_telegram_html(caption)[:1024] if caption else None
                await self._app.bot.send_document(
                    chat_id=target,
                    document=f,
                    caption=html_caption,
                    parse_mode="HTML" if html_caption else None,
                )
            return True
        except Exception:
            logger.exception("Failed to send document %s to Telegram", file_path)
            return False

    async def start(self) -> None:
        """Start the Telegram bot (non-blocking)."""
        if not TELEGRAM_BOT_TOKEN:
            logger.info("No TELEGRAM_BOT_TOKEN set, skipping Telegram transport")
            return

        self._app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._app.add_handler(CommandHandler("authorize", self._handle_authorize))
        self._app.add_handler(CommandHandler("revoke", self._handle_revoke))
        # Non-slash text → graph. Slash commands not caught by CommandHandlers
        # above are routed via a separate handler to on_command callback.
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self._app.add_handler(MessageHandler(filters.COMMAND, self._handle_message))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_document))
        self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

        await self._app.initialize()
        await self._app.start()

        # Resolve bot identity for swarm mention detection
        bot_info = await self._app.bot.get_me()
        self._bot_username = bot_info.username or ""
        self._bot_id = bot_info.id
        logger.info("Bot identity: @%s (id=%d)", self._bot_username, self._bot_id)

        # In group mode, allow the bot to see all messages (disable privacy mode)
        if self._swarm_mode:
            await self._app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
            )
        else:
            await self._app.updater.start_polling(drop_pending_updates=True)

        # Register command menu with Telegram so users see autocomplete
        from telegram import BotCommand
        commands = [
            BotCommand("help", "Show available commands"),
            BotCommand("clear", "Reset conversation (new thread)"),
            BotCommand("compact", "Free context space"),
            BotCommand("context", "Context window usage"),
            BotCommand("agents", "List loaded agents"),
            BotCommand("skills", "List / manage skills"),
            BotCommand("mcp", "MCP server status / management"),
            BotCommand("model", "Show current model"),
            BotCommand("reload", "Hot-reload graph and config"),
            BotCommand("restart", "Cold-restart process"),
            BotCommand("authorize", "Authorize a user (owner only)"),
            BotCommand("revoke", "Revoke user access (owner only)"),
        ]
        if self._swarm_mode:
            commands.append(BotCommand("swarm", "Swarm status"))
        await self._app.bot.set_my_commands(commands)

        if self._swarm_mode:
            logger.info(
                "Telegram bot started (swarm group mode, role=%s, name=%s)",
                self._swarm_role, self._swarm_name,
            )
        else:
            logger.info("Telegram bot started")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            try:
                await self._app.updater.stop()
            except BaseException:
                logger.debug("Telegram updater stop error (ignored)", exc_info=True)
            try:
                await self._app.stop()
            except BaseException:
                logger.debug("Telegram app stop error (ignored)", exc_info=True)
            try:
                await self._app.shutdown()
            except BaseException:
                logger.debug("Telegram app shutdown error (ignored)", exc_info=True)
            logger.info("Telegram bot stopped")


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into Telegram-safe chunks."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
