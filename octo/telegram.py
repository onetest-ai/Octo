"""Telegram bot transport — shared thread with console."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

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
    ) -> None:
        self.graph_app = graph_app
        self.thread_id = thread_id
        self.on_message = on_message    # called when user message arrives
        self.on_response = on_response  # called when AI response is ready
        self.callbacks = callbacks or [] # LangChain callbacks (e.g. CLI tool panels)
        self.graph_lock = graph_lock     # shared lock to prevent races with heartbeat/cron
        self._app: Application | None = None

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
            if isinstance(user_content, str):
                tagged = f"[Channel: Telegram | User: {sender_name}]\n\n{user_content}" if sender_name else user_content
            else:
                # Multimodal — prepend tag to the first text block
                tagged = list(user_content)
                if tagged and tagged[0].get("type") == "text" and sender_name:
                    tagged[0] = dict(tagged[0])
                    tagged[0]["text"] = f"[Channel: Telegram | User: {sender_name}]\n\n{tagged[0]['text']}"

            from octo.retry import invoke_with_retry

            if self.graph_lock:
                await self.graph_lock.acquire()
            try:
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
                    if msg.content.strip().lower() in _HANDOFF_PHRASES:
                        continue
                    return msg.content
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
            await update.message.reply_text("Not authorized. Ask the owner to run /authorize.")
            return

        user_text = update.message.text
        sender = self._sender_name(update)
        logger.info("Telegram message from %s: %s", sender, user_text[:100])

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

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming voice messages: transcribe → graph → voice reply."""
        if not update.message or not update.message.voice:
            return

        sender_id = str(update.message.from_user.id) if update.message.from_user else ""
        if not _is_authorized(sender_id):
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

    async def send_proactive(self, text: str, source: str = "") -> None:
        """Send a proactive message to the owner (not a reply to incoming)."""
        if not self._app or not TELEGRAM_OWNER_ID:
            return

        chat_id = int(TELEGRAM_OWNER_ID)

        if source:
            text = f"<b>[{source}]</b>\n\n{text}"

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
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_document))
        self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
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
