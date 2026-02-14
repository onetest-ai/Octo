"""Telegram-specific tools — send_file.

Extracted from graph.py. The transport instance is registered at runtime
by cli.py after the Telegram bot is initialized.
"""
from __future__ import annotations

from langchain_core.tools import tool

# Module-level ref set by cli.py after Telegram transport is initialized
_telegram_transport = None


def set_telegram_transport(tg) -> None:
    """Register the TelegramTransport instance for file-sending tools."""
    global _telegram_transport
    _telegram_transport = tg


@tool
async def send_file(file_path: str, caption: str = "") -> str:
    """Send a file to the user via Telegram.

    Use this when the user asks for a research report, document, or any file
    that an agent has produced. The file is sent as a Telegram document
    attachment — much better than pasting long content inline.

    Common file locations:
    - Research workspace: .octo/workspace/<date>/<filename>
    - Use the Glob or Read tool first to find the file if unsure of the path.

    Args:
        file_path: Path to the file to send (absolute or relative to workspace).
        caption: Short description shown with the file (1-2 sentences).
    """
    import os
    from octo.config import WORKSPACE

    # Resolve relative paths against workspace root
    if not os.path.isabs(file_path):
        file_path = str(WORKSPACE / file_path)

    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"

    if _telegram_transport is None:
        return "Telegram transport not available. The file exists at: " + file_path

    sent = await _telegram_transport.send_document(file_path, caption=caption)
    if sent:
        filename = os.path.basename(file_path)
        return f"File '{filename}' sent to Telegram successfully."
    return f"Failed to send file to Telegram. The file exists at: {file_path}"


TELEGRAM_TOOLS = [send_file]
