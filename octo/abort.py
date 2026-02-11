"""ESC-to-abort: raw terminal listener for cancelling graph execution."""
from __future__ import annotations

import asyncio
import os
import select
import sys
import termios
import tty
from contextlib import asynccontextmanager
from typing import AsyncIterator


@asynccontextmanager
async def esc_listener(abort_event: asyncio.Event) -> AsyncIterator[None]:
    """Listen for bare ESC on stdin and set *abort_event* when detected.

    Temporarily switches stdin to raw mode so individual key bytes can be
    read.  Terminal settings are always restored on exit.

    No-op when stdin is not a TTY (Telegram, piped input, tests).
    """
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        loop = asyncio.get_running_loop()

        async def _reader() -> None:
            while not abort_event.is_set():
                # Poll stdin with 200ms timeout (non-blocking for the event loop)
                try:
                    ready = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: select.select([fd], [], [], 0.2)[0],
                        ),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                if not ready:
                    continue

                byte = os.read(fd, 1)

                if byte == b"\x1b":
                    # Got ESC byte — wait 50ms for a follow-up.
                    # If nothing follows it's a bare ESC press.
                    try:
                        follow = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, lambda: select.select([fd], [], [], 0.05)[0],
                            ),
                            timeout=0.1,
                        )
                    except asyncio.TimeoutError:
                        follow = []

                    if follow:
                        # Part of an escape sequence (arrows, etc.) — drain & ignore
                        _drain_sync(fd)
                        continue

                    # Bare ESC — abort
                    abort_event.set()
                    return

                if byte == b"\x03":
                    # Ctrl+C in raw mode (SIGINT won't fire)
                    abort_event.set()
                    return

        reader_task = asyncio.create_task(_reader())
        try:
            yield
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
    finally:
        # Always restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _drain_sync(fd: int) -> None:
    """Consume remaining bytes of an escape sequence (up to 8, 10ms each)."""
    for _ in range(8):
        ready, _, _ = select.select([fd], [], [], 0.01)
        if ready:
            os.read(fd, 1)
        else:
            break
