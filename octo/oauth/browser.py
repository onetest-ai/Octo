"""Browser-based OAuth redirect handler and local callback server."""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# HTML page served after receiving the OAuth callback
_SUCCESS_HTML = b"""\
HTTP/1.1 200 OK\r
Content-Type: text/html; charset=utf-8\r
Connection: close\r
\r
<!DOCTYPE html>
<html><body style="font-family:system-ui;text-align:center;padding-top:80px">
<h2>Authentication successful</h2>
<p>You can close this tab and return to the terminal.</p>
</body></html>
"""

_ERROR_HTML = b"""\
HTTP/1.1 400 Bad Request\r
Content-Type: text/html; charset=utf-8\r
Connection: close\r
\r
<!DOCTYPE html>
<html><body style="font-family:system-ui;text-align:center;padding-top:80px">
<h2>Authentication failed</h2>
<p>No authorization code received. Please try again.</p>
</body></html>
"""


async def open_browser(url: str) -> None:
    """Open the authorization URL in the default browser.

    Used as ``redirect_handler`` for ``OAuthClientProvider``.
    Prints a visible message so the user knows auth is needed — important
    when this fires mid-chat on a 401.
    """
    import webbrowser

    from rich.console import Console

    console = Console()
    console.print()
    console.print(
        "  [bold yellow]Authentication required[/bold yellow] — "
        "a browser window is opening."
    )
    console.print("  [dim]Complete the login in your browser, then return here.[/dim]")
    console.print()

    logger.info("Opening browser for OAuth: %s", url)
    webbrowser.open(url)


def make_callback_handler(port: int = 9876):
    """Create a callback handler bound to a specific port.

    Returns an async callable ``() -> (authorization_code, state | None)``
    suitable for ``OAuthClientProvider(callback_handler=...)``.
    """

    async def wait_for_callback() -> tuple[str, str | None]:
        """Start a one-shot HTTP server and wait for the OAuth callback."""
        result: asyncio.Future[tuple[str, str | None]] = asyncio.get_event_loop().create_future()

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                request_str = request_line.decode("utf-8", errors="replace")

                # Parse "GET /callback?code=xxx&state=yyy HTTP/1.1"
                parts = request_str.split(" ")
                if len(parts) >= 2:
                    parsed = urlparse(parts[1])
                    qs = parse_qs(parsed.query)
                    code = qs.get("code", [None])[0]
                    state = qs.get("state", [None])[0]

                    if code:
                        writer.write(_SUCCESS_HTML)
                        await writer.drain()
                        if not result.done():
                            result.set_result((code, state))
                    else:
                        writer.write(_ERROR_HTML)
                        await writer.drain()
                else:
                    writer.write(_ERROR_HTML)
                    await writer.drain()
            except Exception as exc:
                logger.warning("Callback handler error: %s", exc)
                if not result.done():
                    result.set_exception(exc)
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(_handle, "127.0.0.1", port)
        logger.info("OAuth callback server listening on http://127.0.0.1:%d", port)

        try:
            code, state = await asyncio.wait_for(result, timeout=300.0)

            from rich.console import Console

            Console().print("  [bold green]Authentication received.[/bold green] Resuming...")

            return code, state
        finally:
            server.close()
            await server.wait_closed()

    return wait_for_callback
