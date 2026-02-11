"""CLI commands for managing MCP server OAuth authentication."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from octo.config import MCP_CONFIG_PATH, OAUTH_DIR
from octo.oauth.storage import FileTokenStorage

console = Console()


def _get_auth_servers() -> dict[str, dict]:
    """Return servers from .mcp.json that have an ``auth`` block."""
    if not MCP_CONFIG_PATH.is_file():
        return {}
    raw = json.loads(MCP_CONFIG_PATH.read_text("utf-8"))
    servers = raw.get("mcpServers", {})
    return {
        name: spec["auth"]
        for name, spec in servers.items()
        if spec.get("auth")
    }


async def handle_auth(action: str, server_name: str | None) -> None:
    """Dispatch auth subcommands."""
    if action == "login":
        if not server_name:
            console.print("[red]Usage: octo auth login <server_name>[/red]")
            return
        await _login(server_name)
    elif action == "status":
        _status()
    elif action == "logout":
        if not server_name:
            console.print("[red]Usage: octo auth logout <server_name>[/red]")
            return
        _logout(server_name)


async def _login(server_name: str) -> None:
    """Trigger OAuth flow for a specific MCP server."""
    auth_servers = _get_auth_servers()
    if server_name not in auth_servers:
        console.print(f"[red]Server '{server_name}' not found or has no auth config.[/red]")
        available = list(auth_servers.keys())
        if available:
            console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        return

    auth_config = auth_servers[server_name]
    auth_type = auth_config.get("type", "")

    # Read the server URL from .mcp.json
    raw = json.loads(MCP_CONFIG_PATH.read_text("utf-8"))
    server_url = raw["mcpServers"][server_name].get("url", "")

    if auth_type == "oauth":
        await _login_oauth(server_name, auth_config, server_url)
    elif auth_type == "client_credentials":
        await _login_client_credentials(server_name, auth_config, server_url)
    else:
        console.print(f"[red]Unknown auth type '{auth_type}'[/red]")


async def _login_oauth(server_name: str, auth_config: dict, server_url: str) -> None:
    """Run Authorization Code + PKCE flow (opens browser)."""
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from octo.oauth.browser import make_callback_handler, open_browser

    redirect_uri = auth_config.get("redirect_uri", "http://localhost:9876/callback")
    port = int(redirect_uri.rsplit(":", 1)[-1].split("/")[0]) if ":" in redirect_uri else 9876

    storage = FileTokenStorage(server_name, OAUTH_DIR)
    metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl(redirect_uri)],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=auth_config.get("scopes"),
        client_name=f"Octo MCP ({server_name})",
    )

    # Plain client_id → pre-seed storage; URL → CIMD
    client_id = auth_config.get("client_id", "")
    client_metadata_url = None
    if client_id.startswith("https://"):
        client_metadata_url = client_id
    elif client_id:
        from octo.loaders.mcp_loader import _preseed_client_info
        _preseed_client_info(storage, client_id, redirect_uri)

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=open_browser,
        callback_handler=make_callback_handler(port),
        timeout=300.0,
        client_metadata_url=client_metadata_url,
    )

    console.print(f"[yellow]Starting OAuth flow for '{server_name}'...[/yellow]")
    console.print("[dim]A browser window will open. Complete the login there.[/dim]")

    try:
        # Trigger the auth flow by calling ensure_token
        await provider.ensure_token()
        console.print(f"[green]Authenticated '{server_name}' successfully.[/green]")
    except Exception as exc:
        console.print(f"[red]Authentication failed: {exc}[/red]")


async def _login_client_credentials(
    server_name: str, auth_config: dict, server_url: str
) -> None:
    """Fetch tokens using client_credentials grant (no browser)."""
    import os

    from mcp.client.auth.extensions.client_credentials import (
        ClientCredentialsOAuthProvider,
    )

    secret_env = auth_config.get("client_secret_env", "")
    client_secret = os.getenv(secret_env, "") if secret_env else auth_config.get("client_secret", "")

    if not client_secret:
        console.print(
            f"[red]No client secret found. "
            f"Set the '{secret_env}' environment variable.[/red]"
        )
        return

    storage = FileTokenStorage(server_name, OAUTH_DIR)
    provider = ClientCredentialsOAuthProvider(
        server_url=server_url,
        storage=storage,
        client_id=auth_config.get("client_id", ""),
        client_secret=client_secret,
        token_endpoint_auth_method=auth_config.get(
            "token_endpoint_auth_method", "client_secret_basic"
        ),
        scopes=auth_config.get("scopes"),
    )

    console.print(f"[yellow]Fetching tokens for '{server_name}'...[/yellow]")

    try:
        await provider.ensure_token()
        console.print(f"[green]Authenticated '{server_name}' successfully.[/green]")
    except Exception as exc:
        console.print(f"[red]Authentication failed: {exc}[/red]")


def _status() -> None:
    """Show auth status for all OAuth-configured servers."""
    auth_servers = _get_auth_servers()
    if not auth_servers:
        console.print("[dim]No MCP servers with auth config found in .mcp.json[/dim]")
        return

    table = Table(
        title="MCP OAuth Status",
        border_style="cyan",
        box=box.SIMPLE,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Server", style="bold yellow", no_wrap=True)
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    for name, auth_config in auth_servers.items():
        auth_type = auth_config.get("type", "unknown")
        storage = FileTokenStorage(name, OAUTH_DIR)
        if storage.has_tokens():
            status = "[green]authenticated[/green]"
        else:
            status = "[red]not authenticated[/red]"
        table.add_row(name, auth_type, status)

    console.print(table)
    console.print()


def _logout(server_name: str) -> None:
    """Delete stored tokens for a server."""
    auth_servers = _get_auth_servers()
    if server_name not in auth_servers:
        console.print(f"[red]Server '{server_name}' not found or has no auth config.[/red]")
        return

    storage = FileTokenStorage(server_name, OAUTH_DIR)
    if storage.has_tokens():
        storage.clear()
        console.print(f"[green]Logged out of '{server_name}'.[/green]")
    else:
        console.print(f"[dim]'{server_name}' was not authenticated.[/dim]")
