"""MCP server management — add, remove, disable, enable, status."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm, Prompt

from octo.config import MCP_CONFIG_PATH

console = Console()


# ── JSON I/O ────────────────────────────────────────────────────────────

def _read_mcp_json() -> dict[str, Any]:
    """Read .mcp.json, return full dict. Creates empty structure if missing."""
    if MCP_CONFIG_PATH.is_file():
        return json.loads(MCP_CONFIG_PATH.read_text("utf-8"))
    return {"mcpServers": {}}


def _write_mcp_json(data: dict[str, Any]) -> None:
    """Write .mcp.json with pretty formatting."""
    MCP_CONFIG_PATH.write_text(json.dumps(data, indent=4) + "\n", "utf-8")


# ── Status ──────────────────────────────────────────────────────────────

def mcp_get_status(tools_by_server: dict[str, list]) -> list[dict[str, Any]]:
    """Return status info for all configured servers."""
    data = _read_mcp_json()
    servers = data.get("mcpServers", {})

    result = []
    for name, spec in servers.items():
        server_type = spec.get("type", "stdio")
        disabled = spec.get("disabled", False)
        tool_count = len(tools_by_server.get(name, []))

        if server_type == "stdio":
            detail = f"{spec.get('command', '')} {' '.join(spec.get('args', []))}"
        else:
            detail = spec.get("url", "")

        # Truncate long details
        if len(detail) > 50:
            detail = detail[:47] + "..."

        result.append({
            "name": name,
            "type": server_type,
            "disabled": disabled,
            "tool_count": tool_count,
            "detail": detail.strip(),
        })
    return result


# ── Disable / Enable ────────────────────────────────────────────────────

def mcp_disable(name: str) -> bool:
    """Disable a server. Returns True if found."""
    data = _read_mcp_json()
    servers = data.get("mcpServers", {})
    if name not in servers:
        console.print(f"[red]Server '{name}' not found in .mcp.json[/red]")
        return False
    if servers[name].get("disabled"):
        console.print(f"[dim]'{name}' is already disabled.[/dim]")
        return False
    servers[name]["disabled"] = True
    _write_mcp_json(data)
    console.print(f"[yellow]Disabled '{name}'.[/yellow]")
    return True


def mcp_enable(name: str) -> bool:
    """Enable a disabled server. Returns True if found."""
    data = _read_mcp_json()
    servers = data.get("mcpServers", {})
    if name not in servers:
        console.print(f"[red]Server '{name}' not found in .mcp.json[/red]")
        return False
    if not servers[name].get("disabled"):
        console.print(f"[dim]'{name}' is already enabled.[/dim]")
        return False
    servers[name].pop("disabled", None)
    _write_mcp_json(data)
    console.print(f"[green]Enabled '{name}'.[/green]")
    return True


# ── Remove ──────────────────────────────────────────────────────────────

def mcp_remove(name: str) -> bool:
    """Remove a server from .mcp.json. Returns True if removed."""
    data = _read_mcp_json()
    servers = data.get("mcpServers", {})
    if name not in servers:
        console.print(f"[red]Server '{name}' not found in .mcp.json[/red]")
        return False
    if not Confirm.ask(f"Remove '{name}' from .mcp.json?", default=False):
        return False
    del servers[name]
    _write_mcp_json(data)
    console.print(f"[green]Removed '{name}'.[/green]")
    return True


# ── Add wizard ──────────────────────────────────────────────────────────

def mcp_add_wizard() -> str | None:
    """Guided wizard to add a new MCP server. Returns server name or None."""
    console.print()
    console.print("[bold cyan]Add MCP Server[/bold cyan]")
    console.print()

    # 1. Server name
    name = Prompt.ask("  Server name (e.g. outlook, jira)").strip()
    if not name:
        console.print("[red]Name required.[/red]")
        return None

    data = _read_mcp_json()
    servers = data.get("mcpServers", {})
    if name in servers:
        if not Confirm.ask(f"  '{name}' already exists. Overwrite?", default=False):
            return None

    # 2. Transport type
    transport = Prompt.ask(
        "  Transport",
        choices=["stdio", "http"],
        default="stdio",
    )

    spec: dict[str, Any] = {"type": transport}

    if transport == "stdio":
        _collect_stdio(spec)
    else:
        _collect_http(spec)

    # Tool filtering
    if Confirm.ask("  Limit which tools are loaded?", default=False):
        tools_str = Prompt.ask("  Tool names (comma-separated, or empty to skip)")
        if tools_str.strip():
            spec["include_tools"] = [t.strip() for t in tools_str.split(",") if t.strip()]

    # Write
    servers[name] = spec
    data["mcpServers"] = servers
    _write_mcp_json(data)

    console.print(f"[green]Added '{name}' to .mcp.json[/green]")
    return name


def _collect_stdio(spec: dict[str, Any]) -> None:
    """Collect stdio server config."""
    spec["command"] = Prompt.ask("  Command", default="npx")
    args_str = Prompt.ask("  Args (space-separated, e.g. -y @org/package)")
    spec["args"] = args_str.split() if args_str.strip() else []

    if Confirm.ask("  Add environment variables?", default=False):
        env: dict[str, str] = {}
        while True:
            kv = Prompt.ask("  KEY=VALUE (or empty to finish)")
            if not kv.strip():
                break
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
        if env:
            spec["env"] = env


def _collect_http(spec: dict[str, Any]) -> None:
    """Collect HTTP/streamable_http server config."""
    spec["url"] = Prompt.ask("  Server URL")

    auth_type = Prompt.ask(
        "  Authentication",
        choices=["none", "headers", "oauth", "client_credentials"],
        default="none",
    )

    if auth_type == "headers":
        headers: dict[str, str] = {}
        console.print("  [dim]Enter headers as KEY=VALUE, empty to finish:[/dim]")
        while True:
            kv = Prompt.ask("  KEY=VALUE (or empty to finish)")
            if not kv.strip():
                break
            if "=" in kv:
                k, v = kv.split("=", 1)
                headers[k.strip()] = v.strip()
        if headers:
            spec["headers"] = headers

    elif auth_type == "oauth":
        auth: dict[str, Any] = {"type": "oauth"}
        auth["client_id"] = Prompt.ask("  OAuth client ID")
        scopes = Prompt.ask("  Scopes (space-separated, or empty)")
        if scopes.strip():
            auth["scopes"] = scopes.strip()
        redirect = Prompt.ask("  Redirect URI", default="http://localhost:9876/callback")
        auth["redirect_uri"] = redirect
        spec["auth"] = auth

    elif auth_type == "client_credentials":
        auth = {"type": "client_credentials"}
        auth["client_id"] = Prompt.ask("  Client ID")
        auth["client_secret_env"] = Prompt.ask("  Env var for client secret (e.g. JIRA_CLIENT_SECRET)")
        scopes = Prompt.ask("  Scopes (space-separated, or empty)")
        if scopes.strip():
            auth["scopes"] = scopes.strip()
        spec["auth"] = auth
