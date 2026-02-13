"""MCP server management — add, remove, disable, enable, status, registry."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from rich import box
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

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


# ── MCP Registry ───────────────────────────────────────────────────────

def _registry_fetch(endpoint: str, cache_key: str = "") -> Any:
    """Fetch a JSON endpoint from the MCP registry, with optional file cache."""
    from octo.config import MCP_REGISTRY_CACHE_DIR, MCP_REGISTRY_CACHE_TTL

    cache_path = MCP_REGISTRY_CACHE_DIR / cache_key if cache_key else None

    if cache_path and cache_path.is_file():
        age = time.time() - cache_path.stat().st_mtime
        if age < MCP_REGISTRY_CACHE_TTL:
            try:
                return json.loads(cache_path.read_text("utf-8"))
            except json.JSONDecodeError:
                pass  # stale/corrupt cache, re-fetch

    req = urllib.request.Request(endpoint)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2) + "\n", "utf-8")

    return data


def mcp_registry_search(query: str, limit: int = 20) -> list[dict]:
    """Search the official MCP registry. Returns list of server summaries."""
    from octo.config import MCP_REGISTRY_BASE_URL

    url = (
        f"{MCP_REGISTRY_BASE_URL}/v0/servers"
        f"?search={urllib.parse.quote(query)}&limit={limit}"
    )
    safe_key = re.sub(r"[^\w-]", "_", query)[:40]
    raw = _registry_fetch(url, f"search_{safe_key}.json")

    results: list[dict] = []
    for entry in raw.get("servers", []):
        srv = entry.get("server", {})
        meta = entry.get("_meta", {})
        official = meta.get("io.modelcontextprotocol.registry/official", {})
        if not official.get("isLatest", True):
            continue

        packages = srv.get("packages", [])
        remotes = srv.get("remotes", [])
        reg_types = sorted({p.get("registryType", "?") for p in packages})

        results.append({
            "name": srv.get("name", ""),
            "description": srv.get("description", "")[:80],
            "version": srv.get("version", "?"),
            "has_packages": bool(packages),
            "has_remotes": bool(remotes),
            "registry_types": reg_types,
        })

    return results


def mcp_registry_get_server(server_name: str) -> dict | None:
    """Fetch full server details from the MCP registry."""
    from octo.config import MCP_REGISTRY_BASE_URL

    encoded = urllib.parse.quote(server_name, safe="")
    url = f"{MCP_REGISTRY_BASE_URL}/v0.1/servers/{encoded}/versions/latest"
    safe_key = re.sub(r"[^\w-]", "_", server_name)[:60]

    try:
        raw = _registry_fetch(url, f"server_{safe_key}.json")
        return raw.get("server", raw)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    except (urllib.error.URLError, OSError):
        raise

    # Fallback: search by exact name
    url = (
        f"{MCP_REGISTRY_BASE_URL}/v0/servers"
        f"?search={urllib.parse.quote(server_name)}&limit=10"
    )
    try:
        raw = _registry_fetch(url, f"search_exact_{safe_key}.json")
    except (urllib.error.URLError, OSError):
        return None

    for entry in raw.get("servers", []):
        srv = entry.get("server", {})
        if srv.get("name") == server_name:
            meta = entry.get("_meta", {})
            official = meta.get(
                "io.modelcontextprotocol.registry/official", {}
            )
            if official.get("isLatest", True):
                return srv
    return None


# ── Install wizard ─────────────────────────────────────────────────────

def _derive_local_name(server_name: str) -> str:
    """Derive a short local name from a registry server name.

    'io.github.getsentry/sentry-mcp' -> 'sentry-mcp'
    'ai.waystation/postgres'         -> 'postgres'
    'com.notion/mcp'                 -> 'notion-mcp'
    """
    if "/" in server_name:
        local = server_name.rsplit("/", 1)[-1]
    else:
        local = server_name

    if local == "mcp" and "/" in server_name:
        parts = server_name.split("/")
        prefix = parts[-2].split(".")[-1] if len(parts) >= 2 else ""
        local = f"{prefix}-mcp" if prefix else "mcp"

    return local


def _prompt_env_var(evar: dict) -> str | None:
    """Prompt for a single environment variable value."""
    name = evar.get("name", "VAR")
    desc = evar.get("description", "")
    is_secret = evar.get("isSecret", False)
    placeholder = evar.get("placeholder", "")
    choices = evar.get("choices")
    is_required = evar.get("isRequired", False)
    default_val = evar.get("value", "")

    label = f"  {name}"
    if desc:
        console.print(f"    [dim]{desc}[/dim]")

    if choices and isinstance(choices, list):
        return Prompt.ask(label, choices=choices, default=default_val or choices[0])

    kwargs: dict[str, Any] = {}
    if is_secret:
        kwargs["password"] = True
    if default_val:
        kwargs["default"] = default_val
    elif placeholder and not is_secret:
        console.print(f"    [dim]e.g. {placeholder}[/dim]")

    value = Prompt.ask(label, **kwargs).strip()

    if not value and is_required:
        console.print(f"    [red]{name} is required.[/red]")
        value = Prompt.ask(label, **kwargs).strip()

    return value if value else None


def _collect_env_vars(env_defs: list[dict]) -> dict[str, str]:
    """Interactively collect environment variable values from registry defs."""
    env: dict[str, str] = {}
    required = [e for e in env_defs if e.get("isRequired")]
    optional = [e for e in env_defs if not e.get("isRequired")]

    if required:
        console.print("  [bold]Required environment variables:[/bold]")
        for evar in required:
            val = _prompt_env_var(evar)
            if val is not None:
                env[evar["name"]] = val

    if optional:
        if Confirm.ask("  Configure optional environment variables?", default=False):
            for evar in optional:
                val = _prompt_env_var(evar)
                if val is not None:
                    env[evar["name"]] = val

    return env


def _collect_package_args(arg_defs: list[dict]) -> list[str]:
    """Collect package arguments (positional and named) from registry defs."""
    args: list[str] = []
    for adef in arg_defs:
        name = adef.get("name", "")
        desc = adef.get("description", "")
        is_required = adef.get("isRequired", False)
        arg_type = adef.get("type", "positional")
        placeholder = adef.get("placeholder", "")

        if desc:
            console.print(f"    [dim]{desc}[/dim]")

        label = f"  --{name}" if arg_type == "named" else f"  {name}"
        kwargs: dict[str, Any] = {}
        if placeholder:
            kwargs["default"] = placeholder

        value = Prompt.ask(label, **kwargs).strip()
        if not value and is_required:
            console.print(f"    [red]{name} is required.[/red]")
            value = Prompt.ask(label, **kwargs).strip()

        if value:
            if arg_type == "named":
                args.extend([f"--{name}", value])
            else:
                args.append(value)

    return args


def _collect_remote_headers(header_defs: list[dict]) -> dict[str, str]:
    """Collect header values for a remote server."""
    headers: dict[str, str] = {}
    for hdef in header_defs:
        name = hdef.get("name", "")
        desc = hdef.get("description", "")
        is_secret = hdef.get("isSecret", False)
        value_template = hdef.get("value", "")
        is_required = hdef.get("isRequired", False)

        if desc:
            console.print(f"    [dim]{desc}[/dim]")

        # Resolve {placeholder} tokens in value templates
        if "{" in value_template and "}" in value_template:
            placeholders = re.findall(r"\{(\w+)\}", value_template)
            filled = value_template
            for ph in placeholders:
                ph_val = Prompt.ask(f"  {ph}", password=is_secret).strip()
                filled = filled.replace(f"{{{ph}}}", ph_val)
            if filled:
                headers[name] = filled
        else:
            kwargs: dict[str, Any] = {}
            if is_secret:
                kwargs["password"] = True
            if value_template:
                kwargs["default"] = value_template
            val = Prompt.ask(f"  {name}", **kwargs).strip()
            if val:
                headers[name] = val
            elif is_required:
                console.print(f"    [red]{name} is required.[/red]")
                val = Prompt.ask(f"  {name}", **kwargs).strip()
                if val:
                    headers[name] = val

    return headers


def _build_stdio_spec(
    package: dict, env: dict[str, str], pkg_args: list[str],
) -> dict[str, Any]:
    """Build a .mcp.json stdio entry from a registry package."""
    reg_type = package.get("registryType", "npm")
    identifier = package.get("identifier", "")

    if reg_type == "npm":
        spec: dict[str, Any] = {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", identifier] + pkg_args,
        }
    elif reg_type == "pypi":
        spec = {
            "type": "stdio",
            "command": "uvx",
            "args": [identifier] + pkg_args,
        }
    else:
        console.print(f"[yellow]Package type '{reg_type}' — manual setup may be needed.[/yellow]")
        spec = {
            "type": "stdio",
            "command": "FILL_IN",
            "args": [identifier] + pkg_args,
        }

    if env:
        spec["env"] = env
    return spec


def _build_remote_spec(remote: dict, headers: dict[str, str]) -> dict[str, Any]:
    """Build a .mcp.json HTTP entry from a registry remote."""
    spec: dict[str, Any] = {
        "type": "http",
        "url": remote.get("url", ""),
    }
    if headers:
        spec["headers"] = headers
    return spec


def _pick_package(packages: list[dict]) -> dict | None:
    """Let user pick a package if multiple exist."""
    if len(packages) == 1:
        pkg = packages[0]
        reg = pkg.get("registryType", "?")
        ident = pkg.get("identifier", "?")
        console.print(f"  Package: [cyan]{ident}[/cyan] ({reg})")
        return pkg

    console.print("  [bold]Available packages:[/bold]")
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("#", style="bold yellow", width=3)
    table.add_column("Type", style="cyan")
    table.add_column("Identifier")
    table.add_column("Transport", style="dim")

    for i, pkg in enumerate(packages, 1):
        transport = pkg.get("transport", {}).get("type", "stdio")
        table.add_row(
            str(i),
            pkg.get("registryType", "?"),
            pkg.get("identifier", "?"),
            transport,
        )
    console.print(table)

    choices = [str(i) for i in range(1, len(packages) + 1)]
    pick = Prompt.ask("  Select package", choices=choices, default="1")
    return packages[int(pick) - 1]


def _pick_remote(remotes: list[dict]) -> dict | None:
    """Let user pick a remote if multiple exist."""
    if len(remotes) == 1:
        r = remotes[0]
        console.print(f"  Remote: [cyan]{r.get('url', '?')}[/cyan]")
        return r

    console.print("  [bold]Available remotes:[/bold]")
    for i, r in enumerate(remotes, 1):
        console.print(
            f"  [bold yellow]{i}[/bold yellow]  "
            f"{r.get('type', '?')}: {r.get('url', '?')}"
        )

    choices = [str(i) for i in range(1, len(remotes) + 1)]
    pick = Prompt.ask("  Select remote", choices=choices, default="1")
    return remotes[int(pick) - 1]


def mcp_install_wizard(server_name: str) -> str | None:
    """Install an MCP server from the official registry.

    Fetches server details, guides user through configuration,
    writes the entry to .mcp.json. Returns local server name or None.
    """
    console.print()
    console.print("[bold cyan]Install MCP Server[/bold cyan]")
    console.print()

    # 1. Fetch server details
    try:
        srv = mcp_registry_get_server(server_name)
    except (urllib.error.URLError, OSError) as e:
        console.print(f"[red]Failed to fetch server details: {e}[/red]")
        return None

    if not srv:
        console.print(f"[red]Server '{server_name}' not found in MCP registry.[/red]")
        console.print("[dim]Tip: use /mcp find <query> to search.[/dim]")
        return None

    # 2. Display server info
    console.print(f"  [bold]{srv.get('name', server_name)}[/bold]")
    console.print(f"  {srv.get('description', '')}")
    console.print(f"  Version: {srv.get('version', '?')}")
    repo = srv.get("repository", {})
    if isinstance(repo, dict) and repo.get("url"):
        console.print(f"  Repository: [dim]{repo['url']}[/dim]")
    console.print()

    packages = srv.get("packages", [])
    remotes = srv.get("remotes", [])

    if not packages and not remotes:
        console.print("[red]Server has no installable packages or remotes.[/red]")
        return None

    # 3. Choose installation method
    chosen_package = None
    chosen_remote = None

    if packages and remotes:
        choice = Prompt.ask(
            "  Install as",
            choices=["local", "remote"],
            default="local",
        )
        if choice == "local":
            chosen_package = _pick_package(packages)
        else:
            chosen_remote = _pick_remote(remotes)
    elif packages:
        chosen_package = _pick_package(packages)
    else:
        chosen_remote = _pick_remote(remotes)

    if chosen_package is None and chosen_remote is None:
        return None

    # 4. Derive local name
    default_name = _derive_local_name(server_name)
    local_name = Prompt.ask("  Local server name", default=default_name).strip()

    data = _read_mcp_json()
    servers = data.get("mcpServers", {})
    if local_name in servers:
        if not Confirm.ask(f"  '{local_name}' already exists. Overwrite?", default=False):
            return None

    # 5. Collect configuration and build spec
    if chosen_package:
        env_defs = chosen_package.get("environmentVariables", [])
        arg_defs = chosen_package.get("packageArguments", [])
        env = _collect_env_vars(env_defs) if env_defs else {}
        pkg_args = _collect_package_args(arg_defs) if arg_defs else []
        spec = _build_stdio_spec(chosen_package, env, pkg_args)
    else:
        header_defs = chosen_remote.get("headers", [])
        headers = _collect_remote_headers(header_defs) if header_defs else {}
        spec = _build_remote_spec(chosen_remote, headers)

    # 6. Write to .mcp.json
    servers[local_name] = spec
    data["mcpServers"] = servers
    _write_mcp_json(data)

    console.print()
    console.print(f"  [green]Installed '{local_name}' to .mcp.json[/green]")
    console.print(f"  [dim]{json.dumps(spec, indent=2)}[/dim]")

    return local_name
