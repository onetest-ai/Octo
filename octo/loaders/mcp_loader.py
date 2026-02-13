"""Load MCP servers from .mcp.json → LangChain tools."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from octo.config import MCP_CONFIG_PATH, OAUTH_DIR

logger = logging.getLogger(__name__)


def _preseed_client_info(storage, client_id: str, redirect_uri: str) -> None:
    """Write client_info to storage if not already present.

    This lets the MCP SDK skip dynamic client registration for
    pre-registered OAuth apps (e.g. Microsoft Entra, Atlassian).
    """
    if storage._client_path.is_file():
        return  # already seeded or registered — don't overwrite

    from mcp.shared.auth import OAuthClientInformationFull

    info = OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[redirect_uri],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    storage._client_path.parent.mkdir(parents=True, exist_ok=True)
    storage._client_path.write_text(info.model_dump_json(indent=2), "utf-8")


def _build_auth_provider(
    server_name: str, auth_config: dict[str, Any], server_url: str
):
    """Build an httpx.Auth provider from an ``auth`` block in .mcp.json.

    Returns an ``httpx.Auth`` instance or ``None`` on error.
    """
    auth_type = auth_config.get("type", "")

    if auth_type == "oauth":
        try:
            from mcp.client.auth import OAuthClientProvider
            from mcp.shared.auth import OAuthClientMetadata
            from pydantic import AnyUrl

            from octo.oauth.browser import make_callback_handler, open_browser
            from octo.oauth.storage import FileTokenStorage

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

            # client_id handling:
            # - URL (https://...) → CIMD (Client ID Metadata Document)
            # - plain string     → pre-registered app, pre-seed storage so SDK
            #                      skips dynamic registration
            client_id = auth_config.get("client_id", "")
            client_metadata_url = None
            if client_id.startswith("https://"):
                client_metadata_url = client_id
            elif client_id:
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
            return provider
        except Exception as exc:
            logger.warning("Failed to build OAuth provider for '%s': %s", server_name, exc)
            return None

    elif auth_type == "client_credentials":
        try:
            from mcp.client.auth.extensions.client_credentials import (
                ClientCredentialsOAuthProvider,
            )

            from octo.oauth.storage import FileTokenStorage

            secret_env = auth_config.get("client_secret_env", "")
            client_secret = os.getenv(secret_env, "") if secret_env else auth_config.get("client_secret", "")

            if not client_secret:
                logger.warning(
                    "OAuth client_credentials for '%s': no secret found "
                    "(checked env var '%s')",
                    server_name,
                    secret_env,
                )
                return None

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
            return provider
        except Exception as exc:
            logger.warning(
                "Failed to build client_credentials provider for '%s': %s",
                server_name,
                exc,
            )
            return None

    else:
        if auth_type:
            logger.warning("Unknown auth type '%s' for server '%s'", auth_type, server_name)
        return None


def _parse_mcp_config(path: Path) -> dict[str, dict[str, Any]]:
    """Read .mcp.json and convert to MultiServerMCPClient format.

    Returns an empty dict (with no crash) if the file is missing,
    unreadable, or contains invalid JSON.
    """
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        raise  # let caller handle with user-friendly message
    servers = raw.get("mcpServers", {})

    configs: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
        if spec.get("disabled"):
            continue
        server_type = spec.get("type", "stdio")

        if server_type == "stdio":
            configs[name] = {
                "transport": "stdio",
                "command": spec["command"],
                "args": spec.get("args", []),
            }
            if spec.get("env"):
                configs[name]["env"] = spec["env"]

        elif server_type in ("http", "streamable_http"):
            configs[name] = {
                "transport": "streamable_http",
                "url": spec["url"],
            }
            if spec.get("headers"):
                configs[name]["headers"] = spec["headers"]
            if spec.get("auth"):
                auth_provider = _build_auth_provider(name, spec["auth"], spec["url"])
                if auth_provider:
                    configs[name]["auth"] = auth_provider

    return configs


def get_mcp_configs() -> dict[str, dict[str, Any]]:
    """Parse .mcp.json and return server configs (without connecting)."""
    return _parse_mcp_config(MCP_CONFIG_PATH)


def get_tool_filters() -> dict[str, dict[str, list[str]]]:
    """Return per-server tool filters from .mcp.json.

    Reads ``include_tools`` and ``exclude_tools`` arrays from each server
    block. Example::

        "outlook": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@softeria/ms-365-mcp-server"],
            "include_tools": ["mail_list", "mail_read", "mail_send"]
        }

    Returns ``{"outlook": {"include": [...], "exclude": [...]}}``
    """
    if not MCP_CONFIG_PATH.is_file():
        return {}

    raw = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers", {})

    filters: dict[str, dict[str, list[str]]] = {}
    for name, spec in servers.items():
        has_inc = "include_tools" in spec
        has_exc = "exclude_tools" in spec
        if has_inc or has_exc:
            filters[name] = {
                "include": spec.get("include_tools", []),
                "exclude": spec.get("exclude_tools", []),
                "_has_include": has_inc,
            }
    return filters


def filter_tools(tools: list, server_name: str, filters: dict) -> list:
    """Apply include/exclude filters to a tool list for one server.

    When ``include_tools`` key exists in .mcp.json (even if empty),
    only the listed tools are kept.  An empty list means **no tools**.
    """
    if server_name not in filters:
        return tools

    f = filters[server_name]
    inc = set(f.get("include", []))
    exc = set(f.get("exclude", []))

    # If include_tools key is present, whitelist mode — empty means none
    if f.get("_has_include"):
        tools = [t for t in tools if t.name in inc]
    if exc:
        tools = [t for t in tools if t.name not in exc]

    return tools


def _has_broken_refs(schema: dict, root: dict | None = None) -> str | None:
    """Walk a JSON schema and check for unresolvable $ref pointers.

    Returns the broken ref string, or None if all refs are valid.
    """
    if root is None:
        root = schema

    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if ref and isinstance(ref, str) and ref.startswith("#/"):
            # Resolve JSON pointer
            parts = ref[2:].split("/")
            node = root
            for part in parts:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, list):
                    try:
                        node = node[int(part)]
                    except (ValueError, IndexError):
                        return ref
                else:
                    return ref
        for v in schema.values():
            result = _has_broken_refs(v, root)
            if result:
                return result
    elif isinstance(schema, list):
        for item in schema:
            result = _has_broken_refs(item, root)
            if result:
                return result
    return None


def validate_tool_schemas(tools: list, server_name: str) -> list:
    """Filter out tools with broken JSON schemas and log warnings."""
    valid = []
    for t in tools:
        schema = getattr(t, "args_schema", None)
        # Try to get the raw schema dict
        raw = None
        if schema is not None:
            if hasattr(schema, "schema"):
                try:
                    raw = schema.schema()
                except Exception:
                    pass
            elif hasattr(schema, "model_json_schema"):
                try:
                    raw = schema.model_json_schema()
                except Exception:
                    pass
            elif isinstance(schema, dict):
                raw = schema

        if raw and isinstance(raw, dict):
            broken = _has_broken_refs(raw)
            if broken:
                logger.warning(
                    "Skipping tool '%s' from '%s': broken schema ref %s",
                    t.name, server_name, broken,
                )
                continue
        valid.append(t)
    return valid


def create_mcp_client(configs: dict[str, dict[str, Any]] | None = None) -> MultiServerMCPClient:
    """Create a MultiServerMCPClient. Must be used as async context manager.

    Usage:
        async with create_mcp_client() as client:
            tools = client.get_tools()
    """
    if configs is None:
        configs = get_mcp_configs()
    return MultiServerMCPClient(configs)


class MCPSessionPool:
    """Keeps STDIO MCP servers alive as persistent sessions.

    Instead of spawning a new subprocess per tool call, opens a persistent
    ``ClientSession`` for each STDIO server and keeps it alive for the CLI
    session lifetime.  Tools are bound to the live session so every
    ``ainvoke()`` reuses the same subprocess.

    Configs are stored so that dead sessions can be reconnected automatically.
    """

    def __init__(self) -> None:
        self._stacks: dict[str, Any] = {}
        self._configs: dict[str, dict[str, Any]] = {}

    async def connect(self, name: str, config: dict[str, Any]) -> list:
        """Open persistent session, return LangChain tools bound to it."""
        from contextlib import AsyncExitStack

        from langchain_mcp_adapters.sessions import create_session
        from langchain_mcp_adapters.tools import load_mcp_tools

        stack = AsyncExitStack()
        try:
            session = await stack.enter_async_context(create_session(config))
            await session.initialize()
            tools = await load_mcp_tools(session, server_name=name)
        except BaseException:
            # Clean up the stack so dying subprocess doesn't leak cancel scopes
            try:
                await stack.aclose()
            except BaseException:
                pass
            raise
        self._stacks[name] = stack
        self._configs[name] = config
        return tools

    async def reconnect(self, name: str) -> list:
        """Close a dead session and open a fresh one.

        Returns new tools bound to the fresh session, or empty list
        if the server config is unknown or reconnection fails.
        """
        config = self._configs.get(name)
        if not config:
            return []
        await self.close(name)
        return await self.connect(name, config)

    async def close(self, name: str) -> None:
        """Close a specific server's persistent session."""
        stack = self._stacks.pop(name, None)
        if stack:
            try:
                await stack.aclose()
            except BaseException:
                pass

    async def close_all(self) -> None:
        """Close all persistent sessions (for shutdown or reload)."""
        for name in list(self._stacks):
            await self.close(name)
