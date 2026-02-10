"""Load MCP servers from .mcp.json → LangChain tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from octo.config import MCP_CONFIG_PATH


def _parse_mcp_config(path: Path) -> dict[str, dict[str, Any]]:
    """Read .mcp.json and convert to MultiServerMCPClient format."""
    if not path.is_file():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers", {})

    configs: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
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

    return configs


def get_mcp_configs() -> dict[str, dict[str, Any]]:
    """Parse .mcp.json and return server configs (without connecting)."""
    return _parse_mcp_config(MCP_CONFIG_PATH)


def create_mcp_client(configs: dict[str, dict[str, Any]] | None = None) -> MultiServerMCPClient:
    """Create a MultiServerMCPClient. Must be used as async context manager.

    Usage:
        async with create_mcp_client() as client:
            tools = client.get_tools()
    """
    if configs is None:
        configs = get_mcp_configs()
    return MultiServerMCPClient(configs)
