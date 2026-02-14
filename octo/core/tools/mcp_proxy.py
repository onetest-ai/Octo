"""Deferred MCP tool proxy — find_tools and call_mcp_tool.

Extracted from graph.py. The MCP tool registry is populated at startup
by _register_mcp_tools() and tools are invoked lazily via call_mcp_tool.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# --- Deferred MCP tool registry -------------------------------------------

_mcp_tool_registry: dict[str, Any] = {}
_mcp_server_summaries: list[dict[str, str]] = []
_tool_to_server: dict[str, str] = {}  # reverse index: tool_name → server_name
_session_pool: Any = None  # MCPSessionPool ref, set from cli.py


def set_session_pool(pool: Any) -> None:
    """Register the MCPSessionPool for auto-reconnect on session errors."""
    global _session_pool
    _session_pool = pool


def get_mcp_tool(name: str) -> Any | None:
    """Return a registered MCP tool by exact name, or None."""
    return _mcp_tool_registry.get(name)


def get_mcp_server_summaries() -> list[dict[str, str]]:
    """Return the list of MCP server summaries for prompt building."""
    return _mcp_server_summaries


def register_mcp_tools(tools_by_server: dict[str, list]) -> None:
    """Populate the deferred MCP tool registry from per-server tool lists."""
    _mcp_tool_registry.clear()
    _mcp_server_summaries.clear()
    _tool_to_server.clear()
    for server_name, tools in tools_by_server.items():
        for t in tools:
            _mcp_tool_registry[t.name] = t
            _tool_to_server[t.name] = server_name
        tool_names = [t.name.split("__")[-1] for t in tools[:5]]
        summary = ", ".join(tool_names)
        if len(tools) > 5:
            summary += f" (+{len(tools) - 5} more)"
        _mcp_server_summaries.append({
            "server": server_name,
            "tools": len(tools),
            "summary": summary,
        })


@tool
def find_tools(query: str) -> str:
    """Search available MCP tools by keyword. Returns matching tool names,
    descriptions, and parameter schemas. Use call_mcp_tool() to execute.

    Args:
        query: Search keyword (e.g. 'github issues', 'search', 'calendar')
    """
    words = query.lower().split()
    scored: list[tuple[int, dict]] = []
    for name, t in _mcp_tool_registry.items():
        desc = t.description or ""
        haystack = f"{name.lower()} {desc.lower()}"
        hits = sum(1 for w in words if w in haystack)
        if hits == 0:
            continue
        schema: dict = {}
        if hasattr(t, "args_schema") and t.args_schema:
            try:
                schema = t.args_schema.model_json_schema()
            except Exception:
                pass
        scored.append((hits, {
            "name": name,
            "description": desc[:200],
            "parameters": schema,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [entry for _, entry in scored]

    if not matches:
        servers = ", ".join(s["server"] for s in _mcp_server_summaries)
        return f"No tools found matching '{query}'. Available servers: {servers}"

    return json.dumps(matches[:15], indent=2)


def _is_session_error(exc: Exception) -> bool:
    """Check if an exception indicates a dead MCP session."""
    if isinstance(exc, (ConnectionError, EOFError, OSError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("closed", "broken pipe", "eof", "disconnect"))


@tool
async def call_mcp_tool(tool_name: str, arguments: dict | None = None) -> str:
    """Execute an MCP tool by name. Use find_tools() first to discover
    available tools and their parameter schemas.

    Args:
        tool_name: Exact tool name (from find_tools results)
        arguments: Tool arguments as a dictionary
    """
    t = _mcp_tool_registry.get(tool_name)
    if not t:
        close = [n for n in _mcp_tool_registry if tool_name.lower() in n.lower()][:5]
        hint = f" Similar: {', '.join(close)}" if close else ""
        return f"[Error] Tool '{tool_name}' not found.{hint} Use find_tools() to search."

    try:
        result = await t.ainvoke(arguments or {})
        return str(result)
    except Exception as e:
        # Auto-reconnect for dead STDIO sessions
        server = _tool_to_server.get(tool_name)
        if server and _session_pool and _is_session_error(e):
            try:
                logger.warning("MCP session '%s' died, reconnecting...", server)
                new_tools = await _session_pool.reconnect(server)
                # Update registry with fresh session-bound tools
                for nt in new_tools:
                    _mcp_tool_registry[nt.name] = nt
                    _tool_to_server[nt.name] = server
                # Retry with new tool
                t2 = _mcp_tool_registry.get(tool_name)
                if t2:
                    result = await t2.ainvoke(arguments or {})
                    return str(result)
            except Exception as reconnect_err:
                return (
                    f"[Tool error] {tool_name}: session died "
                    f"({type(e).__name__}: {e}), "
                    f"reconnect failed ({type(reconnect_err).__name__}: {reconnect_err}). "
                    f"Try /mcp reload."
                )
        return f"[Tool error] {tool_name}: {type(e).__name__}: {e}"


MCP_PROXY_TOOLS = [find_tools, call_mcp_tool]
