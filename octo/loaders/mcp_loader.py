"""Backward-compat shim — moved to octo.core.loaders.mcp_loader."""
from octo.core.loaders.mcp_loader import *  # noqa: F401,F403
from octo.core.loaders.mcp_loader import (
    MCPSessionPool,
    _preseed_client_info,
    create_mcp_client,
    filter_tools,
    get_mcp_configs,
    get_tool_filters,
    validate_tool_schemas,
)
