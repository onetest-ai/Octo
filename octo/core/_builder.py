"""Graph builder for OctoEngine — constructs LangGraph app from OctoConfig.

This bridges the gap between the new OctoConfig-based API and the existing
graph.py machinery. As the split matures, more of graph.py will move here.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def build_engine_graph(config: Any) -> tuple[Any, Any]:
    """Build a compiled LangGraph app from an OctoConfig.

    Currently delegates to the existing graph.py build_graph() for the
    actual agent/supervisor assembly. The OctoConfig is used to configure:
    - Model selection (provider, credentials, tiers)
    - Checkpointing backend
    - Storage backend for tools
    - Context management parameters

    Returns:
        Tuple of (compiled_app, checkpointer)
    """
    from octo.core.config import OctoConfig
    assert isinstance(config, OctoConfig)

    # --- Configure models ---
    # Temporarily override the config module's globals so that make_model()
    # picks up our config values. This is a bridge until models.py is
    # refactored to accept config directly.
    import octo.config as legacy_config
    import octo.models as models_mod

    if config.llm_provider:
        models_mod.LLM_PROVIDER = config.llm_provider
    if config.default_model:
        models_mod.DEFAULT_MODEL = config.default_model
    if config.high_tier_model:
        models_mod.HIGH_TIER_MODEL = config.high_tier_model
    if config.low_tier_model:
        models_mod.LOW_TIER_MODEL = config.low_tier_model

    # Map credentials to the module-level vars models.py reads
    creds = config.llm_credentials
    if "api_key" in creds:
        if config.llm_provider == "anthropic":
            models_mod.ANTHROPIC_API_KEY = creds["api_key"]
        elif config.llm_provider == "openai":
            models_mod.OPENAI_API_KEY = creds["api_key"]
        elif config.llm_provider == "azure":
            models_mod.AZURE_OPENAI_API_KEY = creds["api_key"]
        elif config.llm_provider == "github":
            models_mod.GITHUB_TOKEN = creds["api_key"]
    if "endpoint" in creds and config.llm_provider == "azure":
        models_mod.AZURE_OPENAI_ENDPOINT = creds["endpoint"]
    if "region" in creds and config.llm_provider == "bedrock":
        models_mod.AWS_REGION = creds["region"]
    if "access_key_id" in creds:
        models_mod.AWS_ACCESS_KEY_ID = creds["access_key_id"]
    if "secret_access_key" in creds:
        models_mod.AWS_SECRET_ACCESS_KEY = creds["secret_access_key"]

    # --- Configure middleware thresholds ---
    legacy_config.TOOL_RESULT_LIMIT = config.tool_result_limit
    legacy_config.SUPERVISOR_MSG_CHAR_LIMIT = config.supervisor_msg_char_limit
    legacy_config.SUMMARIZATION_TRIGGER_TOKENS = config.summarization_trigger_tokens
    legacy_config.SUMMARIZATION_KEEP_TOKENS = config.summarization_keep_tokens

    # --- Build checkpointer ---
    checkpointer = await _make_checkpointer(config)

    # --- Build the graph ---
    # Use preloaded_tools if provided; otherwise the graph will load MCP tools
    mcp_tools = config.preloaded_tools or []
    mcp_tools_by_server: dict[str, list] = {}
    if mcp_tools:
        mcp_tools_by_server["preloaded"] = mcp_tools

    # Import and call the existing build_graph
    from octo.graph import build_graph
    app_tuple = await build_graph(
        mcp_tools=mcp_tools,
        mcp_tools_by_server=mcp_tools_by_server,
    )
    # build_graph returns (app, agents, skills) but the app is compiled
    # with its own checkpointer. We need to recompile with ours.
    # For now, the app from build_graph already has its checkpointer.
    # TODO: refactor build_graph to return the workflow before compilation
    # so we can inject our own checkpointer cleanly.
    app = app_tuple[0]

    return app, checkpointer


async def _make_checkpointer(config: Any) -> Any:
    """Create a checkpointer based on config.

    Args:
        config: OctoConfig instance.

    Returns:
        A LangGraph checkpoint saver instance.
    """
    backend = config.checkpoint_backend

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:
            raise ImportError(
                "PostgreSQL checkpointing requires the postgres extra. "
                "Install with: pip install octo-agent[postgres]"
            )
        dsn = config.checkpoint_config.get("dsn", "")
        if not dsn:
            raise ValueError("PostgreSQL checkpointer requires 'dsn' in checkpoint_config")
        checkpointer = AsyncPostgresSaver.from_conn_string(dsn)
        await checkpointer.setup()
        return checkpointer

    # Default: SQLite
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = config.checkpoint_config.get("path", "")
    if not db_path:
        # Use in-memory if no path given
        db_path = ":memory:"

    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    return checkpointer
