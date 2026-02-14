"""Graph builder for OctoEngine — constructs LangGraph app from OctoConfig.

This bridges the gap between the new OctoConfig-based API and the existing
graph.py machinery. As the split matures, more of graph.py will move here.

⚠️ KNOWN LIMITATION: Thread Safety / Global State Mutation
   This builder patches module-level variables in octo.models and octo.config
   to inject OctoConfig values into the legacy code path. This means:
   - NOT thread-safe
   - Only ONE OctoEngine config can be active per process
   - For multi-tenant scenarios, use one engine per process/worker

   This will be resolved when models.py and config.py are refactored to
   accept config objects directly instead of reading module globals.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel to detect concurrent engine builds
_build_lock_holder: str | None = None


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

    Raises:
        RuntimeError: If another build is in progress (concurrent builds
            would corrupt global state).
    """
    global _build_lock_holder

    from octo.core.config import OctoConfig
    assert isinstance(config, OctoConfig)

    # Detect concurrent builds that would corrupt global state
    build_id = f"{config.llm_provider}:{config.default_model}:{id(config)}"
    if _build_lock_holder is not None:
        raise RuntimeError(
            f"Concurrent OctoEngine build detected. Another build is in progress "
            f"({_build_lock_holder}). Only one engine can be built at a time per process. "
            f"See _builder.py docstring for details."
        )

    _build_lock_holder = build_id
    try:
        return await _build_engine_graph_impl(config)
    finally:
        _build_lock_holder = None


async def _build_engine_graph_impl(config: Any) -> tuple[Any, Any]:
    """Internal build implementation."""

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

    # Import and call the existing build_graph, passing our checkpointer
    from octo.graph import build_graph
    app_tuple = await build_graph(
        mcp_tools=mcp_tools,
        mcp_tools_by_server=mcp_tools_by_server,
        checkpointer=checkpointer,
    )
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
