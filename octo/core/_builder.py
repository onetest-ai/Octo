"""Graph builder for OctoEngine — constructs LangGraph app from OctoConfig.

This bridges the gap between the new OctoConfig-based API and the existing
graph.py machinery.  Uses config injection into make_model() instead of
mutating module-level globals.
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
    - Context management parameters

    Returns:
        Tuple of (compiled_app, checkpointer)

    Raises:
        RuntimeError: If another build is in progress.
    """
    global _build_lock_holder

    from octo.core.config import OctoConfig
    assert isinstance(config, OctoConfig)

    # Detect concurrent builds
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


def _build_model_config(config: Any) -> dict:
    """Build a config dict for make_model() from OctoConfig."""
    creds = config.llm_credentials
    model_config: dict[str, str] = {
        "provider": config.llm_provider,
        "default_model": config.default_model,
    }
    if config.high_tier_model:
        model_config["high_tier_model"] = config.high_tier_model
    if config.low_tier_model:
        model_config["low_tier_model"] = config.low_tier_model

    # Map credentials to config keys based on provider
    if "api_key" in creds:
        model_config["api_key"] = creds["api_key"]
    if config.llm_provider == "openai" and "api_key" in creds:
        model_config["openai_api_key"] = creds["api_key"]
    if config.llm_provider == "azure":
        if "api_key" in creds:
            model_config["azure_api_key"] = creds["api_key"]
        if "endpoint" in creds:
            model_config["azure_endpoint"] = creds["endpoint"]
        if "api_version" in creds:
            model_config["azure_api_version"] = creds["api_version"]
    if config.llm_provider == "bedrock":
        if "region" in creds:
            model_config["region"] = creds["region"]
        if "access_key_id" in creds:
            model_config["access_key_id"] = creds["access_key_id"]
        if "secret_access_key" in creds:
            model_config["secret_access_key"] = creds["secret_access_key"]
    if config.llm_provider == "github":
        if "api_key" in creds:
            model_config["github_token"] = creds["api_key"]
        if "base_url" in creds:
            model_config["github_base_url"] = creds["base_url"]
        if "anthropic_base_url" in creds:
            model_config["github_anthropic_base_url"] = creds["anthropic_base_url"]
    if config.llm_provider == "gemini":
        if "api_key" in creds:
            model_config["google_api_key"] = creds["api_key"]
    if config.llm_provider == "local":
        if "base_url" in creds:
            model_config["openai_api_base"] = creds["base_url"]
        if "api_key" in creds:
            model_config["openai_api_key"] = creds["api_key"]

    return model_config


async def _build_engine_graph_impl(config: Any) -> tuple[Any, Any]:
    """Internal build implementation.

    Passes all configuration as parameters to build_graph() instead of
    mutating module-level globals. This makes engine builds safe for
    multi-tenant environments (multiple engines in the same process).
    """
    # Build model config dict (provider, credentials, model names)
    model_config = _build_model_config(config)

    # Build context limits dict (no more global mutation!)
    context_limits = {
        "tool_result_limit": config.tool_result_limit,
        "supervisor_msg_char_limit": config.supervisor_msg_char_limit,
        "summarization_trigger_tokens": config.summarization_trigger_tokens,
        "summarization_keep_tokens": config.summarization_keep_tokens,
    }

    # Build checkpointer
    from octo.core.checkpointing import make_checkpointer
    checkpointer = await make_checkpointer(config)

    # Use preloaded_tools if provided; otherwise the graph will load MCP tools
    mcp_tools = config.preloaded_tools or []
    mcp_tools_by_server: dict[str, list] = {}
    if mcp_tools:
        mcp_tools_by_server["preloaded"] = mcp_tools

    # Resolve agent and skill configs from OctoConfig.
    # Priority: explicit config > storage-based loading > None (fallback to CLI mode)
    agent_configs = config.agent_configs if config.agent_configs else None
    skill_configs = config.skill_configs if config.skill_configs else None

    # If no pre-loaded configs but storage is available, load from storage (S3/filesystem)
    if agent_configs is None and config.storage is not None:
        from octo.core.loaders.agent_loader import load_agents_from_storage
        agent_configs = await load_agents_from_storage(config.storage, prefix="agents")
        if not agent_configs:
            agent_configs = None  # Let build_graph fall back to CLI mode

    if skill_configs is None and config.storage is not None:
        from octo.core.loaders.skill_loader import load_skills_from_storage
        skill_configs = await load_skills_from_storage(config.storage, prefix="skills")
        if not skill_configs:
            skill_configs = None  # Let build_graph fall back to CLI mode

    # Resolve persona files from OctoConfig.
    # Priority: explicit config > storage-based loading > None (fallback to CLI mode)
    persona_files = config.persona_files if config.persona_files else None
    if persona_files is None and config.storage is not None:
        from octo.core.loaders.persona_loader import load_persona_from_storage
        persona_files = await load_persona_from_storage(config.storage, prefix="persona")
        if not persona_files:
            persona_files = None  # Let build_graph fall back to CLI mode

    # Import and call build_graph with full config injection (no globals needed)
    from octo.core.graph import build_graph
    app_tuple = await build_graph(
        mcp_tools=mcp_tools,
        mcp_tools_by_server=mcp_tools_by_server,
        checkpointer=checkpointer,
        storage=config.storage,
        model_config=model_config,
        context_limits=context_limits,
        agent_configs=agent_configs,
        skill_configs=skill_configs,
        persona_files=persona_files,
        engine_mode=True,
        disable_builtin_tools=config.disable_builtin_tools,
        builtin_tools_override=config.builtin_tools_override,
    )
    app = app_tuple[0]

    return app, checkpointer



# _make_checkpointer moved to octo.core.checkpointing.make_checkpointer
