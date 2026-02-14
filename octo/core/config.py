"""OctoConfig — engine configuration dataclass.

This is the pure-data configuration for the Octo engine. No env vars,
no dotenv, no side effects at import time. The CLI layer (octo.config)
reads environment and builds an OctoConfig from it.

For embedding in services (e.g. OneTest Gateway), callers construct
OctoConfig directly with their own values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OctoConfig:
    """Engine configuration. Caller provides everything — no env var reading.

    Required:
        llm_provider: Provider name ("anthropic", "openai", "bedrock", "azure", "github")
        llm_credentials: Provider-specific credentials dict

    Optional:
        All other fields have sensible defaults.
    """

    # --- LLM ---
    llm_provider: str = ""                       # "anthropic" | "openai" | "bedrock" | "azure" | "github"
    llm_credentials: dict[str, str] = field(default_factory=dict)
    # {"api_key": "...", "endpoint": "...", "region": "...", etc.}

    default_model: str = "claude-sonnet-4-5-20250929"
    high_tier_model: str = ""    # empty = same as default_model
    low_tier_model: str = ""     # empty = same as default_model
    model_profile: str = "balanced"  # "quality" | "balanced" | "budget"

    # --- Storage ---
    # StorageBackend instance for files, memory, skills. None = in-memory only.
    storage: Any = None  # typed as Any to avoid circular imports; should be StorageBackend

    # --- Checkpointing ---
    checkpoint_backend: str = "sqlite"  # "sqlite" | "postgres"
    checkpoint_config: dict = field(default_factory=dict)
    # sqlite: {"path": "/path/to/db"}
    # postgres: {"dsn": "postgresql://..."}

    # --- Context management ---
    context_limit: int = 200_000
    tool_result_limit: int = 40_000
    supervisor_msg_char_limit: int = 30_000
    summarization_trigger_tokens: int = 40_000
    summarization_keep_tokens: int = 8_000

    # --- Agents & Skills ---
    agent_configs: list[Any] = field(default_factory=list)  # list[AgentConfig]
    mcp_servers: dict = field(default_factory=dict)
    preloaded_tools: list = field(default_factory=list)
    skill_configs: list = field(default_factory=list)

    # --- System prompt ---
    system_prompt: str = ""
    persona_files: dict[str, str] = field(default_factory=dict)
    # {"SOUL.md": "content...", "USER.md": "content...", etc.}

    # --- Workspace ---
    workspace_path: str = ""  # project root (for CLI: the directory containing .octo/)

    def effective_high_model(self) -> str:
        return self.high_tier_model or self.default_model

    def effective_low_model(self) -> str:
        return self.low_tier_model or self.default_model
