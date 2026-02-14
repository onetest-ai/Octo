"""OctoConfig — engine configuration dataclass.

This is the pure-data configuration for the Octo engine. No env vars,
no dotenv, no side effects at import time. The CLI layer (octo.config)
reads environment and builds an OctoConfig from it.

For embedding in services (e.g. OneTest Gateway), callers construct
OctoConfig directly with their own values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from octo.core.storage.base import StorageBackend

VALID_PROVIDERS = {"anthropic", "openai", "bedrock", "azure", "github"}
VALID_CHECKPOINT_BACKENDS = {"sqlite", "postgres"}
VALID_MODEL_PROFILES = {"quality", "balanced", "budget"}


class OctoConfigError(ValueError):
    """Raised when OctoConfig validation fails."""


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
    storage: StorageBackend | None = None

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

    def validate(self) -> None:
        """Validate configuration. Raises OctoConfigError on problems.

        Called automatically by OctoEngine before building the graph.
        Can also be called manually for early validation.
        """
        errors: list[str] = []

        if not self.llm_provider:
            errors.append("llm_provider is required (e.g. 'anthropic', 'openai')")
        elif self.llm_provider not in VALID_PROVIDERS:
            errors.append(
                f"llm_provider '{self.llm_provider}' not recognized. "
                f"Valid: {', '.join(sorted(VALID_PROVIDERS))}"
            )

        if not self.llm_credentials:
            errors.append("llm_credentials is required (e.g. {'api_key': '...'})")
        else:
            # Provider-specific credential checks
            if self.llm_provider in ("anthropic", "openai", "github"):
                if "api_key" not in self.llm_credentials:
                    errors.append(f"{self.llm_provider} requires 'api_key' in llm_credentials")
            elif self.llm_provider == "azure":
                if "api_key" not in self.llm_credentials:
                    errors.append("azure requires 'api_key' in llm_credentials")
                if "endpoint" not in self.llm_credentials:
                    errors.append("azure requires 'endpoint' in llm_credentials")

        if not self.default_model:
            errors.append("default_model is required")

        if self.model_profile not in VALID_MODEL_PROFILES:
            errors.append(
                f"model_profile '{self.model_profile}' not recognized. "
                f"Valid: {', '.join(sorted(VALID_MODEL_PROFILES))}"
            )

        if self.checkpoint_backend not in VALID_CHECKPOINT_BACKENDS:
            errors.append(
                f"checkpoint_backend '{self.checkpoint_backend}' not recognized. "
                f"Valid: {', '.join(sorted(VALID_CHECKPOINT_BACKENDS))}"
            )
        elif self.checkpoint_backend == "postgres":
            if "dsn" not in self.checkpoint_config:
                errors.append("PostgreSQL checkpoint_backend requires 'dsn' in checkpoint_config")

        if self.context_limit <= 0:
            errors.append("context_limit must be positive")

        if errors:
            raise OctoConfigError(
                f"OctoConfig validation failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
