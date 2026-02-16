"""Model factory — create LLM instances for any supported provider.

Supports two usage modes:

1. **CLI mode** (default): Reads credentials from module-level globals
   populated by ``octo.config`` at import time.  No ``config`` param needed.

2. **Engine mode**: Pass a ``config`` dict to ``make_model()`` with provider
   credentials.  Bypasses module globals entirely — safe for embedding.

   Expected config keys (all optional, provider-dependent):
     provider, api_key, endpoint, api_version, region,
     access_key_id, secret_access_key, github_token,
     github_base_url, github_anthropic_base_url,
     default_model, high_tier_model, low_tier_model
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

# Patch langgraph's _should_bind_tools to handle Pydantic tool objects
# (ChatBedrockConverse stores bound tools as Pydantic objects, not dicts)
def _patch_should_bind_tools():
    from langgraph.prebuilt import chat_agent_executor as _cae
    _orig = _cae._should_bind_tools

    def _patched(model, tools, num_builtin=0):
        try:
            return _orig(model, tools, num_builtin=num_builtin)
        except AttributeError:
            # StructuredTool doesn't have .get() — tools are already bound
            return False

    _cae._should_bind_tools = _patched

_patch_should_bind_tools()

from dataclasses import dataclass
from typing import Callable

from octo.config import (
    ANTHROPIC_API_KEY,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    GITHUB_TOKEN,
    GITHUB_MODELS_BASE_URL,
    GITHUB_MODELS_ANTHROPIC_BASE_URL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    DEFAULT_MODEL,
    HIGH_TIER_MODEL,
    LOW_TIER_MODEL,
)


# ---------------------------------------------------------------------------
# Provider registry — single source of truth for factories + default models
# ---------------------------------------------------------------------------

@dataclass
class ProviderSpec:
    """Registration for an LLM provider."""
    factory: Callable  # fn(name, *, config=None) -> BaseChatModel
    default: str = ""  # default tier model ID
    high: str = ""     # high tier model ID
    low: str = ""      # low tier model ID


# ---------------------------------------------------------------------------
# Helper: resolve a value from config dict or fall back to module global
# ---------------------------------------------------------------------------

def _cfg(config: dict | None, key: str, default: str = "") -> str:
    """Get a config value, falling back to module-level globals."""
    if config and key in config:
        return config[key]
    # Map config keys to module-level globals
    _GLOBALS = {
        "provider": LLM_PROVIDER,
        "api_key": ANTHROPIC_API_KEY,
        "region": AWS_REGION,
        "access_key_id": AWS_ACCESS_KEY_ID,
        "secret_access_key": AWS_SECRET_ACCESS_KEY,
        "openai_api_key": OPENAI_API_KEY,
        "openai_api_base": OPENAI_API_BASE,
        "azure_api_key": AZURE_OPENAI_API_KEY,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "azure_api_version": AZURE_OPENAI_API_VERSION,
        "github_token": GITHUB_TOKEN,
        "github_base_url": GITHUB_MODELS_BASE_URL,
        "github_anthropic_base_url": GITHUB_MODELS_ANTHROPIC_BASE_URL,
        "google_api_key": GOOGLE_API_KEY,
        "default_model": DEFAULT_MODEL,
        "high_tier_model": HIGH_TIER_MODEL,
        "low_tier_model": LOW_TIER_MODEL,
    }
    return _GLOBALS.get(key, default)


def _detect_provider(model_name: str, *, config: dict | None = None) -> str:
    """Auto-detect provider from model name pattern.

    Priority:
      1. Explicit provider in config or LLM_PROVIDER env var
      2. Universal ``provider/`` prefix (loops over _REGISTRY keys)
      3. Legacy model-name heuristics (backward compat)
      4. Credential availability fallback
    """
    explicit = _cfg(config, "provider")
    if explicit:
        return explicit

    # Universal provider prefix: anthropic/claude-*, gemini/gemini-*, local/llama3, etc.
    for provider in _REGISTRY:
        if model_name.startswith(f"{provider}/"):
            return provider

    # Legacy heuristics (backward compat — unprefixed model names)
    if ".anthropic." in model_name or ".amazon." in model_name or ".meta." in model_name:
        return "bedrock"
    if model_name.startswith(("gpt-", "o1-", "o3-", "o4-")):
        if _cfg(config, "azure_endpoint"):
            return "azure"
        return "openai"
    if model_name.startswith("claude-"):
        return "anthropic"
    if model_name.startswith("gemini-"):
        return "gemini"

    # Fallback: check which credentials are available
    if _cfg(config, "region") and _cfg(config, "access_key_id"):
        return "bedrock"
    if _cfg(config, "azure_endpoint") and _cfg(config, "azure_api_key"):
        return "azure"
    if _cfg(config, "google_api_key"):
        return "gemini"
    if _cfg(config, "openai_api_key"):
        return "openai"
    if _cfg(config, "api_key"):
        return "anthropic"
    if _cfg(config, "github_token"):
        return "github"

    return "anthropic"


def resolve_model_name(
    model_name: str = "",
    tier: str = "default",
    *,
    config: dict | None = None,
) -> str:
    """Resolve model name from explicit override or tier."""
    if model_name:
        return model_name
    if tier == "high":
        return _cfg(config, "high_tier_model") or _cfg(config, "default_model")
    if tier == "low":
        return _cfg(config, "low_tier_model") or _cfg(config, "default_model")
    return _cfg(config, "default_model")


def _make_anthropic(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=name,
        api_key=_cfg(config, "api_key"),
        max_tokens=8192,
    )


_bedrock_client = None


def reset_bedrock_client() -> None:
    """Discard the cached Bedrock client so the next call creates a fresh one."""
    global _bedrock_client
    _bedrock_client = None


def _get_bedrock_client(*, config: dict | None = None):
    """Get or create a Bedrock runtime client.

    When ``config`` is provided, always creates a fresh (non-cached) client
    with the given credentials — safe for multi-config embedding.
    When ``config`` is None (CLI mode), uses the module-level cache.
    """
    global _bedrock_client

    if config:
        # Engine mode: create a fresh client per config, no caching
        import boto3
        from botocore.config import Config as BotoConfig

        return boto3.client(
            "bedrock-runtime",
            region_name=_cfg(config, "region"),
            aws_access_key_id=_cfg(config, "access_key_id"),
            aws_secret_access_key=_cfg(config, "secret_access_key"),
            config=BotoConfig(
                read_timeout=300,
                connect_timeout=10,
                retries={"max_attempts": 0},
            ),
        )

    # CLI mode: cached singleton
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=BotoConfig(
                read_timeout=300,
                connect_timeout=10,
                retries={"max_attempts": 0},
            ),
        )
    return _bedrock_client


def _make_bedrock(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=name,
        client=_get_bedrock_client(config=config),
        max_tokens=8192,
    )


def _make_openai(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": name,
        "api_key": _cfg(config, "openai_api_key") or _cfg(config, "api_key"),
    }
    base_url = _cfg(config, "openai_api_base")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _make_azure(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=name,
        azure_endpoint=_cfg(config, "azure_endpoint"),
        api_key=_cfg(config, "azure_api_key") or _cfg(config, "api_key"),
        api_version=_cfg(config, "azure_api_version") or "2024-12-01-preview",
    )


# --- GitHub Models ---
# Claude model names on GitHub use Anthropic's native API; everything else
# goes through the OpenAI-compatible chat/completions endpoint.
_GITHUB_CLAUDE_PREFIXES = ("claude-", "anthropic/claude-")


def _is_github_claude(model: str) -> bool:
    """Check if a GitHub Models model name is a Claude variant."""
    return any(model.startswith(p) for p in _GITHUB_CLAUDE_PREFIXES)


def _make_github(name: str, *, config: dict | None = None) -> BaseChatModel:
    """Create a LangChain model via GitHub Models.

    Supports two API formats:
      - Claude models   -> ChatAnthropic  (Anthropic Messages API)
      - Everything else  -> ChatOpenAI     (OpenAI Chat Completions API)

    Model names can be prefixed with "github/" — the prefix is stripped
    before calling the API.
    """
    model_id = name.removeprefix("github/")
    token = _cfg(config, "github_token") or _cfg(config, "api_key")

    if _is_github_claude(model_id):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_id,
            api_key=token,
            base_url=_cfg(config, "github_anthropic_base_url")
            or "https://models.inference.ai.azure.com",
            max_tokens=8192,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            api_key=token,
            base_url=_cfg(config, "github_base_url")
            or "https://models.inference.ai.azure.com",
        )


def _make_gemini(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = _cfg(config, "google_api_key") or _cfg(config, "api_key")
    kwargs: dict = {"model": name, "temperature": 0, "max_tokens": 8192}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatGoogleGenerativeAI(**kwargs)


def _make_local(name: str, *, config: dict | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    base_url = _cfg(config, "openai_api_base") or "http://localhost:8000/v1"
    api_key = _cfg(config, "openai_api_key") or _cfg(config, "api_key") or "not-needed"
    return ChatOpenAI(model=name, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Provider registry — single source of truth for factories + default models
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        _make_anthropic,
        "claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
    ),
    "bedrock": ProviderSpec(
        _make_bedrock,
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    "openai": ProviderSpec(_make_openai, "gpt-4o", "gpt-4o", "gpt-4o-mini"),
    "azure": ProviderSpec(_make_azure, "gpt-4o", "gpt-4o", "gpt-4o-mini"),
    "github": ProviderSpec(
        _make_github,
        "github/openai/gpt-4.1", "github/openai/gpt-4.1", "github/openai/gpt-4o-mini",
    ),
    "gemini": ProviderSpec(
        _make_gemini, "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
    ),
    "local": ProviderSpec(_make_local),
}

# Backward compat: flat factory dict
_PROVIDERS = {k: v.factory for k, v in _REGISTRY.items()}


def make_model(
    model_name: str = "",
    tier: str = "default",
    *,
    config: dict | None = None,
) -> BaseChatModel:
    """Create an LLM instance for the appropriate provider.

    Args:
        model_name: Explicit model ID. If empty, resolved from tier.
        tier: One of "high", "default", "low".
        config: Optional credentials dict for engine mode.
            When provided, bypasses module-level globals.

    Returns:
        A LangChain chat model instance.
    """
    name = resolve_model_name(model_name, tier, config=config)
    provider = _detect_provider(name, config=config)

    # Central prefix stripping — remove "provider/" if present
    prefix = f"{provider}/"
    if name.startswith(prefix):
        name = name[len(prefix):]

    factory = _PROVIDERS.get(provider)
    if not factory:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDERS)}"
        )

    return factory(name, config=config)
