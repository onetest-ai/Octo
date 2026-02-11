"""Model factory — create LLM instances for any supported provider."""
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

from octo.config import (
    ANTHROPIC_API_KEY,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    GITHUB_TOKEN,
    GITHUB_MODELS_BASE_URL,
    GITHUB_MODELS_ANTHROPIC_BASE_URL,
    LLM_PROVIDER,
    DEFAULT_MODEL,
    HIGH_TIER_MODEL,
    LOW_TIER_MODEL,
)


def _detect_provider(model_name: str) -> str:
    """Auto-detect provider from model name pattern.

    Priority:
      1. Explicit LLM_PROVIDER env var
      2. Model name heuristics
      3. Credential availability
    """
    if LLM_PROVIDER:
        return LLM_PROVIDER

    # GitHub Models — model names prefixed with "github/"
    if model_name.startswith("github/"):
        return "github"

    # Bedrock model IDs contain region prefix like "eu.anthropic." or "us.anthropic."
    if ".anthropic." in model_name or ".amazon." in model_name or ".meta." in model_name:
        return "bedrock"

    # OpenAI models
    if model_name.startswith(("gpt-", "o1-", "o3-", "o4-")):
        if AZURE_OPENAI_ENDPOINT:
            return "azure"
        return "openai"

    # Claude models via direct API
    if model_name.startswith("claude-"):
        return "anthropic"

    # Fallback: check which credentials are available
    if AWS_REGION and AWS_ACCESS_KEY_ID:
        return "bedrock"
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        return "azure"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GITHUB_TOKEN:
        return "github"

    return "anthropic"


def resolve_model_name(model_name: str = "", tier: str = "default") -> str:
    """Resolve model name from explicit override or tier."""
    if model_name:
        return model_name
    if tier == "high":
        return HIGH_TIER_MODEL
    if tier == "low":
        return LOW_TIER_MODEL
    return DEFAULT_MODEL


def _make_anthropic(name: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=name,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=8192,
    )


_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        from botocore.config import Config

        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=Config(
                read_timeout=300,       # 5 min — large contexts need time
                connect_timeout=10,
                retries={"max_attempts": 0},  # we handle retries ourselves
            ),
        )
    return _bedrock_client


def _make_bedrock(name: str) -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=name,
        client=_get_bedrock_client(),
        max_tokens=8192,
    )


def _make_openai(name: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=name,
        api_key=OPENAI_API_KEY,
    )


def _make_azure(name: str) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=name,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# --- GitHub Models ---
# Claude model names on GitHub use Anthropic's native API; everything else
# goes through the OpenAI-compatible chat/completions endpoint.
_GITHUB_CLAUDE_PREFIXES = ("claude-", "anthropic/claude-")


def _is_github_claude(model: str) -> bool:
    """Check if a GitHub Models model name is a Claude variant."""
    return any(model.startswith(p) for p in _GITHUB_CLAUDE_PREFIXES)


def _make_github(name: str) -> BaseChatModel:
    """Create a LangChain model via GitHub Models.

    Supports two API formats:
      • Claude models   → ChatAnthropic  (Anthropic Messages API)
      • Everything else  → ChatOpenAI     (OpenAI Chat Completions API)

    Model names can be prefixed with "github/" — the prefix is stripped
    before calling the API.  Examples:
      github/gpt-4o              → ChatOpenAI("gpt-4o")
      github/openai/gpt-4.1      → ChatOpenAI("openai/gpt-4.1")
      github/claude-sonnet-4-5-20250929 → ChatAnthropic("claude-sonnet-4-5-20250929")
      github/anthropic/claude-sonnet-4  → ChatAnthropic("anthropic/claude-sonnet-4")
    """
    # Strip "github/" prefix if present
    model_id = name.removeprefix("github/")

    if _is_github_claude(model_id):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_id,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_MODELS_ANTHROPIC_BASE_URL,
            max_tokens=8192,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_MODELS_BASE_URL,
        )


_PROVIDERS = {
    "anthropic": _make_anthropic,
    "bedrock": _make_bedrock,
    "openai": _make_openai,
    "azure": _make_azure,
    "github": _make_github,
}


def make_model(model_name: str = "", tier: str = "default") -> BaseChatModel:
    """Create an LLM instance for the appropriate provider.

    Args:
        model_name: Explicit model ID. If empty, resolved from tier.
        tier: One of "high", "default", "low".

    Returns:
        A LangChain chat model instance.
    """
    name = resolve_model_name(model_name, tier)
    provider = _detect_provider(name)

    factory = _PROVIDERS.get(provider)
    if not factory:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(_PROVIDERS)}"
        )

    return factory(name)
