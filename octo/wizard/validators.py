"""Provider credential validation — shared by init wizard and doctor."""
from __future__ import annotations


def validate_provider(
    provider: str,
    creds: dict[str, str],
    model_name: str = "",
) -> tuple[bool, str]:
    """Test provider connectivity with a minimal API call.

    Returns ``(success, message)``.
    """
    try:
        fn = {
            "anthropic": _validate_anthropic,
            "bedrock": _validate_bedrock,
            "openai": _validate_openai,
            "azure": _validate_azure,
            "github": _validate_github,
            "gemini": _validate_gemini,
            "local": _validate_local,
        }.get(provider)
        if fn is None:
            return False, f"Unknown provider: {provider}"
        return fn(creds, model_name)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _validate_anthropic(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_anthropic import ChatAnthropic

    model = model_name or "claude-haiku-4-5-20251001"
    llm = ChatAnthropic(
        model=model,
        api_key=creds.get("ANTHROPIC_API_KEY", ""),
        max_tokens=16,
    )
    llm.invoke("Say 'ok'")
    return True, f"Connected to Anthropic ({model})"


def _validate_bedrock(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    import boto3
    from langchain_aws import ChatBedrockConverse

    model = model_name or "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    client = boto3.client(
        "bedrock-runtime",
        region_name=creds.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY", ""),
    )
    llm = ChatBedrockConverse(model_id=model, client=client, max_tokens=16)
    llm.invoke("Say 'ok'")
    return True, f"Connected to Bedrock ({model})"


def _validate_openai(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_openai import ChatOpenAI

    model = model_name or "gpt-4o-mini"
    llm = ChatOpenAI(
        model=model,
        api_key=creds.get("OPENAI_API_KEY", ""),
        max_tokens=16,
    )
    llm.invoke("Say 'ok'")
    return True, f"Connected to OpenAI ({model})"


def _validate_azure(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_openai import AzureChatOpenAI

    model = model_name or "gpt-4o-mini"
    llm = AzureChatOpenAI(
        azure_deployment=model,
        azure_endpoint=creds.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=creds.get("AZURE_OPENAI_API_KEY", ""),
        api_version=creds.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        max_tokens=16,
    )
    llm.invoke("Say 'ok'")
    return True, f"Connected to Azure OpenAI ({model})"


def _validate_github(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_openai import ChatOpenAI

    # Strip github/ prefix for validation; default to a cheap model
    model = (model_name or "github/openai/gpt-4o-mini").removeprefix("github/")
    llm = ChatOpenAI(
        model=model,
        api_key=creds.get("GITHUB_TOKEN", ""),
        base_url="https://models.inference.ai.azure.com",
        max_tokens=16,
    )
    llm.invoke("Say 'ok'")
    return True, f"Connected to GitHub Models ({model})"


def _validate_gemini(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = model_name or "gemini-2.5-flash-lite"
    api_key = creds.get("GOOGLE_API_KEY", "")
    kwargs: dict = {"model": model, "max_tokens": 16}
    if api_key:
        kwargs["api_key"] = api_key
    llm = ChatGoogleGenerativeAI(**kwargs)
    llm.invoke("Say 'ok'")
    return True, f"Connected to Google Gemini ({model})"


def _validate_local(creds: dict[str, str], model_name: str) -> tuple[bool, str]:
    from langchain_openai import ChatOpenAI

    model = model_name or "test"
    base_url = creds.get("OPENAI_API_BASE", "") or "http://localhost:8000/v1"
    api_key = creds.get("OPENAI_API_KEY", "") or "not-needed"
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=16,
    )
    llm.invoke("Say 'ok'")
    return True, f"Connected to local endpoint ({base_url}, model={model})"
