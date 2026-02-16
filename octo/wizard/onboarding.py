"""Interactive setup wizard for Octo CLI (``octo init``)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# ── Default models per provider — sourced from models._REGISTRY ────────

_PROVIDERS = [
    ("1", "anthropic", "Anthropic (Claude)", "Direct API — requires ANTHROPIC_API_KEY"),
    ("2", "bedrock", "AWS Bedrock", "Uses AWS credentials — supports Claude, Llama, etc."),
    ("3", "openai", "OpenAI", "GPT-4o, o1, o3 — requires OPENAI_API_KEY"),
    ("4", "azure", "Azure OpenAI", "Azure-hosted models — requires endpoint + key"),
    ("5", "github", "GitHub Models", "GPT, Claude, Mistral, Llama via GitHub PAT"),
    ("6", "gemini", "Google Gemini", "Gemini 2.5 Flash/Pro — requires GOOGLE_API_KEY"),
    ("7", "local", "Local / Custom", "vLLM, Ollama, llama.cpp — OpenAI-compatible endpoint"),
]

_MCP_TEMPLATES: dict[str, dict[str, Any]] = {
    "playwright": {
        "type": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
    },
    "context7": {
        "type": "http",
        "url": "https://mcp.context7.com/mcp",
    },
}


# ── Public entry point ──────────────────────────────────────────────────


def run_init(
    *,
    quick: bool = False,
    provider: str | None = None,
    skip_validation: bool = False,
    skip_persona: bool = False,
    force: bool = False,
) -> None:
    """Run the interactive setup wizard."""
    from octo.config import WORKSPACE

    _print_init_banner()

    env_path = WORKSPACE / ".env"

    # Step 1: existing config guard
    if not _check_existing_config(env_path, force):
        return

    # Step 2: track selection
    track = "quick" if quick else _select_track()

    # Step 3: provider
    selected_provider = _select_provider(provider)

    # Step 4: credentials
    creds = _collect_credentials(selected_provider)

    # Step 5: model config
    models = _collect_model_config(selected_provider, track)

    # Step 6: build env vars
    env_vars: dict[str, str] = {}
    env_vars["LLM_PROVIDER"] = selected_provider
    env_vars.update(creds)
    env_vars.update(models)
    env_vars["MODEL_PROFILE"] = "balanced"

    # Step 7: optional config (advanced only)
    if track == "advanced":
        env_vars.update(_collect_optional_config())

    # Step 8: write .env
    _write_env_file(env_path, env_vars)
    console.print(f"  [green]Created {env_path}[/green]")

    # Step 9: validate credentials
    if not skip_validation:
        _run_validation(selected_provider, creds, models)

    # Step 10: persona scaffolding
    created_files: list[str] = []
    if not skip_persona:
        user_name = ""
        if track == "advanced":
            user_name = Prompt.ask("\nYour name (for USER.md)", default="")
        created_files = _scaffold_persona(user_name)

    # Step 11: MCP scaffolding (advanced)
    if track == "advanced":
        _scaffold_mcp(WORKSPACE)

    # Step 12: summary
    _print_summary(env_path, created_files, selected_provider)


# ── Wizard steps ────────────────────────────────────────────────────────


def _print_init_banner() -> None:
    console.print()
    console.print(
        Panel(
            "[bold cyan]Octo Setup Wizard[/bold cyan]\n"
            "[dim]Configure LLM provider, credentials, and workspace.[/dim]",
            box=box.DOUBLE,
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _check_existing_config(env_path: Path, force: bool) -> bool:
    """Return True to proceed, False to abort."""
    if not env_path.is_file():
        return True

    if force:
        console.print("  [yellow]--force: overwriting existing .env[/yellow]")
        return True

    console.print(f"  [yellow]Found existing {env_path}[/yellow]")
    choice = Prompt.ask(
        "  What would you like to do?",
        choices=["overwrite", "abort"],
        default="abort",
    )
    return choice == "overwrite"


def _select_track() -> str:
    console.print()
    console.print("[bold cyan]Setup mode:[/bold cyan]")
    console.print("  [bold yellow]1[/bold yellow]  QuickStart  [dim]— pick provider, paste key, done[/dim]")
    console.print("  [bold yellow]2[/bold yellow]  Advanced    [dim]— model tiers, Telegram, MCP, persona[/dim]")
    choice = Prompt.ask("  Mode", choices=["1", "2"], default="1")
    return "quick" if choice == "1" else "advanced"


def _select_provider(preselected: str | None) -> str:
    if preselected:
        console.print(f"\n  [dim]Provider: {preselected}[/dim]")
        return preselected

    console.print()
    console.print("[bold cyan]Select your LLM provider:[/bold cyan]")
    console.print()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("", style="bold yellow", width=3)
    table.add_column("Provider", style="bold cyan")
    table.add_column("Description", style="dim")
    for num, _key, name, desc in _PROVIDERS:
        table.add_row(num, name, desc)
    console.print(table)

    choice = Prompt.ask("  Provider", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")
    return {p[0]: p[1] for p in _PROVIDERS}[choice]


def _collect_credentials(provider: str) -> dict[str, str]:
    console.print()
    console.print(f"[bold cyan]Configure {provider} credentials:[/bold cyan]")

    creds: dict[str, str] = {}

    if provider == "anthropic":
        existing = os.environ.get("ANTHROPIC_API_KEY", "")
        if existing and not existing.startswith("sk-ant-oat"):
            console.print("  [dim]Using ANTHROPIC_API_KEY from environment[/dim]")
            creds["ANTHROPIC_API_KEY"] = existing
        else:
            creds["ANTHROPIC_API_KEY"] = Prompt.ask("  Anthropic API key", password=True)

    elif provider == "bedrock":
        creds["AWS_REGION"] = Prompt.ask(
            "  AWS region",
            default=os.environ.get("AWS_REGION", "us-east-1"),
        )
        existing_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        if existing_key:
            console.print("  [dim]Using AWS credentials from environment[/dim]")
            creds["AWS_ACCESS_KEY_ID"] = existing_key
            creds["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        else:
            creds["AWS_ACCESS_KEY_ID"] = Prompt.ask("  AWS access key ID", password=True)
            creds["AWS_SECRET_ACCESS_KEY"] = Prompt.ask("  AWS secret access key", password=True)

    elif provider == "openai":
        existing = os.environ.get("OPENAI_API_KEY", "")
        if existing:
            console.print("  [dim]Using OPENAI_API_KEY from environment[/dim]")
            creds["OPENAI_API_KEY"] = existing
        else:
            creds["OPENAI_API_KEY"] = Prompt.ask("  OpenAI API key", password=True)

    elif provider == "azure":
        creds["AZURE_OPENAI_ENDPOINT"] = Prompt.ask(
            "  Azure OpenAI endpoint URL",
            default=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        )
        existing = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if existing:
            console.print("  [dim]Using AZURE_OPENAI_API_KEY from environment[/dim]")
            creds["AZURE_OPENAI_API_KEY"] = existing
        else:
            creds["AZURE_OPENAI_API_KEY"] = Prompt.ask("  Azure OpenAI API key", password=True)
        creds["AZURE_OPENAI_API_VERSION"] = Prompt.ask(
            "  API version",
            default=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    elif provider == "github":
        existing = os.environ.get("GITHUB_TOKEN", "")
        if existing:
            console.print("  [dim]Using GITHUB_TOKEN from environment[/dim]")
            creds["GITHUB_TOKEN"] = existing
        else:
            console.print("  [dim]Create a PAT at github.com/settings/tokens with 'models:read' scope[/dim]")
            creds["GITHUB_TOKEN"] = Prompt.ask("  GitHub Personal Access Token", password=True)

    elif provider == "gemini":
        existing = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if existing:
            console.print("  [dim]Using GOOGLE_API_KEY from environment[/dim]")
            creds["GOOGLE_API_KEY"] = existing
        else:
            console.print("  [dim]Get an API key at aistudio.google.com/apikey[/dim]")
            creds["GOOGLE_API_KEY"] = Prompt.ask("  Google API key", password=True)

    elif provider == "local":
        creds["OPENAI_API_BASE"] = Prompt.ask(
            "  API base URL",
            default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        )
        api_key = Prompt.ask("  API key (optional, press Enter to skip)", default="", password=True)
        if api_key:
            creds["OPENAI_API_KEY"] = api_key

    return creds


def _collect_model_config(provider: str, track: str) -> dict[str, str]:
    from octo.models import _REGISTRY

    spec = _REGISTRY.get(provider)
    defaults = {
        "DEFAULT_MODEL": spec.default if spec else "",
        "HIGH_TIER_MODEL": spec.high if spec else "",
        "LOW_TIER_MODEL": spec.low if spec else "",
    }

    if track == "quick":
        if not defaults["DEFAULT_MODEL"]:
            # Local provider: no defaults — must ask for model name
            model = Prompt.ask("  Model name (e.g. llama3, mistral)")
            return {"DEFAULT_MODEL": model, "HIGH_TIER_MODEL": model, "LOW_TIER_MODEL": model}
        console.print(f"  [dim]Using default models for {provider}[/dim]")
        return dict(defaults)

    console.print()
    console.print("[bold cyan]Model configuration:[/bold cyan]")
    console.print("[dim]  Press Enter to accept defaults.[/dim]")
    console.print()

    models: dict[str, str] = {}
    labels = {
        "DEFAULT_MODEL": "Default model",
        "HIGH_TIER_MODEL": "High-tier model (complex reasoning)",
        "LOW_TIER_MODEL": "Low-tier model (summarization, errors)",
    }
    for var, default in defaults.items():
        models[var] = Prompt.ask(f"  {labels.get(var, var)}", default=default or "")

    return models


def _collect_optional_config() -> dict[str, str]:
    console.print()
    console.print("[bold cyan]Optional configuration:[/bold cyan]")

    optional: dict[str, str] = {}

    # Telegram
    if Confirm.ask("\n  Configure Telegram bot?", default=False):
        optional["TELEGRAM_BOT_TOKEN"] = Prompt.ask("  Bot token", password=True)
        optional["TELEGRAM_OWNER_ID"] = Prompt.ask("  Your Telegram user ID")

    # Agent directories
    if Confirm.ask("\n  Add external agent directories (AGENT_DIRS)?", default=False):
        console.print("  [dim]Colon-separated paths to .claude/agents/ directories[/dim]")
        dirs = Prompt.ask("  AGENT_DIRS")
        if dirs.strip():
            optional["AGENT_DIRS"] = dirs.strip()

    # Profile
    console.print()
    console.print("  [dim]Model profiles: quality (best), balanced (default), budget (cheapest)[/dim]")
    profile = Prompt.ask("  Model profile", choices=["quality", "balanced", "budget"], default="balanced")
    optional["MODEL_PROFILE"] = profile

    return optional


def _run_validation(provider: str, creds: dict[str, str], models: dict[str, str]) -> None:
    console.print()
    with console.status("[yellow]Validating credentials...[/yellow]", spinner="dots"):
        from octo.wizard.validators import validate_provider

        low_model = models.get("LOW_TIER_MODEL", "")
        ok, msg = validate_provider(provider, creds, low_model)

    if ok:
        console.print(f"  [green]{msg}[/green]")
    else:
        console.print(f"  [red]{msg}[/red]")
        console.print("  [dim]Credentials saved anyway. Run [bold]octo doctor[/bold] to re-check.[/dim]")


def _scaffold_persona(user_name: str = "") -> list[str]:
    from octo.config import MEMORY_DIR, PERSONA_DIR
    from octo.wizard.templates import PERSONA_TEMPLATES

    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    for filename, template_fn in PERSONA_TEMPLATES.items():
        path = PERSONA_DIR / filename
        if path.exists():
            continue
        content = template_fn(user_name=user_name)
        path.write_text(content + "\n", encoding="utf-8")
        created.append(str(path))

    if created:
        console.print(f"\n  [green]Created {len(created)} persona file(s)[/green]")
    else:
        console.print("\n  [dim]All persona files already exist — skipped[/dim]")

    return created


def _scaffold_mcp(workspace: Path) -> None:
    mcp_path = workspace / ".mcp.json"
    if mcp_path.is_file():
        console.print("\n  [dim].mcp.json already exists — skipped[/dim]")
        return

    if not Confirm.ask("\n  Create .mcp.json with starter MCP servers?", default=False):
        return

    available = list(_MCP_TEMPLATES.keys())
    console.print(f"  [dim]Available: {', '.join(available)}[/dim]")
    selected = Prompt.ask(
        "  Servers to include (comma-separated, or 'all')",
        default="all",
    )

    if selected.strip().lower() == "all":
        servers = dict(_MCP_TEMPLATES)
    else:
        names = [s.strip() for s in selected.split(",") if s.strip()]
        servers = {n: _MCP_TEMPLATES[n] for n in names if n in _MCP_TEMPLATES}

    if not servers:
        console.print("  [dim]No valid servers selected — skipped[/dim]")
        return

    data = {"mcpServers": servers}
    mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    console.print(f"  [green]Created {mcp_path} ({len(servers)} server(s))[/green]")


def _write_env_file(env_path: Path, env_vars: dict[str, str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[tuple[str, str, list[str]]] = [
        (
            "LLM Provider",
            "Auto-detected from model name if not set.\n# Values: anthropic, bedrock, openai, azure, github, gemini, local",
            ["LLM_PROVIDER"],
        ),
        ("Anthropic", "Direct Anthropic API access", ["ANTHROPIC_API_KEY"]),
        ("AWS Bedrock", "AWS credentials for Bedrock", ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
        ("OpenAI", "OpenAI API access", ["OPENAI_API_KEY"]),
        (
            "Azure OpenAI",
            "Azure-hosted OpenAI models",
            ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"],
        ),
        (
            "GitHub Models",
            "GPT, Claude, Mistral, Llama via GitHub PAT (models:read scope)",
            ["GITHUB_TOKEN", "GITHUB_MODELS_BASE_URL", "GITHUB_MODELS_ANTHROPIC_BASE_URL"],
        ),
        ("Google Gemini", "Gemini 2.5 Flash/Pro", ["GOOGLE_API_KEY"]),
        (
            "Local / Custom",
            "OpenAI-compatible endpoint (vLLM, Ollama, llama.cpp)",
            ["OPENAI_API_BASE"],
        ),
        (
            "Model Tiers",
            "HIGH = complex reasoning, DEFAULT = general, LOW = summarization/errors\n# Use provider/ prefix for mixed providers: anthropic/claude-*, gemini/gemini-*, local/llama3",
            ["DEFAULT_MODEL", "HIGH_TIER_MODEL", "LOW_TIER_MODEL"],
        ),
        ("Model Profile", "Named profiles: quality, balanced, budget", ["MODEL_PROFILE"]),
        (
            "Agent Directories",
            "Colon-separated paths to project .claude/agents/ directories",
            ["AGENT_DIRS"],
        ),
        ("Telegram", "Optional Telegram bot integration", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_OWNER_ID"]),
        ("Voice (ElevenLabs)", "Optional TTS", ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]),
    ]

    lines = [
        f"# Octo CLI Configuration",
        f"# Generated by `octo init` on {now}",
        "",
    ]

    for title, comment, keys in sections:
        section_vars = {k: env_vars[k] for k in keys if k in env_vars}
        if not section_vars:
            continue
        lines.append(f"# --- {title} ---")
        for cmt_line in comment.split("\n"):
            lines.append(f"# {cmt_line}")
        for key, value in section_vars.items():
            lines.append(f"{key}={value}")
        lines.append("")

    env_path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(env_path: Path, created_files: list[str], provider: str) -> None:
    summary = Text()
    summary.append("Setup complete!\n\n", style="bold green")

    summary.append("Created:\n", style="bold white")
    summary.append(f"  {env_path}\n", style="cyan")
    for f in created_files:
        summary.append(f"  {f}\n", style="cyan")

    summary.append(f"\nProvider: ", style="bold white")
    summary.append(f"{provider}\n", style="cyan")

    summary.append("\nNext steps:\n", style="bold white")
    summary.append("  octo          ", style="bold yellow")
    summary.append("Start chatting\n", style="dim")
    summary.append("  octo doctor   ", style="bold yellow")
    summary.append("Verify full configuration\n", style="dim")
    summary.append("  Edit .octo/persona/*.md to customize personality\n", style="dim")

    console.print()
    console.print(
        Panel(summary, title="[bold cyan]Octo Setup[/bold cyan]", border_style="green", box=box.DOUBLE, padding=(1, 2))
    )
    console.print()
