"""Health check — validates Octo configuration and connectivity (``octo doctor``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

console = Console()


class CheckResult:
    """Result of a single health check."""

    def __init__(self, name: str, passed: bool, message: str, hint: str = "") -> None:
        self.name = name
        self.passed = passed
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "message": self.message, "hint": self.hint}


async def run_doctor(*, fix: bool = False, json_output: bool = False) -> None:
    """Run all health checks and display results."""
    results: list[CheckResult] = [
        _check_env_file(),
        _check_provider_credentials(),
        await _check_llm_connectivity(),
        _check_octo_dir(),
        _check_persona_files(),
        _check_mcp_config(),
        _check_agent_dirs(),
        _check_database(),
        _check_oauth_tokens(),
    ]

    if json_output:
        console.print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    _print_results(results)

    if fix and any(not r.passed for r in results):
        console.print()
        if Confirm.ask("Run setup wizard to fix issues?", default=True):
            from octo.wizard.onboarding import run_init

            run_init()


# ── Individual checks ───────────────────────────────────────────────────


def _check_env_file() -> CheckResult:
    from octo.config import WORKSPACE

    env_path = WORKSPACE / ".env"
    if not env_path.is_file():
        return CheckResult(".env file", False, "Not found", "Run `octo init` to create one.")

    content = env_path.read_text()
    if len(content.strip()) < 10:
        return CheckResult(".env file", False, "File exists but appears empty", "Run `octo init` to reconfigure.")

    return CheckResult(".env file", True, f"Found ({len(content)} bytes)")


def _check_provider_credentials() -> CheckResult:
    from octo.config import (
        ANTHROPIC_API_KEY,
        AWS_ACCESS_KEY_ID,
        AWS_REGION,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        GITHUB_TOKEN,
        OPENAI_API_KEY,
    )

    providers: list[str] = []
    warnings: list[str] = []

    if ANTHROPIC_API_KEY:
        if ANTHROPIC_API_KEY.startswith("sk-ant-oat"):
            warnings.append("ANTHROPIC_API_KEY looks like an OAuth token (sk-ant-oat...) — won't work as API key")
        else:
            providers.append("anthropic")
    if AWS_REGION and AWS_ACCESS_KEY_ID:
        providers.append("bedrock")
    if OPENAI_API_KEY:
        providers.append("openai")
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        providers.append("azure")
    if GITHUB_TOKEN:
        providers.append("github")

    if warnings and not providers:
        return CheckResult(
            "Provider credentials",
            False,
            warnings[0],
            "Set a real ANTHROPIC_API_KEY or use another provider.",
        )

    if not providers:
        return CheckResult(
            "Provider credentials",
            False,
            "No provider credentials found",
            "Set ANTHROPIC_API_KEY, AWS creds, OPENAI_API_KEY, or Azure creds in .env",
        )

    msg = f"Configured: {', '.join(providers)}"
    if warnings:
        msg += f" (warning: {warnings[0]})"
    return CheckResult("Provider credentials", True, msg)


async def _check_llm_connectivity() -> CheckResult:
    try:
        from octo.config import (
            ANTHROPIC_API_KEY,
            AWS_ACCESS_KEY_ID,
            AWS_REGION,
            AWS_SECRET_ACCESS_KEY,
            AZURE_OPENAI_API_KEY,
            AZURE_OPENAI_API_VERSION,
            AZURE_OPENAI_ENDPOINT,
            GITHUB_TOKEN,
            LOW_TIER_MODEL,
            OPENAI_API_KEY,
        )
        from octo.models import _detect_provider, resolve_model_name
        from octo.wizard.validators import validate_provider

        model_name = resolve_model_name()
        provider = _detect_provider(model_name)

        creds = {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "AWS_REGION": AWS_REGION,
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
            "GITHUB_TOKEN": GITHUB_TOKEN,
        }

        ok, msg = validate_provider(provider, creds, LOW_TIER_MODEL)
        if ok:
            return CheckResult("LLM connectivity", True, msg)
        return CheckResult("LLM connectivity", False, msg, "Check credentials and model names in .env")
    except Exception as e:
        return CheckResult(
            "LLM connectivity",
            False,
            f"{type(e).__name__}: {str(e)[:120]}",
            "Check credentials in .env",
        )


def _check_octo_dir() -> CheckResult:
    from octo.config import OCTO_DIR

    if not OCTO_DIR.is_dir():
        return CheckResult(
            ".octo/ directory",
            False,
            "Not found",
            "Run `octo init` or start Octo — directories are auto-created.",
        )

    subdirs = ["persona", "agents", "skills", "memory", "projects"]
    existing = [d for d in subdirs if (OCTO_DIR / d).is_dir()]
    return CheckResult(".octo/ directory", True, f"Found ({len(existing)}/{len(subdirs)} subdirs)")


def _check_persona_files() -> CheckResult:
    from octo.config import PERSONA_DIR

    required = ["SOUL.md", "IDENTITY.md"]
    optional = ["USER.md", "AGENTS.md", "MEMORY.md"]

    if not PERSONA_DIR.is_dir():
        return CheckResult(
            "Persona files",
            False,
            "persona/ directory not found",
            "Run `octo init` to scaffold persona files.",
        )

    all_files = required + optional
    existing = [f for f in all_files if (PERSONA_DIR / f).is_file()]
    missing_required = [f for f in required if f not in existing]

    if missing_required:
        return CheckResult(
            "Persona files",
            False,
            f"Missing required: {', '.join(missing_required)}",
            "Run `octo init` to scaffold missing files.",
        )

    return CheckResult("Persona files", True, f"{len(existing)}/{len(all_files)} files present")


def _check_mcp_config() -> CheckResult:
    from octo.config import MCP_CONFIG_PATH

    if not MCP_CONFIG_PATH.is_file():
        return CheckResult("MCP config", True, "Not configured (optional)")

    try:
        data = json.loads(MCP_CONFIG_PATH.read_text())
        servers = data.get("mcpServers", {})
        return CheckResult("MCP config", True, f"{len(servers)} server(s): {', '.join(servers.keys())}")
    except json.JSONDecodeError as e:
        return CheckResult("MCP config", False, f"Invalid JSON: {e}", "Fix syntax in .mcp.json")


def _check_agent_dirs() -> CheckResult:
    from octo.config import AGENT_DIRS

    if not AGENT_DIRS:
        return CheckResult("Agent directories", True, "Using defaults only (.claude/agents/)")

    valid = [d for d in AGENT_DIRS if d.is_dir()]
    invalid = [d for d in AGENT_DIRS if not d.is_dir()]

    if invalid:
        return CheckResult(
            "Agent directories",
            False,
            f"{len(invalid)} missing: {', '.join(str(d) for d in invalid[:3])}",
            "Check AGENT_DIRS in .env — paths must exist.",
        )

    total_agents = sum(len(list(d.glob("*.md"))) for d in valid)
    return CheckResult("Agent directories", True, f"{len(valid)} dir(s), {total_agents} agent file(s)")


def _check_database() -> CheckResult:
    from octo.config import DB_PATH

    if not DB_PATH.is_file():
        return CheckResult("SQLite database", True, "Not yet created (normal for first run)")

    size = DB_PATH.stat().st_size
    size_str = f"{size / 1024:.0f} KB" if size < 1_000_000 else f"{size / 1_048_576:.1f} MB"
    return CheckResult("SQLite database", True, f"Found ({size_str})")


def _check_oauth_tokens() -> CheckResult:
    from octo.config import MCP_CONFIG_PATH, OAUTH_DIR
    from octo.oauth.storage import FileTokenStorage

    if not MCP_CONFIG_PATH.is_file():
        return CheckResult("OAuth tokens", True, "No MCP config (skipped)")

    try:
        data = json.loads(MCP_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return CheckResult("OAuth tokens", True, "Could not read MCP config (skipped)")

    servers = data.get("mcpServers", {})
    auth_servers = {name: spec for name, spec in servers.items() if spec.get("auth")}

    if not auth_servers:
        return CheckResult("OAuth tokens", True, "No servers require OAuth")

    authenticated = []
    missing = []
    for name in auth_servers:
        storage = FileTokenStorage(name, OAUTH_DIR)
        if storage.has_tokens():
            authenticated.append(name)
        else:
            missing.append(name)

    if missing:
        return CheckResult(
            "OAuth tokens",
            True,  # warn, not fail — tokens can be acquired later
            f"{len(authenticated)}/{len(auth_servers)} authenticated"
            + (f" (missing: {', '.join(missing)})" if missing else ""),
            f"Run `octo auth login <server>` for: {', '.join(missing)}",
        )

    return CheckResult("OAuth tokens", True, f"All {len(auth_servers)} server(s) authenticated")


# ── Output ──────────────────────────────────────────────────────────────


def _print_results(results: list[CheckResult]) -> None:
    console.print()
    console.print("[bold cyan]Octo Health Check[/bold cyan]")
    console.print()

    table = Table(box=box.SIMPLE, padding=(0, 2))
    table.add_column("Check", style="bold white")
    table.add_column("Status", width=6)
    table.add_column("Details")

    for r in results:
        status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        details = r.message
        if not r.passed and r.hint:
            details += f"\n[dim]{r.hint}[/dim]"
        table.add_row(r.name, status, details)

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    console.print()
    if passed == total:
        console.print(f"  [bold green]All {total} checks passed.[/bold green]")
    else:
        console.print(f"  [bold yellow]{passed}/{total} checks passed, {total - passed} issue(s) found.[/bold yellow]")
    console.print()
