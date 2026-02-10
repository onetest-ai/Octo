"""Assisted onboarding — setup wizard, health check, first-run detection."""
from __future__ import annotations

from octo.wizard.doctor import run_doctor
from octo.wizard.onboarding import run_init

__all__ = ["run_init", "run_doctor", "check_first_run"]


def check_first_run() -> bool:
    """Check if Octo is configured.  If not, offer to run the setup wizard.

    Returns ``True`` if configuration exists (proceed to chat).
    Returns ``False`` if user declined setup (caller should exit).
    """
    from pathlib import Path

    from rich.console import Console
    from rich.prompt import Confirm

    from octo.config import WORKSPACE

    env_path = WORKSPACE / ".env"
    if env_path.is_file():
        return True

    console = Console()
    console.print()
    console.print("[bold yellow]No .env file found.[/bold yellow]")
    console.print("Octo needs API credentials to connect to an LLM provider.")
    console.print()

    if Confirm.ask("Run setup wizard now?", default=True):
        run_init()
        return env_path.is_file()

    console.print()
    console.print("[dim]Run [bold]octo init[/bold] later to configure.[/dim]")
    return False
