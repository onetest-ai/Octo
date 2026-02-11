"""Dependency installer for skills — Python, npm, MCP, system packages."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys

import click

from octo.config import MCP_CONFIG_PATH


class DependencyInstaller:
    """Install dependencies declared in a skill's YAML frontmatter."""

    def __init__(self, deps: dict, requires: list[dict] | None = None) -> None:
        self._python: list[str] = deps.get("python", [])
        self._npm: list[str] = deps.get("npm", [])
        self._mcp: list[dict] = deps.get("mcp", [])
        self._system: list[str] = deps.get("system", [])
        self._requires: list[dict] = requires or []

    # ------------------------------------------------------------------
    # Pre-checks
    # ------------------------------------------------------------------

    def check_requirements(self) -> list[str]:
        """Verify required commands and env vars. Returns list of problems."""
        problems: list[str] = []
        for req in self._requires:
            if "command" in req:
                if not shutil.which(req["command"]):
                    reason = req.get("reason", "")
                    problems.append(f"Missing command: {req['command']}" + (f" ({reason})" if reason else ""))
            if "env" in req:
                if not os.getenv(req["env"]):
                    reason = req.get("reason", "")
                    problems.append(f"Missing env var: {req['env']}" + (f" ({reason})" if reason else ""))
        return problems

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------

    def install_python(self) -> bool:
        """Install Python pip packages. Returns True if all succeeded."""
        if not self._python:
            return True

        click.echo(f"  Installing Python packages: {', '.join(self._python)}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *self._python],
                check=True,
                capture_output=True,
                text=True,
            )
            click.echo("  Python packages installed.")
            return True
        except subprocess.CalledProcessError as exc:
            click.echo(f"  Failed to install Python packages: {exc.stderr[:200]}")
            return False

    # ------------------------------------------------------------------
    # npm
    # ------------------------------------------------------------------

    def install_npm(self) -> bool:
        """Install npm global packages. Returns True if all succeeded."""
        if not self._npm:
            return True

        if not shutil.which("npm"):
            click.echo("  Warning: npm not found — skipping npm dependencies.")
            return False

        click.echo(f"  Installing npm packages: {', '.join(self._npm)}")
        try:
            subprocess.run(
                ["npm", "install", "-g", *self._npm],
                check=True,
                capture_output=True,
                text=True,
            )
            click.echo("  npm packages installed.")
            return True
        except subprocess.CalledProcessError as exc:
            click.echo(f"  Failed to install npm packages: {exc.stderr[:200]}")
            return False

    # ------------------------------------------------------------------
    # MCP
    # ------------------------------------------------------------------

    def configure_mcp(self) -> bool:
        """Add MCP server configs to .mcp.json. Returns True if successful."""
        if not self._mcp:
            return True

        # Load existing config
        config: dict = {}
        if MCP_CONFIG_PATH.is_file():
            try:
                config = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = {}

        servers = config.setdefault("mcpServers", {})
        added = []

        for mcp_entry in self._mcp:
            server_name = mcp_entry.get("server", "")
            if not server_name:
                continue

            if server_name in servers:
                click.echo(f"  MCP server '{server_name}' already configured, skipping.")
                continue

            servers[server_name] = {
                "command": "npx",
                "args": ["-y", mcp_entry.get("package", ""), *mcp_entry.get("args", [])],
            }
            env = mcp_entry.get("env")
            if env:
                servers[server_name]["env"] = env

            added.append(server_name)

        if added:
            MCP_CONFIG_PATH.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            click.echo(f"  Added MCP servers: {', '.join(added)}")
            click.echo("  Note: restart Octo or run /mcp reload to activate new MCP servers.")

        return True

    # ------------------------------------------------------------------
    # System packages
    # ------------------------------------------------------------------

    def prompt_system_packages(self) -> None:
        """Display system package requirements (never auto-install)."""
        if not self._system:
            return

        click.echo("\n  System packages required (install manually):")
        system = platform.system()
        for pkg in self._system:
            if system == "Darwin":
                click.echo(f"    brew install {pkg}")
            elif system == "Linux":
                click.echo(f"    sudo apt install {pkg}  # or your distro's package manager")
            else:
                click.echo(f"    {pkg}")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def install_all(self) -> bool:
        """Run all installation steps. Returns True if everything succeeded."""
        # Pre-checks
        problems = self.check_requirements()
        if problems:
            click.echo("\n  Requirements not met:")
            for p in problems:
                click.echo(f"    - {p}")
            if not click.confirm("  Continue anyway?", default=False):
                return False

        ok = True
        ok = self.install_python() and ok
        ok = self.install_npm() and ok
        ok = self.configure_mcp() and ok
        self.prompt_system_packages()

        if ok:
            click.echo("\n  All dependencies installed successfully.")
        else:
            click.echo("\n  Some dependencies failed — check messages above.")

        return ok
