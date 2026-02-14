"""Claude Code tool — invoke Claude Code CLI agents via `claude -p`.

Allows Octi to delegate tasks to external Claude Code agents running
in their native project directories with full project context.

Each project has its own CLAUDE_CONFIG_DIR (.claude/ directory) with
settings, agents, rules, etc. The tool resolves the correct config
from the project registry and sets it in the subprocess environment.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain_core.tools import tool

from octo.config import PROJECTS, CLAUDE_CODE_TIMEOUT, get_project_for_agent


def _resolve_project(agent_name: str = "", working_directory: str = ""):
    """Resolve project config, working dir, and env for a claude invocation.

    Returns (cwd: str, env: dict, error: str | None).
    """
    env_overrides: dict[str, str] = {}

    # 1. Try agent name → project registry
    if agent_name:
        proj = get_project_for_agent(agent_name)
        if proj:
            cwd = working_directory or proj.path
            env_overrides.update(proj.env)
            return cwd, env_overrides, None

    # 2. Try working_directory → match a known project
    if working_directory:
        wd = str(Path(working_directory).expanduser().resolve())
        for proj in PROJECTS.values():
            if wd.startswith(proj.path):
                env_overrides.update(proj.env)
                break
        return working_directory, env_overrides, None

    # 3. Fallback: current directory, no env overrides
    return os.getcwd(), env_overrides, None


@tool
async def claude_code(
    prompt: str,
    working_directory: str = "",
    agent: str = "",
    timeout: int = CLAUDE_CODE_TIMEOUT,
) -> str:
    """Run a task using Claude Code CLI in a project directory.

    Use this to delegate work to Claude Code — it has full access to
    the target project's codebase, git history, and local tools.
    Each project's CLAUDE_CONFIG_DIR is set automatically so Claude Code
    picks up the correct settings, agents, and rules.

    Args:
        prompt: The task or question to send to Claude Code.
        working_directory: Project directory to run in. If omitted and
            agent is specified, auto-resolves from project registry.
        agent: Optional agent name (from .claude/agents/) to use.
            This also determines which project to run in.
        timeout: Maximum seconds to wait (default from CLAUDE_CODE_TIMEOUT env, 1800s).

    Returns:
        Claude Code's response text, or error message.
    """
    cwd, env_overrides, err = _resolve_project(agent, working_directory)
    if err:
        return err

    cwd_path = Path(cwd).expanduser().resolve()
    if not cwd_path.is_dir():
        return f"Error: Directory not found: {cwd}"

    # Build subprocess environment: inherit current + project overrides
    proc_env = {**os.environ, **env_overrides}

    # Build command
    cmd = ["claude", "-p", "--output-format", "json"]
    if agent:
        cmd.extend(["--agent", agent])
    cmd.append(prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path),
            env=proc_env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"Error: Claude Code timed out after {timeout}s"
    except FileNotFoundError:
        return (
            "Error: 'claude' CLI not found. "
            "Install Claude Code: npm install -g @anthropic-ai/claude-code"
        )
    except Exception as e:
        return f"Error running Claude Code: {e}"

    # Parse response
    output = stdout.decode("utf-8", errors="replace").strip()
    err_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        parts = []
        if output:
            parts.append(output)
        if err_text:
            parts.append(f"[stderr] {err_text}")
        parts.append(f"[exit code: {proc.returncode}]")
        return "\n".join(parts)

    # JSON output mode — extract the result text
    if output:
        try:
            data = json.loads(output)
            # claude --output-format json returns {result: "...", ...}
            if isinstance(data, dict):
                return data.get("result", output)
        except json.JSONDecodeError:
            pass

    return output or "(no output)"


claude_code_tool = claude_code
