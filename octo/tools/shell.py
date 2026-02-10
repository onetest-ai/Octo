"""Shell tool — Bash command execution.

Named to match Claude Code's Bash tool.
"""
from __future__ import annotations

import asyncio
import subprocess

from langchain_core.tools import tool


@tool
def Bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds (default 120).

    Returns:
        Combined stdout and stderr output, or error message.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None,  # uses current working directory
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(output_parts) if output_parts else "(no output)"
        # Truncate very large outputs
        if len(output) > 50000:
            output = output[:50000] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"


bash_tool = Bash
