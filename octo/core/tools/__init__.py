"""Core tools — built-in tools, memory, planning, MCP proxy, Telegram.

Built-in tools (matching Claude Code names):
  Read, Grep, Glob, Bash, Edit

Delegation:
  claude_code

Agent tools:
  memory, planning, mcp_proxy, telegram_tools
"""
from __future__ import annotations

from octo.core.tools.filesystem import read_tool, grep_tool, glob_tool, edit_tool
from octo.core.tools.shell import bash_tool
from octo.core.tools.claude_code import claude_code_tool
from octo.core.tools.lifecycle import (
    task_complete, escalate_question, AGENT_LIFECYCLE_TOOLS,
)

BUILTIN_TOOLS = [read_tool, grep_tool, glob_tool, edit_tool, bash_tool]
CLAUDE_CODE_TOOL = claude_code_tool

__all__ = [
    "BUILTIN_TOOLS",
    "AGENT_LIFECYCLE_TOOLS",
    "CLAUDE_CODE_TOOL",
    "read_tool",
    "grep_tool",
    "glob_tool",
    "edit_tool",
    "bash_tool",
    "claude_code_tool",
    "task_complete",
    "escalate_question",
]
