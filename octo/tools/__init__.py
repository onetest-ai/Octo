"""Built-in tools for Octo agents.

Filesystem/shell tools (matching Claude Code names):
  Read, Grep, Glob, Bash, Edit

Delegation:
  claude_code
"""
from octo.tools.filesystem import read_tool, grep_tool, glob_tool, edit_tool
from octo.tools.shell import bash_tool
from octo.tools.claude_code import claude_code_tool

BUILTIN_TOOLS = [read_tool, grep_tool, glob_tool, edit_tool, bash_tool]
CLAUDE_CODE_TOOL = claude_code_tool

__all__ = [
    "BUILTIN_TOOLS",
    "CLAUDE_CODE_TOOL",
    "read_tool",
    "grep_tool",
    "glob_tool",
    "edit_tool",
    "bash_tool",
    "claude_code_tool",
]
