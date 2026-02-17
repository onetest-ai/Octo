"""Built-in tools for Octo agents.

Backward-compat shim — actual implementations moved to octo.core.tools.
"""
from octo.core.tools import (  # noqa: F401
    BUILTIN_TOOLS,
    AGENT_LIFECYCLE_TOOLS,
    CLAUDE_CODE_TOOL,
    read_tool,
    grep_tool,
    glob_tool,
    edit_tool,
    bash_tool,
    claude_code_tool,
    task_complete,
    escalate_question,
)

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
