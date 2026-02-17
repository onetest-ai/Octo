"""Agent lifecycle tools — task_complete, escalate_question.

Lightweight versions for regular worker agents (non-background).
These guide the LLM to produce structured output but don't manage
any external state.  Background agents use their own closure-based
versions in background.py that interact with TaskStore.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def task_complete(summary: str) -> str:
    """Signal that your assigned task is fully complete.

    Call this when you have finished all work.  Your summary will be
    relayed to the supervisor and then to the user.

    Args:
        summary: concise description of what was accomplished and any
            key results or findings.
    """
    return summary


@tool
def escalate_question(question: str) -> str:
    """Ask the user a clarifying question when you are blocked.

    Call this when you need information or a decision from the user
    before you can proceed.  Your question will be relayed through
    the supervisor.

    Args:
        question: the specific question you need answered.
    """
    return (
        f"[ESCALATION] The following question needs to be relayed to the user:\n\n"
        f"{question}\n\n"
        f"Please include this question in your response to the user."
    )


AGENT_LIFECYCLE_TOOLS = [task_complete, escalate_question]
