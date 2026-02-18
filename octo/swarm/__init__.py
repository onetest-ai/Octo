"""Swarm — multi-instance Octo collaboration via MCP-over-HTTP."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octo.swarm.runner import SwarmRunner

_swarm_runner: SwarmRunner | None = None


def set_swarm_runner(runner: SwarmRunner) -> None:
    global _swarm_runner
    _swarm_runner = runner


def get_swarm_runner() -> SwarmRunner | None:
    return _swarm_runner
