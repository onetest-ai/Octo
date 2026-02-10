"""Load AGENT.md files → AgentConfig dataclasses.

Loads from two sources:
  1. Project agent dirs (AGENT_DIRS) — Claude Code AGENT.md files
  2. Octo-native agents (.octo/agents/*/AGENT.md) — includes deep_research type
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from octo.config import AGENT_DIRS, AGENTS_DIR


@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = ""  # empty = inherit default
    type: str = ""  # "deep_research" for deepagents, empty for create_agent
    color: str = "cyan"
    source_project: str = ""


def _parse_frontmatter_fallback(raw: str) -> dict[str, str]:
    """Fallback parser for YAML frontmatter with complex multi-line values.

    Claude Code AGENT.md descriptions can contain literal \\n, XML tags, etc.
    that break standard YAML parsing. This extracts top-level key: value pairs.
    """
    meta: dict[str, str] = {}
    current_key = ""
    current_val = ""

    for line in raw.strip().splitlines():
        # Check if this line starts a new key (word followed by colon at start)
        if ":" in line and not line[0].isspace():
            colon_idx = line.index(":")
            candidate_key = line[:colon_idx].strip()
            # Valid YAML keys are simple words
            if candidate_key.isidentifier():
                if current_key:
                    meta[current_key] = current_val.strip()
                current_key = candidate_key
                current_val = line[colon_idx + 1:].strip()
                continue
        # Continuation of previous value
        if current_key:
            current_val += " " + line.strip()

    if current_key:
        meta[current_key] = current_val.strip()

    return meta


def _parse_agent_md(path: Path) -> AgentConfig | None:
    """Parse a single AGENT.md file with YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None

    # Split frontmatter from body
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        # Fallback for complex frontmatter (long descriptions with special chars)
        meta = _parse_frontmatter_fallback(parts[1])

    if not meta:
        return None

    name = meta.get("name", path.stem)
    description = meta.get("description", "")
    # Clean up description — take only first sentence/paragraph for routing
    if len(description) > 200:
        # Cut at first period after 50 chars, or at 200
        cut = description.find(".", 50)
        if cut != -1 and cut < 300:
            description = description[:cut + 1]
        else:
            description = description[:200] + "..."
    body = parts[2].strip()

    # tools can be a list or comma-separated string
    raw_tools = meta.get("tools", [])
    if isinstance(raw_tools, str):
        tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
    else:
        tools = list(raw_tools)

    model = meta.get("model", "")
    if model == "inherit":
        model = ""

    color = meta.get("color", "cyan")

    # Derive source project from parent path
    source = path.parent.parent.parent.name  # .claude/agents/x.md → project dir

    agent_type = meta.get("type", "")

    return AgentConfig(
        name=name,
        description=description,
        system_prompt=body,
        tools=tools,
        model=model,
        type=agent_type,
        color=color,
        source_project=source,
    )


def load_agents() -> list[AgentConfig]:
    """Scan all agent directories and return AgentConfig list."""
    agents: list[AgentConfig] = []
    seen_names: set[str] = set()

    for agent_dir in AGENT_DIRS:
        if not agent_dir.is_dir():
            continue
        for md_file in sorted(agent_dir.glob("*.md")):
            cfg = _parse_agent_md(md_file)
            if cfg and cfg.name not in seen_names:
                agents.append(cfg)
                seen_names.add(cfg.name)

    return agents


def load_octo_agents() -> list[AgentConfig]:
    """Scan .octo/agents/*/AGENT.md and return AgentConfig list.

    These are Octo-native agents (e.g. deep_research type) configured
    directly in the workspace, not loaded from external projects.
    """
    agents: list[AgentConfig] = []

    if not AGENTS_DIR.is_dir():
        return agents

    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        agent_file = agent_dir / "AGENT.md"
        if not agent_file.is_file():
            continue
        cfg = _parse_agent_md(agent_file)
        if cfg:
            cfg.source_project = "octo"
            agents.append(cfg)

    return agents
