"""Engine constants — pure data structures with no side effects.

These are safe for embedding: no env vars, no filesystem ops, no imports
of heavy dependencies.  ``octo.config`` re-exports everything here for
backward compatibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model Profiles
# ---------------------------------------------------------------------------
# Named profiles mapping agent roles to model tiers.
# quality  = best output, highest cost
# balanced = good output, moderate cost (default)
# budget   = acceptable output, lowest cost

BUILTIN_PROFILES: dict[str, dict[str, str]] = {
    "quality": {
        "supervisor": "high",
        "worker_default": "default",
        "worker_high_keywords": "high",
    },
    "balanced": {
        "supervisor": "default",
        "worker_default": "low",
        "worker_high_keywords": "high",
    },
    "budget": {
        "supervisor": "low",
        "worker_default": "low",
        "worker_high_keywords": "default",
    },
}


# ---------------------------------------------------------------------------
# Default middleware thresholds
# ---------------------------------------------------------------------------
DEFAULT_TOOL_RESULT_LIMIT = 20_000
DEFAULT_SUMMARIZATION_TRIGGER_TOKENS = 40_000
DEFAULT_SUMMARIZATION_KEEP_TOKENS = 8_000
DEFAULT_SUPERVISOR_MSG_CHAR_LIMIT = 30_000
DEFAULT_CLAUDE_CODE_TIMEOUT = 2400  # seconds


# ---------------------------------------------------------------------------
# Project registry
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str
    path: str  # project root directory
    config_dir: str  # the .claude/ directory
    env: dict[str, str] = field(default_factory=dict)  # extra env vars for claude -p
    agents: list[str] = field(default_factory=list)  # agent names in this project
    # --- metadata (all optional) ---
    description: str = ""  # short project description
    repo_url: str = ""  # GitHub / GitLab / etc.
    issues_url: str = ""  # Jira, GitHub Issues, Linear, etc.
    tech_stack: list[str] = field(default_factory=list)  # ["python", "react", ...]
    default_branch: str = ""  # e.g. "main", "master", "develop"
    ci_url: str = ""  # Jenkins, GitHub Actions URL, etc.
    docs_url: str = ""  # Confluence, wiki, readthedocs, etc.
    tags: dict[str, str] = field(default_factory=dict)  # freeform key-value metadata


def project_to_dict(proj: ProjectConfig) -> dict:
    """Serialize a ProjectConfig to a JSON-safe dict."""
    d: dict = {
        "name": proj.name,
        "path": proj.path,
        "config_dir": proj.config_dir,
        "env": proj.env,
        "agents": proj.agents,
    }
    # Only persist non-empty metadata fields to keep JSON clean
    if proj.description:
        d["description"] = proj.description
    if proj.repo_url:
        d["repo_url"] = proj.repo_url
    if proj.issues_url:
        d["issues_url"] = proj.issues_url
    if proj.tech_stack:
        d["tech_stack"] = proj.tech_stack
    if proj.default_branch:
        d["default_branch"] = proj.default_branch
    if proj.ci_url:
        d["ci_url"] = proj.ci_url
    if proj.docs_url:
        d["docs_url"] = proj.docs_url
    if proj.tags:
        d["tags"] = proj.tags
    return d


def project_from_dict(data: dict) -> ProjectConfig:
    """Deserialize a dict into a ProjectConfig."""
    return ProjectConfig(
        name=data.get("name", ""),
        path=data.get("path", ""),
        config_dir=data.get("config_dir", ""),
        env=data.get("env", {}),
        agents=data.get("agents", []),
        description=data.get("description", ""),
        repo_url=data.get("repo_url", ""),
        issues_url=data.get("issues_url", ""),
        tech_stack=data.get("tech_stack", []),
        default_branch=data.get("default_branch", ""),
        ci_url=data.get("ci_url", ""),
        docs_url=data.get("docs_url", ""),
        tags=data.get("tags", {}),
    )


_PROJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def validate_project_name(name: str) -> str | None:
    """Return an error message if the project name is invalid, else None."""
    if not name:
        return "Project name cannot be empty."
    if not _PROJECT_NAME_RE.match(name):
        return (
            f"Invalid project name '{name}'. "
            "Use only letters, digits, hyphens, underscores, and dots."
        )
    if name in (".", ".."):
        return f"Invalid project name '{name}'."
    return None


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------

def parse_interval(spec: str) -> int:
    """Parse interval like '30m', '1h', '60s', or bare '120' (seconds)."""
    spec = spec.strip()
    m = re.match(r"^(\d+)\s*(s|m|h)$", spec, re.I)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        return val * {"s": 1, "m": 60, "h": 3600}[unit]
    if spec.isdigit():
        return int(spec)
    return 1800  # default 30 minutes


def parse_time_str(spec: str):
    """Parse time string like '08:00' into datetime.time."""
    from datetime import time as _time
    parts = spec.strip().split(":")
    return _time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
