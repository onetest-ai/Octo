"""Configuration — loads .env, resolves workspace, model config."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _default_workspace() -> Path:
    """Return the platform-appropriate default workspace root.

    The workspace root is the directory that *contains* ``.octo/``,
    ``.env``, and ``.mcp.json``.

    - ``OCTO_HOME`` env var overrides — points to the workspace root.
    - POSIX default: ``~`` (state lives at ``~/.octo``).
    - Windows default: ``%LOCALAPPDATA%/octo`` (state at
      ``%LOCALAPPDATA%/octo/.octo``).
    """
    env = os.getenv("OCTO_HOME")
    if env:
        return Path(env).expanduser().resolve()

    if os.name == "nt":
        local = os.getenv("LOCALAPPDATA")
        if local:
            return Path(local) / "octo"
        return Path.home() / "octo"

    return Path.home()


def _find_workspace() -> Path:
    """Walk up from cwd to find directory containing .octo/ or .env.

    Resolution order:
    1. Walk up from cwd looking for an existing ``.octo/`` dir or ``.env`` file
       (project-local workspace).
    2. Fall back to the platform default (``~`` on POSIX,
       ``%LOCALAPPDATA%/octo`` on Windows).  Overridable with ``OCTO_HOME``.
    """
    p = Path.cwd()
    while p != p.parent:
        if (p / ".octo").is_dir() or (p / ".env").is_file():
            return p
        p = p.parent
    # No project workspace found — use the user-global default.
    ws = _default_workspace()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


WORKSPACE = _find_workspace()

# --- Internal storage (.octo/) ---
OCTO_DIR = WORKSPACE / ".octo"
OCTO_DIR.mkdir(exist_ok=True)

# .env: prefer OCTO_DIR/.env (global install keeps everything under ~/.octo/),
# then fall back to WORKSPACE/.env (project-local).  load_dotenv won't
# override vars already set by the first call.
load_dotenv(OCTO_DIR / ".env")
load_dotenv(WORKSPACE / ".env")
DB_PATH = OCTO_DIR / "octo.db"
PERSONA_DIR = OCTO_DIR / "persona"
MEMORY_DIR = OCTO_DIR / "memory"
SKILLS_DIR = OCTO_DIR / "skills"
# Tell skills.sh CLI to use .octo/ as CODEX_HOME so `-a codex -g` installs
# to .octo/skills/ directly.
os.environ.setdefault("CODEX_HOME", str(OCTO_DIR))
# Additional dirs scanned for skills (skills.sh / Agent Skills ecosystem)
EXTERNAL_SKILLS_DIRS: list[Path] = [
    WORKSPACE / ".agents" / "skills",   # skills.sh universal path
    WORKSPACE / ".claude" / "skills",   # Claude Code skills path
]
SKILLS_REGISTRY_URL = os.getenv(
    "SKILLS_REGISTRY_URL",
    "https://raw.githubusercontent.com/onetest-ai/skills/main/registry.json",
)
SKILLS_CACHE_DIR = OCTO_DIR / "cache"
SKILLS_CACHE_DIR.mkdir(exist_ok=True)
SKILLS_CACHE_TTL = int(os.getenv("SKILLS_CACHE_TTL", "3600"))  # seconds

# --- MCP Registry (official) ---
MCP_REGISTRY_BASE_URL = os.getenv(
    "MCP_REGISTRY_URL",
    "https://registry.modelcontextprotocol.io",
)
MCP_REGISTRY_CACHE_DIR = OCTO_DIR / "cache" / "mcp-registry"
MCP_REGISTRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MCP_REGISTRY_CACHE_TTL = int(os.getenv("MCP_REGISTRY_CACHE_TTL", "1800"))  # 30 min

AGENTS_DIR = OCTO_DIR / "agents"
AGENTS_DIR.mkdir(exist_ok=True)
OAUTH_DIR = OCTO_DIR / "oauth"  # created on demand by FileTokenStorage

# --- Auth ---
# Prefer ANTHROPIC_API_KEY; fall back to CLAUDE_CODE_OAUTH_TOKEN
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")

# --- AWS Bedrock ---
AWS_REGION = os.getenv("AWS_REGION", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Azure OpenAI ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --- GitHub Models ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Base URLs — override if using a custom proxy or Azure AI Foundry resource
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com"
)
GITHUB_MODELS_ANTHROPIC_BASE_URL = os.getenv(
    "GITHUB_MODELS_ANTHROPIC_BASE_URL", "https://models.inference.ai.azure.com"
)

# --- Provider override (optional) ---
# Auto-detected from model name if not set. Values: anthropic, bedrock, openai, azure, github
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")

# --- Model tiers ---
# HIGH  = complex reasoning, architecture, multi-step planning
# DEFAULT = supervisor routing, general chat, tool use
# LOW   = summarization, simple workers, cost-sensitive tasks
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
HIGH_TIER_MODEL = os.getenv("HIGH_TIER_MODEL", DEFAULT_MODEL)
LOW_TIER_MODEL = os.getenv("LOW_TIER_MODEL", "claude-haiku-4-5-20251001")

# --- Project state ---
STATE_PATH = OCTO_DIR / "STATE.md"
PLANS_DIR = OCTO_DIR / "plans"
PLANS_DIR.mkdir(exist_ok=True)

# --- Research workspace ---
# Shared workspace for agent research and temporary files, organized by date.
RESEARCH_WORKSPACE = OCTO_DIR / "workspace"
RESEARCH_WORKSPACE.mkdir(exist_ok=True)

# --- Middleware ---
# Tool result truncation — max chars before a single tool result is cut
TOOL_RESULT_LIMIT = int(os.getenv("TOOL_RESULT_LIMIT", "40000"))

# Summarization — fraction of context window that triggers compaction
SUMMARIZATION_TRIGGER_FRACTION = float(os.getenv("SUMMARIZATION_TRIGGER_FRACTION", "0.7"))
# Summarization — token count trigger (whichever fires first with fraction)
SUMMARIZATION_TRIGGER_TOKENS = int(os.getenv("SUMMARIZATION_TRIGGER_TOKENS", "100000"))
# Summarization — how many tokens of recent history to keep after compaction
SUMMARIZATION_KEEP_TOKENS = int(os.getenv("SUMMARIZATION_KEEP_TOKENS", "20000"))
# Supervisor pre-model hook — max chars per message before truncation (safety net)
SUPERVISOR_MSG_CHAR_LIMIT = int(os.getenv("SUPERVISOR_MSG_CHAR_LIMIT", "30000"))

# Claude Code tool — default subprocess timeout in seconds.
# Claude Code tasks can run for up to an hour; 2400s (40min) is a safe default.
CLAUDE_CODE_TIMEOUT = int(os.getenv("CLAUDE_CODE_TIMEOUT", "2400"))

# --- Model Profiles ---
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

_active_profile: str = os.getenv("MODEL_PROFILE", "balanced")


def get_active_profile() -> str:
    return _active_profile


def set_active_profile(name: str) -> bool:
    global _active_profile
    if name in BUILTIN_PROFILES:
        _active_profile = name
        return True
    return False


def get_profile_tiers() -> dict[str, str]:
    return BUILTIN_PROFILES.get(_active_profile, BUILTIN_PROFILES["balanced"])


# --- MCP ---
# Prefer OCTO_DIR/.mcp.json (keeps global config inside ~/.octo/),
# fall back to WORKSPACE/.mcp.json (project-local).
_mcp_in_octo = OCTO_DIR / ".mcp.json"
_mcp_in_ws = WORKSPACE / ".mcp.json"
MCP_CONFIG_PATH = _mcp_in_octo if _mcp_in_octo.is_file() else _mcp_in_ws

# --- Agent directories ---
_extra = os.getenv("AGENT_DIRS", "")
AGENT_DIRS: list[Path] = [WORKSPACE / ".claude" / "agents"]
if _extra:
    AGENT_DIRS.extend(Path(p.strip()) for p in _extra.split(":") if p.strip())

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_OWNER_ID = os.getenv("TELEGRAM_OWNER_ID", "")

# --- Heartbeat ---
HEARTBEAT_INTERVAL = os.getenv("HEARTBEAT_INTERVAL", "30m")
HEARTBEAT_ACTIVE_START = os.getenv("HEARTBEAT_ACTIVE_HOURS_START", "08:00")
HEARTBEAT_ACTIVE_END = os.getenv("HEARTBEAT_ACTIVE_HOURS_END", "22:00")
HEARTBEAT_PATH = PERSONA_DIR / "HEARTBEAT.md"

# --- Cron ---
CRON_PATH = OCTO_DIR / "cron.json"

# --- Virtual Persona ---
VP_DIR = OCTO_DIR / "virtual-persona"
# VP_DIR.mkdir() is intentionally NOT called here — created lazily on first /vp enable
VP_ENABLED = os.getenv("VP_ENABLED", "false").lower() in ("true", "1", "yes")
VP_POLL_INTERVAL = os.getenv("VP_POLL_INTERVAL", "2m")
VP_ACTIVE_START = os.getenv("VP_ACTIVE_HOURS_START", "08:00")
VP_ACTIVE_END = os.getenv("VP_ACTIVE_HOURS_END", "22:00")
VP_SELF_EMAILS = [
    e.strip().lower()
    for e in os.getenv("VP_SELF_EMAILS", "").split(",")
    if e.strip()
]


def _parse_interval(spec: str) -> int:
    """Parse heartbeat interval like '30m', '1h', '60s', or bare '120' (seconds)."""
    import re as _re
    spec = spec.strip()
    m = _re.match(r"^(\d+)\s*(s|m|h)$", spec, _re.I)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        return val * {"s": 1, "m": 60, "h": 3600}[unit]
    # Bare number → treat as seconds
    if spec.isdigit():
        return int(spec)
    return 1800  # default 30 minutes


def _parse_time_str(spec: str):
    """Parse time string like '08:00' into datetime.time."""
    from datetime import time as _time
    parts = spec.strip().split(":")
    return _time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


HEARTBEAT_INTERVAL_SECONDS = _parse_interval(HEARTBEAT_INTERVAL)
HEARTBEAT_ACTIVE_START_TIME = _parse_time_str(HEARTBEAT_ACTIVE_START)
HEARTBEAT_ACTIVE_END_TIME = _parse_time_str(HEARTBEAT_ACTIVE_END)

VP_POLL_INTERVAL_SECONDS = _parse_interval(VP_POLL_INTERVAL)
VP_ACTIVE_START_TIME = _parse_time_str(VP_ACTIVE_START)
VP_ACTIVE_END_TIME = _parse_time_str(VP_ACTIVE_END)

# --- Voice ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")


# --- Project registry ---
# Each project is a JSON file under .octo/projects/<name>.json.
# Auto-seeded from AGENT_DIRS on first run; edit the files to customise.

PROJECTS_DIR = OCTO_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)


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


def _project_to_dict(proj: ProjectConfig) -> dict:
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


def _project_from_dict(data: dict) -> ProjectConfig:
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


def _autodiscover_project_metadata(project_dir: Path) -> dict:
    """Auto-detect metadata from a project directory.

    Discovers: repo_url, default_branch, tech_stack, description.
    Returns a dict of non-empty fields to merge into ProjectConfig.
    """
    import subprocess

    meta: dict = {}

    # --- git remote URL ---
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            meta["repo_url"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # --- default branch ---
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=project_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # refs/remotes/origin/main → main
            meta["default_branch"] = result.stdout.strip().split("/")[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # --- tech stack detection ---
    stack: list[str] = []
    markers = {
        "pyproject.toml": "python", "setup.py": "python", "requirements.txt": "python",
        "package.json": "javascript", "tsconfig.json": "typescript",
        "Cargo.toml": "rust", "go.mod": "go", "pom.xml": "java",
        "build.gradle": "java", "Gemfile": "ruby", "mix.exs": "elixir",
        "Dockerfile": "docker", "docker-compose.yml": "docker",
        "Makefile": "make", ".terraform": "terraform",
    }
    for filename, tech in markers.items():
        if (project_dir / filename).exists() and tech not in stack:
            stack.append(tech)
    if stack:
        meta["tech_stack"] = stack

    # --- description from pyproject.toml or package.json ---
    pyproject = project_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text())
            desc = data.get("project", {}).get("description", "")
            if desc:
                meta["description"] = desc
        except Exception:
            pass

    if "description" not in meta:
        pkg_json = project_dir / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text())
                desc = data.get("description", "")
                if desc:
                    meta["description"] = desc
            except Exception:
                pass

    return meta


def _seed_projects_from_agent_dirs() -> None:
    """Create project JSON files from AGENT_DIRS for any project not yet on disk."""
    for agent_dir in AGENT_DIRS:
        if not agent_dir.is_dir():
            continue
        config_dir = agent_dir.parent          # .claude/
        project_dir = config_dir.parent        # project root
        name = project_dir.name
        project_file = PROJECTS_DIR / f"{name}.json"

        if project_file.exists():
            continue  # don't overwrite user edits

        agent_names = [md.stem for md in sorted(agent_dir.glob("*.md"))]
        auto = _autodiscover_project_metadata(project_dir)

        proj = ProjectConfig(
            name=name,
            path=str(project_dir),
            config_dir=str(config_dir),
            env={"CLAUDE_CONFIG_DIR": str(config_dir)},
            agents=agent_names,
            description=auto.get("description", ""),
            repo_url=auto.get("repo_url", ""),
            tech_stack=auto.get("tech_stack", []),
            default_branch=auto.get("default_branch", ""),
        )
        project_file.write_text(
            json.dumps(_project_to_dict(proj), indent=2) + "\n"
        )


def _load_projects() -> dict[str, ProjectConfig]:
    """Load all project configs from .octo/projects/*.json."""
    _seed_projects_from_agent_dirs()

    projects: dict[str, ProjectConfig] = {}
    for f in sorted(PROJECTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            proj = _project_from_dict(data)
            if not proj.name:
                proj.name = f.stem
            projects[proj.name] = proj
        except (json.JSONDecodeError, TypeError):
            continue
    return projects


PROJECTS: dict[str, ProjectConfig] = _load_projects()


def get_project_for_agent(agent_name: str) -> ProjectConfig | None:
    """Find which project owns a given agent name."""
    for proj in PROJECTS.values():
        if agent_name in proj.agents:
            return proj
    return None


def reload_projects() -> dict[str, ProjectConfig]:
    """Re-read all project JSON files from disk and refresh PROJECTS.

    This is a *light* reload — it refreshes the in-memory dict so that
    future ``claude_code`` calls and system-prompt builds pick up changes,
    but does **not** rebuild the LangGraph graph.

    Mutates the existing PROJECTS dict in-place so that all modules holding
    a reference (e.g. ``from octo.config import PROJECTS``) see the update.
    """
    fresh = _load_projects()
    PROJECTS.clear()
    PROJECTS.update(fresh)
    return PROJECTS


import re as _re

_PROJECT_NAME_RE = _re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _validate_project_name(name: str) -> str | None:
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


def save_project(proj: ProjectConfig) -> Path:
    """Persist a ProjectConfig to disk and update the in-memory registry."""
    err = _validate_project_name(proj.name)
    if err:
        raise ValueError(err)
    project_file = PROJECTS_DIR / f"{proj.name}.json"
    project_file.write_text(
        json.dumps(_project_to_dict(proj), indent=2) + "\n"
    )
    PROJECTS[proj.name] = proj
    return project_file


def remove_project(name: str) -> bool:
    """Remove a project from disk and the in-memory registry.

    Returns True if the project existed and was removed.
    """
    project_file = PROJECTS_DIR / f"{name}.json"
    if project_file.exists():
        project_file.unlink()
    removed = PROJECTS.pop(name, None) is not None
    return removed
