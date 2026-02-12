"""Configuration — loads .env, resolves workspace, model config."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _find_workspace() -> Path:
    """Walk up from cwd to find directory containing .octo/ or .env."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".octo").is_dir() or (p / ".env").is_file():
            return p
        p = p.parent
    # fallback: directory of this package's parent
    return Path(__file__).resolve().parent.parent


WORKSPACE = _find_workspace()
load_dotenv(WORKSPACE / ".env")

# --- Internal storage (.octo/) ---
OCTO_DIR = WORKSPACE / ".octo"
OCTO_DIR.mkdir(exist_ok=True)
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
MCP_CONFIG_PATH = WORKSPACE / ".mcp.json"

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


def _parse_heartbeat_interval(spec: str) -> int:
    """Parse heartbeat interval like '30m', '1h', '60s' into seconds."""
    import re as _re
    m = _re.match(r"^(\d+)\s*(s|m|h)$", spec.strip(), _re.I)
    if not m:
        return 1800  # default 30 minutes
    val = int(m.group(1))
    unit = m.group(2).lower()
    return val * {"s": 1, "m": 60, "h": 3600}[unit]


def _parse_time_str(spec: str):
    """Parse time string like '08:00' into datetime.time."""
    from datetime import time as _time
    parts = spec.strip().split(":")
    return _time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


HEARTBEAT_INTERVAL_SECONDS = _parse_heartbeat_interval(HEARTBEAT_INTERVAL)
HEARTBEAT_ACTIVE_START_TIME = _parse_time_str(HEARTBEAT_ACTIVE_START)
HEARTBEAT_ACTIVE_END_TIME = _parse_time_str(HEARTBEAT_ACTIVE_END)

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


def _project_to_dict(proj: ProjectConfig) -> dict:
    return {
        "name": proj.name,
        "path": proj.path,
        "config_dir": proj.config_dir,
        "env": proj.env,
        "agents": proj.agents,
    }


def _project_from_dict(data: dict) -> ProjectConfig:
    return ProjectConfig(
        name=data.get("name", ""),
        path=data.get("path", ""),
        config_dir=data.get("config_dir", ""),
        env=data.get("env", {}),
        agents=data.get("agents", []),
    )


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

        proj = ProjectConfig(
            name=name,
            path=str(project_dir),
            config_dir=str(config_dir),
            env={"CLAUDE_CONFIG_DIR": str(config_dir)},
            agents=agent_names,
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
