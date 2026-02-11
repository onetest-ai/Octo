"""Load SKILL.md files → SkillConfig dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from octo.config import SKILLS_DIR


@dataclass
class SkillConfig:
    name: str
    description: str
    body: str  # full markdown content for injection into conversation
    model_invocation: bool = True  # False = user-only (/slash command), not offered to LLM
    # --- marketplace fields (optional, backward-compatible) ---
    version: str = "0.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: dict = field(default_factory=dict)
    requires: list[dict] = field(default_factory=list)
    permissions: dict = field(default_factory=dict)
    source: str = "local"  # "local" | "marketplace"


def _parse_skill_md(path: Path) -> SkillConfig | None:
    """Parse a single SKILL.md with YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None

    name = meta.get("name", path.parent.name)
    description = meta.get("description", "")
    body = parts[2].strip()

    # model-invocation: true (default) / false (user-only slash command)
    model_invocation = meta.get("model-invocation", True)
    if isinstance(model_invocation, str):
        model_invocation = model_invocation.lower() not in ("false", "no", "0")

    return SkillConfig(
        name=name,
        description=description,
        body=body,
        model_invocation=model_invocation,
        version=str(meta.get("version", "0.0.0")),
        author=meta.get("author", ""),
        tags=meta.get("tags", []),
        dependencies=meta.get("dependencies", {}),
        requires=meta.get("requires", []),
        permissions=meta.get("permissions", {}),
    )


def load_skills() -> list[SkillConfig]:
    """Scan .octo/skills/ directory and return SkillConfig list."""
    skills: list[SkillConfig] = []

    if not SKILLS_DIR.is_dir():
        return skills

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            cfg = _parse_skill_md(skill_file)
            if cfg:
                skills.append(cfg)

    return skills
