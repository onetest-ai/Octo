"""Load SKILL.md files → SkillConfig dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from octo.config import SKILLS_DIR


@dataclass
class SkillConfig:
    name: str
    description: str
    body: str  # full markdown content for injection into conversation
    model_invocation: bool = True  # False = user-only (/slash command), not offered to LLM


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

    return SkillConfig(name=name, description=description, body=body, model_invocation=model_invocation)


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
