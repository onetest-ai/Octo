"""Load SKILL.md files → SkillConfig dataclasses."""
from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from octo.config import SKILLS_DIR

log = logging.getLogger(__name__)

# pip package name → Python import name (only for names that differ)
_PIP_TO_IMPORT: dict[str, str] = {
    "pillow": "PIL",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "beautifulsoup4": "bs4",
    "pdf2image": "pdf2image",
}


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
    # --- progressive disclosure ---
    skill_dir: Path | None = None  # absolute path to skill directory
    references: list[str] = field(default_factory=list)  # relative paths of files in references/
    scripts: list[str] = field(default_factory=list)  # relative paths of files in scripts/


def _catalog_subdir(skill_dir: Path, subdir_name: str) -> list[str]:
    """List files in a skill subdirectory (references/, scripts/), relative to skill_dir."""
    subdir = skill_dir / subdir_name
    if not subdir.is_dir():
        return []
    return sorted(
        str(f.relative_to(skill_dir))
        for f in subdir.rglob("*")
        if f.is_file()
    )


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

    skill_dir = path.parent.resolve()

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
        skill_dir=skill_dir,
        references=_catalog_subdir(skill_dir, "references"),
        scripts=_catalog_subdir(skill_dir, "scripts"),
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


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------

def _pip_name(spec: str) -> str:
    """Extract bare package name from a pip specifier like 'pdfplumber>=0.11'."""
    return re.split(r"[><=!~\[]", spec, maxsplit=1)[0].strip().lower()


def _import_name(pip_pkg: str) -> str:
    """Map a pip package name to its Python import name."""
    key = pip_pkg.lower()
    if key in _PIP_TO_IMPORT:
        return _PIP_TO_IMPORT[key]
    # Default: replace dashes with underscores
    return key.replace("-", "_")


def check_missing_deps(skill: SkillConfig) -> list[str]:
    """Return list of pip specifiers whose packages are not importable."""
    python_deps: list[str] = skill.dependencies.get("python", [])
    if not python_deps:
        return []

    missing: list[str] = []
    for spec in python_deps:
        pkg = _pip_name(spec)
        mod = _import_name(pkg)
        if importlib.util.find_spec(mod) is None:
            missing.append(spec)
    return missing


def verify_skills_deps(skills: list[SkillConfig]) -> dict[str, list[str]]:
    """Check all skills for missing Python deps. Returns {skill_name: [missing_specs]}.

    Also logs warnings for any skills with missing dependencies.
    """
    problems: dict[str, list[str]] = {}
    for sk in skills:
        missing = check_missing_deps(sk)
        if missing:
            problems[sk.name] = missing
            log.warning(
                "Skill '%s' has missing Python deps: %s  "
                "(run: pip install %s)",
                sk.name,
                ", ".join(missing),
                " ".join(_pip_name(s) for s in missing),
            )
    return problems
