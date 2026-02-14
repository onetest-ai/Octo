"""Backward-compat shim — moved to octo.core.loaders.skill_loader."""
from octo.core.loaders.skill_loader import *  # noqa: F401,F403
from octo.core.loaders.skill_loader import (
    SkillConfig,
    check_missing_deps,
    load_skills,
    verify_skills_deps,
)
