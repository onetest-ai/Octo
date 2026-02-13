"""People profile map — contact directory with roles, topics, and tone prefs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tone presets controlling persona formatting style
TONE_PRESETS = {
    "casual_technical": "Full slang, short, direct. Engineer-to-engineer.",
    "casual": "Friendly, simpler language. Non-technical but familiar.",
    "semi_formal": "Professional but personable. Managers and execs.",
}

_DEFAULT_TONE = "casual_technical"

# Max topics tracked per person
_MAX_TOPICS = 15


class PeopleProfiles:
    """JSON-backed contact map for VP context.

    Each profile stores: name, title, department, discussion topics,
    preferred tone, interaction count, and last interaction timestamp.
    """

    def __init__(self, profiles_path: Path) -> None:
        self._path = profiles_path
        self._profiles: dict[str, dict[str, Any]] = {}
        self._load()

    def get_profile(self, email: str) -> dict[str, Any] | None:
        return self._profiles.get(email.lower())

    def update_profile(self, email: str, **fields: Any) -> None:
        """Merge fields into an existing or new profile."""
        key = email.lower()
        profile = self._profiles.setdefault(key, {"email": key})
        profile.update(fields)
        self._save()

    def ensure_profile(self, email: str, name: str = "") -> dict[str, Any]:
        """Get or create a minimal profile for this email."""
        key = email.lower()
        if key not in self._profiles:
            self._profiles[key] = {
                "email": key,
                "name": name,
                "title": "",
                "department": "",
                "topics": [],
                "tone": _DEFAULT_TONE,
                "interaction_count": 0,
                "last_interaction": "",
                "notes": "",
            }
            self._save()
        elif name and not self._profiles[key].get("name"):
            self._profiles[key]["name"] = name
            self._save()
        return self._profiles[key]

    def record_interaction(
        self, email: str, topic: str = "", category: str = ""
    ) -> None:
        """Update profile after a VP interaction.

        Increments interaction_count, updates last_interaction,
        and appends topic to the topics list (with dedup + cap).
        """
        profile = self.ensure_profile(email)
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1
        profile["last_interaction"] = datetime.now(timezone.utc).isoformat()

        if topic or category:
            topics: list[str] = profile.get("topics", [])
            new_topic = topic or category
            # Move to front if already present, else prepend
            if new_topic in topics:
                topics.remove(new_topic)
            topics.insert(0, new_topic)
            profile["topics"] = topics[:_MAX_TOPICS]

        self._save()

    def set_tone(self, email: str, tone: str) -> bool:
        """Set tone preset for a person. Returns False if invalid preset."""
        if tone not in TONE_PRESETS:
            return False
        profile = self.ensure_profile(email)
        profile["tone"] = tone
        self._save()
        return True

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return all profiles sorted by interaction count (most active first)."""
        return sorted(
            self._profiles.values(),
            key=lambda p: p.get("interaction_count", 0),
            reverse=True,
        )

    async def enrich_from_teams(self, email: str, chat_members: list[dict]) -> None:
        """Enrich profile with Teams member info (name, title, department).

        Args:
            email: The user's email to enrich.
            chat_members: List of member dicts from MCP msteams list-chat-members.
        """
        for member in chat_members:
            member_email = (
                member.get("email", "") or member.get("mail", "")
            ).lower()
            if member_email == email.lower():
                profile = self.ensure_profile(email)
                if member.get("displayName") and not profile.get("name"):
                    profile["name"] = member["displayName"]
                if member.get("jobTitle") and not profile.get("title"):
                    profile["title"] = member["jobTitle"]
                if member.get("department") and not profile.get("department"):
                    profile["department"] = member["department"]
                self._save()
                break

    # --- Internal ---

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._profiles = data
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load VP profiles: %s", e)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._profiles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
