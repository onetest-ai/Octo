"""Access control — YAML-backed allow/block lists with confidence modifiers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {
    "version": "1.0",
    "enabled": False,
    "allow_ai": {
        "users": [],
        "channels": [],
        "default_action": "always_user",
    },
    "always_user": {
        "users": [],
        "channels": [],
        "notify_real_artem": True,
        "auto_create_reminder": True,
    },
    "ignored_chats": [],
    "priority_users": [],
    "audit_log": {"enabled": True},
}


@dataclass
class AccessDecision:
    """Result of an access check."""

    allowed: bool
    action: str  # "allow_ai" | "always_user" | "not_authorized"
    reason: str
    confidence_modifier: int = 0
    notify_priority: str = "normal"


class AccessControl:
    """YAML-backed access control for Virtual Persona.

    Manages two lists:
    - allow_ai: users/channels where VP may respond
    - always_user: users/channels routed to real Artem only
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._config: dict[str, Any] = {}
        self._load()

    # --- Public API ---

    def is_enabled(self) -> bool:
        return bool(self._config.get("enabled", False))

    def set_enabled(self, enabled: bool) -> None:
        self._config["enabled"] = enabled
        self._save()

    def check_access(
        self, user_email: str, channel_id: str = ""
    ) -> AccessDecision:
        """Check whether VP may respond to this user/channel.

        Priority: always_user > allow_ai > default_action.
        """
        # 1. always_user takes precedence
        block = self._find_always_user(user_email, channel_id)
        if block is not None:
            return AccessDecision(
                allowed=False,
                action="always_user",
                reason=block.get("reason", "VIP / blocked user"),
                confidence_modifier=0,
                notify_priority=block.get("notify_priority", "urgent"),
            )

        # 2. Check allow_ai whitelist
        allow = self._find_allow_ai(user_email, channel_id)
        if allow is not None:
            return AccessDecision(
                allowed=True,
                action="allow_ai",
                reason="Whitelisted",
                confidence_modifier=int(allow.get("confidence_modifier", 0)),
                notify_priority="normal",
            )

        # 3. Default action
        default = (
            self._config.get("allow_ai", {}).get("default_action", "always_user")
        )
        return AccessDecision(
            allowed=False,
            action="not_authorized",
            reason=f"Not in allow_ai list (default: {default})",
            confidence_modifier=0,
            notify_priority="normal",
        )

    def add_user(
        self,
        email: str,
        list_name: str = "allow_ai",
        *,
        modifier: int = 0,
        priority: str = "normal",
        notes: str = "",
        name: str = "",
    ) -> None:
        """Add a user to allow_ai or always_user list."""
        section = self._config.setdefault(list_name, {})
        users: list[dict] = section.setdefault("users", [])

        # Remove if already present (to update)
        users[:] = [u for u in users if u.get("email") != email]

        entry: dict[str, Any] = {"email": email}
        if name:
            entry["name"] = name
        if list_name == "allow_ai":
            entry["confidence_modifier"] = modifier
            if notes:
                entry["notes"] = notes
        else:
            entry["reason"] = notes or "Blocked"
            entry["notify_priority"] = priority
        entry["enabled"] = True

        users.append(entry)
        self._save()

    def remove_user(self, email: str) -> bool:
        """Remove a user from all lists. Returns True if found."""
        removed = False
        for section_name in ("allow_ai", "always_user"):
            section = self._config.get(section_name, {})
            users: list[dict] = section.get("users", [])
            before = len(users)
            users[:] = [u for u in users if u.get("email") != email]
            if len(users) < before:
                removed = True
        if removed:
            self._save()
        return removed

    def update_confidence_modifier(self, email: str, modifier: int) -> bool:
        """Update confidence modifier for a user in allow_ai. Returns True if found."""
        users = self._config.get("allow_ai", {}).get("users", [])
        for u in users:
            if u.get("email") == email:
                u["confidence_modifier"] = modifier
                self._save()
                return True
        return False

    def get_allow_list(self) -> list[dict]:
        return list(self._config.get("allow_ai", {}).get("users", []))

    def get_block_list(self) -> list[dict]:
        return list(self._config.get("always_user", {}).get("users", []))

    def reload(self) -> None:
        self._load()

    # --- Ignored chats ---

    def get_ignored_chats(self) -> list[dict]:
        """Return the list of ignored chat entries."""
        return list(self._config.get("ignored_chats", []))

    def is_ignored(self, chat_id: str) -> bool:
        """Check if a chat is in the ignore list."""
        for entry in self._config.get("ignored_chats", []):
            eid = entry.get("id", "") if isinstance(entry, dict) else entry
            if eid == chat_id:
                return True
        return False

    def ignore_chat(self, chat_id: str, label: str = "") -> None:
        """Add a chat to the ignore list."""
        ignored: list[dict] = self._config.setdefault("ignored_chats", [])
        # Don't add duplicates
        if self.is_ignored(chat_id):
            return
        entry: dict[str, str] = {"id": chat_id}
        if label:
            entry["label"] = label
        ignored.append(entry)
        self._save()

    def unignore_chat(self, chat_id: str) -> bool:
        """Remove a chat from the ignore list. Returns True if it was ignored."""
        ignored: list = self._config.get("ignored_chats", [])
        before = len(ignored)
        ignored[:] = [
            e for e in ignored
            if (e.get("id", "") if isinstance(e, dict) else e) != chat_id
        ]
        if len(ignored) < before:
            self._save()
            return True
        return False

    # --- Priority users (never ignore) ---

    def get_priority_users(self) -> list[dict]:
        """Return the list of priority user entries."""
        return list(self._config.get("priority_users", []))

    def is_priority_user(self, email: str) -> bool:
        """Check if a user is in the priority (never-ignore) list."""
        if not email:
            return False
        email_lower = email.lower()
        for entry in self._config.get("priority_users", []):
            e = entry.get("email", "") if isinstance(entry, dict) else entry
            if e.lower() == email_lower:
                return True
        return False

    def add_priority_user(self, email: str, name: str = "", notes: str = "") -> None:
        """Add a user to the priority (never-ignore) list."""
        priority: list[dict] = self._config.setdefault("priority_users", [])
        # Don't add duplicates
        if self.is_priority_user(email):
            return
        entry: dict[str, str] = {"email": email.lower()}
        if name:
            entry["name"] = name
        if notes:
            entry["notes"] = notes
        priority.append(entry)
        self._save()

    def remove_priority_user(self, email: str) -> bool:
        """Remove a user from the priority list. Returns True if found."""
        priority: list = self._config.get("priority_users", [])
        before = len(priority)
        priority[:] = [
            e for e in priority
            if (e.get("email", "") if isinstance(e, dict) else e).lower() != email.lower()
        ]
        if len(priority) < before:
            self._save()
            return True
        return False

    # --- Delegation lock ---

    def get_delegated_threads(self) -> dict[str, dict]:
        """Load delegated thread locks from delegated.json."""
        path = self._path.parent / "delegated.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def is_delegated(self, chat_id: str) -> bool:
        return chat_id in self.get_delegated_threads()

    def lock_thread(self, chat_id: str, reason: str, user_email: str, query_preview: str) -> None:
        """Mark a thread as delegated to real Artem."""
        delegated = self.get_delegated_threads()
        delegated[chat_id] = {
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "user_email": user_email,
            "query_preview": query_preview[:200],
        }
        path = self._path.parent / "delegated.json"
        path.write_text(json.dumps(delegated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def release_thread(self, chat_id: str) -> bool:
        """Release a delegated thread. Returns True if it was locked."""
        delegated = self.get_delegated_threads()
        if chat_id in delegated:
            del delegated[chat_id]
            path = self._path.parent / "delegated.json"
            path.write_text(json.dumps(delegated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return True
        return False

    def release_all_threads(self) -> int:
        """Release all delegated threads. Returns count released."""
        delegated = self.get_delegated_threads()
        count = len(delegated)
        if count:
            path = self._path.parent / "delegated.json"
            path.write_text("{}\n", encoding="utf-8")
        return count

    # --- Internal ---

    def _load(self) -> None:
        if self._path.is_file():
            try:
                import yaml

                self._config = yaml.safe_load(
                    self._path.read_text(encoding="utf-8")
                ) or {}
            except Exception as e:
                logger.warning("Failed to load VP access config: %s", e)
                self._config = dict(_DEFAULT_CONFIG)
        else:
            self._config = dict(_DEFAULT_CONFIG)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml

            self._path.write_text(
                yaml.safe_dump(self._config, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save VP access config: %s", e)

    def _find_always_user(
        self, user_email: str, channel_id: str
    ) -> dict | None:
        section = self._config.get("always_user", {})
        # Check users
        for u in section.get("users", []):
            if u.get("email", "").lower() == user_email.lower():
                return u
        # Check channels
        if channel_id:
            for ch in section.get("channels", []):
                if self._match_channel(ch.get("id", ""), channel_id):
                    return ch
        return None

    def _find_allow_ai(
        self, user_email: str, channel_id: str
    ) -> dict | None:
        section = self._config.get("allow_ai", {})
        # Check users
        for u in section.get("users", []):
            if (
                u.get("email", "").lower() == user_email.lower()
                and u.get("enabled", True)
            ):
                return u
        # Check channels
        if channel_id:
            for ch in section.get("channels", []):
                if ch.get("enabled", True) and self._match_channel(
                    ch.get("id", ""), channel_id
                ):
                    return ch
        return None

    @staticmethod
    def _match_channel(pattern: str, channel_id: str) -> bool:
        """Match channel ID with wildcard support (e.g. 'teams:exec_*')."""
        if "*" in pattern or "?" in pattern:
            return fnmatch(channel_id, pattern)
        return pattern == channel_id
