"""Lightweight MS Teams MCP server.

Provides chat and team tools via Microsoft Graph API with user-consent-only
scopes (no admin approval needed).  Runs as a STDIO MCP server using FastMCP.

Ships a pre-registered public client ID so users can authenticate immediately
without registering their own Azure AD application.  Override with env vars
if you prefer your own app registration.

Environment variables:
    TEAMS_CLIENT_ID   — Azure AD application (client) ID
                        (default: pre-registered public client)
    TEAMS_TENANT_ID   — Azure AD directory (tenant) ID    (default: "common")
    TEAMS_TOKEN_CACHE — Path to token cache file
                        (default: ~/.octo/teams_token_cache.json)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import msal
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Pre-registered public client application (same as Softeria ms-365-mcp-server).
# Users can override with TEAMS_CLIENT_ID env var for their own app registration.
DEFAULT_CLIENT_ID = "084a3e9f-a9f4-43f7-89f9-d229cf97853e"

SCOPES = [
    "User.Read",
    "Chat.Read",
    "ChatMessage.Send",
    "Team.ReadBasic.All",
    "Channel.ReadBasic.All",
]

# ---------------------------------------------------------------------------
# Contacts / chat cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("TEAMS_CACHE_DIR", Path.home() / ".octo"))
_CONTACTS_PATH = _CACHE_DIR / "teams_contacts.json"


def _load_contacts() -> dict[str, Any]:
    """Load the contacts/chat cache from disk."""
    if _CONTACTS_PATH.is_file():
        try:
            return json.loads(_CONTACTS_PATH.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"chats": {}, "users": {}}


def _save_contacts(data: dict[str, Any]) -> None:
    """Persist the contacts/chat cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CONTACTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")


def _update_contacts_from_chats(chats: list[dict[str, Any]]) -> None:
    """Update the contacts cache with chat info (members, topics)."""
    contacts = _load_contacts()
    for chat in chats:
        cid = chat.get("id", "")
        if not cid:
            continue
        entry = contacts["chats"].get(cid, {})
        entry["id"] = cid
        entry["chatType"] = chat.get("chatType")
        entry["topic"] = chat.get("topic")
        if "members" in chat:
            entry["members"] = chat["members"]
            # Index users
            for m in chat["members"]:
                if m.get("displayName"):
                    contacts["users"][m["displayName"].lower()] = {
                        "displayName": m["displayName"],
                        "chatIds": list(set(
                            contacts["users"]
                            .get(m["displayName"].lower(), {})
                            .get("chatIds", []) + [cid]
                        )),
                    }
        if chat.get("lastMessage", {}).get("from"):
            name = chat["lastMessage"]["from"]
            entry.setdefault("members", [])
            if not any(m.get("displayName") == name for m in entry.get("members", [])):
                entry["members"].append({"displayName": name})
            contacts["users"].setdefault(name.lower(), {
                "displayName": name, "chatIds": [],
            })
            if cid not in contacts["users"][name.lower()]["chatIds"]:
                contacts["users"][name.lower()]["chatIds"].append(cid)
        contacts["chats"][cid] = entry
    _save_contacts(contacts)


def _update_contacts_from_members(chat_id: str, members: list[dict[str, Any]]) -> None:
    """Update the contacts cache with chat member info."""
    contacts = _load_contacts()
    entry = contacts["chats"].get(chat_id, {"id": chat_id})
    entry["members"] = members
    contacts["chats"][chat_id] = entry
    for m in members:
        name = m.get("displayName", "")
        if name:
            user = contacts["users"].get(name.lower(), {
                "displayName": name, "chatIds": [],
            })
            user["displayName"] = name
            if chat_id not in user.get("chatIds", []):
                user.setdefault("chatIds", []).append(chat_id)
            contacts["users"][name.lower()] = user
    _save_contacts(contacts)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


class TeamsAuth:
    """MSAL device-code auth with persistent token cache."""

    def __init__(self) -> None:
        self._client_id = os.environ.get("TEAMS_CLIENT_ID", "") or DEFAULT_CLIENT_ID
        self._tenant_id = os.environ.get("TEAMS_TENANT_ID", "common")
        self._cache_path = Path(
            os.environ.get(
                "TEAMS_TOKEN_CACHE",
                Path.home() / ".octo" / "teams_token_cache.json",
            )
        )
        self._cache = msal.SerializableTokenCache()
        self._load_cache()
        self._app = msal.PublicClientApplication(
            self._client_id,
            authority=f"https://login.microsoftonline.com/{self._tenant_id}",
            token_cache=self._cache,
        )

    # -- cache persistence --

    def _load_cache(self) -> None:
        if self._cache_path.is_file():
            self._cache.deserialize(self._cache_path.read_text("utf-8"))

    def _save_cache(self) -> None:
        if self._cache.has_state_changed:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(self._cache.serialize(), "utf-8")
            try:
                self._cache_path.chmod(0o600)
            except OSError:
                pass

    # -- token acquisition --

    def get_token(self) -> str | None:
        """Try silent token acquisition from cache. Returns token or None."""
        accounts = self._app.get_accounts()
        if not accounts:
            return None
        result = self._app.acquire_token_silent(SCOPES, account=accounts[0])
        self._save_cache()
        if result and "access_token" in result:
            return result["access_token"]
        return None

    async def device_code_login(self) -> dict[str, Any]:
        """Run device code flow. Blocks until user completes browser auth."""
        flow = self._app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            return {"error": flow.get("error_description", "Failed to start device flow")}

        import sys
        print(flow["message"], file=sys.stderr, flush=True)

        result = await asyncio.to_thread(
            self._app.acquire_token_by_device_flow, flow
        )
        self._save_cache()

        if "access_token" in result:
            account = result.get("id_token_claims", {}).get("preferred_username", "")
            return {"status": "authenticated", "account": account}
        return {"error": result.get("error_description", "Authentication failed")}

    def logout(self) -> None:
        """Clear all cached tokens."""
        if self._cache_path.is_file():
            self._cache_path.unlink()
        self._cache = msal.SerializableTokenCache()
        self._app = msal.PublicClientApplication(
            self._client_id,
            authority=f"https://login.microsoftonline.com/{self._tenant_id}",
            token_cache=self._cache,
        )


# ---------------------------------------------------------------------------
# Graph API helpers
# ---------------------------------------------------------------------------

_NOT_AUTHENTICATED = "Not authenticated. Call the 'login' tool first."


async def _graph_get(
    auth: TeamsAuth, endpoint: str, params: dict[str, str] | None = None
) -> dict[str, Any]:
    """GET from Microsoft Graph API."""
    token = auth.get_token()
    if not token:
        return {"error": _NOT_AUTHENTICATED}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{GRAPH_BASE}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
        )
    if resp.status_code == 401:
        return {"error": "Token expired or invalid. Call 'login' to re-authenticate."}
    if resp.status_code == 403:
        return {
            "error": (
                f"Permission denied for {endpoint}. "
                "This scope may require admin consent in your organization."
            )
        }
    resp.raise_for_status()
    return resp.json()


async def _graph_post(
    auth: TeamsAuth, endpoint: str, body: dict[str, Any]
) -> dict[str, Any]:
    """POST to Microsoft Graph API."""
    token = auth.get_token()
    if not token:
        return {"error": _NOT_AUTHENTICATED}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{GRAPH_BASE}{endpoint}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if resp.status_code == 401:
        return {"error": "Token expired or invalid. Call 'login' to re-authenticate."}
    if resp.status_code == 403:
        return {
            "error": (
                f"Permission denied for {endpoint}. "
                "This scope may require admin consent in your organization."
            )
        }
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# FastMCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "teams",
    instructions=(
        "MS Teams MCP server. Call 'login' first to authenticate via device code flow. "
        "Then use chat and team tools to read and send messages.\n\n"
        "Use 'find-chat' to search for a chat by person name before reading messages. "
        "Use 'list-chat-members' to see who is in a chat. "
        "The server maintains a contacts cache so repeated lookups are fast."
    ),
)
auth = TeamsAuth()


@mcp.tool()
async def login() -> dict[str, Any]:
    """Authenticate to Microsoft Teams via device code flow.

    Opens a URL where you enter a code in your browser.
    Blocks until authentication completes (up to 15 minutes).
    """
    return await auth.device_code_login()


@mcp.tool()
def logout() -> dict[str, str]:
    """Log out of Microsoft Teams and clear cached tokens."""
    auth.logout()
    return {"status": "logged out", "message": "Token cache cleared. Call 'login' to re-authenticate."}


@mcp.tool(name="list-chats")
async def list_chats(limit: int = 30) -> dict[str, Any]:
    """List your recent Teams chats with last message preview and members.

    Args:
        limit: Number of chats to return (1-50, default 30).
    """
    limit = max(1, min(limit, 50))
    data = await _graph_get(auth, "/me/chats", {
        "$top": str(limit),
        "$expand": "lastMessagePreview,members",
        "$orderby": "lastMessagePreview/createdDateTime desc",
    })
    if "error" in data:
        return data
    chats = []
    for c in data.get("value", []):
        members = [
            {"displayName": m.get("displayName", ""), "email": m.get("email", "")}
            for m in c.get("members", [])
            if m.get("@odata.type", "") == "#microsoft.graph.aadUserConversationMember"
        ]
        chat = {
            "id": c.get("id"),
            "topic": c.get("topic"),
            "chatType": c.get("chatType"),
            "createdDateTime": c.get("createdDateTime"),
            "members": members,
        }
        preview = c.get("lastMessagePreview")
        if preview:
            chat["lastMessage"] = {
                "from": (preview.get("from") or {}).get("user", {}).get("displayName"),
                "content": (preview.get("body") or {}).get("content", "")[:200],
                "createdDateTime": preview.get("createdDateTime"),
            }
        chats.append(chat)
    _update_contacts_from_chats(chats)
    return {"chats": chats, "count": len(chats)}


@mcp.tool(name="list-chat-messages")
async def list_chat_messages(chatId: str, limit: int = 30) -> dict[str, Any]:
    """Read messages from a specific chat.

    Args:
        chatId: The chat ID (from list-chats or find-chat results).
        limit: Number of messages to return (1-50, default 30).
    """
    limit = max(1, min(limit, 50))
    data = await _graph_get(auth, f"/chats/{chatId}/messages", {"$top": str(limit)})
    if "error" in data:
        return data
    messages = []
    for m in data.get("value", []):
        msg = {
            "id": m.get("id"),
            "from": (m.get("from") or {}).get("user", {}).get("displayName"),
            "body": (m.get("body") or {}).get("content", ""),
            "contentType": (m.get("body") or {}).get("contentType"),
            "createdDateTime": m.get("createdDateTime"),
            "messageType": m.get("messageType"),
        }
        messages.append(msg)
    return {"messages": messages, "count": len(messages)}


@mcp.tool(name="list-chat-members")
async def list_chat_members(chatId: str) -> dict[str, Any]:
    """List members of a specific chat.

    Args:
        chatId: The chat ID (from list-chats or find-chat results).
    """
    data = await _graph_get(auth, f"/chats/{chatId}/members")
    if "error" in data:
        return data
    members = []
    for m in data.get("value", []):
        members.append({
            "displayName": m.get("displayName", ""),
            "email": m.get("email", ""),
            "roles": m.get("roles", []),
        })
    _update_contacts_from_members(chatId, members)
    return {"members": members, "count": len(members)}


@mcp.tool(name="find-chat")
async def find_chat(query: str) -> dict[str, Any]:
    """Find a chat by person name, topic, or keyword. Searches the local
    contacts cache first, then falls back to fetching recent chats from Teams.

    Args:
        query: Person name, chat topic, or keyword to search for.
    """
    query_lower = query.lower()
    contacts = _load_contacts()
    matches: list[dict[str, Any]] = []

    # Search cached users by name
    for name_key, user_info in contacts.get("users", {}).items():
        if query_lower in name_key:
            for cid in user_info.get("chatIds", []):
                chat_entry = contacts["chats"].get(cid, {})
                if chat_entry and not any(m["id"] == cid for m in matches):
                    matches.append({
                        "id": cid,
                        "topic": chat_entry.get("topic"),
                        "chatType": chat_entry.get("chatType"),
                        "members": chat_entry.get("members", []),
                        "matchedUser": user_info.get("displayName"),
                    })

    # Search cached chats by topic
    for cid, chat_entry in contacts.get("chats", {}).items():
        topic = (chat_entry.get("topic") or "").lower()
        if query_lower in topic and not any(m["id"] == cid for m in matches):
            matches.append({
                "id": cid,
                "topic": chat_entry.get("topic"),
                "chatType": chat_entry.get("chatType"),
                "members": chat_entry.get("members", []),
            })

    if matches:
        return {"matches": matches[:10], "count": len(matches), "source": "cache"}

    # Cache miss — fetch recent chats and retry
    fresh = await list_chats(limit=50)
    if "error" in fresh:
        return fresh

    # Search again after cache is updated
    contacts = _load_contacts()
    for name_key, user_info in contacts.get("users", {}).items():
        if query_lower in name_key:
            for cid in user_info.get("chatIds", []):
                chat_entry = contacts["chats"].get(cid, {})
                if chat_entry and not any(m["id"] == cid for m in matches):
                    matches.append({
                        "id": cid,
                        "topic": chat_entry.get("topic"),
                        "chatType": chat_entry.get("chatType"),
                        "members": chat_entry.get("members", []),
                        "matchedUser": user_info.get("displayName"),
                    })

    for cid, chat_entry in contacts.get("chats", {}).items():
        topic = (chat_entry.get("topic") or "").lower()
        if query_lower in topic and not any(m["id"] == cid for m in matches):
            matches.append({
                "id": cid,
                "topic": chat_entry.get("topic"),
                "chatType": chat_entry.get("chatType"),
                "members": chat_entry.get("members", []),
            })

    if matches:
        return {"matches": matches[:10], "count": len(matches), "source": "fresh"}
    return {"matches": [], "count": 0, "message": f"No chats found matching '{query}'."}


@mcp.tool(name="send-chat-message")
async def send_chat_message(
    chatId: str, content: str, contentType: str = "text"
) -> dict[str, Any]:
    """Send a message to a Teams chat.

    Args:
        chatId: The chat ID (from list-chats or find-chat results).
        content: Message content (text or HTML).
        contentType: 'text' or 'html' (default: 'text').
    """
    body = {"body": {"contentType": contentType, "content": content}}
    data = await _graph_post(auth, f"/chats/{chatId}/messages", body)
    if "error" in data:
        return data
    return {
        "status": "sent",
        "messageId": data.get("id"),
        "createdDateTime": data.get("createdDateTime"),
    }


@mcp.tool(name="reply-to-chat-message")
async def reply_to_chat_message(
    chatId: str, messageId: str, content: str, contentType: str = "text"
) -> dict[str, Any]:
    """Reply to a specific message in a Teams chat.

    Args:
        chatId: The chat ID.
        messageId: The message ID to reply to.
        content: Reply content (text or HTML).
        contentType: 'text' or 'html' (default: 'text').
    """
    body = {"body": {"contentType": contentType, "content": content}}
    data = await _graph_post(
        auth, f"/chats/{chatId}/messages/{messageId}/replies", body
    )
    if "error" in data:
        return data
    return {
        "status": "sent",
        "messageId": data.get("id"),
        "createdDateTime": data.get("createdDateTime"),
    }


@mcp.tool(name="list-joined-teams")
async def list_joined_teams() -> dict[str, Any]:
    """List Teams you belong to."""
    data = await _graph_get(auth, "/me/joinedTeams")
    if "error" in data:
        return data
    teams = [
        {
            "id": t.get("id"),
            "displayName": t.get("displayName"),
            "description": t.get("description"),
        }
        for t in data.get("value", [])
    ]
    return {"teams": teams, "count": len(teams)}


@mcp.tool(name="list-team-channels")
async def list_team_channels(teamId: str) -> dict[str, Any]:
    """List channels in a specific Team.

    Args:
        teamId: The team ID (from list-joined-teams results).
    """
    data = await _graph_get(auth, f"/teams/{teamId}/channels")
    if "error" in data:
        return data
    channels = [
        {
            "id": ch.get("id"),
            "displayName": ch.get("displayName"),
            "description": ch.get("description"),
            "membershipType": ch.get("membershipType"),
        }
        for ch in data.get("value", [])
    ]
    return {"channels": channels, "count": len(channels)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Teams MCP server on STDIO."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
