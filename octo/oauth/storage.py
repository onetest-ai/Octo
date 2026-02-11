"""File-based token storage implementing the MCP SDK TokenStorage protocol."""
from __future__ import annotations

from pathlib import Path

from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class FileTokenStorage(TokenStorage):
    """Persistent token storage for a single MCP server.

    Stores tokens and client info as JSON files under
    ``<oauth_dir>/<server_name>/``.
    """

    def __init__(self, server_name: str, oauth_dir: Path) -> None:
        self._dir = oauth_dir / server_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._token_path = self._dir / "tokens.json"
        self._client_path = self._dir / "client_info.json"

    # -- tokens --

    async def get_tokens(self) -> OAuthToken | None:
        if not self._token_path.is_file():
            return None
        try:
            return OAuthToken.model_validate_json(self._token_path.read_text("utf-8"))
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._token_path.write_text(tokens.model_dump_json(indent=2), "utf-8")

    # -- client info --

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        if not self._client_path.is_file():
            return None
        try:
            return OAuthClientInformationFull.model_validate_json(
                self._client_path.read_text("utf-8")
            )
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_path.write_text(client_info.model_dump_json(indent=2), "utf-8")

    # -- helpers --

    def has_tokens(self) -> bool:
        """Check if tokens exist on disk (sync, for status display)."""
        return self._token_path.is_file()

    def clear(self) -> None:
        """Delete stored tokens and client info (sync, for logout)."""
        for p in (self._token_path, self._client_path):
            if p.is_file():
                p.unlink()
