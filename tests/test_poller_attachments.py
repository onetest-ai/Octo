"""Tests for VP poller attachment extraction."""
from __future__ import annotations

import asyncio
import json
from datetime import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from octo.virtual_persona.poller import VPPoller


@pytest.fixture
def poller(tmp_path: Path) -> VPPoller:
    """Create a minimal VPPoller for testing."""
    vp_graph = MagicMock()
    octo_app = MagicMock()
    lock = asyncio.Lock()
    return VPPoller(
        vp_graph=vp_graph,
        octo_app=octo_app,
        graph_lock=lock,
        interval=60,
        active_start=time(8, 0),
        active_end=time(22, 0),
    )


class TestExtractAttachments:
    """VPPoller._extract_attachments — downloads attachments to workspace."""

    def test_no_attachments_returns_empty(self, poller: VPPoller):
        msg = {"body": {"content": "hello"}}
        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat1", "msg1")
        )
        assert result == []

    def test_empty_attachments_returns_empty(self, poller: VPPoller):
        msg = {"attachments": []}
        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat1", "msg1")
        )
        assert result == []

    def test_reference_attachment_saves_url(self, poller: VPPoller, tmp_path: Path, monkeypatch):
        """Reference attachments (links/images) save a .url file."""
        monkeypatch.setattr(
            "octo.virtual_persona.poller.RESEARCH_WORKSPACE",
            tmp_path,
            raising=False,
        )
        # Patch the import inside the method
        import octo.config
        original = octo.config.RESEARCH_WORKSPACE
        octo.config.RESEARCH_WORKSPACE = tmp_path

        msg = {
            "attachments": [{
                "contentType": "reference",
                "name": "screenshot.png",
                "contentUrl": "https://graph.microsoft.com/v1.0/drives/abc/items/def/content",
            }]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat123", "msg456")
        )

        assert len(result) == 1
        assert result[0]["name"] == "screenshot.png"
        assert result[0]["content_type"] == "reference_url"
        assert Path(result[0]["path"]).exists()
        assert Path(result[0]["path"]).read_text() == (
            "https://graph.microsoft.com/v1.0/drives/abc/items/def/content"
        )

        octo.config.RESEARCH_WORKSPACE = original

    def test_inline_content_attachment_saves_file(self, poller: VPPoller, tmp_path: Path):
        """Text content attachments are saved as files."""
        import octo.config
        original = octo.config.RESEARCH_WORKSPACE
        octo.config.RESEARCH_WORKSPACE = tmp_path

        msg = {
            "attachments": [{
                "contentType": "text/plain",
                "name": "notes.txt",
                "content": "These are my meeting notes...",
            }]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat789", "msg012")
        )

        assert len(result) == 1
        assert result[0]["name"] == "notes.txt"
        assert result[0]["content_type"] == "text/plain"
        saved = Path(result[0]["path"]).read_text()
        assert "meeting notes" in saved

        octo.config.RESEARCH_WORKSPACE = original

    def test_adaptive_card_attachment_saves_json(self, poller: VPPoller, tmp_path: Path):
        """Adaptive card attachments are saved as JSON."""
        import octo.config
        original = octo.config.RESEARCH_WORKSPACE
        octo.config.RESEARCH_WORKSPACE = tmp_path

        card_content = {
            "type": "AdaptiveCard",
            "body": [{"type": "TextBlock", "text": "Hello from card"}],
        }
        msg = {
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "name": "card",
                "content": card_content,
            }]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chatABC", "msgDEF")
        )

        assert len(result) == 1
        assert result[0]["name"] == "card"
        assert result[0]["path"].endswith(".json")
        data = json.loads(Path(result[0]["path"]).read_text())
        assert data["type"] == "AdaptiveCard"

        octo.config.RESEARCH_WORKSPACE = original

    def test_url_only_attachment_returns_url_ref(self, poller: VPPoller):
        """Attachments with just a URL but no content or reference type."""
        msg = {
            "attachments": [{
                "contentType": "application/pdf",
                "name": "report.pdf",
                "contentUrl": "https://example.com/report.pdf",
            }]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat1", "msg1")
        )

        assert len(result) == 1
        assert result[0]["name"] == "report.pdf"
        assert result[0]["path"] == "https://example.com/report.pdf"
        assert result[0]["content_type"] == "application/pdf"

    def test_multiple_attachments(self, poller: VPPoller, tmp_path: Path):
        """Multiple attachments of different types."""
        import octo.config
        original = octo.config.RESEARCH_WORKSPACE
        octo.config.RESEARCH_WORKSPACE = tmp_path

        msg = {
            "attachments": [
                {
                    "contentType": "reference",
                    "name": "image.png",
                    "contentUrl": "https://example.com/image.png",
                },
                {
                    "contentType": "text/plain",
                    "name": "log.txt",
                    "content": "error at line 42",
                },
            ]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat_multi", "msg_multi")
        )

        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "image.png" in names
        assert "log.txt" in names

        octo.config.RESEARCH_WORKSPACE = original

    def test_sanitizes_filenames(self, poller: VPPoller, tmp_path: Path):
        """Unsafe characters in filenames are replaced."""
        import octo.config
        original = octo.config.RESEARCH_WORKSPACE
        octo.config.RESEARCH_WORKSPACE = tmp_path

        msg = {
            "attachments": [{
                "contentType": "text/plain",
                "name": "../../etc/passwd",
                "content": "not really",
            }]
        }

        result = asyncio.get_event_loop().run_until_complete(
            poller._extract_attachments(msg, "chat1", "msg1")
        )

        assert len(result) == 1
        path = Path(result[0]["path"])
        # Path should be safely within workspace, not ../etc/passwd
        assert "etc" not in str(path.parent)
        assert path.exists()

        octo.config.RESEARCH_WORKSPACE = original
