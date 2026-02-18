"""Peer registry — JSON file-backed management of swarm peers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """A known swarm peer."""

    name: str
    url: str  # e.g. "http://192.168.1.42:9100/mcp/"
    capabilities: list[str] = field(default_factory=list)
    status: str = "unknown"  # unknown | online | offline
    last_seen: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Only include non-empty optional fields
        if not d["last_seen"]:
            del d["last_seen"]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PeerInfo:
        return cls(
            name=data["name"],
            url=data["url"],
            capabilities=data.get("capabilities", []),
            status=data.get("status", "unknown"),
            last_seen=data.get("last_seen", ""),
        )


class PeerRegistry:
    """JSON file-backed peer registry at .octo/swarm/peers.json."""

    def __init__(self, swarm_dir: Path) -> None:
        self._dir = swarm_dir
        self._path = swarm_dir / "peers.json"

    def load(self) -> list[PeerInfo]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text("utf-8"))
            return [PeerInfo.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load peers.json: %s", exc)
            return []

    def save(self, peers: list[PeerInfo]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([p.to_dict() for p in peers], indent=2) + "\n",
            encoding="utf-8",
        )

    def add_peer(
        self, name: str, url: str, capabilities: list[str] | None = None,
    ) -> PeerInfo:
        peers = self.load()
        # Remove existing peer with same name
        peers = [p for p in peers if p.name != name]
        peer = PeerInfo(name=name, url=url, capabilities=capabilities or [])
        peers.append(peer)
        self.save(peers)
        logger.info("Added peer %s at %s", name, url)
        return peer

    def remove_peer(self, name: str) -> bool:
        peers = self.load()
        new_peers = [p for p in peers if p.name != name]
        if len(new_peers) == len(peers):
            return False
        self.save(new_peers)
        logger.info("Removed peer %s", name)
        return True

    def update_status(self, name: str, status: str, last_seen: str = "") -> None:
        peers = self.load()
        for p in peers:
            if p.name == name:
                p.status = status
                if last_seen:
                    p.last_seen = last_seen
                break
        self.save(peers)

    def get_peer(self, name: str) -> PeerInfo | None:
        for p in self.load():
            if p.name == name:
                return p
        return None
