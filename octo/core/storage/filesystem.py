"""Filesystem storage backend — the default for CLI usage.

Maps relative paths to a root directory on the local filesystem.
Always available (no extra dependencies).
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path


class FilesystemStorage:
    """Local filesystem storage backend.

    Args:
        root: Base directory. All paths are resolved relative to this.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path against root."""
        resolved = (self.root / path).resolve()
        # Safety: prevent path traversal outside root
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    async def read(self, path: str) -> str:
        """Read a file's text content."""
        p = self._resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> None:
        """Write text content to a file."""
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def append(self, path: str, content: str) -> None:
        """Append text to a file."""
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)

    async def exists(self, path: str) -> bool:
        """Check if a file exists."""
        return self._resolve(path).exists()

    async def list_dir(self, prefix: str = "") -> list[str]:
        """List files under a prefix (non-recursive)."""
        p = self._resolve(prefix)
        if not p.is_dir():
            return []
        return [
            str(Path(prefix) / item.name)
            for item in sorted(p.iterdir())
            if item.is_file()
        ]

    async def delete(self, path: str) -> None:
        """Delete a file. No error if missing."""
        p = self._resolve(path)
        if p.is_file():
            p.unlink()

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern relative to root."""
        results = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(self.root))
                if fnmatch.fnmatch(rel, pattern):
                    results.append(rel)
        return results

    def __repr__(self) -> str:
        return f"FilesystemStorage(root={self.root!r})"
