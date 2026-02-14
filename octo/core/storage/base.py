"""StorageBackend protocol — abstract interface for file storage.

Implementations:
- FilesystemStorage (always available)
- S3Storage (requires [s3] extra)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage for files, memory, skills, state.

    All paths are relative (e.g. "memory/2026-02-14.md").
    The backend resolves them against its root (filesystem path, S3 prefix, etc.).
    """

    async def read(self, path: str) -> str:
        """Read a file's text content. Raises FileNotFoundError if missing."""
        ...

    async def write(self, path: str, content: str) -> None:
        """Write text content to a file (creates parent dirs as needed)."""
        ...

    async def append(self, path: str, content: str) -> None:
        """Append text to a file (creates if missing)."""
        ...

    async def exists(self, path: str) -> bool:
        """Check if a file exists."""
        ...

    async def list_dir(self, prefix: str = "") -> list[str]:
        """List files under a prefix (non-recursive)."""
        ...

    async def delete(self, path: str) -> None:
        """Delete a file. No error if missing."""
        ...

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern."""
        ...
