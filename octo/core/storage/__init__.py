"""Storage backends for Octo engine.

Public API::

    from octo.core.storage import StorageBackend, FilesystemStorage
    from octo.core.storage.s3 import S3Storage  # requires [s3] extra
"""
from __future__ import annotations

from octo.core.storage.base import StorageBackend
from octo.core.storage.filesystem import FilesystemStorage

__all__ = ["StorageBackend", "FilesystemStorage"]
