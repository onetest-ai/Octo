"""Octo core — embeddable agent engine.

Public API::

    from octo.core import OctoEngine, OctoConfig
    from octo.core.storage import StorageBackend, FilesystemStorage

    config = OctoConfig(
        llm_provider="anthropic",
        llm_credentials={"api_key": "sk-..."},
        storage=FilesystemStorage(root="/path/to/.octo"),
    )
    engine = OctoEngine(config)
    response = await engine.invoke("Hello!", thread_id="conv-123")
"""
from __future__ import annotations

from octo.core.config import OctoConfig, OctoConfigError
from octo.core.engine import OctoEngine, OctoEngineError, OctoResponse
from octo.core.storage import FilesystemStorage, StorageBackend

__all__ = [
    "FilesystemStorage",
    "OctoConfig",
    "OctoConfigError",
    "OctoEngine",
    "OctoEngineError",
    "OctoResponse",
    "StorageBackend",
]
