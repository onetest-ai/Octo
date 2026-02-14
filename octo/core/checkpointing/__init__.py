"""Checkpointing backends for OctoEngine.

Public API::

    from octo.core.checkpointing import make_checkpointer

    checkpointer = await make_checkpointer(config)
"""
from __future__ import annotations

from typing import Any


async def make_checkpointer(config: Any) -> Any:
    """Create a checkpointer based on OctoConfig.

    Supports:
      - ``sqlite`` (default): Local SQLite via aiosqlite.
      - ``postgres``: PostgreSQL via langgraph checkpoint-postgres.

    Args:
        config: OctoConfig instance with ``checkpoint_backend`` and
            ``checkpoint_config`` fields.

    Returns:
        A LangGraph checkpoint saver instance.

    Raises:
        ImportError: If the required backend extra is not installed.
        ValueError: If required config keys are missing.
    """
    backend = config.checkpoint_backend

    if backend == "postgres":
        return await _make_postgres_checkpointer(config)

    # Default: SQLite
    return await _make_sqlite_checkpointer(config)


async def _make_sqlite_checkpointer(config: Any) -> Any:
    """Create a SQLite checkpointer."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = config.checkpoint_config.get("path", "")
    if not db_path:
        db_path = ":memory:"

    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    return checkpointer


async def _make_postgres_checkpointer(config: Any) -> Any:
    """Create a PostgreSQL checkpointer."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        raise ImportError(
            "PostgreSQL checkpointing requires the postgres extra. "
            "Install with: pip install octo-agent[postgres]"
        )

    dsn = config.checkpoint_config.get("dsn", "")
    if not dsn:
        raise ValueError(
            "PostgreSQL checkpointer requires 'dsn' in checkpoint_config"
        )

    checkpointer = AsyncPostgresSaver.from_conn_string(dsn)
    await checkpointer.setup()
    return checkpointer
