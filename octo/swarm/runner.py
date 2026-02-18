"""SwarmRunner — lifecycle manager for the swarm MCP server + peer monitoring."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Health-check interval for peers (seconds)
_HEALTH_INTERVAL = 60


class SwarmRunner:
    """Manages the swarm: starts the MCP server, monitors peer health.

    Follows the HeartbeatRunner lifecycle pattern: ``start()`` / ``stop()``.
    """

    def __init__(
        self,
        instance_name: str,
        port: int,
        capabilities: list[str],
        swarm_dir: Path,
        graph_app: Any,
        graph_lock: asyncio.Lock,
        main_loop: asyncio.AbstractEventLoop,
        worker_pool: Any | None = None,
        task_store: Any | None = None,
    ) -> None:
        self._name = instance_name
        self._port = port
        self._capabilities = capabilities
        self._swarm_dir = swarm_dir
        self._app = graph_app
        self._lock = graph_lock
        self._main_loop = main_loop
        self._worker_pool = worker_pool
        self._task_store = task_store

        self._server_thread: Any | None = None
        self._uvicorn_server: Any | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    # -- public properties -----------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def port(self) -> int:
        return self._port

    @property
    def running(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Start the MCP server in a daemon thread and the health monitor."""
        from octo.swarm.server import build_swarm_mcp_server, start_mcp_server_in_thread

        self._swarm_dir.mkdir(parents=True, exist_ok=True)

        mcp_server = build_swarm_mcp_server(
            instance_name=self._name,
            capabilities=self._capabilities,
            graph_app=self._app,
            graph_lock=self._lock,
            main_loop=self._main_loop,
            worker_pool=self._worker_pool,
            task_store=self._task_store,
        )

        try:
            self._server_thread, self._uvicorn_server = start_mcp_server_in_thread(
                mcp_server, self._port,
            )
        except OSError as exc:
            logger.error("Failed to start swarm server on port %d: %s", self._port, exc)
            return

        # Start the peer health monitor
        self._stop_event.clear()
        self._monitor_task = asyncio.create_task(
            self._health_loop(), name="swarm-health-monitor",
        )
        logger.info("Swarm started: %s on port %d", self._name, self._port)

    async def stop(self) -> None:
        """Stop the MCP server and health monitor."""
        self._stop_event.set()
        if self._monitor_task:
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._monitor_task.cancel()
            self._monitor_task = None
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        logger.info("Swarm stopped")

    def update_graph(self, new_app: Any) -> None:
        """Update the graph reference after a rebuild."""
        self._app = new_app

    # -- health monitoring -----------------------------------------------

    async def _health_loop(self) -> None:
        """Periodically check peer health."""
        while not self._stop_event.is_set():
            try:
                await self._check_peers()
            except Exception:
                logger.debug("Peer health check failed", exc_info=True)
            # Wait for interval or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=_HEALTH_INTERVAL,
                )
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # interval elapsed, loop again

    async def _check_peers(self) -> None:
        """Ping each peer's /health endpoint."""
        import httpx
        from datetime import datetime, timezone
        from octo.swarm.registry import PeerRegistry

        registry = PeerRegistry(self._swarm_dir)
        peers = registry.load()
        if not peers:
            return

        async with httpx.AsyncClient(timeout=5.0) as client:
            for peer in peers:
                try:
                    # Derive health URL from MCP URL
                    health_url = peer.url.rsplit("/", 1)[0] + "/health"
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        peer.status = "online"
                        peer.last_seen = datetime.now(timezone.utc).isoformat()
                    else:
                        peer.status = "offline"
                except Exception:
                    peer.status = "offline"

        registry.save(peers)
