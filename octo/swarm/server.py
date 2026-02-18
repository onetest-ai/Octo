"""MCP-over-HTTP server — exposes this Octo instance to swarm peers."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def build_swarm_mcp_server(
    instance_name: str,
    capabilities: list[str],
    graph_app: Any,
    graph_lock: asyncio.Lock,
    main_loop: asyncio.AbstractEventLoop,
    worker_pool: Any | None = None,
    task_store: Any | None = None,
) -> Any:
    """Build a FastMCP server exposing this Octo instance to swarm peers.

    The server runs in a separate thread, so all graph/pool access must
    be scheduled on *main_loop* via ``run_coroutine_threadsafe``.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name=f"octo-{instance_name}",
        instructions=(
            f"Octo swarm peer '{instance_name}'. "
            "Use ask() for quick questions, dispatch_task() for long-running work."
        ),
        host="0.0.0.0",
        port=9100,  # overridden by caller via uvicorn config
        stateless_http=True,
    )

    # -- helpers to run coroutines on the main event loop ---------------

    def _run_on_main(coro: Any, timeout: float = 120.0) -> Any:
        """Schedule *coro* on the main event loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, main_loop)
        return future.result(timeout=timeout)

    async def _run_on_main_async(coro: Any, timeout: float = 120.0) -> Any:
        """Await *coro* on the main event loop from the server's loop."""
        future = asyncio.run_coroutine_threadsafe(coro, main_loop)
        return await asyncio.wrap_future(future)

    # -- tools ----------------------------------------------------------

    @mcp.tool()
    async def ask(question: str, context: str = "") -> str:  # noqa: D401
        """Ask this Octo instance a question and get an answer synchronously."""
        from langchain_core.messages import HumanMessage

        prompt = question
        if context:
            prompt = f"Context from a peer Octo instance:\n{context}\n\nQuestion: {question}"

        async def _invoke() -> str:
            thread_id = f"swarm-{instance_name}-{uuid.uuid4().hex[:8]}"
            async with graph_lock:
                result = await graph_app.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            # Extract the last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    return str(msg.content)
            return "(no response)"

        try:
            return await _run_on_main_async(_invoke(), timeout=120.0)
        except TimeoutError:
            return "Error: request timed out (120s)"
        except Exception as exc:
            logger.exception("ask() failed")
            return f"Error: {exc}"

    @mcp.tool()
    async def dispatch_task(
        task: str, context: str = "", priority: str = "normal",
    ) -> str:  # noqa: D401
        """Queue a long-running task for background execution. Returns a task_id."""
        if worker_pool is None:
            return json.dumps({"error": "Background worker pool not available"})

        from octo.background import BackgroundTask

        full_prompt = task
        if context:
            full_prompt = f"Context from peer:\n{context}\n\nTask: {task}"

        bg_task = BackgroundTask(
            id=f"swarm-{uuid.uuid4().hex[:8]}",
            type="agent",
            prompt=full_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            task_id = await _run_on_main_async(
                worker_pool.dispatch(bg_task), timeout=10.0,
            )
            return json.dumps({"task_id": task_id, "status": "dispatched"})
        except Exception as exc:
            logger.exception("dispatch_task() failed")
            return json.dumps({"error": str(exc)})

    @mcp.tool()
    async def check_task(task_id: str) -> str:  # noqa: D401
        """Check the status and result of a previously dispatched task."""
        if task_store is None:
            return json.dumps({"error": "Task store not available"})

        bg_task = task_store.load(task_id)
        if bg_task is None:
            return json.dumps({"error": f"Task {task_id} not found"})

        return json.dumps({
            "task_id": bg_task.id,
            "status": bg_task.status,
            "result": bg_task.result or None,
            "error": bg_task.error or None,
        })

    @mcp.tool()
    async def get_info() -> str:  # noqa: D401
        """Get information about this Octo instance."""
        return json.dumps({
            "name": instance_name,
            "capabilities": capabilities,
            "status": "online",
        })

    return mcp


def start_mcp_server_in_thread(
    mcp_server: Any,
    port: int,
) -> tuple[threading.Thread, Any]:
    """Run the FastMCP streamable-HTTP server in a daemon thread.

    Returns ``(thread, uvicorn_server)`` so the caller can signal shutdown
    via ``uvicorn_server.should_exit = True``.
    """
    import uvicorn
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # Build ASGI app
    app = mcp_server.streamable_http_app()

    # Add /health endpoint
    async def _health(request: Any) -> JSONResponse:
        return JSONResponse({"status": "ok", "name": mcp_server.name})

    app.routes.append(Route("/health", _health))

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="swarm-mcp-server")
    thread.start()
    return thread, server
