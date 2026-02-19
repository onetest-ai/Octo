"""Background task execution — process and agent workers.

Allows the supervisor to dispatch long-running tasks that run independently
and deliver results via the proactive notification channel (CLI + Telegram).

Two task types:
- **process**: subprocess (claude -p, shell commands) — done when process exits
- **agent**: standalone LangGraph agent — done when task_complete tool called
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ── Task data ─────────────────────────────────────────────────────

@dataclass
class BackgroundTask:
    """Persistent state for a background task."""

    id: str
    type: str                       # "process" | "agent"
    status: str = "pending"         # pending | running | completed | failed | paused | cancelled

    # Process mode
    command: str = ""
    cwd: str = ""

    # Agent mode
    prompt: str = ""
    agent_name: str = ""
    thread_id: str = ""
    max_turns: int = 50

    # Environment overrides (project env, CLAUDE_CONFIG_DIR, etc.)
    env_overrides: dict[str, str] = field(default_factory=dict)

    # Common
    timeout: int = 0  # 0 = no timeout (fire-and-forget)
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    error: str = ""

    # Escalation (agent mode)
    paused_question: str = ""
    paused_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BackgroundTask:
        # Filter to only known fields to survive schema changes
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── JSON file store ───────────────────────────────────────────────

class TaskStore:
    """Persist tasks as JSON files in .octo/tasks/."""

    def __init__(self, tasks_dir: Path):
        self._dir = tasks_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, task: BackgroundTask) -> None:
        path = self._dir / f"{task.id}.json"
        path.write_text(json.dumps(task.to_dict(), indent=2) + "\n")

    def load(self, task_id: str) -> BackgroundTask | None:
        path = self._dir / f"{task_id}.json"
        if not path.is_file():
            return None
        try:
            return BackgroundTask.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    def list_all(self) -> list[BackgroundTask]:
        tasks = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                tasks.append(BackgroundTask.from_dict(json.loads(path.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue
        return tasks

    def delete(self, task_id: str) -> bool:
        path = self._dir / f"{task_id}.json"
        if path.is_file():
            path.unlink()
            return True
        return False


# ── Worker pool ───────────────────────────────────────────────────

class BackgroundWorkerPool:
    """Manages concurrent background task execution.

    Does NOT use the shared graph_lock — background tasks run independently
    with their own semaphore-based concurrency control.
    """

    def __init__(
        self,
        store: TaskStore,
        on_complete: Callable[[str, str, str], Awaitable[None]],
        max_concurrent: int = 3,
    ):
        self._store = store
        self._on_complete = on_complete
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task] = {}

    async def dispatch(self, task: BackgroundTask) -> str:
        """Start a background task. Returns task ID."""
        self._store.save(task)
        self._running[task.id] = asyncio.create_task(
            self._run(task), name=f"bg:{task.id}"
        )
        logger.info("Dispatched background task %s (%s)", task.id, task.type)
        return task.id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or pending task."""
        if task_id in self._running:
            self._running[task_id].cancel()
            # Update status after cancellation propagates
            task = self._store.load(task_id)
            if task:
                task.status = "cancelled"
                task.completed_at = _now()
                self._store.save(task)
            return True
        task = self._store.load(task_id)
        if task and task.status in ("pending", "paused"):
            task.status = "cancelled"
            task.completed_at = _now()
            self._store.save(task)
            return True
        return False

    async def resume_task(self, task_id: str, answer: str) -> bool:
        """Resume a paused agent task with user's answer."""
        task = self._store.load(task_id)
        if not task or task.status != "paused" or task.type != "agent":
            return False

        task.status = "running"
        task.paused_question = ""
        task.paused_at = ""
        self._store.save(task)

        # Re-dispatch — the agent runner injects the answer and continues
        task._resume_answer = answer  # type: ignore[attr-defined]
        self._running[task.id] = asyncio.create_task(
            self._run(task, resume_answer=answer), name=f"bg:{task.id}"
        )
        return True

    async def follow_up(self, task_id: str, instruction: str) -> BackgroundTask | None:
        """Create a follow-up task that carries the original task's result as context.

        Used when replying to a completed/failed task notification.
        Returns the new task, or None if original not found.
        """
        original = self._store.load(task_id)
        if not original:
            return None

        new_id = uuid.uuid4().hex[:8]
        # Build a prompt that includes the original task context
        context_parts = [
            f"## Previous task ({original.id})",
            f"**Original objective**: {original.prompt or original.command}",
        ]
        if original.result:
            # Cap context to avoid blowing the window
            result_text = original.result[:8000]
            if len(original.result) > 8000:
                result_text += "\n... (truncated)"
            context_parts.append(f"**Result**:\n{result_text}")
        if original.error:
            context_parts.append(f"**Error**: {original.error}")

        context_parts.append(f"\n## Follow-up instruction\n{instruction}")

        new_task = BackgroundTask(
            id=new_id,
            type="agent",
            status="pending",
            created_at=_now(),
            prompt="\n".join(context_parts),
            agent_name=original.agent_name,
            thread_id=f"bg:{new_id}",
            cwd=original.cwd,
            timeout=original.timeout,
            max_turns=original.max_turns,
        )

        await self.dispatch(new_task)
        return new_task

    def get_task(self, task_id: str) -> BackgroundTask | None:
        return self._store.load(task_id)

    def list_tasks(self, status: str = "") -> list[BackgroundTask]:
        tasks = self._store.list_all()
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    async def shutdown(self) -> None:
        """Cancel all running tasks gracefully."""
        for worker in list(self._running.values()):
            worker.cancel()
        if self._running:
            try:
                await asyncio.wait(
                    list(self._running.values()),
                    timeout=5.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
            except Exception:
                pass  # tasks may already be cancelled by MCP teardown
        self._running.clear()

    # ── Internal runners ──────────────────────────────────────────

    async def _run(self, task: BackgroundTask, resume_answer: str = "") -> None:
        """Execute a task within the semaphore."""
        async with self._semaphore:
            try:
                task.status = "running"
                task.started_at = task.started_at or _now()
                self._store.save(task)

                if task.type == "process":
                    await self._run_process(task)
                elif task.type == "agent":
                    await self._run_agent(task, resume_answer)
                else:
                    task.status = "failed"
                    task.error = f"Unknown task type: {task.type}"
                    self._store.save(task)

            except (asyncio.CancelledError, BaseException) as exc:
                if isinstance(exc, asyncio.CancelledError) or "cancel" in type(exc).__name__.lower():
                    task.status = "cancelled"
                    task.completed_at = _now()
                    self._store.save(task)
                    return  # don't re-raise — let shutdown() collect cleanly
                # Regular exception
                logger.exception("Background task %s failed", task.id)
                task.status = "failed"
                task.error = str(exc)
                task.completed_at = _now()
                self._store.save(task)
            finally:
                self._running.pop(task.id, None)

    async def _run_process(self, task: BackgroundTask) -> None:
        """Execute a subprocess task. Done = process exit."""
        from octo.config import ADDITIONAL_CLAUDE_ARGS, CLAUDE_CODE_TIMEOUT

        command = task.command
        env = os.environ.copy()

        # Use env_overrides stored on the task (set by dispatch_background).
        # Fall back to path-based resolution for backward compat (old tasks
        # persisted without env_overrides).
        if task.env_overrides:
            env_overrides = task.env_overrides
            resolved_cwd = task.cwd
        else:
            from octo.core.tools.claude_code import _resolve_project
            resolved_cwd, env_overrides, _ = _resolve_project(
                agent_name=task.agent_name,
                working_directory=task.cwd,
            )
        env.update(env_overrides)

        is_claude = _is_claude_command(command)

        # GUARD: refuse to run claude without CLAUDE_CONFIG_DIR
        if is_claude and "CLAUDE_CONFIG_DIR" not in env_overrides:
            task.status = "failed"
            task.error = (
                "Refused: claude command launched without CLAUDE_CONFIG_DIR. "
                "Register the project in .octo/projects/ or use the `project` "
                "parameter in dispatch_background."
            )
            task.completed_at = _now()
            self._store.save(task)
            await self._on_complete(task.id, task.status, task.error)
            return

        # GUARD: enforce minimum timeout for claude commands
        if is_claude and 0 < task.timeout < CLAUDE_CODE_TIMEOUT:
            logger.warning(
                "Task %s: timeout=%d too low for claude, overriding to %d",
                task.id, task.timeout, CLAUDE_CODE_TIMEOUT,
            )
            task.timeout = CLAUDE_CODE_TIMEOUT
            self._store.save(task)
        # GUARD: enforce minimum timeout for test/long-running commands
        elif _is_long_running_command(command) and 0 < task.timeout < _LONG_RUNNING_MIN_TIMEOUT:
            logger.warning(
                "Task %s: timeout=%d too low for test command, overriding to %d",
                task.id, task.timeout, _LONG_RUNNING_MIN_TIMEOUT,
            )
            task.timeout = _LONG_RUNNING_MIN_TIMEOUT
            self._store.save(task)

        # Inject ADDITIONAL_CLAUDE_ARGS into claude commands
        if ADDITIONAL_CLAUDE_ARGS and is_claude:
            parts = shlex.split(command)
            extra = shlex.split(ADDITIONAL_CLAUDE_ARGS)
            # Insert after "claude" but before other args
            parts[1:1] = extra
            command = shlex.join(parts)

        cwd = task.cwd or resolved_cwd

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            if task.timeout > 0:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=task.timeout
                )
            else:
                stdout, stderr = await proc.communicate()
        except asyncio.TimeoutError:
            task.status = "failed"
            task.error = f"Timed out after {task.timeout}s"
            task.completed_at = _now()
            self._store.save(task)
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            await self._on_complete(task.id, task.status, task.error)
            return

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            task.status = "completed"
            task.result = out_text or "(no output)"
            if err_text:
                task.result += f"\n\n[stderr]\n{err_text}"
        else:
            task.status = "failed"
            task.error = f"Exit code {proc.returncode}"
            if err_text:
                task.error += f"\n{err_text}"
            task.result = out_text

        task.completed_at = _now()
        self._store.save(task)
        await self._on_complete(task.id, task.status, task.result or task.error)

    async def _run_agent(self, task: BackgroundTask, resume_answer: str = "") -> None:
        """Execute an agent task. Done = task_complete tool or max turns.

        When task.agent_name matches a loaded AgentConfig, uses that agent's
        system_prompt, model, and tool configuration. Otherwise falls back
        to generic defaults.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain.agents import create_agent

        # Lazy imports for model + tools
        from octo.models import make_model

        # Sentinel flags — the closure-based tools set these
        _done = {"completed": False, "paused": False}

        # -- Build per-task tools --
        def _make_task_tools():
            from langchain_core.tools import tool as lc_tool

            @lc_tool
            def task_complete(summary: str) -> str:
                """Call when your task is fully complete.

                Args:
                    summary: concise result for the user
                """
                task.status = "completed"
                task.result = summary
                task.completed_at = _now()
                self._store.save(task)
                _done["completed"] = True
                return "Task marked complete."

            @lc_tool
            def escalate_question(question: str) -> str:
                """Ask the user for clarification. Execution pauses until they respond.

                Args:
                    question: what you need to know
                """
                task.status = "paused"
                task.paused_question = question
                task.paused_at = _now()
                self._store.save(task)
                _done["paused"] = True
                # Delivery happens in the runner after the loop breaks —
                # we can't call async from this sync tool (runs in thread pool).
                return "Question sent to user. Pausing execution."

            return [task_complete, escalate_question]

        task_tools = _make_task_tools()

        # -- Resolve agent config from AGENT.md files --
        agent_cfg = None
        if task.agent_name:
            agent_cfg = _find_agent_config(task.agent_name)

        # -- Build tools --
        from octo.core.tools import BUILTIN_TOOLS
        try:
            from octo.core.tools.mcp_proxy import find_tools, call_mcp_tool
            mcp_tools_available = [find_tools, call_mcp_tool]
        except Exception:
            mcp_tools_available = []

        if agent_cfg and agent_cfg.tools:
            # Agent specifies tool names — resolve from builtin + MCP proxy
            builtin_by_name = {t.name: t for t in BUILTIN_TOOLS}
            all_mcp_by_name = {t.name: t for t in mcp_tools_available}
            agent_tools = [builtin_by_name[n] for n in agent_cfg.tools if n in builtin_by_name]
            agent_tools += [all_mcp_by_name[n] for n in agent_cfg.tools if n in all_mcp_by_name]
            # If agent requests tools not in builtin/mcp, they get MCP proxy as fallback
            if not agent_tools:
                agent_tools = list(BUILTIN_TOOLS) + mcp_tools_available
        else:
            agent_tools = list(BUILTIN_TOOLS) + mcp_tools_available

        all_tools = agent_tools + task_tools

        # -- Resolve model --
        if agent_cfg:
            from octo.core.graph import _agent_tier, _resolve_agent_model
            tier = _agent_tier(task.agent_name)
            model = _resolve_agent_model(agent_cfg.model, tier)
        else:
            model = make_model(tier="high")

        # -- Build system prompt --
        bg_rules = (
            "\n\n## Background Task Rules\n"
            "- You are running as an autonomous background task.\n"
            "- When your task is fully complete, call `task_complete` with a summary.\n"
            "- If you need clarification from the user, call `escalate_question`.\n"
            "- Work autonomously. Do not wait for user input unless truly stuck.\n"
        )

        if agent_cfg and agent_cfg.system_prompt:
            system_prompt = (
                agent_cfg.system_prompt
                + f"\n\n## Current Objective\n{task.prompt}"
                + bg_rules
            )
        else:
            system_prompt = (
                f"You are running as a background task.\n\n"
                f"## Objective\n{task.prompt}"
                + bg_rules
                + "- You have filesystem tools (read, grep, glob, edit, bash) and MCP tools available.\n"
            )

        # Build standalone agent (no supervisor overhead)
        from langgraph.checkpoint.memory import MemorySaver
        from octo.middleware import ToolErrorMiddleware
        checkpointer = MemorySaver()

        agent = create_agent(
            model=model,
            tools=all_tools,
            name=task.agent_name or f"bg-{task.id}",
            system_prompt=system_prompt,
            middleware=[ToolErrorMiddleware()],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": task.thread_id}}

        # Build input
        if resume_answer:
            input_msg = {"messages": [HumanMessage(content=resume_answer)]}
        else:
            input_msg = {"messages": [HumanMessage(content=task.prompt)]}

        turn = 0
        while turn < task.max_turns:
            result = await agent.ainvoke(input_msg, config=config)
            turn += 1

            # Check sentinel flags from tool closures
            if _done["completed"] or _done["paused"]:
                break

            # Check if agent stopped calling tools (natural end)
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
                    # Agent finished without calling task_complete — use its last message
                    task.status = "completed"
                    task.result = last.content if isinstance(last.content, str) else str(last.content)
                    task.completed_at = _now()
                    self._store.save(task)
                    break

            # For subsequent turns, pass empty to let agent continue from checkpoint
            input_msg = {"messages": []}

        else:
            # Hit max turns
            task.status = "failed"
            task.error = f"Max turns ({task.max_turns}) exceeded"
            task.completed_at = _now()
            self._store.save(task)

        if _done["paused"]:
            await self._on_complete(
                task.id,
                "paused",
                f"**Question from background task {task.id}:**\n\n"
                f"{task.paused_question}\n\n"
                f"Reply with: `/task {task.id} resume <your answer>`",
            )
        elif _done["completed"] or task.status == "completed":
            await self._on_complete(task.id, "completed", task.result)
        elif task.status == "failed":
            await self._on_complete(task.id, "failed", task.error)


# ── Agent config lookup ──────────────────────────────────────────

def _find_agent_config(agent_name: str):
    """Look up an AgentConfig by name from all loaded agent sources.

    Returns AgentConfig if found, None otherwise. Searches both project
    agents (AGENT_DIRS) and Octo-native agents (.octo/agents/).
    """
    from octo.loaders.agent_loader import load_agents, load_octo_agents

    for cfg in load_agents() + load_octo_agents():
        if cfg.name == agent_name:
            return cfg
    return None


# ── Module-level singleton (set by cli.py) ────────────────────────

_worker_pool: BackgroundWorkerPool | None = None


def set_worker_pool(pool: BackgroundWorkerPool) -> None:
    global _worker_pool
    _worker_pool = pool


def get_worker_pool() -> BackgroundWorkerPool | None:
    return _worker_pool


# ── Supervisor tool factory ───────────────────────────────────────

def make_dispatch_background_tool():
    """Build the dispatch_background tool for the supervisor. Call after set_worker_pool()."""
    from langchain_core.tools import tool

    @tool
    async def dispatch_background(
        task_type: str,
        command: str = "",
        prompt: str = "",
        project: str = "",
        agent_name: str = "",
        cwd: str = "",
        timeout: int = 0,
        max_turns: int = 50,
    ) -> str:
        """Dispatch a long-running task to background execution.

        The task runs independently — you return immediately with a task ID.
        The user is notified when the task completes (or if it needs input).

        Use this when a task will take several minutes or more. Do NOT use
        for quick operations that finish in seconds.

        IMPORTANT — timeout rules:
        - For `claude -p` commands: ALWAYS use timeout=0 (no timeout). Claude Code
          tasks routinely run 30-90 minutes. Setting 600 or 900 WILL kill the task
          prematurely. The system enforces a minimum of CLAUDE_CODE_TIMEOUT for
          claude commands — low values are auto-corrected.
        - For test commands (pytest, robot, playwright, etc.): ALWAYS use timeout=0.
          Test suites with Playwright/browser automation routinely run 30-60 minutes.
          The system enforces a minimum of 3600s for test commands.
        - For simple/quick shell commands (curl, ls, grep): a timeout is acceptable.
        - timeout=0 means "run until done" — this is the CORRECT default for
          ALL background tasks. Only set a non-zero timeout if you are CERTAIN the
          command will finish quickly. When in doubt, use 0.

        IMPORTANT — project requirement for claude commands:
        - Claude commands MUST have a resolvable project (via `project` param or
          `cwd` matching a registered project). Running claude without
          CLAUDE_CONFIG_DIR is prohibited.

        Args:
            task_type: "process" for subprocess (claude -p, shell), "agent" for LangGraph agent
            command: Shell command for process mode (e.g., "claude -p 'analyze codebase'")
            prompt: Task description for agent mode
            project: Project name from the registry (e.g., "onetest"). Auto-sets cwd
                and CLAUDE_CONFIG_DIR for that project. Use this instead of cwd.
            agent_name: Optional agent name — also resolves the project automatically
            cwd: Working directory (overridden by project if both given)
            timeout: Max seconds before kill (0 = no timeout, runs until done). Default 0.
                For claude -p commands, ALWAYS use 0. Low values are auto-corrected.
            max_turns: Max agent conversation turns (default 50, agent mode only)
        """
        pool = get_worker_pool()
        if not pool:
            return "Error: background worker pool not initialized."

        if task_type not in ("process", "agent"):
            return f"Error: task_type must be 'process' or 'agent', got '{task_type}'."
        if task_type == "process" and not command:
            return "Error: 'command' is required for process mode."
        if task_type == "agent" and not prompt:
            return "Error: 'prompt' is required for agent mode."

        # Resolve project → cwd + env from registry (direct, no path re-discovery)
        task_env: dict[str, str] = {}
        if project:
            from octo.config import PROJECTS
            proj = PROJECTS.get(project)
            if not proj:
                return f"Error: project '{project}' not found. Available: {', '.join(PROJECTS)}"
            cwd = cwd or proj.path
            task_env = dict(proj.env)  # copy — includes CLAUDE_CONFIG_DIR if set
        elif cwd or agent_name:
            # No explicit project — fall back to path-based resolution
            from octo.core.tools.claude_code import _resolve_project
            resolved_cwd, env_overrides, _ = _resolve_project(
                agent_name=agent_name,
                working_directory=cwd,
            )
            cwd = cwd or resolved_cwd
            task_env = env_overrides

        # Enforce rules for claude commands
        _is_claude_cmd = _is_claude_command(command)
        if _is_claude_cmd:
            from octo.config import CLAUDE_CODE_TIMEOUT
            if "CLAUDE_CONFIG_DIR" not in task_env:
                return (
                    "Error: claude commands require a registered project with "
                    "CLAUDE_CONFIG_DIR. Either pass `project` parameter with a "
                    f"registered project name, or ensure `cwd` matches a known "
                    f"project path. Running claude without project context is "
                    f"prohibited."
                )
            # Enforce minimum timeout — low values kill claude prematurely
            if 0 < timeout < CLAUDE_CODE_TIMEOUT:
                logger.warning(
                    "dispatch_background: timeout=%d too low for claude command, "
                    "overriding to %d (CLAUDE_CODE_TIMEOUT)",
                    timeout, CLAUDE_CODE_TIMEOUT,
                )
                timeout = CLAUDE_CODE_TIMEOUT
        elif _is_long_running_command(command):
            # Test suites (pytest, playwright, etc.) can run 30-60+ minutes
            if 0 < timeout < _LONG_RUNNING_MIN_TIMEOUT:
                logger.warning(
                    "dispatch_background: timeout=%d too low for test command, "
                    "overriding to %d",
                    timeout, _LONG_RUNNING_MIN_TIMEOUT,
                )
                timeout = _LONG_RUNNING_MIN_TIMEOUT

        task_id = uuid.uuid4().hex[:8]
        task = BackgroundTask(
            id=task_id,
            type=task_type,
            status="pending",
            created_at=_now(),
            command=command,
            prompt=prompt,
            agent_name=agent_name,
            thread_id=f"bg:{task_id}" if task_type == "agent" else "",
            cwd=cwd,
            env_overrides=task_env,
            timeout=timeout,
            max_turns=max_turns,
        )

        await pool.dispatch(task)

        mode_info = f"Command: `{command}`" if task_type == "process" else f"Prompt: {prompt[:100]}"
        return (
            f"Background task dispatched.\n"
            f"- **ID**: {task_id}\n"
            f"- **Type**: {task_type}\n"
            f"- {mode_info}\n\n"
            f"The user will be notified when it completes. "
            f"They can check status with `/tasks` or `/task {task_id}`."
        )

    return dispatch_background


# ── Helpers ───────────────────────────────────────────────────────

def _is_claude_command(command: str) -> bool:
    """Check if a shell command invokes Claude Code CLI."""
    if not command:
        return False
    # Handle commands like "cd /foo && claude -p ...", "claude -p ...",
    # or full paths like "/home/user/.local/bin/claude -p ..."
    stripped = command.lstrip()
    # Direct invocation
    if stripped.startswith("claude ") or stripped.startswith("claude\t"):
        return True
    # Full path invocation (e.g., /usr/local/bin/claude)
    if "/claude " in stripped or "/claude\t" in stripped:
        return True
    # After cd/env/etc: "cd /foo && claude -p ..."
    if "&& claude " in stripped or "&& claude\t" in stripped:
        return True
    if "| claude " in stripped or "; claude " in stripped:
        return True
    return False


# Minimum timeout for long-running commands (seconds).
# pytest with Playwright/ReportPortal, robot, behave, etc. can run 30+ min.
_LONG_RUNNING_MIN_TIMEOUT = 3600  # 1 hour

# Patterns that indicate a long-running test/build command.
_LONG_RUNNING_PATTERNS = (
    "pytest", "python -m pytest",
    "robot", "behave", "cucumber",
    "npm test", "npm run test", "npx playwright",
    "mvn test", "gradle test",
    "make test", "make check",
)


def _is_long_running_command(command: str) -> bool:
    """Check if a command is known to be potentially long-running (tests, builds)."""
    if not command:
        return False
    # Normalize: look at the actual command after cd/env prefixes
    # e.g., "cd /foo && pytest ..." or "/path/to/pytest ..."
    lower = command.lower()
    for pattern in _LONG_RUNNING_PATTERNS:
        if pattern in lower:
            return True
    return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
