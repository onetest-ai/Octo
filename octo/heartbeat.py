"""Proactive AI — heartbeat timer and cron scheduler.

Heartbeat: periodic background check that fires every N minutes,
reads HEARTBEAT.md for standing instructions, and delivers messages
to the user when something needs attention.

Cron: persistent scheduled tasks (one-shot, interval, or cron expression)
that the agent or user can create.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ── Suppression ──────────────────────────────────────────────────────

HEARTBEAT_OK_PATTERN = re.compile(r"^\s*HEARTBEAT_OK\s*$", re.IGNORECASE | re.MULTILINE)


def _is_heartbeat_ok(text: str) -> bool:
    """Check if the response is a heartbeat acknowledgement (suppress)."""
    return bool(HEARTBEAT_OK_PATTERN.match(text.strip()))


# ── Interval / time helpers ──────────────────────────────────────────

def _parse_interval_td(spec: str) -> timedelta:
    """Wrapper that builds timedelta cleanly."""
    match = re.match(r"^(\d+)\s*(s|m|h|d)", spec.strip(), re.I)
    if not match:
        raise ValueError(f"Invalid interval: {spec!r}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return timedelta(seconds=value * multipliers[unit])


def _parse_at_time(spec: str) -> datetime:
    """Parse an 'at' spec into a UTC datetime.

    Supports:
      - Relative: "in 2h", "in 30m", "in 1d", "2h", "30m"
      - Time-of-day: "15:00", "08:30" (next occurrence, local)
      - ISO datetime: "2024-02-11T15:00:00Z", "2024-02-11T15:00"
    """
    s = spec.strip().lower()

    # Relative: "in 2h", "in 30m", "2h", "30m"
    rel = re.match(r"^(?:in\s+)?(\d+)\s*(s|m|h|d)", s, re.I)
    if rel:
        delta = _parse_interval_td(rel.group(1) + rel.group(2))
        return datetime.now(timezone.utc) + delta

    # Time-of-day: "15:00", "08:30"
    tod = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if tod:
        h, m = int(tod.group(1)), int(tod.group(2))
        now = datetime.now()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        # Convert local to UTC
        return target.astimezone(timezone.utc)

    # ISO datetime
    try:
        dt = datetime.fromisoformat(s.rstrip("z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    raise ValueError(f"Cannot parse time spec: {spec!r}")


def _next_cron_run(cron_expr: str, after: datetime) -> datetime:
    """Calculate next run time for a 5-field cron expression.

    Fields: minute hour day_of_month month day_of_week
    Supports: *, ranges (1-5), lists (1,3,5), steps (*/5)
    Day of week: 0=Mon ... 6=Sun (also MON-SUN)
    """
    DOW_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

    def _parse_field(field_str: str, min_val: int, max_val: int) -> set[int]:
        result: set[int] = set()
        for part in field_str.split(","):
            part = part.strip().lower()
            # Replace day names
            for name, num in DOW_MAP.items():
                part = part.replace(name, str(num))

            if "/" in part:
                base, step_str = part.split("/", 1)
                step = int(step_str)
                if base == "*":
                    start = min_val
                elif "-" in base:
                    start = int(base.split("-")[0])
                else:
                    start = int(base)
                result.update(range(start, max_val + 1, step))
            elif "-" in part:
                lo, hi = part.split("-", 1)
                result.update(range(int(lo), int(hi) + 1))
            elif part == "*":
                result.update(range(min_val, max_val + 1))
            else:
                result.add(int(part))
        return result

    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(fields)}: {cron_expr!r}")

    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    doms = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    dows = _parse_field(fields[4], 0, 6)

    # Start searching from the next minute after 'after'
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Search up to 2 years ahead
    max_candidate = after + timedelta(days=730)
    while candidate < max_candidate:
        if (candidate.month in months
                and candidate.day in doms
                and candidate.weekday() in dows
                and candidate.hour in hours
                and candidate.minute in minutes):
            return candidate
        candidate += timedelta(minutes=1)

    raise ValueError(f"No matching time found for cron expression: {cron_expr!r}")


# ── Cron data structures ─────────────────────────────────────────────

class CronJobType(str, Enum):
    AT = "at"
    EVERY = "every"
    CRON = "cron"


@dataclass
class CronJob:
    id: str
    task: str
    type: CronJobType
    spec: str
    isolated: bool = False
    created_at: str = ""
    next_run: str = ""
    last_run: str | None = None
    paused: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> CronJob:
        data = dict(data)
        data["type"] = CronJobType(data.get("type", "at"))
        return cls(**data)


# ── Cron persistence ─────────────────────────────────────────────────

class CronStore:
    """Read/write cron jobs from a JSON file."""

    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[CronJob]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [CronJob.from_dict(d) for d in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def save(self, jobs: list[CronJob]) -> None:
        self._path.write_text(
            json.dumps([j.to_dict() for j in jobs], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def add(self, job: CronJob) -> None:
        jobs = self.load()
        jobs.append(job)
        self.save(jobs)

    def remove(self, job_id: str) -> bool:
        jobs = self.load()
        before = len(jobs)
        jobs = [j for j in jobs if j.id != job_id]
        if len(jobs) < before:
            self.save(jobs)
            return True
        return False

    def update(self, job: CronJob) -> None:
        jobs = self.load()
        for i, j in enumerate(jobs):
            if j.id == job.id:
                jobs[i] = job
                break
        self.save(jobs)

    def toggle_pause(self, job_id: str) -> bool | None:
        """Toggle paused state. Returns new paused value, or None if not found."""
        jobs = self.load()
        for j in jobs:
            if j.id == job_id:
                j.paused = not j.paused
                self.save(jobs)
                return j.paused
        return None


# ── Heartbeat runner ─────────────────────────────────────────────────

class HeartbeatRunner:
    """Background asyncio task that periodically invokes the agent."""

    def __init__(
        self,
        graph_app: Any,
        get_thread_id: Callable[[], str],
        interval: int,
        active_start: time,
        active_end: time,
        heartbeat_path: Path,
        on_message: Callable[[str], Awaitable[None]],
        graph_lock: asyncio.Lock,
        callbacks: list | None = None,
    ) -> None:
        self._app = graph_app
        self._get_thread_id = get_thread_id
        self._interval = interval
        self._active_start = active_start
        self._active_end = active_end
        self._heartbeat_path = heartbeat_path
        self._on_message = on_message
        self._lock = graph_lock
        self._callbacks = callbacks or []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def active_start(self) -> time:
        return self._active_start

    @property
    def active_end(self) -> time:
        return self._active_end

    def _is_active_hours(self) -> bool:
        now = datetime.now().time()
        if self._active_start <= self._active_end:
            return self._active_start <= now <= self._active_end
        # Overnight range (e.g., 22:00 -> 06:00)
        return now >= self._active_start or now <= self._active_end

    def _load_heartbeat_instructions(self) -> str:
        """Read HEARTBEAT.md, return empty string if file is empty/comments-only."""
        if not self._heartbeat_path.is_file():
            return ""
        text = self._heartbeat_path.read_text(encoding="utf-8").strip()
        # Filter out lines that are only comments or blank
        lines = [
            line for line in text.split("\n")
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("<!--")
        ]
        return "\n".join(lines).strip()

    def _build_heartbeat_prompt(self, instructions: str) -> str:
        from octo.config import STATE_PATH, OCTO_DIR

        now = datetime.now()
        parts = [
            f"[HEARTBEAT \u2014 {now.strftime('%Y-%m-%d %H:%M')} local]",
            "",
            "This is an automatic periodic check. Your standing instructions:",
            "",
            instructions,
            "",
        ]

        # Include plan summary (most recent plan file)
        from octo.config import PLANS_DIR
        plan_files = sorted(PLANS_DIR.glob("plan_*.json"), reverse=True)
        if plan_files:
            try:
                todos = json.loads(plan_files[0].read_text(encoding="utf-8"))
                if todos:
                    completed = sum(1 for t in todos if t.get("status") == "completed")
                    parts.append(f"Current plan: {completed}/{len(todos)} tasks completed.")
            except Exception:
                pass

        # Include STATE.md summary
        if STATE_PATH.is_file():
            state = STATE_PATH.read_text(encoding="utf-8").strip()
            if state:
                parts.append(f"\nProject state:\n{state[:500]}")

        parts.append(
            "\n---\n"
            "If nothing needs attention and there is no message for the user, "
            "reply with exactly: HEARTBEAT_OK\n"
            "Otherwise, provide your message to the user."
        )
        return "\n".join(parts)

    async def _tick(self) -> None:
        """Execute one heartbeat cycle."""
        if not self._is_active_hours():
            logger.debug("Heartbeat skipped: outside active hours")
            return

        instructions = self._load_heartbeat_instructions()
        if not instructions:
            logger.debug("Heartbeat skipped: HEARTBEAT.md is empty")
            return

        prompt = self._build_heartbeat_prompt(instructions)

        # Phase 1: Quick check via low-tier LLM (cheap)
        from octo.models import make_model
        model = make_model(tier="low")

        try:
            response = await model.ainvoke(prompt)
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()
        except Exception as e:
            logger.warning("Heartbeat LLM call failed: %s", e)
            return

        if _is_heartbeat_ok(content):
            logger.debug("Heartbeat: OK (no action needed)")
            return

        # Phase 2: Agent wants to act — route through the full graph
        try:
            acquired = self._lock.locked()
            if acquired:
                logger.debug("Heartbeat skipped: graph lock busy (user is active)")
                return
            await asyncio.wait_for(self._lock.acquire(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.debug("Heartbeat skipped: graph lock busy (user is active)")
            return

        try:
            thread_id = self._get_thread_id()
            config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
            if self._callbacks:
                config["callbacks"] = self._callbacks

            from octo.retry import invoke_with_retry

            result = await invoke_with_retry(
                self._app,
                {"messages": [HumanMessage(content=prompt)]},
                config,
            )

            response_text = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    response_text = msg.content
                    break

            if response_text and not _is_heartbeat_ok(response_text):
                await self._on_message(response_text)
        except Exception as e:
            logger.warning("Heartbeat graph invocation failed: %s", e)
        finally:
            self._lock.release()

    async def force_tick(self) -> None:
        """Force-run a heartbeat tick (for /heartbeat test)."""
        await self._tick()

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await self._tick()
            except Exception:
                logger.exception("Heartbeat tick failed")

    def start(self) -> None:
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("Heartbeat stopped")


# ── Cron scheduler ───────────────────────────────────────────────────

class CronScheduler:
    """Background scheduler that executes cron jobs at their scheduled times."""

    def __init__(
        self,
        store: CronStore,
        graph_app: Any,
        get_thread_id: Callable[[], str],
        on_message: Callable[[str, str], Awaitable[None]],
        graph_lock: asyncio.Lock,
        callbacks: list | None = None,
    ) -> None:
        self._store = store
        self._app = graph_app
        self._get_thread_id = get_thread_id
        self._on_message = on_message
        self._lock = graph_lock
        self._callbacks = callbacks or []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._check_interval = 30  # seconds

    async def _execute_job(self, job: CronJob) -> None:
        thread_id = f"cron:{job.id}" if job.isolated else self._get_thread_id()

        prompt = (
            f"[SCHEDULED TASK \u2014 {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n\n"
            f"Task: {job.task}\n\n"
            "Execute this task and provide the result."
        )

        lock_needed = not job.isolated
        if lock_needed:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Cron job %s skipped: graph lock busy", job.id)
                return

        try:
            config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
            if self._callbacks:
                config["callbacks"] = self._callbacks

            from octo.retry import invoke_with_retry

            result = await invoke_with_retry(
                self._app,
                {"messages": [HumanMessage(content=prompt)]},
                config,
            )

            response_text = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    response_text = msg.content
                    break

            if response_text:
                await self._on_message(job.task, response_text)
        except Exception as e:
            logger.warning("Cron job %s failed: %s", job.id, e)
        finally:
            if lock_needed:
                self._lock.release()

    def _advance_job(self, job: CronJob) -> CronJob | None:
        """Calculate next run time. Returns None for completed one-shots."""
        now = datetime.now(timezone.utc)
        job.last_run = now.isoformat()

        if job.type == CronJobType.AT:
            return None
        elif job.type == CronJobType.EVERY:
            delta = _parse_interval_td(job.spec)
            job.next_run = (now + delta).isoformat()
        elif job.type == CronJobType.CRON:
            job.next_run = _next_cron_run(job.spec, now).isoformat()

        return job

    async def _check(self) -> None:
        jobs = self._store.load()
        now = datetime.now(timezone.utc)

        for job in jobs:
            if job.paused:
                continue
            try:
                next_run = datetime.fromisoformat(job.next_run)
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if next_run <= now:
                logger.info("Executing cron job: %s (%s)", job.id, job.task[:50])
                await self._execute_job(job)

                updated = self._advance_job(job)
                if updated is None:
                    self._store.remove(job.id)
                else:
                    self._store.update(updated)

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._check_interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await self._check()
            except Exception:
                logger.exception("Cron scheduler check failed")

    def start(self) -> None:
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("Cron scheduler started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("Cron scheduler stopped")


# ── schedule_task tool ───────────────────────────────────────────────

_cron_store: CronStore | None = None


def set_cron_store(store: CronStore) -> None:
    global _cron_store
    _cron_store = store


def make_schedule_task_tool():
    """Build the schedule_task LangChain tool. Call after set_cron_store()."""
    from langchain_core.tools import tool

    @tool
    def schedule_task(
        task: str,
        schedule_type: str = "at",
        spec: str = "",
        isolated: bool = False,
    ) -> str:
        """Schedule a task to run later.

        Use this when the user says things like "remind me in 2 hours",
        "check this every morning", or "run this at 3pm".

        Args:
            task: What to do (natural language description).
            schedule_type: "at" (one-shot), "every" (interval), "cron" (expression).
            spec: Schedule spec. Examples:
                - at: "in 2h", "in 30m", "15:00", "2024-02-11T15:00Z"
                - every: "30m", "2h", "1d"
                - cron: "0 9 * * MON-FRI" (5-field: min hour dom month dow)
            isolated: If True, runs in a fresh conversation (no shared context).
        """
        if _cron_store is None:
            return "Error: Cron scheduler not initialized."

        try:
            job_type = CronJobType(schedule_type)
        except ValueError:
            return f"Invalid schedule_type: {schedule_type}. Use: at, every, cron"

        now = datetime.now(timezone.utc)

        try:
            if job_type == CronJobType.AT:
                next_run = _parse_at_time(spec)
            elif job_type == CronJobType.EVERY:
                delta = _parse_interval_td(spec)
                next_run = now + delta
            elif job_type == CronJobType.CRON:
                next_run = _next_cron_run(spec, now)
            else:
                return f"Unknown schedule type: {schedule_type}"
        except ValueError as e:
            return f"Invalid spec: {e}"

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            task=task,
            type=job_type,
            spec=spec,
            isolated=isolated,
            created_at=now.isoformat(),
            next_run=next_run.isoformat(),
        )

        _cron_store.add(job)
        return f"Scheduled: '{task}' \u2014 next run at {next_run.strftime('%Y-%m-%d %H:%M UTC')}"

    return schedule_task
