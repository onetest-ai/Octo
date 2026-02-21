"""CLI entry point — Click group + async chat loop."""
from __future__ import annotations

import asyncio
import logging
import uuid

import click
from langchain_core.messages import HumanMessage

from octo.config import DEFAULT_MODEL, TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Octi — LangGraph multi-agent console."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@main.command()
@click.option("--model", default="", help="Model override")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Debug output (show LLM calls)")
@click.option("--voice", is_flag=True, help="Enable TTS")
@click.option("--thread", default="", help="Resume thread ID")
@click.option("--resume", is_flag=True, help="Resume last session")
@click.option("--no-telegram", is_flag=True, help="Disable Telegram bot")
def chat(model: str, verbose: bool, debug: bool, voice: bool, thread: str, resume: bool, no_telegram: bool) -> None:
    """Start interactive chat session."""
    asyncio.run(_chat_loop(model, verbose, debug, voice, thread, resume, no_telegram))


@main.command(name="init")
@click.option("--quick", is_flag=True, help="QuickStart mode (minimal prompts)")
@click.option("--provider", type=click.Choice(["anthropic", "bedrock", "openai", "azure"]),
              help="Pre-select provider")
@click.option("--no-validate", is_flag=True, help="Skip credential validation")
@click.option("--no-persona", is_flag=True, help="Skip persona file scaffolding")
@click.option("--force", is_flag=True, help="Overwrite existing .env")
def init_cmd(quick: bool, provider: str, no_validate: bool, no_persona: bool, force: bool) -> None:
    """Interactive setup wizard. Creates .env and scaffolds .octo/ structure."""
    from octo.wizard import run_init
    run_init(quick=quick, provider=provider, skip_validation=no_validate,
             skip_persona=no_persona, force=force)


@main.command()
@click.option("--fix", is_flag=True, help="Attempt to fix issues (re-run init)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def doctor(fix: bool, json_output: bool) -> None:
    """Check Octo configuration health."""
    from octo.wizard import run_doctor
    asyncio.run(run_doctor(fix=fix, json_output=json_output))


@main.command(name="auth")
@click.argument("action", type=click.Choice(["login", "status", "logout"]))
@click.argument("server_name", required=False)
def auth_cmd(action: str, server_name: str | None) -> None:
    """Manage MCP server OAuth authentication."""
    from octo.oauth.cli_commands import handle_auth
    asyncio.run(handle_auth(action, server_name))


# --- Skills marketplace subcommand ---
from octo.skills_cli import skills as skills_group
main.add_command(skills_group)


async def _chat_loop(
    model_override: str,
    verbose: bool,
    debug: bool,
    voice_on: bool,
    thread_id: str,
    resume: bool,
    no_telegram: bool,
) -> None:
    from octo.wizard import check_first_run
    if not check_first_run():
        return

    from octo import ui
    from octo.callbacks import create_cli_callback
    from octo.graph import build_graph, context_info, read_todos, set_session_pool, set_telegram_transport
    from octo.loaders.mcp_loader import MCPSessionPool, create_mcp_client, filter_tools, get_mcp_configs, get_tool_filters, validate_tool_schemas
    from octo.loaders.skill_loader import load_skills
    from octo.models import make_model, resolve_model_name, _detect_provider
    from octo.sessions import save_session, get_last_session
    from octo.telegram import TelegramTransport
    from octo import voice as voice_mod

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(name)s: %(message)s",
    )
    # Silence noisy loggers (auto-retried, harmless)
    if not debug:
        for noisy in (
            "telegram.ext.Updater",
            "httpx", "httpcore",
            "mcp", "mcp.server", "mcp.server.lowlevel",
            "mcp.client", "mcp.client.session",
        ):
            logging.getLogger(noisy).setLevel(logging.CRITICAL)

    # Always show VP confidence decisions (even without --debug)
    logging.getLogger("octo.virtual_persona").setLevel(logging.INFO)

    # Resolve model info for display
    active_model = resolve_model_name(model_override)
    provider = _detect_provider(active_model)

    # Thread ID — explicit > --resume > new
    if not thread_id and resume:
        last = get_last_session()
        if last:
            thread_id = last["thread_id"]
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]

    # Persist session immediately
    save_session(thread_id, model=active_model)

    # Persistent session pool for STDIO MCP servers
    session_pool = MCPSessionPool()

    # Load MCP servers + build graph (reloadable via /mcp reload)
    async def _load_mcp_servers():
        """Load MCP servers, return (flat_tools, tools_by_server)."""
        try:
            configs = get_mcp_configs()
            filters = get_tool_filters()
        except Exception as e:
            ui.print_error(
                f"Failed to parse .mcp.json: {e}\n"
                "  Fix the JSON syntax and run /mcp reload"
            )
            return [], {}
        # Inject swarm peers as MCP servers
        from octo.config import SWARM_ENABLED
        if SWARM_ENABLED:
            from octo.config import SWARM_DIR
            from octo.swarm.registry import PeerRegistry
            if SWARM_DIR.is_dir():
                _peer_reg = PeerRegistry(SWARM_DIR)
                for _peer in _peer_reg.load():
                    _peer_key = f"swarm-{_peer.name}"
                    if _peer_key not in configs:
                        configs[_peer_key] = {
                            "transport": "streamable_http",
                            "url": _peer.url,
                        }
        tools_flat = []
        tools_map: dict[str, list] = {}
        for sname, scfg in configs.items():
            try:
                if scfg.get("transport") == "stdio":
                    # Persistent session — subprocess stays alive
                    stools = await session_pool.connect(sname, scfg)
                else:
                    # HTTP/SSE — stateless, ephemeral per-call is fine
                    client = create_mcp_client({sname: scfg})
                    stools = await client.get_tools()
                all_count = len(stools)
                stools = filter_tools(stools, sname, filters)
                stools = validate_tool_schemas(stools, sname)
                tools_map[sname] = stools
                if len(stools) < all_count:
                    ui.print_info(f"MCP '{sname}': {len(stools)}/{all_count} tools (filtered)")
                tools_flat.extend(stools)
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except BaseException as e:
                # Build config context for the error explainer
                cfg_summary = f"server={sname}, transport={scfg.get('transport', 'stdio')}"
                if scfg.get("command"):
                    cfg_summary += f", command={scfg['command']} {' '.join(scfg.get('args', []))}"
                if scfg.get("url"):
                    cfg_summary += f", url={scfg['url']}"
                try:
                    from octo.middleware import explain_error
                    explanation = await explain_error(
                        e,
                        context=f"connecting to MCP server '{sname}'",
                        details=cfg_summary,
                    )
                    ui.print_error(f"MCP server '{sname}': {explanation}")
                except Exception:
                    # If explainer itself fails, fall back to raw error
                    ui.print_error(f"MCP server '{sname}': {e}")
        return tools_flat, tools_map

    mcp_tools, mcp_tools_by_server = await _load_mcp_servers()

    try:
        # Build graph
        app, agent_configs, skills = await build_graph(mcp_tools, mcp_tools_by_server)

        # Register session pool for auto-reconnect on dead STDIO sessions
        set_session_pool(session_pool)

        # Wire live context bar into the toolbar
        ui.set_context_ref(context_info)

        # Print welcome banner
        ui.print_welcome(
            model=active_model,
            provider=provider,
            thread_id=thread_id,
            agent_count=len(agent_configs),
            skill_count=len(skills),
            mcp_tool_count=len(mcp_tools),
            mcp_servers=list(mcp_tools_by_server.keys()) if mcp_tools else None,
        )

        # Voice
        if voice_on:
            voice_mod.toggle_voice(True)
            ui.print_status("Voice enabled", "green")

        # Setup input with slash command completion
        _BASE_SLASH_CMDS = ["/help", "/clear", "/compact", "/context", "/agents", "/skills",
                            "/tools", "/call", "/projects", "/projects show", "/projects create",
                            "/projects update", "/projects remove", "/projects reload",
                            "/sessions", "/plan", "/profile",
                            "/voice", "/model", "/mcp", "/cron", "/heartbeat", "/vp",
                            "/bg", "/tasks", "/task", "/swarm",
                            "/create-agent", "/create-skill", "/reload", "/restart", "/update",
                            "/state", "/memory", "exit", "quit"]
        slash_cmds = (_BASE_SLASH_CMDS
                      + [f"/{s.name}" for s in skills]
                      + [f"/{a.name}" for a in agent_configs])
        ui.setup_input(slash_cmds)

        # Callback handler — shared between CLI and Telegram so both show tool traces
        cli_callback = create_cli_callback(verbose=verbose or True, debug=debug)
        callbacks: list = [cli_callback]

        # Langfuse tracing (opt-in via LANGFUSE_ENABLED=true)
        from octo.config import LANGFUSE_ENABLED
        if LANGFUSE_ENABLED:
            try:
                from langfuse import Langfuse
                from langfuse.langchain import CallbackHandler as LangfuseHandler
                from octo.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
                # Initialize the Langfuse client singleton (v3)
                Langfuse(
                    public_key=LANGFUSE_PUBLIC_KEY,
                    secret_key=LANGFUSE_SECRET_KEY,
                    host=LANGFUSE_HOST,
                )
                langfuse_handler = LangfuseHandler()
                callbacks.append(langfuse_handler)
                ui.print_status("Langfuse tracing enabled", "green")
            except ImportError:
                ui.print_info("LANGFUSE_ENABLED=true but langfuse not installed (pip install langfuse)")
            except Exception as exc:
                ui.print_info(f"Langfuse init failed: {exc}")

        # Shared lock — prevents concurrent graph invocations from CLI, Telegram, heartbeat, and cron
        graph_lock = asyncio.Lock()

        # Telegram transport
        tg: TelegramTransport | None = None
        if not no_telegram and TELEGRAM_BOT_TOKEN:
            # Swarm group mode params
            from octo.config import (
                SWARM_ENABLED, SWARM_TELEGRAM_MODE, SWARM_TELEGRAM_GROUP_ID,
                SWARM_ROLE, SWARM_NAME,
            )
            _tg_swarm_mode = SWARM_ENABLED and SWARM_TELEGRAM_MODE == "group"
            _tg_swarm_name = SWARM_NAME if _tg_swarm_mode else ""

            tg = TelegramTransport(
                graph_app=app,
                thread_id=thread_id,
                on_message=lambda text: ui.print_telegram_message(text),
                on_response=lambda text: ui.print_response(text, source="Octi"),
                callbacks=callbacks,
                graph_lock=graph_lock,
                swarm_mode=_tg_swarm_mode,
                swarm_role=SWARM_ROLE if _tg_swarm_mode else "worker",
                swarm_name=_tg_swarm_name,
                group_chat_id=SWARM_TELEGRAM_GROUP_ID if _tg_swarm_mode else None,
            )
            await tg.start()
            set_telegram_transport(tg)
            if _tg_swarm_mode:
                ui.print_status(
                    f"Telegram bot connected (swarm group mode, role={SWARM_ROLE})", "green",
                )
            else:
                ui.print_status("Telegram bot connected", "green")
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks,
            "metadata": {"langfuse_session_id": thread_id},
        }

        # --- Proactive AI: heartbeat + cron ---
        from octo.config import (
            HEARTBEAT_PATH, HEARTBEAT_INTERVAL_SECONDS,
            HEARTBEAT_ACTIVE_START_TIME, HEARTBEAT_ACTIVE_END_TIME,
            CRON_PATH,
        )
        from octo.heartbeat import HeartbeatRunner, CronScheduler, CronStore, set_cron_store

        async def _deliver_proactive(text: str, source: str = "\U0001f493 Heartbeat") -> None:
            ui.print_response(text, source=f"Octi ({source})")
            if tg:
                await tg.send_proactive(text, source=source)

        async def _deliver_cron(task: str, response: str) -> None:
            msg = f"**Scheduled task**: {task}\n\n{response}"
            await _deliver_proactive(msg, source="\u23f0 Cron")

        heartbeat = HeartbeatRunner(
            graph_app=app,
            get_thread_id=lambda: config["configurable"]["thread_id"],
            interval=HEARTBEAT_INTERVAL_SECONDS,
            active_start=HEARTBEAT_ACTIVE_START_TIME,
            active_end=HEARTBEAT_ACTIVE_END_TIME,
            heartbeat_path=HEARTBEAT_PATH,
            on_message=_deliver_proactive,
            graph_lock=graph_lock,
        )
        heartbeat.start()

        cron_store = CronStore(CRON_PATH)
        set_cron_store(cron_store)

        cron_scheduler = CronScheduler(
            store=cron_store,
            graph_app=app,
            get_thread_id=lambda: config["configurable"]["thread_id"],
            on_message=_deliver_cron,
            graph_lock=graph_lock,
        )
        cron_scheduler.start()

        # --- Background worker pool ---
        from octo.background import BackgroundWorkerPool, TaskStore, set_worker_pool
        from octo.config import BG_MAX_CONCURRENT, OCTO_DIR

        _bg_task_store = TaskStore(OCTO_DIR / "tasks")

        async def _deliver_bg(task_id: str, status: str, text: str) -> None:
            if status == "completed":
                display_msg = f"**Completed:**\n\n{text}"
            elif status == "paused":
                display_msg = text  # already formatted with question
            else:
                display_msg = f"**Failed:**\n\n{text}"

            # Show result to user immediately (CLI + Telegram)
            ui.print_response(display_msg, source=f"Octi (Task {task_id})")
            if tg:
                await tg.send_bg_notification(
                    f"**[Task {task_id}]**\n\n{display_msg}", task_id,
                )

            # Paused tasks — user resumes manually, no history injection
            if status == "paused":
                return

            # Inject result as AIMessage so the agent has context on the next
            # user-initiated turn.  AIMessage = agent's own subsystem reporting
            # back, won't trigger the model to "respond" to itself.
            from langchain_core.messages import AIMessage
            status_label = "completed" if status == "completed" else "failed"
            truncated = text[:4000]
            if len(text) > 4000:
                truncated += "\n... (truncated)"

            # Include original objective so agent can cross-validate on next turn
            task_obj = _bg_task_store.load(task_id)
            objective = ""
            if task_obj:
                objective = task_obj.prompt or task_obj.command or ""

            context_parts = [
                f"[Background task {task_id} {status_label}]",
            ]
            if objective:
                context_parts.append(f"Objective: {objective}")
            context_parts.append(f"Result:\n{truncated}")

            try:
                await app.aupdate_state(config, {
                    "messages": [AIMessage(content="\n".join(context_parts))]
                })
            except Exception:
                pass  # best-effort — result is already shown to user

        _worker_pool = BackgroundWorkerPool(
            store=_bg_task_store,
            on_complete=_deliver_bg,
            max_concurrent=BG_MAX_CONCURRENT,
        )
        set_worker_pool(_worker_pool)

        # --- Swarm ---
        swarm_runner = None
        from octo.config import SWARM_ENABLED
        if SWARM_ENABLED:
            import socket
            from octo.config import SWARM_NAME, SWARM_PORT, SWARM_CAPABILITIES, SWARM_DIR
            from octo.swarm import set_swarm_runner
            from octo.swarm.runner import SwarmRunner

            SWARM_DIR.mkdir(parents=True, exist_ok=True)
            _swarm_name = SWARM_NAME or (socket.gethostname().split(".")[0] + "-octo")

            swarm_runner = SwarmRunner(
                instance_name=_swarm_name,
                port=SWARM_PORT,
                capabilities=SWARM_CAPABILITIES,
                swarm_dir=SWARM_DIR,
                graph_app=app,
                graph_lock=graph_lock,
                main_loop=asyncio.get_running_loop(),
                worker_pool=_worker_pool,
                task_store=_bg_task_store,
            )
            swarm_runner.start()
            set_swarm_runner(swarm_runner)
            ui.print_status(f"Swarm active: {_swarm_name} on port {SWARM_PORT}", "green")

        hb_interval_display = HEARTBEAT_INTERVAL_SECONDS // 60
        ui.print_status(
            f"Heartbeat active (every {hb_interval_display}m, "
            f"{HEARTBEAT_ACTIVE_START_TIME.strftime('%H:%M')}-"
            f"{HEARTBEAT_ACTIVE_END_TIME.strftime('%H:%M')})", "green"
        )
        cron_jobs = cron_store.load()
        if cron_jobs:
            ui.print_status(f"Cron scheduler active ({len(cron_jobs)} jobs)", "green")

        # --- Virtual Persona poller ---
        vp_poller: Any = None
        from octo.config import VP_ENABLED, VP_POLL_INTERVAL_SECONDS, VP_ACTIVE_START_TIME, VP_ACTIVE_END_TIME, VP_SELF_EMAILS
        if VP_ENABLED:
            from octo.virtual_persona.graph import build_vp_graph
            from octo.virtual_persona.poller import VPPoller, set_self_emails
            if VP_SELF_EMAILS:
                set_self_emails(VP_SELF_EMAILS)

            async def _deliver_escalation(
                text: str, teams_chat_id: str = "", teams_message_id: str = "",
            ) -> None:
                ui.print_response(text, source="Octi (VP Escalation)")
                if tg:
                    await tg.send_vp_notification(text, teams_chat_id, teams_message_id)
                elif not tg:
                    # Console-only fallback
                    pass

            vp_graph = build_vp_graph()
            vp_poller = VPPoller(
                vp_graph=vp_graph,
                octo_app=app,
                graph_lock=graph_lock,
                interval=VP_POLL_INTERVAL_SECONDS,
                active_start=VP_ACTIVE_START_TIME,
                active_end=VP_ACTIVE_END_TIME,
                on_escalation=_deliver_escalation,
                octo_config=config,
            )
            vp_poller.start()
            vp_display = f"{VP_POLL_INTERVAL_SECONDS // 60}m" if VP_POLL_INTERVAL_SECONDS >= 60 else f"{VP_POLL_INTERVAL_SECONDS}s"
            ui.print_status(
                f"VP poller active (every {vp_display}, "
                f"{VP_ACTIVE_START_TIME.strftime('%H:%M')}-"
                f"{VP_ACTIVE_END_TIME.strftime('%H:%M')})", "green"
            )

        async def _rebuild_graph():
            nonlocal app, agent_configs, skills, mcp_tools, mcp_tools_by_server
            # Stop background workers BEFORE closing MCP sessions — workers may
            # be mid-execution using MCP tools; tearing down sessions first causes
            # anyio cancel scope propagation that crashes the event loop.
            await _worker_pool.shutdown()
            # Close MCP sessions directly — do NOT use asyncio.shield() here:
            # shield creates an inner task, but anyio requires cancel scopes
            # to exit in the same task that entered them.  Calling close_all()
            # inline keeps everything in the main task.
            await session_pool.close_all()
            mcp_tools, mcp_tools_by_server = await _load_mcp_servers()
            app, agent_configs, skills = await build_graph(mcp_tools, mcp_tools_by_server)
            # Update proactive runners with new graph
            heartbeat._app = app
            cron_scheduler._app = app
            if vp_poller is not None:
                vp_poller._octo_app = app
            if swarm_runner is not None:
                swarm_runner.update_graph(app)
            if tg is not None:
                tg.graph_app = app
            # Refresh tab-completion with new skill and agent names
            ui.setup_input(
                _BASE_SLASH_CMDS
                + [f"/{s.name}" for s in skills]
                + [f"/{a.name}" for a in agent_configs]
            )

        async def _graceful_restart(label: str = "Restarting") -> None:
            """Gracefully shut down all subsystems and exec a fresh process.

            Session continuity is preserved via --thread flag. The new process
            picks up the same conversation from the checkpointer — identical to
            hot reload from the agent's perspective, but clean and reliable.
            """
            import sys as _sys
            import os as _os
            save_session(thread_id, model=active_model)
            ui.print_info(f"{label} Octo (session {thread_id})...")
            await heartbeat.stop()
            await cron_scheduler.stop()
            await _worker_pool.shutdown()
            if swarm_runner is not None:
                await swarm_runner.stop()
            if vp_poller is not None:
                vp_poller.stop()
            if tg:
                await tg.stop()
            await session_pool.close_all()
            restart_args = [_sys.executable, "-m", "octo", "chat",
                            "--thread", thread_id]
            if verbose:
                restart_args.append("--verbose")
            if debug:
                restart_args.append("--debug")
            if voice_mod.is_enabled():
                restart_args.append("--voice")
            if no_telegram:
                restart_args.append("--no-telegram")
            _os.execv(_sys.executable, restart_args)

        async def _tg_command_handler(cmd: str, args: str) -> str | None:
            """Handle slash commands from Telegram. Returns response text or None to pass through."""
            nonlocal thread_id

            if cmd == "help":
                return (
                    "**Telegram commands:**\n"
                    "`/clear` — Reset conversation\n"
                    "`/compact` — Free context space\n"
                    "`/context` — Context window usage\n"
                    "`/agents` — List agents\n"
                    "`/skills` — List skills\n"
                    "`/skills search <q>` — Search skill registry\n"
                    "`/skills install <name>` — Install skill\n"
                    "`/skills remove <name>` — Remove skill\n"
                    "`/mcp` — MCP server status\n"
                    "`/mcp find <q>` — Search MCP registry\n"
                    "`/mcp remove <name>` — Remove MCP server\n"
                    "`/mcp disable <name>` — Disable MCP server\n"
                    "`/mcp enable <name>` — Enable MCP server\n"
                    "`/model` — Show current model\n"
                    "`/reload` — Restart with session restore (re-reads everything)\n"
                    "`/restart` — Same as /reload\n"
                    "`/help` — This message\n\n"
                    "Unrecognized `/commands` pass through to the AI."
                )

            # ── Session management ───────────────────────────────────────

            if cmd == "clear":
                thread_id = str(uuid.uuid4())[:8]
                config["configurable"]["thread_id"] = thread_id
                cli_callback.reset_step_counter()
                save_session(thread_id, model=active_model)
                if tg is not None:
                    tg.thread_id = thread_id
                return f"Conversation cleared. New thread: `{thread_id}`"

            if cmd == "compact":
                from langchain_core.messages import RemoveMessage, SystemMessage
                from langchain_core.messages.utils import count_tokens_approximately
                from octo.retry import _sanitize_compact_boundary, _dump_tool_messages

                state = await app.aget_state(config)
                messages = state.values.get("messages", [])
                if len(messages) < 6:
                    return "Conversation too short to compact."

                before_tokens = count_tokens_approximately(messages)
                keep_count = max(4, len(messages) // 3)
                split_idx = _sanitize_compact_boundary(messages, len(messages) - keep_count)
                old_msgs = messages[:split_idx]
                recent_msgs = messages[split_idx:]

                if not old_msgs:
                    return "Nothing to compact."

                removable = [m for m in old_msgs if getattr(m, "id", None)]
                if not removable:
                    return "No removable messages found."

                _dump_tool_messages(removable, label="compact")

                summary_model = make_model(tier="low")
                summary_lines = []
                for m in old_msgs:
                    role = getattr(m, "type", "unknown")
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    if content.strip():
                        summary_lines.append(f"[{role}]: {content[:300]}")
                summary_prompt = (
                    "Summarize this conversation concisely in 2-3 paragraphs. "
                    "Focus on: decisions made, tasks completed, and current objectives.\n\n"
                    + "\n".join(summary_lines[-100:])
                )
                summary_response = await summary_model.ainvoke(summary_prompt)
                summary_msg = SystemMessage(
                    content=(
                        "[Conversation summary — earlier messages were compacted]\n\n"
                        + summary_response.content
                    )
                )

                remove_ops = [RemoveMessage(id=m.id) for m in removable]
                await app.aupdate_state(config, {"messages": remove_ops + [summary_msg]})

                after_tokens = count_tokens_approximately([summary_msg] + recent_msgs)
                return (
                    f"Compacted: {before_tokens:,} → {after_tokens:,} tokens "
                    f"({len(messages)} → {len(recent_msgs) + 1} messages, "
                    f"{len(removable)} removed)"
                )

            if cmd == "context":
                from octo.graph import context_info
                used = context_info.get("used", 0)
                limit = context_info.get("limit", 200_000)
                pct = (used / limit * 100) if limit > 0 else 0
                if pct < 30:
                    label = "PEAK"
                elif pct < 50:
                    label = "GOOD"
                elif pct < 70:
                    label = "DEGRADING"
                else:
                    label = "POOR"
                bar_len = 20
                filled = int(bar_len * pct / 100)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                return f"`{bar}` {pct:.0f}% ({used:,}/{limit:,} tokens) **{label}**"

            # ── Info ─────────────────────────────────────────────────────

            if cmd == "agents":
                if not agent_configs:
                    return "No agents loaded."
                lines = [f"• `{a.name}` — {a.description or '(no description)'}" for a in agent_configs]
                return "**Agents:**\n" + "\n".join(lines)

            if cmd == "model":
                return f"Current model: `{active_model}`"

            # ── Skills management ────────────────────────────────────────

            if cmd == "skills":
                sub = args.split(maxsplit=1)[0] if args else ""
                sub_arg = args.split(maxsplit=1)[1].strip() if args and " " in args else ""

                if not sub:
                    if not skills:
                        return "No skills loaded."
                    lines = [f"• `{s.name}` — {s.description or '(no description)'}" for s in skills]
                    return "**Skills:**\n" + "\n".join(lines)

                if sub == "search":
                    if not sub_arg:
                        return "Usage: `/skills search <query>`"
                    from octo.skills_cli import _fetch_registry
                    registry = _fetch_registry()
                    results = []
                    q = sub_arg.lower()
                    for entry in registry:
                        if (q in entry["name"].lower()
                                or q in entry.get("description", "").lower()
                                or any(q in t.lower() for t in entry.get("tags", []))):
                            results.append(entry)
                    if not results:
                        return f"No skills found for '{sub_arg}'."
                    lines = [
                        f"• `{e['name']}` v{e.get('version', '?')} — {e.get('description', '')[:60]}"
                        for e in results[:15]
                    ]
                    return f"**{len(results)} skill(s) found:**\n" + "\n".join(lines)

                if sub == "install":
                    if not sub_arg:
                        return "Usage: `/skills install <name>`"
                    from octo.skills_cli import (
                        _download_skill_files, _fetch_registry, _find_in_registry,
                        _install_deps,
                    )
                    from octo.config import SKILLS_DIR
                    import shutil
                    skill_name = sub_arg.split()[0]
                    registry = _fetch_registry()
                    entry = _find_in_registry(registry, skill_name)
                    if not entry:
                        return f"Skill '{skill_name}' not found. Try: `/skills search`"
                    dest = SKILLS_DIR / skill_name
                    if dest.exists():
                        shutil.rmtree(dest)
                    files = entry.get("files", ["SKILL.md"])
                    _download_skill_files(skill_name, files)
                    _install_deps(skill_name)
                    await _rebuild_graph()
                    return (
                        f"Installed `{skill_name}` v{entry.get('version', '?')}. "
                        f"{len(skills)} skill(s) loaded."
                    )

                if sub == "remove":
                    if not sub_arg:
                        return "Usage: `/skills remove <name>`"
                    from octo.config import SKILLS_DIR
                    import shutil
                    skill_name = sub_arg.split()[0]
                    dest = SKILLS_DIR / skill_name
                    if not dest.is_dir():
                        return f"Skill '{skill_name}' is not installed."
                    shutil.rmtree(dest)
                    await _rebuild_graph()
                    return f"Removed `{skill_name}`. {len(skills)} skill(s) loaded."

                return "Usage: `/skills [search <q>|install <name>|remove <name>]`"

            # ── MCP management ───────────────────────────────────────────

            if cmd == "mcp":
                from octo.mcp_manager import (
                    mcp_disable, mcp_enable, mcp_get_status,
                    mcp_registry_search, mcp_remove,
                )
                sub = args.split(maxsplit=1)[0] if args else ""
                sub_arg = args.split(maxsplit=1)[1].strip() if args and " " in args else ""

                if not sub:
                    statuses = mcp_get_status(mcp_tools_by_server)
                    if not statuses:
                        return "No MCP servers configured."
                    lines = []
                    for s in statuses:
                        status = "disabled" if s.get("disabled") else f"{s.get('tools', 0)} tools"
                        lines.append(f"• `{s['name']}` — {status}")
                    return "**MCP servers:**\n" + "\n".join(lines)

                if sub == "find":
                    if not sub_arg:
                        return "Usage: `/mcp find <query>`"
                    results = mcp_registry_search(sub_arg)
                    if not results:
                        return f"No MCP servers found for '{sub_arg}'."
                    lines = [
                        f"• `{r.get('name', '?')}` — {r.get('description', '')[:60]}"
                        for r in results[:10]
                    ]
                    return f"**{len(results)} server(s) found:**\n" + "\n".join(lines)

                if sub == "remove":
                    if not sub_arg:
                        return "Usage: `/mcp remove <name>`"
                    if mcp_remove(sub_arg):
                        await _rebuild_graph()
                        return f"Removed `{sub_arg}`. {len(mcp_tools_by_server)} server(s) remaining."
                    return f"Server '{sub_arg}' not found."

                if sub == "disable":
                    if not sub_arg:
                        return "Usage: `/mcp disable <name>`"
                    if mcp_disable(sub_arg):
                        await _rebuild_graph()
                        return f"Disabled `{sub_arg}`."
                    return f"Server '{sub_arg}' not found."

                if sub == "enable":
                    if not sub_arg:
                        return "Usage: `/mcp enable <name>`"
                    if mcp_enable(sub_arg):
                        await _rebuild_graph()
                        return f"Enabled `{sub_arg}`."
                    return f"Server '{sub_arg}' not found."

                if sub == "reload":
                    await _rebuild_graph()
                    return (
                        f"Reloaded: {len(mcp_tools)} tools from "
                        f"{len(mcp_tools_by_server)} server(s)"
                    )

                if sub == "install":
                    # Interactive wizard can't run over Telegram — route to AI
                    return None

                return "Usage: `/mcp [find <q>|remove <name>|disable <name>|enable <name>|reload]`"

            # ── System ───────────────────────────────────────────────────

            if cmd == "update":
                import sys as _sys, subprocess as _sp
                from pathlib import Path as _Path
                repo_dir = str(_Path(__file__).resolve().parent.parent)
                pull = _sp.run(
                    ["git", "pull", "--ff-only"],
                    cwd=repo_dir, capture_output=True, text=True, timeout=60,
                )
                if pull.returncode != 0:
                    return f"git pull failed:\n```\n{pull.stderr.strip()}\n```"
                install = _sp.run(
                    [_sys.executable, "-m", "pip", "install", "-e", repo_dir, "-q"],
                    capture_output=True, text=True, timeout=120,
                )
                if install.returncode != 0:
                    return f"pip install failed:\n```\n{install.stderr.strip()}\n```"
                await _graceful_restart(label="Updating")

            if cmd in ("reload", "restart"):
                await _graceful_restart(label="Reloading" if cmd == "reload" else "Restarting")

            # Unknown command — return None to pass through to graph as text
            return None

        # Wire up Telegram command handler
        if tg is not None:
            tg.on_command = _tg_command_handler

        async def _auto_handoff(app_ref, cfg, model_factory):
            """Save project state on exit using a cheap LLM summarization."""
            from octo.config import STATE_PATH
            from octo.graph import _todos

            try:
                state = await app_ref.aget_state(cfg)
                messages = state.values.get("messages", [])
                if len(messages) < 3:
                    return  # nothing worth saving

                # Gather recent conversation for context
                recent = messages[-20:]
                summary_lines = []
                for m in recent:
                    role = getattr(m, "type", "unknown")
                    content = m.content if isinstance(m.content, str) else str(m.content)
                    if content.strip():
                        summary_lines.append(f"[{role}]: {content[:300]}")

                # Gather active plan if any
                plan_text = ""
                if _todos:
                    completed = sum(1 for t in _todos if t.get("status") == "completed")
                    plan_text = f"\nActive plan: {completed}/{len(_todos)} tasks completed."

                prompt = (
                    "Based on this conversation extract a brief session handoff note. "
                    "Write 3-5 bullet points covering:\n"
                    "1. What was accomplished this session\n"
                    "2. Current position (what's in progress)\n"
                    "3. Next steps / what to do when resuming\n\n"
                    f"{plan_text}\n\n"
                    "Recent conversation:\n"
                    + "\n".join(summary_lines[-30:])
                )

                ui.print_info("Saving session state...")
                summary_model = model_factory(tier="low")
                response = await summary_model.ainvoke(prompt)

                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                content = (
                    f"# Project State\n\n_Last updated: {now}_\n\n"
                    f"## Session Handoff\n{response.content}\n"
                )
                STATE_PATH.write_text(content, encoding="utf-8")
                ui.print_info("Session state saved to STATE.md")
            except Exception as e:
                logger.debug("Auto-handoff failed: %s", e)

        try:
            while True:
                user_input = await ui.styled_input_async()

                if not user_input:
                    continue

                # --- Slash commands ---
                if user_input.lower() in ("exit", "quit"):
                    # Auto-handoff: save project state before exiting
                    await _auto_handoff(app, config, make_model)
                    break

                if user_input == "/help":
                    ui.print_help()
                    continue

                if user_input == "/clear":
                    thread_id = str(uuid.uuid4())[:8]
                    config["configurable"]["thread_id"] = thread_id
                    cli_callback.reset_step_counter()
                    save_session(thread_id, model=active_model)
                    if tg:
                        tg.thread_id = thread_id
                    ui.print_info(f"Conversation cleared. New thread: {thread_id}")
                    continue

                if user_input == "/agents":
                    ui.print_agents(agent_configs)
                    continue

                if user_input.startswith("/skills"):
                    parts = user_input.split(maxsplit=2)
                    sub = parts[1] if len(parts) > 1 else "list"
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    if sub == "list":
                        ui.print_skills(skills)
                    elif sub == "search":
                        from octo.skills_cli import _fetch_registry
                        query = arg
                        registry = _fetch_registry()
                        results = []
                        for entry in registry:
                            if query:
                                q = query.lower()
                                if not (q in entry["name"].lower()
                                        or q in entry.get("description", "").lower()
                                        or any(q in t.lower() for t in entry.get("tags", []))):
                                    continue
                            results.append(entry)
                        if not results:
                            ui.print_info("No skills found matching your query.")
                        else:
                            for entry in results:
                                tags = ", ".join(entry.get("tags", []))
                                ui.print_info(
                                    f"  {entry['name']:<20} v{entry.get('version', '?'):<8} "
                                    f"{tags[:25]:<26} {entry.get('description', '')[:40]}"
                                )
                            ui.print_info(f"{len(results)} skill(s) found.")
                    elif sub == "install" and arg:
                        from octo.skills_cli import (
                            _download_skill_files, _fetch_registry, _find_in_registry,
                            _install_deps,
                        )
                        skill_name = arg.split()[0]
                        registry = _fetch_registry()
                        entry = _find_in_registry(registry, skill_name)
                        if not entry:
                            ui.print_error(f"Skill '{skill_name}' not found. Try: /skills search")
                        else:
                            from octo.config import SKILLS_DIR
                            import shutil
                            dest = SKILLS_DIR / skill_name
                            if dest.exists():
                                shutil.rmtree(dest)
                            files = entry.get("files", ["SKILL.md"])
                            ui.print_info(f"Installing '{skill_name}' v{entry.get('version', '?')}...")
                            _download_skill_files(skill_name, files)
                            _install_deps(skill_name)
                            ui.print_info(f"Installed '{skill_name}'. Rebuilding graph...")
                            await _rebuild_graph()
                            ui.print_info(
                                f"Done. {len(skills)} skill(s) loaded."
                            )
                    elif sub == "remove" and arg:
                        from octo.config import SKILLS_DIR
                        import shutil
                        skill_name = arg.split()[0]
                        dest = SKILLS_DIR / skill_name
                        if not dest.is_dir():
                            ui.print_error(f"Skill '{skill_name}' is not installed.")
                        else:
                            shutil.rmtree(dest)
                            ui.print_info(f"Removed '{skill_name}'. Rebuilding graph...")
                            await _rebuild_graph()
                            ui.print_info(
                                f"Done. {len(skills)} skill(s) loaded."
                            )
                    elif sub == "import" and arg:
                        # /skills import <owner/repo> [skill-name]
                        # Installs globally via codex agent; CODEX_HOME points
                        # to .octo/ so skills land in .octo/skills/.
                        import asyncio as _aio
                        import shlex
                        from octo.skills_cli import parse_add_output, strip_ansi
                        tokens = shlex.split(arg.strip())
                        source = tokens[0]
                        cmd = ["npx", "skills", "add", source,
                               "-a", "codex", "-g", "-y"]
                        if len(tokens) > 1:
                            cmd.extend(["-s", *tokens[1:]])
                        ui.print_info(f"Importing from skills.sh: {source} ...")
                        proc = await _aio.create_subprocess_exec(
                            *cmd,
                            stdin=_aio.subprocess.DEVNULL,
                            stdout=_aio.subprocess.PIPE,
                            stderr=_aio.subprocess.STDOUT,
                        )
                        stdout, _ = await proc.communicate()
                        raw = stdout.decode(errors="replace").strip()
                        if proc.returncode == 0:
                            installed = parse_add_output(raw)
                            if installed:
                                ui.print_info(
                                    f"Installed {len(installed)} skill(s): "
                                    + ", ".join(installed)
                                )
                            else:
                                ui.print_info(strip_ansi(raw))
                            ui.print_info("Rebuilding graph...")
                            await _rebuild_graph()
                            ui.print_info(
                                f"Done. {len(skills)} skill(s) loaded."
                            )
                        else:
                            ui.print_error(
                                f"Import failed (exit {proc.returncode}). "
                                "Is Node.js installed?"
                            )
                    elif sub == "find":
                        import asyncio as _aio
                        from octo.skills_cli import (
                            parse_find_no_results, parse_find_output,
                        )
                        query = arg.strip()
                        if not query:
                            ui.print_error("Usage: /skills find <query>")
                        else:
                            ui.print_info(f"Searching skills.sh for '{query}'...")
                            proc = await _aio.create_subprocess_exec(
                                "npx", "skills", "find", query,
                                stdin=_aio.subprocess.DEVNULL,
                                stdout=_aio.subprocess.PIPE,
                                stderr=_aio.subprocess.STDOUT,
                            )
                            stdout, _ = await proc.communicate()
                            raw = stdout.decode(errors="replace").strip()
                            no_match = parse_find_no_results(raw)
                            if no_match:
                                ui.print_info(no_match)
                            else:
                                results = parse_find_output(raw)
                                if results:
                                    for r in results:
                                        ui.print_info(f"  {r['handle']}")
                                        if r["url"]:
                                            ui.print_info(f"    {r['url']}")
                                    ui.print_info(
                                        f"\n{len(results)} result(s). "
                                        "Install: /skills import <owner/repo> [skill-name]"
                                    )
                                else:
                                    ui.print_info("No results.")
                    else:
                        ui.print_error(
                            "Usage: /skills [list|search|install|remove|import|find]"
                        )
                    continue

                if user_input == "/tools":
                    ui.print_tools(mcp_tools_by_server)
                    continue

                if user_input.startswith("/mcp"):
                    from octo.mcp_manager import (
                        mcp_add_wizard, mcp_disable, mcp_enable, mcp_get_status,
                        mcp_install_wizard, mcp_registry_search, mcp_remove,
                    )
                    parts = user_input.split(maxsplit=2)
                    sub = parts[1] if len(parts) > 1 else ""
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    if sub == "":
                        ui.print_mcp_status(mcp_get_status(mcp_tools_by_server))
                    elif sub == "reload":
                        ui.print_info("Reloading MCP servers and skills...")
                        await _rebuild_graph()
                        ui.print_info(
                            f"Reloaded: {len(mcp_tools)} tools from "
                            f"{len(mcp_tools_by_server)} server(s), "
                            f"{len(skills)} skills"
                        )
                    elif sub == "find":
                        query = arg.strip()
                        if not query:
                            ui.print_error("Usage: /mcp find <query>")
                        else:
                            try:
                                ui.print_info(f"Searching MCP registry for '{query}'...")
                                results = mcp_registry_search(query)
                                if not results:
                                    ui.print_info(f"No MCP servers found for '{query}'.")
                                else:
                                    ui.print_mcp_search_results(results, query)
                            except Exception as e:
                                ui.print_error(f"Registry search failed: {e}")
                    elif sub == "install":
                        if not arg:
                            ui.print_error("Usage: /mcp install <server-name>")
                            ui.print_info("Tip: use /mcp find <query> to search first.")
                        else:
                            name = mcp_install_wizard(arg.strip())
                            if name:
                                ui.print_info("Reloading MCP servers...")
                                await _rebuild_graph()
                                ui.print_info(
                                    f"Reloaded: {len(mcp_tools)} tools from "
                                    f"{len(mcp_tools_by_server)} server(s)"
                                )
                    elif sub == "add":
                        name = mcp_add_wizard()
                        if name:
                            ui.print_info("Reloading MCP servers...")
                            await _rebuild_graph()
                            ui.print_info(
                                f"Reloaded: {len(mcp_tools)} tools from "
                                f"{len(mcp_tools_by_server)} server(s)"
                            )
                    elif sub == "disable" and arg:
                        if mcp_disable(arg):
                            await _rebuild_graph()
                    elif sub == "enable" and arg:
                        if mcp_enable(arg):
                            await _rebuild_graph()
                    elif sub == "remove" and arg:
                        if mcp_remove(arg):
                            await _rebuild_graph()
                    else:
                        ui.print_error(
                            "Usage: /mcp [find <query>|install <name>|add|remove <name>|"
                            "disable <name>|enable <name>|reload]"
                        )
                    continue

                if user_input.startswith("/call"):
                    # /call server_name tool_name {json_args}
                    # /call server_name tool_name  (no args)
                    # /call tool_name {json_args}  (search all servers)
                    # /call tool_name              (search all servers, no args)
                    import json as _json
                    raw = user_input[len("/call"):].strip()
                    if not raw:
                        ui.print_error("Usage: /call [server] <tool_name> [{\"arg\": \"value\"}]")
                        ui.print_info("Tip: use /tools to see available tool names")
                        continue

                    # Try to extract JSON args from the end
                    call_args: dict = {}
                    json_start = raw.find("{")
                    if json_start >= 0:
                        try:
                            call_args = _json.loads(raw[json_start:])
                            raw = raw[:json_start].strip()
                        except _json.JSONDecodeError as je:
                            ui.print_error(f"Invalid JSON args: {je}")
                            continue

                    tokens = raw.split()
                    tool = None
                    server_label = ""

                    if len(tokens) == 2:
                        # /call server tool
                        srv, tname = tokens
                        srv_tools = mcp_tools_by_server.get(srv, [])
                        tool = next((t for t in srv_tools if t.name == tname), None)
                        if not tool:
                            ui.print_error(f"Tool '{tname}' not found in server '{srv}'")
                            continue
                        server_label = srv
                    elif len(tokens) == 1:
                        # /call tool — search all servers
                        tname = tokens[0]
                        for srv, srv_tools in mcp_tools_by_server.items():
                            tool = next((t for t in srv_tools if t.name == tname), None)
                            if tool:
                                server_label = srv
                                break
                        if not tool:
                            ui.print_error(f"Tool '{tname}' not found in any server")
                            continue
                    else:
                        ui.print_error("Usage: /call [server] <tool_name> [{\"arg\": \"value\"}]")
                        continue

                    ui.print_info(f"Calling {server_label}/{tool.name}...")
                    try:
                        result = await tool.ainvoke(call_args)
                        ui.print_response(str(result), source=f"{server_label}/{tool.name}")
                    except Exception as e:
                        ui.print_error(f"Tool error: {e}")
                    continue

                if user_input.startswith("/projects"):
                    from octo.config import (
                        ProjectConfig, PROJECTS, PROJECTS_DIR,
                        _autodiscover_project_metadata, _project_to_dict,
                        reload_projects, save_project, remove_project,
                    )
                    parts = user_input.split(maxsplit=2)
                    sub = parts[1] if len(parts) > 1 else ""
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    if sub == "":
                        ui.print_projects()

                    elif sub == "show":
                        if not arg:
                            ui.print_error("Usage: /projects show <name>")
                        else:
                            ui.print_project_detail(arg)

                    elif sub == "create":
                        # Interactive create wizard
                        from prompt_toolkit import PromptSession as _WizPS
                        from octo.config import _validate_project_name
                        _wiz = _WizPS()
                        async def _ask(msg: str) -> str:
                            return (await _wiz.prompt_async(msg)).strip()
                        try:
                            ui.print_info("Creating a new project…")
                            name = (arg or await _ask("  Project name: "))
                            if not name:
                                ui.print_error("Name is required.")
                                continue
                            name_err = _validate_project_name(name)
                            if name_err:
                                ui.print_error(name_err)
                                continue
                            if name in PROJECTS:
                                ui.print_error(f"Project '{name}' already exists. Use /projects update {name}.")
                                continue
                            path = await _ask("  Project root path: ")
                            if not path:
                                ui.print_error("Path is required.")
                                continue
                            path = str(Path(path).expanduser().resolve())
                            if not Path(path).is_dir():
                                ui.print_error(f"Directory not found: {path}")
                                continue

                            # Auto-discover what we can
                            auto = _autodiscover_project_metadata(Path(path))
                            ui.print_info(f"Auto-discovered: {', '.join(auto.keys()) or 'nothing'}")

                            # Prompt for optional fields (pre-fill with autodiscovered)
                            description = (
                                await _ask(f"  Description [{auto.get('description', '')}]: ")
                            ) or auto.get("description", "")
                            repo_url = (
                                await _ask(f"  Repo URL [{auto.get('repo_url', '')}]: ")
                            ) or auto.get("repo_url", "")
                            issues_url = await _ask("  Issues URL (Jira/GitHub/Linear): ")
                            ci_url = await _ask("  CI URL: ")
                            docs_url = await _ask("  Docs URL: ")

                            # Config dir — try to detect .claude/
                            config_dir = ""
                            claude_dir = Path(path) / ".claude"
                            if claude_dir.is_dir():
                                config_dir = str(claude_dir)
                                ui.print_info(f"Found .claude/ config dir: {config_dir}")
                            else:
                                config_dir = await _ask("  Config dir (.claude/ path, or empty): ")

                            # Detect agents from config dir
                            agent_names: list[str] = []
                            if config_dir:
                                agents_dir = Path(config_dir) / "agents"
                                if agents_dir.is_dir():
                                    agent_names = [md.stem for md in sorted(agents_dir.glob("*.md"))]
                                    if agent_names:
                                        ui.print_info(f"Found agents: {', '.join(agent_names)}")

                            proj = ProjectConfig(
                                name=name,
                                path=path,
                                config_dir=config_dir,
                                env={"CLAUDE_CONFIG_DIR": config_dir} if config_dir else {},
                                agents=agent_names,
                                description=description,
                                repo_url=repo_url,
                                issues_url=issues_url,
                                tech_stack=auto.get("tech_stack", []),
                                default_branch=auto.get("default_branch", ""),
                                ci_url=ci_url,
                                docs_url=docs_url,
                            )
                            pf = save_project(proj)
                            ui.print_info(f"Project '{name}' created: {pf}")
                            ui.print_project_detail(name)
                        except (KeyboardInterrupt, EOFError):
                            ui.print_info("Cancelled.")

                    elif sub == "update":
                        if not arg:
                            ui.print_error("Usage: /projects update <name>")
                            continue
                        proj = PROJECTS.get(arg)
                        if not proj:
                            ui.print_error(f"Project '{arg}' not found.")
                            continue
                        from prompt_toolkit import PromptSession as _WizPS2
                        from dataclasses import replace as dc_replace
                        _wiz2 = _WizPS2()
                        async def _ask2(msg: str) -> str:
                            return (await _wiz2.prompt_async(msg)).strip()
                        try:
                            ui.print_info(f"Updating project '{arg}' (press Enter to keep current value)…")
                            # Work on a copy to avoid dirty state on cancel
                            edits: dict = {}
                            val = await _ask2(
                                f"  Description [{proj.description}]: "
                            )
                            if val:
                                edits["description"] = val
                            val = await _ask2(
                                f"  Repo URL [{proj.repo_url}]: "
                            )
                            if val:
                                edits["repo_url"] = val
                            val = await _ask2(
                                f"  Issues URL [{proj.issues_url}]: "
                            )
                            if val:
                                edits["issues_url"] = val
                            val = await _ask2(
                                f"  CI URL [{proj.ci_url}]: "
                            )
                            if val:
                                edits["ci_url"] = val
                            val = await _ask2(
                                f"  Docs URL [{proj.docs_url}]: "
                            )
                            if val:
                                edits["docs_url"] = val
                            val = await _ask2(
                                f"  Default branch [{proj.default_branch}]: "
                            )
                            if val:
                                edits["default_branch"] = val
                            tech_input = await _ask2(
                                f"  Tech stack (comma-sep) [{', '.join(proj.tech_stack)}]: "
                            )
                            if tech_input:
                                edits["tech_stack"] = [t.strip() for t in tech_input.split(",") if t.strip()]
                            if edits:
                                updated = dc_replace(proj, **edits)
                                save_project(updated)
                                ui.print_info(f"Project '{arg}' updated.")
                                ui.print_project_detail(arg)
                            else:
                                ui.print_info("No changes.")
                        except (KeyboardInterrupt, EOFError):
                            ui.print_info("Cancelled.")

                    elif sub == "remove":
                        if not arg:
                            ui.print_error("Usage: /projects remove <name>")
                        elif remove_project(arg):
                            ui.print_info(f"Project '{arg}' removed from registry.")
                        else:
                            ui.print_error(f"Project '{arg}' not found.")

                    elif sub == "reload":
                        reloaded = reload_projects()
                        ui.print_info(f"Reloaded {len(reloaded)} project(s) from disk.")
                        ui.print_projects()

                    else:
                        ui.print_error(
                            "Usage: /projects [show <name>|create [name]|update <name>|"
                            "remove <name>|reload]"
                        )
                    continue

                if user_input.startswith("/sessions"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        thread_id = parts[1].strip()
                        config["configurable"]["thread_id"] = thread_id
                        cli_callback.reset_step_counter()
                        save_session(thread_id, model=active_model)
                        if tg:
                            tg.thread_id = thread_id
                        ui.print_info(f"Switched to session: {thread_id}")
                    else:
                        ui.print_sessions(current_thread=thread_id)
                    continue

                if user_input == "/plan":
                    from octo.graph import _todos
                    ui.print_plan(_todos)
                    continue

                if user_input.startswith("/voice"):
                    arg = user_input.split(maxsplit=1)[1] if " " in user_input else ""
                    if arg == "on":
                        voice_mod.toggle_voice(True)
                    elif arg == "off":
                        voice_mod.toggle_voice(False)
                    else:
                        voice_mod.toggle_voice()
                    ui.print_info(f"Voice: {'on' if voice_mod.is_enabled() else 'off'}")
                    continue

                if user_input.startswith("/model"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        ui.print_info(f"Model switching requires restart. Set DEFAULT_MODEL={parts[1]} in .env")
                    else:
                        ui.print_info(f"Current model: {active_model}")
                    continue

                if user_input == "/compact":
                    try:
                        from langchain_core.messages import RemoveMessage, SystemMessage
                        from langchain_core.messages.utils import count_tokens_approximately

                        state = await app.aget_state(config)
                        messages = state.values.get("messages", [])
                        if len(messages) < 6:
                            ui.print_info("Conversation too short to compact.")
                            continue

                        before_tokens = count_tokens_approximately(messages)

                        # Keep last ~33% of messages by count (minimum 4),
                        # adjusting the split point to not orphan tool_call/ToolMessage pairs
                        from octo.retry import _sanitize_compact_boundary, _dump_tool_messages
                        keep_count = max(4, len(messages) // 3)
                        split_idx = _sanitize_compact_boundary(messages, len(messages) - keep_count)
                        old_msgs = messages[:split_idx]
                        recent_msgs = messages[split_idx:]

                        if not old_msgs:
                            ui.print_info("Nothing to compact.")
                            continue

                        # Count removable messages (those with valid IDs)
                        removable = [m for m in old_msgs if getattr(m, "id", None)]
                        if not removable:
                            ui.print_info("No removable messages found (missing IDs).")
                            continue

                        # Dump ToolMessages to disk before removing
                        _dump_tool_messages(removable, label="compact")

                        # Summarize old messages via low-tier LLM
                        summary_model = make_model(tier="low")
                        summary_lines = []
                        for m in old_msgs:
                            role = getattr(m, "type", "unknown")
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            if content.strip():
                                summary_lines.append(f"[{role}]: {content[:300]}")
                        summary_prompt = (
                            "Summarize this conversation concisely in 2-3 paragraphs. "
                            "Focus on: decisions made, tasks completed, and current objectives.\n\n"
                            + "\n".join(summary_lines[-100:])  # cap to avoid huge prompts
                        )
                        ui.print_info("Summarizing conversation...")
                        summary_response = await summary_model.ainvoke(summary_prompt)
                        summary_msg = SystemMessage(
                            content=(
                                "[Conversation summary — earlier messages were compacted]\n\n"
                                + summary_response.content
                            )
                        )

                        # Remove old messages and prepend summary
                        remove_ops = [RemoveMessage(id=m.id) for m in removable]
                        await app.aupdate_state(config, {"messages": remove_ops + [summary_msg]})

                        after_tokens = count_tokens_approximately([summary_msg] + recent_msgs)
                        ui.print_info(
                            f"Compacted: {before_tokens:,} -> {after_tokens:,} tokens "
                            f"({len(messages)} -> {len(recent_msgs) + 1} messages, "
                            f"{len(removable)} removed)"
                        )
                    except Exception as e:
                        ui.print_error(f"Compact failed: {e}")
                        if verbose or debug:
                            logger.exception("Compact error")
                    continue

                if user_input == "/context":
                    from octo.graph import context_info
                    used = context_info.get("used", 0)
                    limit = context_info.get("limit", 200_000)
                    pct = (used / limit * 100) if limit > 0 else 0
                    if pct < 30:
                        color, label = "green", "PEAK"
                    elif pct < 50:
                        color, label = "cyan", "GOOD"
                    elif pct < 70:
                        color, label = "yellow", "DEGRADING"
                    else:
                        color, label = "red", "POOR"
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    usage = cli_callback.get_context_usage()
                    ui.console.print(
                        f"  Context: [{color}]{bar}[/{color}] {pct:.0f}% "
                        f"({used:,}/{limit:,} tokens) [{color}]{label}[/{color}]"
                    )
                    ui.console.print(
                        f"  Session: {usage['total_input']:,} input + "
                        f"{usage['total_output']:,} output tokens"
                    )
                    continue

                if user_input.startswith("/profile"):
                    from octo.config import get_active_profile, set_active_profile, BUILTIN_PROFILES
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        name = parts[1].strip()
                        if set_active_profile(name):
                            ui.print_info(f"Switched to '{name}' profile. Takes full effect on restart.")
                        else:
                            available = ", ".join(BUILTIN_PROFILES.keys())
                            ui.print_error(f"Unknown profile '{name}'. Available: {available}")
                    else:
                        current = get_active_profile()
                        ui.print_info(f"Active profile: {current}")
                        for name, tiers in BUILTIN_PROFILES.items():
                            marker = " *" if name == current else "  "
                            ui.print_info(
                                f"{marker}{name}: supervisor={tiers['supervisor']}, "
                                f"workers={tiers['worker_default']}, "
                                f"high-tier={tiers['worker_high_keywords']}"
                            )
                    continue

                if user_input == "/state":
                    from octo.config import STATE_PATH
                    if STATE_PATH.is_file():
                        content = STATE_PATH.read_text(encoding="utf-8").strip()
                        if content:
                            ui.print_markdown(content)
                        else:
                            ui.print_info("STATE.md is empty. The agent will update it during work.")
                    else:
                        ui.print_info("No project state saved yet. STATE.md will be created during work.")
                    continue

                if user_input.startswith("/memory"):
                    from octo.config import MEMORY_DIR, PERSONA_DIR
                    parts = user_input.split(maxsplit=1)
                    sub = parts[1].strip() if len(parts) > 1 else ""

                    if sub == "long":
                        # Show long-term curated memory
                        ltm_path = PERSONA_DIR / "MEMORY.md"
                        if ltm_path.is_file():
                            ui.print_markdown(ltm_path.read_text(encoding="utf-8"))
                        else:
                            ui.print_info("No long-term memory yet.")
                    elif sub == "daily":
                        # Show recent daily logs
                        ui.print_daily_memories(MEMORY_DIR, days=7)
                    else:
                        # Default: show both
                        ltm_path = PERSONA_DIR / "MEMORY.md"
                        if ltm_path.is_file():
                            content = ltm_path.read_text(encoding="utf-8").strip()
                            if content:
                                ui.console.print("[bold cyan]Long-term Memory[/bold cyan]")
                                ui.print_markdown(content)
                        ui.print_daily_memories(MEMORY_DIR, days=5)
                    continue

                if user_input.startswith("/heartbeat"):
                    parts = user_input.split(maxsplit=1)
                    sub = parts[1].strip() if len(parts) > 1 else ""
                    if sub == "test":
                        ui.print_info("Forcing heartbeat tick...")
                        await heartbeat.force_tick()
                    else:
                        ui.print_info(
                            f"Heartbeat: every {heartbeat.interval // 60}m, "
                            f"active {heartbeat.active_start.strftime('%H:%M')}-"
                            f"{heartbeat.active_end.strftime('%H:%M')}"
                        )
                        instructions = heartbeat._load_heartbeat_instructions()
                        if instructions:
                            ui.print_info(f"HEARTBEAT.md: {len(instructions.splitlines())} active lines")
                        else:
                            ui.print_info("HEARTBEAT.md: empty (heartbeat will skip)")
                    continue

                if user_input.startswith("/cron"):
                    parts = user_input.split(maxsplit=2)
                    sub = parts[1] if len(parts) > 1 else "list"
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    if sub == "list":
                        jobs = cron_store.load()
                        if not jobs:
                            ui.print_info("No scheduled tasks. Use /cron add or ask me to schedule something.")
                        else:
                            ui.print_cron_jobs(jobs)
                    elif sub == "add":
                        if not arg:
                            ui.print_error("Usage: /cron add <at|every|cron> <spec> <task>")
                            ui.print_info("  /cron add at in_2h Remind me to review the PR")
                            ui.print_info("  /cron add every 30m Check for new emails")
                            ui.print_info("  /cron add cron '0 9 * * MON-FRI' Morning briefing")
                        else:
                            cron_parts = arg.split(maxsplit=2)
                            if len(cron_parts) < 3:
                                ui.print_error("Usage: /cron add <type> <spec> <task>")
                            else:
                                from octo.heartbeat import CronJob, CronJobType, _parse_at_time, _parse_interval_td, _next_cron_run
                                import uuid as _uuid
                                from datetime import datetime as _dt, timezone as _tz
                                cron_type_str, cron_spec, cron_task = cron_parts
                                try:
                                    cron_type = CronJobType(cron_type_str)
                                    now = _dt.now(_tz.utc)
                                    if cron_type == CronJobType.AT:
                                        next_run = _parse_at_time(cron_spec)
                                    elif cron_type == CronJobType.EVERY:
                                        delta = _parse_interval_td(cron_spec)
                                        next_run = now + delta
                                    elif cron_type == CronJobType.CRON:
                                        next_run = _next_cron_run(cron_spec, now)
                                    else:
                                        raise ValueError(f"Unknown type: {cron_type_str}")
                                    job = CronJob(
                                        id=str(_uuid.uuid4())[:8],
                                        task=cron_task,
                                        type=cron_type,
                                        spec=cron_spec,
                                        created_at=now.isoformat(),
                                        next_run=next_run.isoformat(),
                                    )
                                    cron_store.add(job)
                                    ui.print_info(
                                        f"Scheduled [{job.id}]: '{cron_task}' — "
                                        f"next run at {next_run.strftime('%Y-%m-%d %H:%M UTC')}"
                                    )
                                except (ValueError, KeyError) as e:
                                    ui.print_error(f"Invalid cron spec: {e}")
                    elif sub == "remove":
                        if not arg:
                            ui.print_error("Usage: /cron remove <job_id>")
                        elif cron_store.remove(arg):
                            ui.print_info(f"Removed cron job: {arg}")
                        else:
                            ui.print_error(f"Job not found: {arg}")
                    elif sub in ("pause", "resume"):
                        if not arg:
                            ui.print_error(f"Usage: /cron {sub} <job_id>")
                        else:
                            result = cron_store.toggle_pause(arg)
                            if result is None:
                                ui.print_error(f"Job not found: {arg}")
                            else:
                                state_str = "paused" if result else "resumed"
                                ui.print_info(f"Job {arg}: {state_str}")
                    else:
                        ui.print_error("Usage: /cron [list|add|remove|pause|resume]")
                    continue

                # ── Background tasks ───────────────────────────────
                if user_input.startswith("/bg "):
                    bg_cmd = user_input[4:].strip()
                    if not bg_cmd:
                        ui.print_error("Usage: /bg <command>")
                        continue
                    import uuid as _uuid
                    from datetime import datetime as _dt, timezone as _tz
                    from octo.background import BackgroundTask
                    _bg_task = BackgroundTask(
                        id=_uuid.uuid4().hex[:8],
                        type="process",
                        status="pending",
                        created_at=_dt.now(_tz.utc).isoformat(),
                        command=bg_cmd,
                    )
                    await _worker_pool.dispatch(_bg_task)
                    ui.print_info(f"Background task dispatched: {_bg_task.id}")
                    continue

                if user_input == "/tasks":
                    _all_tasks = _worker_pool.list_tasks()
                    if not _all_tasks:
                        ui.print_info("No background tasks.")
                    else:
                        from rich.table import Table as RichTable
                        _tbl = RichTable(
                            title="Background Tasks",
                            show_header=True,
                            header_style="bold cyan",
                            border_style="dim",
                        )
                        _tbl.add_column("ID", style="cyan", width=10)
                        _tbl.add_column("Type", style="yellow", width=8)
                        _tbl.add_column("Status", width=12)
                        _tbl.add_column("Detail", style="dim", max_width=50)
                        _tbl.add_column("Created", style="dim", width=20)
                        _status_colors = {
                            "completed": "green",
                            "running": "yellow",
                            "failed": "red",
                            "paused": "magenta",
                            "pending": "dim",
                            "cancelled": "dim red",
                        }
                        for _t in _all_tasks[:20]:
                            _sc = _status_colors.get(_t.status, "white")
                            _detail = _t.command[:50] if _t.type == "process" else (_t.prompt[:50] if _t.prompt else "")
                            _tbl.add_row(
                                _t.id,
                                _t.type,
                                f"[{_sc}]{_t.status}[/{_sc}]",
                                _detail,
                                _t.created_at[:19].replace("T", " "),
                            )
                        from octo.ui import console as _ui_console
                        _ui_console.print(_tbl)
                    continue

                if user_input.startswith("/task "):
                    _parts = user_input.split(maxsplit=2)
                    if len(_parts) < 2:
                        ui.print_error("Usage: /task <id> [cancel|resume <answer>]")
                        continue
                    _tid = _parts[1]
                    _action = _parts[2] if len(_parts) > 2 else ""

                    if _action == "cancel":
                        if await _worker_pool.cancel_task(_tid):
                            ui.print_info(f"Task {_tid} cancelled.")
                        else:
                            ui.print_error(f"Task {_tid}: not found or not cancellable.")
                    elif _action.startswith("resume "):
                        _answer = _action[7:].strip()
                        if not _answer:
                            ui.print_error("Usage: /task <id> resume <your answer>")
                        elif await _worker_pool.resume_task(_tid, _answer):
                            ui.print_info(f"Task {_tid} resumed.")
                        else:
                            ui.print_error(f"Task {_tid}: not found or not paused.")
                    elif _action.startswith("followup "):
                        _instr = _action[9:].strip()
                        if not _instr:
                            ui.print_error("Usage: /task <id> followup <instruction>")
                        else:
                            _new = await _worker_pool.follow_up(_tid, _instr)
                            if _new:
                                ui.print_info(f"Follow-up task {_new.id} dispatched (based on {_tid}).")
                            else:
                                ui.print_error(f"Task {_tid} not found.")
                    else:
                        _tsk = _worker_pool.get_task(_tid)
                        if not _tsk:
                            ui.print_error(f"Task {_tid} not found.")
                        else:
                            _lines = [
                                f"**ID**: {_tsk.id}",
                                f"**Type**: {_tsk.type}",
                                f"**Status**: {_tsk.status}",
                                f"**Created**: {_tsk.created_at[:19]}",
                            ]
                            if _tsk.started_at:
                                _lines.append(f"**Started**: {_tsk.started_at[:19]}")
                            if _tsk.completed_at:
                                _lines.append(f"**Completed**: {_tsk.completed_at[:19]}")
                            if _tsk.type == "process":
                                _lines.append(f"**Command**: `{_tsk.command}`")
                            else:
                                _lines.append(f"**Prompt**: {_tsk.prompt}")
                            if _tsk.paused_question:
                                _lines.append(f"\n**Pending question**: {_tsk.paused_question}")
                            if _tsk.result:
                                _r = _tsk.result[:1000] + ("..." if len(_tsk.result) > 1000 else "")
                                _lines.append(f"\n**Result**:\n{_r}")
                            if _tsk.error:
                                _lines.append(f"\n**Error**: {_tsk.error}")
                            ui.print_info("\n".join(_lines))
                    continue

                if user_input.startswith("/swarm"):
                    from octo.config import SWARM_ENABLED as _sw_enabled
                    if not _sw_enabled:
                        ui.print_info("Swarm is disabled. Set SWARM_ENABLED=true in .env")
                        continue
                    from octo.swarm import get_swarm_runner
                    from octo.swarm.registry import PeerRegistry
                    from octo.config import SWARM_DIR as _sw_dir

                    _sw_parts = user_input.split(maxsplit=2)
                    _sw_sub = _sw_parts[1] if len(_sw_parts) > 1 else ""
                    _sw_arg = _sw_parts[2].strip() if len(_sw_parts) > 2 else ""
                    _sw_runner = get_swarm_runner()

                    if not _sw_sub:
                        # /swarm — show status
                        _sw_status = "running" if _sw_runner and _sw_runner.running else "stopped"
                        _sw_reg = PeerRegistry(_sw_dir)
                        _sw_peers = _sw_reg.load()
                        _sw_lines = [f"**Swarm**: {_sw_status}"]
                        if _sw_runner:
                            _sw_lines.append(f"  Name: `{_sw_runner.name}`, Port: `{_sw_runner.port}`")
                        if _sw_peers:
                            _sw_lines.append(f"  **Peers** ({len(_sw_peers)}):")
                            for _p in _sw_peers:
                                _sw_lines.append(f"    - `{_p.name}` ({_p.url}) [{_p.status}]")
                        else:
                            _sw_lines.append("  No peers configured. Add with: /swarm add <name> <url>")
                        ui.print_info("\n".join(_sw_lines))

                    elif _sw_sub == "peers":
                        _sw_reg = PeerRegistry(_sw_dir)
                        _sw_peers = _sw_reg.load()
                        if not _sw_peers:
                            ui.print_info("No peers configured.")
                        else:
                            for _p in _sw_peers:
                                ui.print_info(f"  {_p.name}: {_p.url} [{_p.status}]")

                    elif _sw_sub == "add" and _sw_arg:
                        _sw_tokens = _sw_arg.split(maxsplit=1)
                        if len(_sw_tokens) < 2:
                            ui.print_error("Usage: /swarm add <name> <url>")
                        else:
                            _sw_reg = PeerRegistry(_sw_dir)
                            _sw_reg.add_peer(_sw_tokens[0], _sw_tokens[1])
                            ui.print_info(f"Added peer '{_sw_tokens[0]}'. Rebuilding graph...")
                            await _rebuild_graph()
                            ui.print_info("Done. Peer tools should now be available.")

                    elif _sw_sub == "remove" and _sw_arg:
                        _sw_reg = PeerRegistry(_sw_dir)
                        if _sw_reg.remove_peer(_sw_arg):
                            ui.print_info(f"Removed peer '{_sw_arg}'. Rebuilding graph...")
                            await _rebuild_graph()
                        else:
                            ui.print_error(f"Peer '{_sw_arg}' not found.")

                    elif _sw_sub == "ping":
                        if _sw_runner:
                            ui.print_info("Checking peers...")
                            await _sw_runner._check_peers()
                            _sw_reg = PeerRegistry(_sw_dir)
                            _sw_peers = _sw_reg.load()
                            for _p in _sw_peers:
                                ui.print_info(f"  {_p.name}: {_p.status}")
                        else:
                            ui.print_info("Swarm not running.")
                    else:
                        ui.print_error("Usage: /swarm [peers|add <name> <url>|remove <name>|ping]")
                    continue

                if user_input.startswith("/vp"):
                    from octo.virtual_persona.commands import handle_vp_command
                    vp_args = user_input[3:].strip()
                    new_poller = await handle_vp_command(
                        vp_args,
                        vp_poller=vp_poller,
                        octo_app=app,
                        octo_config=config,
                        graph_lock=graph_lock,
                        telegram=tg,
                    )
                    if new_poller is not None:
                        vp_poller = new_poller
                    continue

                if user_input == "/create-agent":
                    from octo.agent_wizard import create_agent_wizard
                    result = await create_agent_wizard(mcp_tools_by_server)
                    if result:
                        ui.print_info(f"Agent '{result}' created. Rebuilding graph...")
                        await _rebuild_graph()
                        ui.print_info(f"Done. {len(agent_configs)} agent(s) loaded.")
                    continue

                if user_input == "/create-skill":
                    from octo.skill_wizard import create_skill_wizard
                    result = await create_skill_wizard()
                    if result:
                        ui.print_info(f"Skill '{result}' created. Rebuilding graph...")
                        await _rebuild_graph()
                        ui.print_info(f"Done. {len(skills)} skill(s) loaded.")
                    continue

                if user_input == "/update":
                    import sys as _sys, subprocess as _sp
                    from pathlib import Path as _Path
                    repo_dir = str(_Path(__file__).resolve().parent.parent)
                    ui.print_info(f"Pulling latest code from {repo_dir}...")
                    pull = _sp.run(
                        ["git", "pull", "--ff-only"],
                        cwd=repo_dir, capture_output=True, text=True, timeout=60,
                    )
                    if pull.returncode != 0:
                        ui.print_error(f"git pull failed:\n{pull.stderr.strip()}")
                        continue
                    ui.print_info(pull.stdout.strip())
                    ui.print_info("Installing updated package...")
                    install = _sp.run(
                        [_sys.executable, "-m", "pip", "install", "-e", repo_dir, "-q"],
                        capture_output=True, text=True, timeout=120,
                    )
                    if install.returncode != 0:
                        ui.print_error(f"pip install failed:\n{install.stderr.strip()}")
                        continue
                    ui.print_info("Update complete. Restarting...")
                    await _graceful_restart(label="Updating")

                if user_input in ("/reload", "/restart"):
                    label = "Reloading" if user_input == "/reload" else "Restarting"
                    await _graceful_restart(label=label)

                # Check for skill or agent invocation
                if user_input.startswith("/"):
                    cmd_name = user_input.split()[0][1:]  # strip /
                    skill = next((s for s in skills if s.name == cmd_name), None)
                    if skill:
                        args = user_input[len(cmd_name) + 2:].strip()
                        injected = f"[Skill: {skill.name}]\n\n{skill.body}"
                        if args:
                            injected += f"\n\nUser request: {args}"
                        # Progressive disclosure: list available references and scripts
                        if skill.references or skill.scripts:
                            injected += "\n\n---\n**Bundled resources** (use Read/Bash tools to access on demand):"
                            if skill.references:
                                injected += "\nReference docs:"
                                for ref in skill.references:
                                    injected += f"\n- `{skill.skill_dir}/{ref}`"
                            if skill.scripts:
                                injected += "\nScripts:"
                                for scr in skill.scripts:
                                    injected += f"\n- `{skill.skill_dir}/{scr}`"
                        user_input = injected
                    elif next((a for a in agent_configs if a.name == cmd_name), None):
                        # Direct agent invocation — route to specific agent
                        prompt = user_input[len(cmd_name) + 2:].strip()
                        if not prompt:
                            ui.print_error(f"Usage: /{cmd_name} <your request>")
                            continue
                        user_input = (
                            f"[IMPORTANT: Route this request directly to the "
                            f"'{cmd_name}' agent. Do not delegate to any other "
                            f"agent.]\n\n{prompt}"
                        )
                    else:
                        ui.print_error(f"Unknown command: {user_input}")
                        continue

                # --- Invoke graph ---
                # Merge Ctrl+V pasted attachments with any detected file paths
                from octo.attachments import process_user_input as _process_attachments
                from octo.attachments import process_pasted_attachments
                pasted = ui.get_pending_attachments()
                if pasted:
                    msg_content, uploaded = process_pasted_attachments(user_input, pasted)
                else:
                    msg_content, uploaded = _process_attachments(user_input)
                if uploaded:
                    names = [p.rsplit("/", 1)[-1] for p in uploaded]
                    ui.print_info(f"Attached: {', '.join(names)}")

                # Update session with latest user message preview
                preview_text = user_input if isinstance(msg_content, str) else user_input
                save_session(thread_id, preview=preview_text, model=active_model)

                try:
                    # Start thinking spinner — callback will stop it on first tool call
                    spinner_text = "[yellow]Thinking...[/yellow]"
                    if cli_callback.active_task:
                        spinner_text = f"[yellow]Working on: {cli_callback.active_task}[/yellow]"
                    status = ui.console.status(spinner_text, spinner="dots")
                    status.start()
                    cli_callback.status = status

                    try:
                        from octo.abort import esc_listener
                        from octo.retry import invoke_with_retry

                        async def _on_retry(msg: str, attempt: int) -> None:
                            try:
                                status.update(f"[yellow]{msg}[/yellow]")
                            except Exception:
                                pass

                        abort_event = asyncio.Event()
                        async with graph_lock:
                            invoke_task = asyncio.create_task(
                                invoke_with_retry(
                                    app,
                                    {"messages": [HumanMessage(content=msg_content)]},
                                    config,
                                    on_retry=_on_retry,
                                )
                            )
                            async with esc_listener(abort_event):
                                abort_waiter = asyncio.create_task(abort_event.wait())
                                done, _ = await asyncio.wait(
                                    {invoke_task, abort_waiter},
                                    return_when=asyncio.FIRST_COMPLETED,
                                )

                                if abort_event.is_set():
                                    invoke_task.cancel()
                                    try:
                                        await invoke_task
                                    except asyncio.CancelledError:
                                        pass
                                    result = None
                                else:
                                    abort_waiter.cancel()
                                    result = invoke_task.result()
                    finally:
                        # Ensure spinner is stopped
                        try:
                            status.stop()
                        except Exception:
                            pass

                    if result is None:
                        ui.print_info("Aborted.")
                    else:
                        # Extract last AI response
                        response_text = ""
                        for msg in reversed(result.get("messages", [])):
                            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                                response_text = msg.content
                                break

                        if response_text:
                            ui.print_response(response_text)

                            if voice_mod.is_enabled():
                                await voice_mod.speak(response_text)

                except KeyboardInterrupt:
                    ui.print_info("\nInterrupted. Type 'exit' to quit.")
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str or "timed out" in error_str:
                        ui.print_error("Request timed out. Try again — it may be a transient issue.")
                    elif "rate limit" in error_str or "throttling" in error_str:
                        ui.print_error("Rate limited. Wait a moment and try again.")
                    elif "too long" in error_str or "context length" in error_str:
                        ui.print_error("Context too long. Use /compact to free space, or /clear to reset.")
                    else:
                        ui.print_error(f"Error: {e}")
                    if verbose or debug:
                        logger.exception("Graph invocation error")

        finally:
            await heartbeat.stop()
            await cron_scheduler.stop()
            await _worker_pool.shutdown()
            if swarm_runner is not None:
                await swarm_runner.stop()
            if vp_poller is not None:
                vp_poller.stop()
            if tg:
                await tg.stop()
            # Flush Langfuse before exit to prevent hanging threads
            if LANGFUSE_ENABLED:
                try:
                    from langfuse import Langfuse
                    Langfuse().flush()
                    Langfuse().shutdown()
                except Exception:
                    pass
            ui.print_info("Goodbye!")

    finally:
        await session_pool.close_all()
