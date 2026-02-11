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
    from octo.graph import build_graph, read_todos, set_telegram_transport
    from octo.loaders.mcp_loader import create_mcp_client, filter_tools, get_mcp_configs, get_tool_filters, validate_tool_schemas
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

    # Load MCP servers + build graph (reloadable via /mcp reload)
    async def _load_mcp_servers():
        """Load MCP servers, return (flat_tools, tools_by_server)."""
        configs = get_mcp_configs()
        filters = get_tool_filters()
        tools_flat = []
        tools_map: dict[str, list] = {}
        for sname, scfg in configs.items():
            try:
                client = create_mcp_client({sname: scfg})
                stools = await client.get_tools()
                all_count = len(stools)
                stools = filter_tools(stools, sname, filters)
                stools = validate_tool_schemas(stools, sname)
                tools_map[sname] = stools
                if len(stools) < all_count:
                    ui.print_info(f"MCP '{sname}': {len(stools)}/{all_count} tools (filtered)")
                tools_flat.extend(stools)
            except Exception as e:
                detail = str(e)
                if hasattr(e, "exceptions"):
                    detail = "; ".join(str(sub) for sub in e.exceptions)
                ui.print_error(f"MCP server '{sname}': {detail}")
        return tools_flat, tools_map

    mcp_tools, mcp_tools_by_server = await _load_mcp_servers()

    try:
        # Build graph
        app, agent_configs, skills = await build_graph(mcp_tools)

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
        slash_cmds = ["/help", "/clear", "/compact", "/context", "/agents", "/skills", "/tools",
                      "/call", "/projects", "/sessions", "/plan", "/profile", "/voice", "/model",
                      "/mcp", "/cron", "/heartbeat", "exit", "quit"]
        slash_cmds += [f"/{s.name}" for s in skills]
        ui.setup_input(slash_cmds)

        # Callback handler — shared between CLI and Telegram so both show tool traces
        cli_callback = create_cli_callback(verbose=verbose or True, debug=debug)

        # Shared lock — prevents concurrent graph invocations from CLI, Telegram, heartbeat, and cron
        graph_lock = asyncio.Lock()

        # Telegram transport
        tg: TelegramTransport | None = None
        if not no_telegram and TELEGRAM_BOT_TOKEN:
            tg = TelegramTransport(
                graph_app=app,
                thread_id=thread_id,
                on_message=lambda text: ui.print_telegram_message(text),
                on_response=lambda text: ui.print_response(text, source="Octi"),
                callbacks=[cli_callback],
                graph_lock=graph_lock,
            )
            await tg.start()
            set_telegram_transport(tg)
            ui.print_status("Telegram bot connected", "green")
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [cli_callback],
        }

        # --- Proactive AI: heartbeat + cron ---
        from octo.config import (
            HEARTBEAT_PATH, HEARTBEAT_INTERVAL_SECONDS,
            HEARTBEAT_ACTIVE_START_TIME, HEARTBEAT_ACTIVE_END_TIME,
            CRON_PATH,
        )
        from octo.heartbeat import HeartbeatRunner, CronScheduler, CronStore, set_cron_store

        async def _deliver_proactive(text: str, source: str = "Heartbeat") -> None:
            ui.print_response(text, source=f"Octi ({source})")
            if tg:
                await tg.send_proactive(text, source=source)

        async def _deliver_cron(task: str, response: str) -> None:
            msg = f"**Scheduled task**: {task}\n\n{response}"
            await _deliver_proactive(msg, source="Cron")

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

        hb_interval_display = HEARTBEAT_INTERVAL_SECONDS // 60
        ui.print_status(
            f"Heartbeat active (every {hb_interval_display}m, "
            f"{HEARTBEAT_ACTIVE_START_TIME.strftime('%H:%M')}-"
            f"{HEARTBEAT_ACTIVE_END_TIME.strftime('%H:%M')})", "green"
        )
        cron_jobs = cron_store.load()
        if cron_jobs:
            ui.print_status(f"Cron scheduler active ({len(cron_jobs)} jobs)", "green")

        try:
            while True:
                user_input = await ui.styled_input_async()

                if not user_input:
                    continue

                # --- Slash commands ---
                if user_input.lower() in ("exit", "quit"):
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

                if user_input == "/skills":
                    ui.print_skills(skills)
                    continue

                if user_input == "/tools":
                    ui.print_tools(mcp_tools_by_server)
                    continue

                if user_input.startswith("/mcp"):
                    from octo.mcp_manager import (
                        mcp_add_wizard, mcp_disable, mcp_enable, mcp_get_status, mcp_remove,
                    )
                    parts = user_input.split(maxsplit=2)
                    sub = parts[1] if len(parts) > 1 else ""
                    arg = parts[2].strip() if len(parts) > 2 else ""

                    async def _rebuild_graph():
                        nonlocal app, agent_configs, skills, mcp_tools, mcp_tools_by_server
                        mcp_tools, mcp_tools_by_server = await _load_mcp_servers()
                        app, agent_configs, skills = await build_graph(mcp_tools)
                        # Update proactive runners with new graph
                        heartbeat._app = app
                        cron_scheduler._app = app

                    if sub == "":
                        ui.print_mcp_status(mcp_get_status(mcp_tools_by_server))
                    elif sub == "reload":
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
                            "Usage: /mcp [add|remove <name>|disable <name>|enable <name>|reload]"
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

                if user_input == "/projects":
                    ui.print_projects()
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

                        # Keep last ~33% of messages by count (minimum 4)
                        keep_count = max(4, len(messages) // 3)
                        old_msgs = messages[:-keep_count]
                        recent_msgs = messages[-keep_count:]

                        if not old_msgs:
                            ui.print_info("Nothing to compact.")
                            continue

                        # Count removable messages (those with valid IDs)
                        removable = [m for m in old_msgs if getattr(m, "id", None)]
                        if not removable:
                            ui.print_info("No removable messages found (missing IDs).")
                            continue

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

                # Check for skill invocation
                if user_input.startswith("/"):
                    skill_name = user_input.split()[0][1:]  # strip /
                    skill = next((s for s in skills if s.name == skill_name), None)
                    if skill:
                        args = user_input[len(skill_name) + 2:].strip()
                        injected = f"[Skill: {skill.name}]\n\n{skill.body}"
                        if args:
                            injected += f"\n\nUser request: {args}"
                        user_input = injected
                    else:
                        ui.print_error(f"Unknown command: {user_input}")
                        continue

                # --- Invoke graph ---
                # Update session with latest user message preview
                save_session(thread_id, preview=user_input, model=active_model)

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
                                    {"messages": [HumanMessage(content=user_input)]},
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
            if tg:
                await tg.stop()
            ui.print_info("Goodbye!")

    finally:
        pass
