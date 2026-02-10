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
    from octo.graph import build_graph, read_todos
    from octo.loaders.mcp_loader import create_mcp_client, get_mcp_configs
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

    # Parse MCP config (names only, no connections yet)
    mcp_configs = get_mcp_configs()
    mcp_server_names = list(mcp_configs.keys())

    # Connect to MCP servers — try each individually so one failure doesn't block the rest
    mcp_tools = []
    if mcp_configs:
        for server_name, server_cfg in mcp_configs.items():
            try:
                client = create_mcp_client({server_name: server_cfg})
                tools = await client.get_tools()
                mcp_tools.extend(tools)
            except Exception as e:
                # Extract sub-exception details from ExceptionGroups
                detail = str(e)
                if hasattr(e, "exceptions"):
                    detail = "; ".join(str(sub) for sub in e.exceptions)
                ui.print_error(f"MCP server '{server_name}': {detail}")

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
            mcp_servers=mcp_server_names if mcp_tools else None,
        )

        # Voice
        if voice_on:
            voice_mod.toggle_voice(True)
            ui.print_status("Voice enabled", "green")

        # Setup input with slash command completion
        slash_cmds = ["/help", "/clear", "/compact", "/context", "/agents", "/skills",
                      "/projects", "/sessions", "/plan", "/profile", "/voice", "/model", "/thread", "exit", "quit"]
        slash_cmds += [f"/{s.name}" for s in skills]
        ui.setup_input(slash_cmds)

        # Callback handler — shared between CLI and Telegram so both show tool traces
        cli_callback = create_cli_callback(verbose=verbose or True, debug=debug)

        # Telegram transport
        tg: TelegramTransport | None = None
        if not no_telegram and TELEGRAM_BOT_TOKEN:
            tg = TelegramTransport(
                graph_app=app,
                thread_id=thread_id,
                on_message=lambda text: ui.print_telegram_message(text),
                on_response=lambda text: ui.print_response(text, source="Octi"),
                callbacks=[cli_callback],
            )
            await tg.start()
            ui.print_status("Telegram bot connected", "green")
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [cli_callback],
        }

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

                if user_input == "/projects":
                    ui.print_projects()
                    continue

                if user_input == "/sessions":
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

                if user_input.startswith("/thread"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        thread_id = parts[1]
                        config["configurable"]["thread_id"] = thread_id
                        save_session(thread_id, model=active_model)
                        if tg:
                            tg.thread_id = thread_id
                        ui.print_info(f"Switched to thread: {thread_id}")
                    else:
                        ui.print_info(f"Current thread: {thread_id}")
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
                        result = await app.ainvoke(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                        )
                    finally:
                        # Ensure spinner is stopped
                        try:
                            status.stop()
                        except Exception:
                            pass

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
                    ui.print_error(f"Error: {e}")
                    if verbose or debug:
                        logger.exception("Graph invocation error")

        finally:
            if tg:
                await tg.stop()
            ui.print_info("Goodbye!")

    finally:
        pass
