"""VP slash-command handlers — /vp subcommand dispatch."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def handle_vp_command(
    args: str,
    *,
    vp_poller: Any = None,
    octo_app: Any = None,
    octo_config: dict[str, Any] | None = None,
    console: Any = None,
) -> None:
    """Dispatch /vp subcommands.

    Args:
        args: Everything after ``/vp `` (e.g. "enable", "allow foo@bar.com +10").
        vp_poller: Optional VPPoller instance (for enable/disable).
        octo_app: Optional Octo supervisor graph (for persona generation).
        octo_config: Optional Octo config dict (for persona generation).
        console: Rich console for output.
    """
    from octo.config import VP_DIR
    from octo import ui

    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""

    handlers = {
        "status": _cmd_status,
        "enable": _cmd_enable,
        "disable": _cmd_disable,
        "allow": _cmd_allow,
        "block": _cmd_block,
        "remove": _cmd_remove,
        "list": _cmd_list,
        "test": _cmd_test,
        "stats": _cmd_stats,
        "audit": _cmd_audit,
        "confidence": _cmd_confidence,
        "sync": _cmd_sync,
        "profile": _cmd_profile,
        "threads": _cmd_threads,
        "delegated": _cmd_delegated,
        "release": _cmd_release,
        "ignore": _cmd_ignore,
        "unignore": _cmd_unignore,
        "ignored": _cmd_ignored,
        "priority": _cmd_priority,
        "persona": _cmd_persona,
    }

    handler = handlers.get(subcmd)
    if handler is None:
        ui.print_error(f"Unknown /vp subcommand: {subcmd}")
        ui.print_info(f"Available: {', '.join(sorted(handlers))}")
        return

    await handler(rest, vp_dir=VP_DIR, vp_poller=vp_poller, octo_app=octo_app, octo_config=octo_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ac(vp_dir: Path):
    from octo.virtual_persona.access_control import AccessControl
    return AccessControl(vp_dir / "access-control.yaml")


def _get_stats(vp_dir: Path):
    from octo.virtual_persona.stats import VPStats
    return VPStats(vp_dir / "stats.json", vp_dir / "audit.jsonl")


def _get_profiles(vp_dir: Path):
    from octo.virtual_persona.profiles import PeopleProfiles
    return PeopleProfiles(vp_dir / "profiles.json")


def _get_knowledge(vp_dir: Path):
    from octo.virtual_persona.knowledge import ConversationKnowledge
    return ConversationKnowledge(vp_dir / "knowledge")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

async def _cmd_status(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ac = _get_ac(vp_dir)
    stats = _get_stats(vp_dir)
    s = stats.get_stats(days=1)

    enabled = ac.is_enabled()
    polling = vp_poller is not None and getattr(vp_poller, "_running", False)

    ui.print_status(f"VP enabled: {'yes' if enabled else 'no'}", "green" if enabled else "yellow")
    ui.print_status(f"Poller running: {'yes' if polling else 'no'}", "green" if polling else "dim")
    ui.print_info(f"Allow list: {len(ac.get_allow_list())} users")
    ui.print_info(f"Block list: {len(ac.get_block_list())} users")
    ui.print_info(f"Messages today: {s.get('total', 0)} (respond: {s.get('respond', 0)}, "
                  f"monitor: {s.get('monitor', 0)}, skip: {s.get('skip', 0)}, "
                  f"escalate: {s.get('escalate', 0)})")

    ignored = ac.get_ignored_chats()
    if ignored:
        ui.print_info(f"Ignored chats: {len(ignored)}")
    delegated = ac.get_delegated_threads()
    if delegated:
        ui.print_info(f"Delegated threads: {len(delegated)}")


async def _cmd_enable(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    _scaffold_data_files(vp_dir)
    ac = _get_ac(vp_dir)
    ac.set_enabled(True)

    if vp_poller is not None and hasattr(vp_poller, "start"):
        vp_poller.start()
        ui.print_status("VP enabled and poller started")
    else:
        ui.print_status("VP enabled (poller will start on next CLI restart)")


async def _cmd_disable(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ac = _get_ac(vp_dir)
    ac.set_enabled(False)

    if vp_poller is not None and hasattr(vp_poller, "stop"):
        vp_poller.stop()

    ui.print_status("VP disabled", "yellow")


async def _cmd_allow(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp allow <email> [+modifier] [notes]")
        return

    parts = rest.split()
    email = parts[0]
    modifier = 0
    notes_parts = []

    for p in parts[1:]:
        if p.startswith("+") or p.startswith("-"):
            try:
                modifier = int(p)
                continue
            except ValueError:
                pass
        notes_parts.append(p)

    notes = " ".join(notes_parts)
    ac = _get_ac(vp_dir)
    ac.add_user(email, "allow_ai", modifier=modifier, notes=notes)
    ui.print_status(f"Added {email} to allow_ai (modifier: {modifier:+d})")


async def _cmd_block(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp block <email> [urgent|normal] [notes]")
        return

    parts = rest.split()
    email = parts[0]
    priority = "normal"
    notes_parts = []

    for p in parts[1:]:
        if p.lower() in ("urgent", "normal", "low"):
            priority = p.lower()
        else:
            notes_parts.append(p)

    notes = " ".join(notes_parts)
    ac = _get_ac(vp_dir)
    ac.add_user(email, "always_user", priority=priority, notes=notes or "Blocked")
    ui.print_status(f"Added {email} to always_user (priority: {priority})")


async def _cmd_remove(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp remove <email>")
        return

    email = rest.strip().split()[0]
    ac = _get_ac(vp_dir)
    if ac.remove_user(email):
        ui.print_status(f"Removed {email} from all VP lists")
    else:
        ui.print_info(f"{email} not found in any list")


async def _cmd_list(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ac = _get_ac(vp_dir)

    allow = ac.get_allow_list()
    block = ac.get_block_list()

    if not allow and not block:
        ui.print_info("No users in VP lists. Use /vp allow or /vp block to add.")
        return

    if allow:
        ui.print_status(f"Allow AI ({len(allow)} users):", "green")
        for u in allow:
            mod = u.get("confidence_modifier", 0)
            mod_str = f" ({mod:+d})" if mod else ""
            notes = u.get("notes", "")
            notes_str = f" — {notes}" if notes else ""
            enabled = u.get("enabled", True)
            flag = "" if enabled else " [dim](disabled)[/dim]"
            ui.print_info(f"  {u.get('email', '?')}{mod_str}{notes_str}{flag}")

    if block:
        ui.print_status(f"Always User ({len(block)} users):", "yellow")
        for u in block:
            prio = u.get("notify_priority", "normal")
            reason = u.get("reason", "")
            ui.print_info(f"  {u.get('email', '?')} [{prio}] {reason}")


async def _cmd_test(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    parts = rest.split(None, 1)
    if not parts:
        ui.print_error("Usage: /vp test <email> [query]")
        return

    email = parts[0]
    query = parts[1] if len(parts) > 1 else ""

    ac = _get_ac(vp_dir)
    decision = ac.check_access(email)
    ui.print_status(f"Access: {decision.action} (modifier: {decision.confidence_modifier:+d})")

    if query:
        from octo.virtual_persona.confidence import calculate_confidence

        profiles = _get_profiles(vp_dir)
        profile = profiles.get_profile(email)
        knowledge = _get_knowledge(vp_dir)
        thread_ctx = None  # No chat_id for test

        result = await calculate_confidence(
            query=query,
            user_email=email,
            user_name=email.split("@")[0],
            user_profile=profile,
            thread_context=thread_ctx,
            confidence_modifier=decision.confidence_modifier,
        )
        ui.print_status(
            f"Confidence: {result.confidence:.0f}% → {result.decision}",
            "green" if result.decision == "respond"
            else "yellow" if result.decision == "disclaim"
            else "red"
        )
        ui.print_info(f"Category: {result.category}")
        if result.escalation_flags:
            ui.print_info(f"Flags: {', '.join(result.escalation_flags)}")
        ui.print_info(f"Reasoning: {result.reasoning}")


async def _cmd_stats(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    days = 7
    if rest.strip().isdigit():
        days = int(rest.strip())

    stats = _get_stats(vp_dir)
    s = stats.get_stats(days=days)

    ui.print_status(f"VP Stats (last {days} days):")
    ui.print_info(f"Total: {s.get('total', 0)}")
    ui.print_info(f"Respond: {s.get('respond', 0)}, Disclaim: {s.get('disclaim', 0)}, "
                  f"Monitor: {s.get('monitor', 0)}, Escalate: {s.get('escalate', 0)}, "
                  f"Skip: {s.get('skip', 0)}")
    ui.print_info(f"Avg confidence: {s.get('avg_confidence', 0)}%")
    ui.print_info(f"Escalation rate: {s.get('escalation_rate', 0):.0%}")

    by_user = s.get("by_user", {})
    if by_user:
        ui.print_info("Top users:")
        for email, count in list(by_user.items())[:5]:
            ui.print_info(f"  {email}: {count}")


async def _cmd_audit(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    n = 10
    if rest.strip().isdigit():
        n = int(rest.strip())

    stats = _get_stats(vp_dir)
    entries = stats.get_audit_log(n=n)

    if not entries:
        ui.print_info("No audit entries yet.")
        return

    ui.print_status(f"Last {len(entries)} VP audit entries:")
    for e in entries:
        ts = e.get("timestamp", "")[:19]
        decision = e.get("decision", "?")
        conf = e.get("confidence", 0)
        user = e.get("user_email", "?")
        preview = e.get("query_preview", "")[:60]
        color = {"respond": "green", "disclaim": "yellow", "escalate": "red", "skip": "dim"}.get(decision, "dim")
        ui.print_info(f"  {ts} [{color}]{decision}[/{color}] {conf:.0f}% {user} | {preview}")


async def _cmd_confidence(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    parts = rest.split()
    if len(parts) < 2:
        ui.print_error("Usage: /vp confidence <email> <modifier>")
        return

    email = parts[0]
    try:
        mod = int(parts[1])
    except ValueError:
        ui.print_error("Modifier must be an integer (e.g. +10, -5)")
        return

    ac = _get_ac(vp_dir)
    if ac.update_confidence_modifier(email, mod):
        ui.print_status(f"Updated {email} confidence modifier to {mod:+d}")
    else:
        ui.print_error(f"{email} not found in allow_ai list")


async def _cmd_sync(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ui.print_info("Syncing conversation knowledge... (this uses Teams MCP tools)")
    if vp_poller is not None and hasattr(vp_poller, "full_sync"):

        # Live progress callback
        _last_printed = [0]  # mutable ref for closure

        def _on_progress(synced: int, skipped: int, errors: int, total: int) -> None:
            done = synced + skipped + errors
            # Print every 5 chats processed (was 10 — too infrequent)
            if done - _last_printed[0] >= 5:
                parts = [f"{synced} synced"]
                if skipped:
                    parts.append(f"{skipped} skipped")
                if errors:
                    parts.append(f"{errors} errors")
                ui.print_info(f"  ... {done}/{total} processed ({', '.join(parts)})")
                _last_printed[0] = done

        result = await vp_poller.full_sync(on_progress=_on_progress)
        total = result.get("total", 0)
        synced = result.get("synced", 0)
        skipped = result.get("skipped", 0)
        ignored = result.get("ignored", 0)
        errors = result.get("errors", 0)

        if total == 0:
            ui.print_info("No Teams chats found. Check /mcp status for msteams server.")
        else:
            color = "green" if synced > 0 else "yellow"
            ui.print_status(f"Synced {synced}/{total} threads", color)
            if ignored:
                ui.print_info(f"  {ignored} ignored chats")
            if skipped:
                ui.print_info(f"  {skipped} inaccessible (legacy/bot — normal)")
            if errors:
                ui.print_info(f"  {errors} error(s) — check logs for details")
    else:
        ui.print_info("Full sync requires running poller. Use /vp enable first.")


async def _cmd_profile(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    parts = rest.split()
    if not parts:
        ui.print_error("Usage: /vp profile <email> [tone <preset>]")
        return

    email = parts[0]
    profiles = _get_profiles(vp_dir)

    # Check for tone subcommand
    if len(parts) >= 3 and parts[1].lower() == "tone":
        from octo.virtual_persona.profiles import TONE_PRESETS

        tone = parts[2]
        if profiles.set_tone(email, tone):
            ui.print_status(f"Set {email} tone to '{tone}'")
        else:
            ui.print_error(f"Invalid tone. Options: {', '.join(TONE_PRESETS)}")
        return

    profile = profiles.get_profile(email)
    if not profile:
        ui.print_info(f"No profile for {email}")
        return

    ui.print_status(f"Profile: {profile.get('name', email)}")
    if profile.get("title"):
        ui.print_info(f"  Title: {profile['title']}")
    if profile.get("department"):
        ui.print_info(f"  Dept: {profile['department']}")
    ui.print_info(f"  Tone: {profile.get('tone', 'default')}")
    ui.print_info(f"  Interactions: {profile.get('interaction_count', 0)}")
    if profile.get("last_interaction"):
        ui.print_info(f"  Last: {profile['last_interaction'][:19]}")
    topics = profile.get("topics", [])
    if topics:
        ui.print_info(f"  Topics: {', '.join(topics[:5])}")
    if profile.get("notes"):
        ui.print_info(f"  Notes: {profile['notes']}")


async def _cmd_threads(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    n = 10
    if rest.strip().isdigit():
        n = int(rest.strip())

    knowledge = _get_knowledge(vp_dir)
    threads = knowledge.list_threads(n=n)

    if not threads:
        ui.print_info("No classified threads yet. Run /vp sync first.")
        return

    ui.print_status(f"Recent threads ({len(threads)}):")
    for t in threads:
        topic = t.get("topic", "Unknown")
        count = t.get("message_count", 0)
        updated = (t.get("last_updated", ""))[:10]
        participants = t.get("participants", [])
        people = ", ".join(p.split("@")[0] for p in participants[:3]) if participants else ""
        ui.print_info(f"  [{updated}] {topic} ({count} msgs) {people}")


async def _cmd_delegated(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ac = _get_ac(vp_dir)
    delegated = ac.get_delegated_threads()

    if not delegated:
        ui.print_info("No delegated threads.")
        return

    ui.print_status(f"Delegated threads ({len(delegated)}):")
    for chat_id, info in delegated.items():
        ts = info.get("locked_at", "")[:19]
        user = info.get("user_email", "?")
        reason = info.get("reason", "")
        preview = info.get("query_preview", "")[:50]
        ui.print_info(f"  {chat_id} | {ts} | {user} | {reason}")
        if preview:
            ui.print_info(f"    {preview}")


async def _cmd_release(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp release <chat_id> or /vp release all")
        return

    ac = _get_ac(vp_dir)

    if rest.strip().lower() == "all":
        count = ac.release_all_threads()
        ui.print_status(f"Released {count} delegated threads")
    else:
        chat_id = rest.strip()
        if ac.release_thread(chat_id):
            ui.print_status(f"Released thread {chat_id}")
        else:
            ui.print_info(f"Thread {chat_id} was not delegated")


async def _cmd_ignore(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp ignore <chat_id> [label]")
        ui.print_info("Tip: use /vp sync first, then copy a chat ID from the logs.")
        return

    parts = rest.strip().split(None, 1)
    chat_id = parts[0]
    label = parts[1] if len(parts) > 1 else ""

    ac = _get_ac(vp_dir)
    if ac.is_ignored(chat_id):
        ui.print_info(f"Chat {chat_id[:30]}... already ignored")
        return

    ac.ignore_chat(chat_id, label=label)
    display = label or chat_id[:30]
    ui.print_status(f"Ignoring chat: {display}")


async def _cmd_unignore(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    if not rest:
        ui.print_error("Usage: /vp unignore <chat_id> or /vp unignore all")
        return

    ac = _get_ac(vp_dir)

    if rest.strip().lower() == "all":
        count = len(ac.get_ignored_chats())
        if count:
            ac._config["ignored_chats"] = []
            ac._save()
            ui.print_status(f"Removed {count} chats from ignore list")
        else:
            ui.print_info("No ignored chats")
    else:
        chat_id = rest.strip().split()[0]
        if ac.unignore_chat(chat_id):
            ui.print_status(f"Chat {chat_id[:30]}... removed from ignore list")
        else:
            ui.print_info(f"Chat {chat_id[:30]}... was not ignored")


async def _cmd_ignored(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    from octo import ui

    ac = _get_ac(vp_dir)
    ignored = ac.get_ignored_chats()

    if not ignored:
        ui.print_info("No ignored chats. Use /vp ignore <chat_id> to add.")
        return

    ui.print_status(f"Ignored chats ({len(ignored)}):")
    for entry in ignored:
        if isinstance(entry, dict):
            cid = entry.get("id", "?")
            label = entry.get("label", "")
            display = f"  {cid[:40]}{'...' if len(cid) > 40 else ''}"
            if label:
                display += f"  ({label})"
            ui.print_info(display)
        else:
            ui.print_info(f"  {entry}")


async def _cmd_priority(rest: str, *, vp_dir: Path, vp_poller: Any, **_kw: Any) -> None:
    """Manage the priority (never-ignore) user list.

    Usage:
        /vp priority                  — list priority users
        /vp priority add <email>      — add user
        /vp priority remove <email>   — remove user
    """
    from octo import ui

    ac = _get_ac(vp_dir)
    parts = rest.strip().split(None, 2)
    sub = parts[0].lower() if parts else "list"

    if sub == "add":
        if len(parts) < 2:
            ui.print_error("Usage: /vp priority add <email> [name]")
            return
        email = parts[1].lower()
        name = parts[2] if len(parts) > 2 else ""
        ac.add_priority_user(email, name=name)
        ui.print_status(f"Added {email} to priority list (never ignored)")
        return

    if sub == "remove":
        if len(parts) < 2:
            ui.print_error("Usage: /vp priority remove <email>")
            return
        email = parts[1].lower()
        if ac.remove_priority_user(email):
            ui.print_status(f"Removed {email} from priority list")
        else:
            ui.print_info(f"{email} was not in priority list")
        return

    # Default: list
    users = ac.get_priority_users()
    if not users:
        ui.print_info("No priority users. Use /vp priority add <email> to add.")
        return
    ui.print_status(f"Priority users ({len(users)}) — never ignored:")
    for entry in users:
        email = entry.get("email", "?") if isinstance(entry, dict) else entry
        name = entry.get("name", "") if isinstance(entry, dict) else ""
        display = f"  {email}"
        if name:
            display += f"  ({name})"
        ui.print_info(display)


async def _cmd_persona(rest: str, *, vp_dir: Path, vp_poller: Any, octo_app: Any = None, octo_config: dict[str, Any] | None = None, **_kw: Any) -> None:
    from octo import ui

    parts = rest.strip().split(None, 1)
    sub = parts[0].lower() if parts else "show"

    if sub == "generate":
        from octo.virtual_persona.persona import full_persona_pipeline

        name = parts[1].strip() if len(parts) > 1 else ""
        _scaffold_data_files(vp_dir)
        ui.print_status("Starting persona generation pipeline...")
        analysis, prompt = await full_persona_pipeline(
            vp_dir, user_name=name, octo_app=octo_app, octo_config=octo_config,
        )
        if prompt:
            ui.print_status("Persona generated! Review and edit at:")
            ui.print_info(f"  Prompt: {vp_dir / 'system-prompt.md'}")
            ui.print_info(f"  Analysis: {vp_dir / 'communication-analysis.json'}")
        elif analysis:
            ui.print_info("Analysis completed but prompt generation failed. Retry with /vp persona generate")

    elif sub == "show":
        prompt_path = vp_dir / "system-prompt.md"
        if prompt_path.is_file():
            text = prompt_path.read_text(encoding="utf-8")
            # Show first ~20 lines
            lines = text.strip().split("\n")
            ui.print_status("Current persona prompt:")
            for line in lines[:20]:
                ui.print_info(f"  {line}")
            if len(lines) > 20:
                ui.print_info(f"  ... ({len(lines) - 20} more lines)")
            ui.print_info(f"\n  File: {prompt_path}")
        else:
            ui.print_info("No persona prompt yet. Run /vp persona generate to create one.")

    elif sub == "analysis":
        analysis_path = vp_dir / "communication-analysis.json"
        if analysis_path.is_file():
            import json
            data = json.loads(analysis_path.read_text(encoding="utf-8"))
            ui.print_status("Communication analysis:")
            ui.print_info(f"  Tone: {data.get('tone', {}).get('formality', '?')} / "
                          f"{data.get('tone', {}).get('directness', '?')}")
            ui.print_info(f"  Language: {data.get('language', {}).get('primary', '?')} "
                          f"(code-switching: {data.get('language', {}).get('code_switching', '?')})")
            ui.print_info(f"  Style: {data.get('message_style', {}).get('typical_length', '?')} / "
                          f"{data.get('message_style', {}).get('structure', '?')}")
            traits = data.get("personality_traits", [])
            if traits:
                ui.print_info(f"  Traits: {', '.join(traits[:5])}")
            topics = data.get("expertise_topics", [])
            if topics:
                ui.print_info(f"  Topics: {', '.join(topics[:5])}")
            anti = data.get("anti_patterns", [])
            if anti:
                ui.print_info(f"  Anti-patterns: {', '.join(anti[:3])}")
            summary = data.get("summary", "")
            if summary:
                ui.print_info(f"  Summary: {summary}")
            ui.print_info(f"\n  File: {analysis_path}")
        else:
            ui.print_info("No analysis yet. Run /vp persona generate first.")

    elif sub == "path":
        ui.print_info(f"Prompt: {vp_dir / 'system-prompt.md'}")
        ui.print_info(f"Analysis: {vp_dir / 'communication-analysis.json'}")

    else:
        ui.print_error("Usage: /vp persona [generate [name]|show|analysis|path]")


# ---------------------------------------------------------------------------
# Data file scaffolding
# ---------------------------------------------------------------------------


def _scaffold_data_files(vp_dir: Path) -> None:
    """Create default data files if they don't exist."""
    vp_dir.mkdir(parents=True, exist_ok=True)

    # access-control.yaml
    ac_path = vp_dir / "access-control.yaml"
    if not ac_path.is_file():
        ac_path.write_text(
            "version: '1.0'\n"
            "enabled: true\n"
            "allow_ai:\n"
            "  users: []\n"
            "  channels: []\n"
            "  default_action: always_user\n"
            "always_user:\n"
            "  users: []\n"
            "  channels: []\n"
            "  notify_real_artem: true\n"
            "  auto_create_reminder: true\n"
            "audit_log:\n"
            "  enabled: true\n",
            encoding="utf-8",
        )

    # system-prompt.md
    prompt_path = vp_dir / "system-prompt.md"
    if not prompt_path.is_file():
        _write_default_prompt(prompt_path)

    # profiles.json
    profiles_path = vp_dir / "profiles.json"
    if not profiles_path.is_file():
        profiles_path.write_text("{}\n", encoding="utf-8")

    # knowledge dir
    knowledge_dir = vp_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    threads_path = knowledge_dir / "threads.json"
    if not threads_path.is_file():
        threads_path.write_text("{}\n", encoding="utf-8")

    # message-cache.json
    cache_path = vp_dir / "message-cache.json"
    if not cache_path.is_file():
        cache_path.write_text("{}\n", encoding="utf-8")

    # delegated.json
    delegated_path = vp_dir / "delegated.json"
    if not delegated_path.is_file():
        delegated_path.write_text("{}\n", encoding="utf-8")


def _write_default_prompt(path: Path) -> None:
    """Write a generic default VP system prompt.

    This is a placeholder — users should run `/vp persona generate` to create
    a personalized prompt based on their actual communication patterns.
    """
    prompt = """\
You are a virtual assistant representing a real person in their work chat.
Your goal is to respond in their authentic voice based on their communication style.

# IMPORTANT
This is a default placeholder prompt. Run `/vp persona generate` to create a \
personalized prompt based on your actual Teams conversations. The generator will \
analyze your messages and build a prompt that captures your real voice.

# GENERAL RULES
- Add 🤖 at end of responses (transparency marker)
- Match the language the sender uses
- Keep responses concise and direct
- Be technically accurate
- When uncertain, stay silent (escalation is invisible to the sender)

# ESCALATION (invisible to sender)
These topics should be routed to the real person silently:
- Personal decisions, scheduling, commitments
- Confidential or sensitive information
- Real-time status questions
- Topics outside known expertise
- Anything requiring personal judgment

# STYLE
- Direct and helpful
- No corporate speak
- Sound like a real person, not a chatbot
- Start with the answer, not preamble
"""
    path.write_text(prompt, encoding="utf-8")
