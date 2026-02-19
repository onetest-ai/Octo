"""Rich console UI — banner, styled input, help table, response rendering.

Styled after Alita CLI — bordered input box, markdown-aware responses,
two-column welcome banner, tab completion via prompt_toolkit.
"""
from __future__ import annotations

from typing import Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

import os
import signal

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from octo.loaders.agent_loader import AgentConfig
from octo.loaders.skill_loader import SkillConfig

# Remove stale COLUMNS/LINES from environ so Rich uses os.get_terminal_size()
# (ioctl) for live terminal dimensions.  These env vars are snapshots from
# process start and don't update on terminal resize.
os.environ.pop("COLUMNS", None)
os.environ.pop("LINES", None)

console = Console()


def _handle_winch(signum, frame):
    """Handle terminal resize — clear stale env vars and cached dimensions.

    Rich checks COLUMNS/LINES env vars *after* os.get_terminal_size(), and
    the env vars override the ioctl result. They're also checked in __init__
    where they can freeze _width/_height permanently. We clear both.
    """
    os.environ.pop("COLUMNS", None)
    os.environ.pop("LINES", None)
    # Reset any cached dimensions so Rich re-reads via ioctl
    console._width = None
    console._height = None


try:
    signal.signal(signal.SIGWINCH, _handle_winch)
except (OSError, ValueError):
    pass  # not on main thread or signal not available on this platform

# ── Context usage (updated by graph.py pre_model_hook) ───────────────
# Module-level ref — graph.py sets this dict; the toolbar reads it.
_context_ref: dict[str, int] | None = None


def set_context_ref(ref: dict[str, int]) -> None:
    """Register the context_info dict from graph.py for live toolbar display."""
    global _context_ref
    _context_ref = ref


# ── Logo ──────────────────────────────────────────────────────────────

LOGO = [
    " ██████╗  ██████╗████████╗ ██████╗ ",
    "██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗",
    "██║   ██║██║        ██║   ██║   ██║",
    "██║   ██║██║        ██║   ██║   ██║",
    "╚██████╔╝╚██████╗   ██║   ╚██████╔╝",
    " ╚═════╝  ╚═════╝   ╚═╝    ╚═════╝ ",
]


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("octo-agent")
    except Exception:
        return "0.1.x"


def _shorten_model(model: str) -> str:
    """Shorten Bedrock model IDs for display."""
    if "." in model:
        model = model.split(".")[-1]
    parts = model.split("-")
    for i, p in enumerate(parts):
        cleaned = p.split(":")[0]
        if len(cleaned) == 8 and cleaned.isdigit():
            return "-".join(parts[:i])
    return model


# ── Welcome banner ────────────────────────────────────────────────────

def print_welcome(
    *,
    model: str = "",
    provider: str = "",
    thread_id: str = "",
    agent_count: int = 0,
    skill_count: int = 0,
    mcp_tool_count: int = 0,
    mcp_servers: list[str] | None = None,
) -> None:
    """Print the two-column welcome screen with logo + session info + commands."""
    version = _get_version()

    # Left column: logo + version + session info
    left = Text()
    for line in LOGO:
        left.append(line + "\n", style="bold cyan")
    left.append("        CLI ", style="dim")
    left.append(f"v{version}\n\n", style="bold white")

    if provider and model:
        left.append("● ", style="bold green")
        left.append("Model: ", style="bold white")
        left.append(f"{_shorten_model(model)}", style="cyan")
        left.append(" | ", style="dim")
        left.append(f"{provider}\n", style="cyan")
    elif model:
        left.append("● ", style="bold green")
        left.append("Model: ", style="bold white")
        left.append(f"{_shorten_model(model)}\n", style="cyan")

    if agent_count:
        left.append("● ", style="bold green")
        left.append("Agents: ", style="bold white")
        left.append(f"{agent_count}\n", style="cyan")
    if skill_count:
        left.append("● ", style="bold green")
        left.append("Skills: ", style="bold white")
        left.append(f"{skill_count}\n", style="cyan")
    if mcp_servers:
        left.append("● ", style="bold green")
        left.append("MCP:    ", style="bold white")
        left.append(f"{mcp_tool_count} tools\n", style="cyan")
    if thread_id:
        left.append("● ", style="bold green")
        left.append("Thread: ", style="bold white")
        left.append(f"{thread_id}\n", style="cyan")

    # Right column: commands
    right = Text()
    right.append("\n")
    cmds = [
        ("/help", "Show all commands"),
        ("/agents", "List agents"),
        ("/skills", "Skills management"),
        ("/tools", "MCP tools"),
        ("/mcp", "MCP servers"),
        ("/create-agent", "Create agent"),
        ("/create-skill", "Create skill"),
        ("/plan", "Task plan"),
        ("/context", "Context usage"),
        ("/state", "Project state"),
        ("/memory", "View memories"),
        ("/profile", "Model profile"),
        ("/clear", "New conversation"),
        ("exit", "End session"),
    ]
    for cmd, desc in cmds:
        right.append(f"{cmd:<14}", style="bold yellow")
        right.append(f"{desc}\n", style="dim")

    columns = Columns([left, right], padding=(0, 4), expand=False)

    console.print()
    console.print(Panel(
        columns,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


# ── Status / info / error ────────────────────────────────────────────

def print_status(text: str, style: str = "green") -> None:
    """Print a status line with a colored bullet."""
    console.print(f"  [{style}]●[/{style}] {text}")


def print_info(text: str) -> None:
    console.print(f"  [dim]{text}[/dim]")


def print_error(text: str) -> None:
    console.print(f"  [bold red]✗ {text}[/bold red]")


# ── Response display ──────────────────────────────────────────────────

def print_response(text: str, source: str = "Octi") -> None:
    """Print an AI response with markdown rendering when appropriate."""
    console.print()
    console.print(f"[bold bright_cyan]{source}:[/bold bright_cyan]")
    console.print()
    if any(m in text for m in ["```", "**", "##", "- ", "* "]):
        console.print(Markdown(text))
    else:
        console.print(text)
    console.print()


def print_telegram_message(user_text: str, source: str = "Telegram") -> None:
    """Show an incoming Telegram message in the console (like a user message)."""
    console.print()
    console.print(f"  [bold blue]{source}[/bold blue] [dim]>[/dim] {user_text}")
    console.print()


def print_telegram_echo(user_text: str, response_text: str) -> None:
    """Show Telegram message activity in console (legacy callback)."""
    print_telegram_message(user_text)
    print_response(response_text, source="Octi")


def print_markdown(text: str) -> None:
    console.print(Markdown(text))


def print_daily_memories(memory_dir, days: int = 5) -> None:
    """Print recent daily memory log files."""
    from datetime import date, timedelta
    from pathlib import Path

    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        print_info("No daily memories found.")
        return

    today = date.today()
    found = False
    for i in range(days):
        d = today - timedelta(days=i)
        path = memory_dir / f"{d.isoformat()}.md"
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                if not found:
                    console.print("[bold cyan]Daily Memories[/bold cyan]")
                found = True
                console.print(Markdown(content))

    if not found:
        print_info(f"No daily memories in the last {days} days.")


# ── Agent / skill / help tables ──────────────────────────────────────

def print_agents(agents: Sequence[AgentConfig]) -> None:
    from octo.config import PROJECTS

    # Octo-native agents (LangGraph workers)
    octo = [a for a in agents if a.source_project == "octo"]
    # Project agents (reached via project workers → claude_code)
    project = [a for a in agents if a.source_project != "octo"]

    if octo:
        table = Table(
            title="Octo Agents",
            border_style="cyan",
            box=box.SIMPLE,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("Name", style="bold yellow", no_wrap=True)
        table.add_column("Type", style="magenta", no_wrap=True)
        table.add_column("Description", max_width=50)
        for a in octo:
            table.add_row(a.name, a.type or "agent", a.description[:50])
        console.print(table)

    if PROJECTS:
        table = Table(
            title="Project Workers",
            border_style="magenta",
            box=box.SIMPLE,
            header_style="bold magenta",
            padding=(0, 1),
        )
        table.add_column("Project", style="bold yellow", no_wrap=True)
        table.add_column("Path", style="dim", max_width=40)
        table.add_column("Sub-agents", max_width=45)
        for proj in PROJECTS.values():
            sub = [a.name for a in project if a.source_project == proj.name]
            sub_str = ", ".join(sub) if sub else "(general only)"
            table.add_row(proj.name, proj.path, sub_str)
        console.print(table)

    console.print()


def print_skills(skills: Sequence[SkillConfig]) -> None:
    table = Table(
        title="Skills",
        border_style="green",
        box=box.SIMPLE,
        header_style="bold green",
        padding=(0, 1),
    )
    table.add_column("Command", style="bold yellow", no_wrap=True)
    table.add_column("Source", style="dim", no_wrap=True)
    table.add_column("Description", max_width=55)
    for s in skills:
        table.add_row(f"/{s.name}", s.source, s.description[:55])
    console.print(table)
    console.print()


def print_tools(tools_by_server: dict[str, list]) -> None:
    """Print MCP tools grouped by server — useful for building include/exclude lists."""
    if not tools_by_server:
        print_info("No MCP tools loaded.")
        return

    total = sum(len(tools) for tools in tools_by_server.values())

    for server_name, tools in tools_by_server.items():
        table = Table(
            title=f"{server_name} ({len(tools)} tools)",
            border_style="blue",
            box=box.SIMPLE,
            header_style="bold blue",
            padding=(0, 1),
        )
        table.add_column("Tool name", style="bold yellow", no_wrap=True)
        table.add_column("Description", max_width=60)
        for t in sorted(tools, key=lambda x: x.name):
            desc = getattr(t, "description", "") or ""
            # First line only
            desc = desc.split("\n")[0][:60]
            table.add_row(t.name, desc)
        console.print(table)

    console.print(f"  [dim]{total} tools total across {len(tools_by_server)} server(s)[/dim]")
    console.print(f"  [dim]Tip: add \"include_tools\" or \"exclude_tools\" to .mcp.json to filter[/dim]")
    console.print()


def print_projects() -> None:
    """Print the project registry table."""
    from octo.config import PROJECTS

    if not PROJECTS:
        print_info("No projects registered. Use /projects create to add one.")
        return

    table = Table(
        title="Projects",
        border_style="magenta",
        box=box.SIMPLE,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Project", style="bold yellow", no_wrap=True)
    table.add_column("Description", max_width=40)
    table.add_column("Path", style="dim", max_width=40)
    table.add_column("Tech", style="cyan", max_width=20)
    table.add_column("Agents", max_width=30)

    for proj in PROJECTS.values():
        agents = ", ".join(proj.agents[:5])
        if len(proj.agents) > 5:
            agents += f" (+{len(proj.agents) - 5})"
        tech = ", ".join(proj.tech_stack[:4]) if proj.tech_stack else ""
        if len(proj.tech_stack) > 4:
            tech += " …"
        desc = (proj.description[:37] + "…") if len(proj.description) > 40 else proj.description
        table.add_row(proj.name, desc, proj.path, tech, agents or "(general)")
    console.print(table)
    console.print(f"  [dim]{len(PROJECTS)} project(s). Use /projects show <name> for details.[/dim]")
    console.print()


def print_project_detail(name: str) -> None:
    """Print detailed info for a single project."""
    from octo.config import PROJECTS

    proj = PROJECTS.get(name)
    if not proj:
        print_error(f"Project '{name}' not found. Use /projects to list.")
        return

    from rich.panel import Panel
    from rich.text import Text

    lines = Text()
    lines.append(f"Name:           ", style="bold")
    lines.append(f"{proj.name}\n")
    if proj.description:
        lines.append(f"Description:    ", style="bold")
        lines.append(f"{proj.description}\n")
    lines.append(f"Path:           ", style="bold")
    lines.append(f"{proj.path}\n")
    lines.append(f"Config dir:     ", style="bold")
    lines.append(f"{proj.config_dir}\n")
    if proj.default_branch:
        lines.append(f"Default branch: ", style="bold")
        lines.append(f"{proj.default_branch}\n")
    if proj.repo_url:
        lines.append(f"Repo URL:       ", style="bold")
        lines.append(f"{proj.repo_url}\n")
    if proj.issues_url:
        lines.append(f"Issues URL:     ", style="bold")
        lines.append(f"{proj.issues_url}\n")
    if proj.ci_url:
        lines.append(f"CI URL:         ", style="bold")
        lines.append(f"{proj.ci_url}\n")
    if proj.docs_url:
        lines.append(f"Docs URL:       ", style="bold")
        lines.append(f"{proj.docs_url}\n")
    if proj.tech_stack:
        lines.append(f"Tech stack:     ", style="bold")
        lines.append(f"{', '.join(proj.tech_stack)}\n")
    if proj.agents:
        lines.append(f"Agents:         ", style="bold")
        lines.append(f"{', '.join(proj.agents)}\n")
    if proj.tags:
        lines.append(f"Tags:           ", style="bold")
        lines.append(f"{', '.join(f'{k}={v}' for k, v in proj.tags.items())}\n")
    if proj.env:
        lines.append(f"Env overrides:  ", style="bold")
        lines.append(f"{', '.join(f'{k}={v}' for k, v in proj.env.items())}\n")

    console.print(Panel(lines, title=f"[bold magenta]Project: {proj.name}[/bold magenta]",
                        border_style="magenta"))
    console.print()


def print_sessions(current_thread: str = "") -> None:
    """Print recent sessions table."""
    from octo.sessions import list_sessions

    sessions = list_sessions()
    if not sessions:
        print_info("No saved sessions.")
        return

    table = Table(
        title="Sessions",
        border_style="blue",
        box=box.SIMPLE,
        header_style="bold blue",
        padding=(0, 1),
    )
    table.add_column("Thread", style="bold yellow", no_wrap=True)
    table.add_column("Updated", style="dim", no_wrap=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Preview", max_width=50)

    for s in sessions:
        tid = s.get("thread_id", "")
        marker = " *" if tid == current_thread else ""
        updated = s.get("updated_at", "")[:16].replace("T", " ")
        table.add_row(
            tid + marker,
            updated,
            _shorten_model(s.get("model", "")),
            s.get("preview", ""),
        )
    console.print(table)
    console.print()


def print_plan(todos: list[dict[str, str]]) -> None:
    """Print task plan with progress bar and completion stats."""
    if not todos:
        print_info("No active plan.")
        return

    total = len(todos)
    completed = sum(1 for t in todos if t.get("status") == "completed")
    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    pending = total - completed - in_progress
    pct = (completed / total * 100) if total > 0 else 0

    # Progress bar
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    color = "green" if pct == 100 else "yellow"

    console.print()
    console.print(f"  [bold cyan]Plan Progress[/bold cyan]  [{color}]{bar}[/{color}]  {pct:.0f}%")
    console.print(
        f"  [green]{completed} done[/green]  "
        f"[yellow]{in_progress} active[/yellow]  "
        f"[dim]{pending} pending[/dim]  ({total} total)"
    )
    console.print()

    table = Table(border_style="dim", box=box.SIMPLE, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", width=12)
    table.add_column("Task")

    status_styles = {
        "completed": "[green]done[/green]",
        "in_progress": "[yellow]active[/yellow]",
        "pending": "[dim]pending[/dim]",
    }

    for i, todo in enumerate(todos, 1):
        status = todo.get("status", "pending")
        style = status_styles.get(status, f"[dim]{status}[/dim]")
        task = todo.get("task", "(no description)")
        table.add_row(str(i), style, task)

    console.print(table)
    console.print()


def print_mcp_status(servers: list[dict]) -> None:
    """Print MCP server status table."""
    if not servers:
        print_info("No MCP servers configured. Use /mcp add to add one.")
        return

    table = Table(
        title="MCP Servers",
        border_style="blue",
        box=box.SIMPLE,
        header_style="bold blue",
        padding=(0, 1),
    )
    table.add_column("Server", style="bold yellow", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Tools", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", style="dim", max_width=50)

    for s in servers:
        if s["disabled"]:
            status = "[red]disabled[/red]"
            tools = "[dim]-[/dim]"
        else:
            status = "[green]enabled[/green]"
            tools = str(s["tool_count"])
        table.add_row(s["name"], s["type"], tools, status, s["detail"])

    console.print(table)
    console.print("  [dim]Tip: /mcp find <query>, /mcp install <name>, /mcp add, /mcp disable <name>, /mcp reload[/dim]")
    console.print()


def print_mcp_search_results(results: list[dict], query: str) -> None:
    """Print MCP registry search results in a Rich table."""
    table = Table(
        title=f"MCP Registry: '{query}' ({len(results)} results)",
        border_style="blue",
        box=box.SIMPLE,
        header_style="bold blue",
        padding=(0, 1),
    )
    table.add_column("Server Name", style="bold yellow", max_width=45)
    table.add_column("Ver", style="dim", no_wrap=True, width=8)
    table.add_column("Type", style="cyan", no_wrap=True, width=12)
    table.add_column("Description", max_width=50)

    for r in results:
        if r.get("registry_types"):
            type_str = ", ".join(r["registry_types"])
        elif r.get("has_remotes"):
            type_str = "remote"
        else:
            type_str = "?"

        table.add_row(
            r["name"],
            r.get("version", "?"),
            type_str,
            r.get("description", ""),
        )
    console.print(table)
    console.print("  [dim]Install: /mcp install <server-name>[/dim]")
    console.print()


def print_cron_jobs(jobs: list) -> None:
    """Print cron job table."""
    table = Table(
        title="Scheduled Tasks",
        border_style="magenta",
        box=box.SIMPLE,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("ID", style="bold yellow", no_wrap=True, width=8)
    table.add_column("Type", style="cyan", no_wrap=True, width=6)
    table.add_column("Spec", style="dim", width=16)
    table.add_column("Next Run", style="green", width=18)
    table.add_column("Status", width=8)
    table.add_column("Task", max_width=40)

    for job in jobs:
        status = "[red]paused[/red]" if job.paused else "[green]active[/green]"
        next_run = job.next_run[:16].replace("T", " ") if job.next_run else "-"
        type_val = job.type.value if hasattr(job.type, "value") else str(job.type)
        table.add_row(
            job.id,
            type_val,
            job.spec,
            next_run,
            status,
            job.task[:40],
        )
    console.print(table)
    console.print()


def print_help() -> None:
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("Command", style="bold yellow", no_wrap=True, width=16)
    table.add_column("Description", style="white")

    commands = [
        ("/help", "Show this help"),
        ("/clear", "Reset conversation (new thread)"),
        ("/compact", "Summarize older messages to free context"),
        ("/context", "Show context window usage"),
        ("/state", "Show project state (STATE.md)"),
        ("/memory [sub]", "View memories (daily/long, default: both)"),
        ("/agents", "List loaded agents"),
        ("/skills [cmd]", "Skills (list/search/install/remove/import/find)"),
        ("/tools", "List MCP tools by server"),
        ("/call [srv] <tool>", "Call MCP tool directly: /call github get_me"),
        ("/mcp [cmd]", "MCP servers (find/install/add/remove/disable/enable/reload)"),
        ("/projects [cmd]", "Projects (show/create/update/remove/reload)"),
        ("/sessions [id]", "List sessions or switch to one"),
        ("/plan", "Show current task plan with progress"),
        ("/profile [name]", "Show/switch model profile (quality/balanced/budget)"),
        ("/heartbeat [test]", "Heartbeat status or force a tick"),
        ("/cron [cmd]", "Scheduled tasks (list/add/remove/pause/resume)"),
        ("/bg <command>", "Run command in background"),
        ("/tasks", "List background tasks"),
        ("/task <id> [cmd]", "Task details / cancel / resume"),
        ("/vp [cmd]", "Virtual Persona (status/enable/disable/allow/block/test/stats)"),
        ("/swarm [cmd]", "Swarm (status/peers/add/remove/ping)"),
        ("/create-agent", "AI-assisted agent creation wizard"),
        ("/create-skill", "AI-assisted skill creation wizard"),
        ("/reload", "Restart with session restore (re-reads everything)"),
        ("/restart", "Same as /reload"),
        ("/voice on|off", "Toggle TTS"),
        ("/model <name>", "Switch model"),
        ("/<agent> <prompt>", "Send prompt directly to a specific agent"),
        ("/<skill>", "Invoke a skill"),
        ("ESC", "Abort running agent"),
        ("exit", "End session (auto-saves state)"),
        ("", ""),
        ("octo init", "Run setup wizard"),
        ("octo doctor", "Check configuration health"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


# ── Input handler (prompt_toolkit) ────────────────────────────────────

_session: PromptSession | None = None
_pending_attachments: list[str] = []


def get_pending_attachments() -> list[str]:
    """Return and clear pending attachments added via Ctrl+V."""
    paths = list(_pending_attachments)
    _pending_attachments.clear()
    return paths


def _clipboard_paste_image() -> str | None:
    """Check macOS clipboard for image data, save to uploads if found.

    Returns the saved file path, or None if clipboard has no image.
    """
    import platform
    if platform.system() != "Darwin":
        return None

    import subprocess
    import tempfile
    from octo.attachments import copy_to_uploads

    # Use osascript to save clipboard image as PNG (no extra deps needed)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    script = f'''
set theFile to POSIX file "{tmp_path}"
try
    set imgData to the clipboard as «class PNGf»
    set fileRef to open for access theFile with write permission
    write imgData to fileRef
    close access fileRef
    return "ok"
on error
    return "no_image"
end try
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() == "ok":
            import os
            if os.path.getsize(tmp_path) > 0:
                dest = copy_to_uploads(tmp_path, filename="clipboard.png")
                os.unlink(tmp_path)
                return dest
        # Clean up temp file
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except Exception:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return None


def _build_key_bindings() -> KeyBindings:
    """Key bindings: Enter submits; Ctrl+Enter / Esc+Enter insert newline.

    macOS terminals send backslash (0x5c) + carriage return (0x0d) for
    Ctrl+Enter.  We bind that two-key sequence to newline insertion.

    Ctrl+V does smart paste: checks clipboard for image data first,
    falls back to text paste.
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _esc_newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("\\", "c-m", eager=True)
    def _ctrl_newline(event):
        """Ctrl+Enter → newline (macOS sends backslash + CR)."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-v", eager=True)
    def _smart_paste(event):
        """Ctrl+V — paste image from clipboard, or text if no image."""
        image_path = _clipboard_paste_image()
        if image_path:
            _pending_attachments.append(image_path)
            filename = image_path.rsplit("/", 1)[-1]
            count = len(_pending_attachments)
            # Show a short tag in the input buffer
            tag = f"[{filename}]" if count == 1 else f"[{filename} +{count - 1}]"
            event.current_buffer.insert_text(tag + " ")
        else:
            # Fall back to normal text paste from clipboard
            data = event.app.clipboard.get_data()
            event.current_buffer.insert_text(data.text if data else "")

    _last_ctrl_c: list[float] = [0.0]

    @kb.add("c-c", eager=True)
    def _ctrl_c(event):
        """First Ctrl+C clears the line; second within 1s exits."""
        import time
        now = time.monotonic()
        buf = event.current_buffer

        if buf.text.strip():
            # Line has content — clear it (and pending attachments)
            buf.reset()
            _pending_attachments.clear()
            _last_ctrl_c[0] = 0.0
        elif now - _last_ctrl_c[0] < 1.0:
            # Double Ctrl+C on empty line — exit
            buf.text = "exit"
            buf.validate_and_handle()
        else:
            # First Ctrl+C on empty line — note timestamp
            _last_ctrl_c[0] = now

    return kb


def setup_input(slash_commands: list[str]) -> None:
    """Initialise the prompt_toolkit session with completion and history.

    Call once after all slash commands are known.
    """
    global _session
    completer = WordCompleter(slash_commands, sentence=True)
    _session = PromptSession(
        completer=completer,
        history=InMemoryHistory(),
        key_bindings=_build_key_bindings(),
        multiline=False,          # Enter submits
        enable_open_in_editor=False,
        mouse_support=False,
    )


async def styled_input_async() -> str:
    """Get user input with multiline support (async).

    - Enter sends the message
    - Escape+Enter inserts a newline
    - Pasted multiline text is preserved (bracket paste)
    - Tab completes slash commands
    """
    if _session is None:
        setup_input([])

    def _toolbar():
        attach = ""
        if _pending_attachments:
            n = len(_pending_attachments)
            names = ", ".join(p.rsplit("/", 1)[-1] for p in _pending_attachments)
            attach = f'  <style fg="ansicyan">📎 {names}</style>'
        keys = (
            " <b>Enter</b> send  "
            "<b>Ctrl+Enter</b> newline  "
            "<b>Ctrl+V</b> paste  "
            "<b>Esc</b> abort"
            + attach
        )
        if _context_ref:
            used = _context_ref.get("used", 0)
            limit = _context_ref.get("limit", 200_000)
            if used > 0 and limit > 0:
                pct = used / limit * 100
                if pct < 30:
                    color = "ansigreen"
                elif pct < 50:
                    color = "ansicyan"
                elif pct < 70:
                    color = "ansiyellow"
                else:
                    color = "ansired"
                bar_len = 10
                filled = int(bar_len * pct / 100)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                used_k = f"{used // 1000}K" if used >= 1000 else str(used)
                limit_k = f"{limit // 1000}K"
                ctx = f'  <style fg="{color}">{bar} {pct:.0f}% ({used_k}/{limit_k})</style>'
                return HTML(keys + ctx)
        return HTML(keys)

    try:
        text = await _session.prompt_async(
            HTML("<b>&gt;</b> "),
            bottom_toolbar=_toolbar,
        )
        return text.strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"
