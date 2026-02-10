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

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from octo.loaders.agent_loader import AgentConfig
from octo.loaders.skill_loader import SkillConfig

console = Console()

# ── Logo ──────────────────────────────────────────────────────────────

LOGO = [
    " ██████╗  ██████╗████████╗██╗",
    "██╔═══██╗██╔════╝╚══██╔══╝██║",
    "██║   ██║██║        ██║   ██║",
    "██║   ██║██║        ██║   ██║",
    "╚██████╔╝╚██████╗   ██║   ██║",
    " ╚═════╝  ╚═════╝   ╚═╝   ╚═╝",
]


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("octo")
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
        ("/skills", "List skills"),
        ("/plan", "Task plan"),
        ("/context", "Context usage"),
        ("/profile", "Model profile"),
        ("/clear", "New conversation"),
        ("/sessions", "List sessions"),
        ("exit", "End session"),
    ]
    for cmd, desc in cmds:
        right.append(f"{cmd:<12}", style="bold yellow")
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
    table.add_column("Description", max_width=60)
    for s in skills:
        table.add_row(f"/{s.name}", s.description[:60])
    console.print(table)
    console.print()


def print_projects() -> None:
    """Print the project registry table."""
    from octo.config import PROJECTS

    table = Table(
        title="Projects",
        border_style="magenta",
        box=box.SIMPLE,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Project", style="bold yellow", no_wrap=True)
    table.add_column("Path", style="dim", max_width=50)
    table.add_column("Config Dir", style="dim", max_width=50)
    table.add_column("Agents", max_width=40)

    for proj in PROJECTS.values():
        agents = ", ".join(proj.agents[:5])
        if len(proj.agents) > 5:
            agents += f" (+{len(proj.agents) - 5})"
        table.add_row(proj.name, proj.path, proj.config_dir, agents)
    console.print(table)
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
        ("/agents", "List loaded agents"),
        ("/skills", "List loaded skills"),
        ("/projects", "Show project registry"),
        ("/sessions", "List saved sessions"),
        ("/plan", "Show current task plan with progress"),
        ("/profile [name]", "Show/switch model profile (quality/balanced/budget)"),
        ("/voice on|off", "Toggle TTS"),
        ("/model <name>", "Switch model"),
        ("/thread [id]", "Show or switch thread"),
        ("exit", "End session"),
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


def _build_key_bindings() -> KeyBindings:
    """Key bindings: Enter submits, Escape+Enter inserts newline."""
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _newline(event):
        """Insert newline on Escape+Enter."""
        event.current_buffer.insert_text("\n")

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

    try:
        text = await _session.prompt_async(
            HTML("<b>&gt;</b> "),
            bottom_toolbar=HTML(
                " <b>Enter</b> send  "
                "<b>Esc+Enter</b> newline  "
                "<b>Tab</b> complete"
            ),
        )
        return text.strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"
