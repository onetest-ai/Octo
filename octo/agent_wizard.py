"""Interactive wizard for creating new Octo agents."""
from __future__ import annotations

import json
import re

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from octo.config import AGENTS_DIR

console = Console()

_AGENT_COLORS = [
    "cyan", "green", "yellow", "magenta", "blue",
    "red", "bright_cyan", "bright_green", "bright_yellow",
]


# ── Phase 1: Interactive collection ──────────────────────────────────


def _print_wizard_banner() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]Create Agent[/bold cyan]\n"
        "[dim]Build a new specialist agent for your Octo team.[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    ))


def _ask_agent_name() -> str | None:
    """Prompt for agent name. Returns None if user cancels."""
    console.print()
    console.print("  [dim]Use lowercase letters, numbers, and dashes (e.g. code-reviewer)[/dim]")
    name = Prompt.ask("  Agent name").strip()

    if not re.match(r"^[a-z][a-z0-9-]*$", name):
        console.print("[red]  Invalid name. Use lowercase letters, numbers, and dashes.[/red]")
        return None

    agent_dir = AGENTS_DIR / name
    if agent_dir.exists():
        if not Confirm.ask(f"  Agent '{name}' already exists. Overwrite?", default=False):
            return None

    return name


def _ask_description() -> str:
    return Prompt.ask("  Description (1-2 sentences, for routing)").strip()


def _ask_agent_type() -> str:
    console.print()
    console.print("  [bold cyan]Agent type:[/bold cyan]")
    console.print("    [bold yellow]1[/bold yellow]  Standard      [dim]— general-purpose (tools + system prompt)[/dim]")
    console.print("    [bold yellow]2[/bold yellow]  Deep Research  [dim]— persistent workspace + sub-agents[/dim]")
    choice = Prompt.ask("  Type", choices=["1", "2"], default="1")
    return "" if choice == "1" else "deep_research"


def _ask_detailed_purpose() -> str:
    console.print()
    console.print("  [bold cyan]What should this agent do?[/bold cyan]")
    console.print("  [dim]Describe role, expertise, tasks. The AI will pick tools and write the prompt.[/dim]")
    return Prompt.ask("  Purpose").strip()


def _ask_color() -> str:
    console.print()
    colors_display = ", ".join(_AGENT_COLORS)
    console.print(f"  [dim]Colors: {colors_display}[/dim]")
    return Prompt.ask("  Agent color", default="cyan").strip()


# ── Tool inventory helpers ───────────────────────────────────────────


def _build_full_tool_inventory(mcp_tools_by_server: dict[str, list]) -> list[dict]:
    """Build a flat list of all available tools with metadata."""
    from octo.tools import BUILTIN_TOOLS

    inventory: list[dict] = []
    idx = 1

    for t in BUILTIN_TOOLS:
        desc = (getattr(t, "description", "") or "").split("\n")[0][:80]
        inventory.append({"idx": idx, "name": t.name, "source": "built-in", "desc": desc})
        idx += 1

    for server_name, tools in sorted(mcp_tools_by_server.items()):
        for t in sorted(tools, key=lambda x: x.name):
            desc = (getattr(t, "description", "") or "").split("\n")[0][:80]
            inventory.append({"idx": idx, "name": t.name, "source": server_name, "desc": desc})
            idx += 1

    return inventory


def _print_tool_table(
    inventory: list[dict],
    selected_names: set[str],
    title: str = "Tools",
) -> None:
    """Print tool table with checkmarks for selected tools."""
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), title=title)
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("", width=2)
    table.add_column("Tool", style="bold cyan", no_wrap=True)
    table.add_column("Source", style="dim", width=14)
    table.add_column("Description", max_width=50)

    for item in inventory:
        check = "[green]✓[/green]" if item["name"] in selected_names else " "
        table.add_row(str(item["idx"]), check, item["name"], item["source"], item["desc"][:50])

    console.print(table)


# ── AI tool proposal ─────────────────────────────────────────────────


def _build_tool_proposal_prompt(
    purpose: str,
    agent_type: str,
    inventory: list[dict],
) -> str:
    """Build prompt for LLM to propose tools based on agent purpose."""
    tool_list = "\n".join(
        f"- {item['name']} ({item['source']}): {item['desc']}"
        for item in inventory
    )

    return f"""\
You are helping configure an AI agent. Based on the agent's purpose, select the most \
relevant tools from the available inventory.

## Agent Purpose
{purpose}

## Agent Type
{"Deep research agent with persistent workspace" if agent_type == "deep_research" else "Standard agent"}

## Available Tools
{tool_list}

## Rules
- Select ONLY tools the agent will actually need for its described purpose
- For standard agents: filesystem tools (Read, Grep, Glob, Edit, Bash) are useful for code-related agents
- For deep research agents: web search/extract tools are essential, filesystem tools less so
- Don't over-select — fewer focused tools is better than giving everything
- Return a JSON array of tool names, nothing else

## Response Format
Return ONLY a JSON array of selected tool names. Example: ["Read", "Grep", "Bash"]"""


async def _propose_tools(
    purpose: str,
    agent_type: str,
    inventory: list[dict],
) -> list[str]:
    """Call LLM to propose tools based on agent purpose."""
    from langchain_core.messages import HumanMessage

    from octo.models import make_model

    prompt = _build_tool_proposal_prompt(purpose, agent_type, inventory)
    model = make_model(tier="low")
    response = await model.ainvoke([HumanMessage(content=prompt)])

    # Parse JSON array from response
    text = response.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    try:
        proposed = json.loads(text)
        if isinstance(proposed, list):
            valid_names = {item["name"] for item in inventory}
            return [name for name in proposed if name in valid_names]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return builtin tools
    return [item["name"] for item in inventory if item["source"] == "built-in"]


def _edit_tool_selection(
    inventory: list[dict],
    selected: set[str],
) -> set[str]:
    """Let user add/remove tools from the proposed selection."""
    while True:
        console.print()
        console.print("  [dim]Enter numbers to toggle, 'all'/'none'/'builtin', or 'done' to confirm[/dim]")
        raw = Prompt.ask("  Edit tools", default="done").strip().lower()

        if raw == "done":
            return selected
        if raw == "all":
            selected = {item["name"] for item in inventory}
            _print_tool_table(inventory, selected, title="Tools (updated)")
            continue
        if raw == "none":
            selected = set()
            _print_tool_table(inventory, selected, title="Tools (updated)")
            continue
        if raw == "builtin":
            selected = {item["name"] for item in inventory if item["source"] == "built-in"}
            _print_tool_table(inventory, selected, title="Tools (updated)")
            continue

        # Toggle by number or name
        changed = False
        for part in raw.split(","):
            part = part.strip()
            match = None
            try:
                num = int(part)
                match = next((item for item in inventory if item["idx"] == num), None)
            except ValueError:
                match = next((item for item in inventory if item["name"] == part), None)

            if match:
                if match["name"] in selected:
                    selected.discard(match["name"])
                else:
                    selected.add(match["name"])
                changed = True

        if changed:
            _print_tool_table(inventory, selected, title="Tools (updated)")


# ── Phase 2: AI generation ───────────────────────────────────────────


def _build_tool_descriptions(
    selected_tools: list[str],
    mcp_tools_by_server: dict[str, list],
) -> str:
    """Format selected tools with their descriptions for the meta-prompt."""
    from octo.tools import BUILTIN_TOOLS

    builtin_by_name = {t.name: t for t in BUILTIN_TOOLS}
    mcp_by_name: dict[str, object] = {}
    for tools in mcp_tools_by_server.values():
        for t in tools:
            mcp_by_name[t.name] = t

    lines: list[str] = []
    for name in selected_tools:
        tool = builtin_by_name.get(name) or mcp_by_name.get(name)
        if tool:
            desc = (getattr(tool, "description", "") or "").split("\n")[0]
            lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(f"- **{name}**: (no description)")

    return "\n".join(lines) if lines else "(no tools selected)"


def _load_example_agents(agent_type: str) -> str:
    """Load 1-2 existing AGENT.md files as examples, matching the requested type."""
    if not AGENTS_DIR.is_dir():
        return ""

    examples: list[str] = []
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        agent_file = agent_dir / "AGENT.md"
        if not agent_file.is_file():
            continue
        content = agent_file.read_text(encoding="utf-8")
        is_deep = "type: deep_research" in content
        if agent_type == "deep_research" and is_deep:
            examples.append(content)
        elif agent_type != "deep_research" and not is_deep:
            examples.append(content)
        if len(examples) >= 2:
            break

    # Fallback: take any agent if none matched
    if not examples:
        for agent_dir in sorted(AGENTS_DIR.iterdir()):
            agent_file = agent_dir / "AGENT.md"
            if agent_file.is_file():
                examples.append(agent_file.read_text(encoding="utf-8"))
                break

    if not examples:
        return ""

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}:\n```markdown\n{ex}\n```")
    return "\n\n".join(parts)


def _build_meta_prompt(
    *,
    name: str,
    description: str,
    agent_type: str,
    tools: list[str],
    tool_descriptions: str,
    detailed_purpose: str,
    examples: str,
) -> str:
    """Construct the meta-prompt for system prompt generation."""
    if agent_type == "deep_research":
        type_guidance = (
            "This is a DEEP RESEARCH agent. It has a persistent workspace at "
            "`.octo/workspace/<today's date>/` for saving notes, drafts, and reports. "
            "It has built-in middleware for planning, filesystem access, sub-agent "
            "spawning, and summarization. Include a '## Workspace' section explaining "
            "the workspace convention and a '## Research Workflow' section with steps."
        )
    else:
        type_guidance = (
            "This is a STANDARD agent. It runs as a LangGraph worker with the "
            "listed tools. Focus the prompt on its specialty, methodology, and "
            "output format."
        )

    examples_section = ""
    if examples:
        examples_section = f"\n## Examples of existing agent prompts\n\n{examples}\n"

    return f"""\
You are an expert at writing system prompts for AI agents. Generate a system prompt \
for an Octo agent with the following specifications.

## Agent Specifications

- **Name**: {name}
- **Description**: {description}
- **Type**: {agent_type or "standard"}
- **Detailed purpose**: {detailed_purpose}

## Available Tools

The agent has access to these tools:
{tool_descriptions}

## Agent Type Guidance

{type_guidance}

## Format

Write ONLY the markdown body of the system prompt (everything AFTER the YAML frontmatter). \
Do NOT include the YAML frontmatter (---/name/description/etc.) — that will be added separately.

The prompt should:
1. Start with a clear role statement ("You are a ...")
2. Include a methodology section with numbered steps
3. Include an output format section
4. Be specific and actionable — not vague platitudes
5. Reference the actual tool names the agent has access to where relevant
6. Be between 200-600 words
{examples_section}
Generate the system prompt now. Output ONLY the prompt text, no explanations or wrapper."""


def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        return "\n".join(lines[1:-1]).strip()
    return stripped


async def _generate_system_prompt(
    *,
    name: str,
    description: str,
    agent_type: str,
    tools: list[str],
    detailed_purpose: str,
    mcp_tools_by_server: dict[str, list],
) -> str:
    """Call LLM to generate the agent's system prompt."""
    from langchain_core.messages import HumanMessage

    from octo.models import make_model

    tool_descriptions = _build_tool_descriptions(tools, mcp_tools_by_server)
    examples = _load_example_agents(agent_type)
    meta_prompt = _build_meta_prompt(
        name=name,
        description=description,
        agent_type=agent_type,
        tools=tools,
        tool_descriptions=tool_descriptions,
        detailed_purpose=detailed_purpose,
        examples=examples,
    )

    model = make_model(tier="default")
    response = await model.ainvoke([HumanMessage(content=meta_prompt)])
    return _strip_code_fences(response.content)


# ── Phase 3: Preview & save ──────────────────────────────────────────


def _compose_agent_md(
    *,
    name: str,
    description: str,
    agent_type: str,
    tools: list[str],
    color: str,
    system_prompt: str,
) -> str:
    """Compose the full AGENT.md file content."""
    tools_str = "[" + ", ".join(tools) + "]" if tools else "[]"
    # Quote description if it contains YAML-special characters
    safe_desc = description
    if any(c in description for c in ":{}[]&*?|>!%@`"):
        safe_desc = '"' + description.replace('"', '\\"') + '"'

    return f"""\
---
name: {name}
description: {safe_desc}
model: inherit
tools: {tools_str}
type: {agent_type}
color: {color}
---

{system_prompt}
"""


def _preview_and_save(content: str, name: str) -> bool:
    """Show preview and save if confirmed. Returns True on save."""
    console.print()
    console.print(Panel(
        content,
        title=f"[bold cyan]AGENT.md Preview: {name}[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(1, 2),
    ))

    console.print()
    choice = Prompt.ask("  Action", choices=["save", "cancel"], default="save")

    if choice == "cancel":
        console.print("  [dim]Cancelled.[/dim]")
        return False

    agent_dir = AGENTS_DIR / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_file = agent_dir / "AGENT.md"
    agent_file.write_text(content, encoding="utf-8")

    console.print(f"  [green]Created {agent_file}[/green]")
    console.print(f"  [dim]Edit the prompt anytime: {agent_file}[/dim]")
    return True


# ── Main entry point ─────────────────────────────────────────────────


async def create_agent_wizard(
    mcp_tools_by_server: dict[str, list],
) -> str | None:
    """Guided wizard to create a new Octo agent.

    Returns the agent name on success, or None if cancelled.
    """
    _print_wizard_banner()

    # Phase 1: Collect inputs
    name = _ask_agent_name()
    if not name:
        return None

    description = _ask_description()
    if not description:
        console.print("  [red]Description is required.[/red]")
        return None

    agent_type = _ask_agent_type()

    detailed_purpose = _ask_detailed_purpose()
    if not detailed_purpose:
        console.print("  [red]Purpose is required.[/red]")
        return None

    # AI-assisted tool selection
    inventory = _build_full_tool_inventory(mcp_tools_by_server)
    console.print()
    with console.status("[yellow]Analyzing purpose and selecting tools...[/yellow]", spinner="dots"):
        proposed = await _propose_tools(detailed_purpose, agent_type, inventory)

    selected = set(proposed)
    console.print()
    console.print(f"  [bold cyan]AI proposed {len(selected)} tool(s):[/bold cyan]")
    _print_tool_table(inventory, selected, title="Proposed Tools")

    # Let user edit
    selected = _edit_tool_selection(inventory, selected)
    tools = [item["name"] for item in inventory if item["name"] in selected]

    color = _ask_color()

    # Phase 2: Generate system prompt
    console.print()
    with console.status("[yellow]Generating system prompt...[/yellow]", spinner="dots"):
        system_prompt = await _generate_system_prompt(
            name=name,
            description=description,
            agent_type=agent_type,
            tools=tools,
            detailed_purpose=detailed_purpose,
            mcp_tools_by_server=mcp_tools_by_server,
        )

    # Phase 3: Preview and save
    content = _compose_agent_md(
        name=name,
        description=description,
        agent_type=agent_type,
        tools=tools,
        color=color,
        system_prompt=system_prompt,
    )

    if _preview_and_save(content, name):
        return name
    return None
