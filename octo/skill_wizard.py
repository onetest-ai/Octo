"""Interactive wizard for creating new Octo skills."""
from __future__ import annotations

import re

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from octo.config import SKILLS_DIR

console = Console()


# ── Phase 1: Interactive collection ──────────────────────────────────


def _print_wizard_banner() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]Create Skill[/bold cyan]\n"
        "[dim]Build a new workflow skill for Octo.[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2),
    ))


def _ask_skill_name() -> str | None:
    """Prompt for skill name. Returns None if user cancels."""
    console.print()
    console.print("  [dim]Use lowercase letters, numbers, and dashes (e.g. code-reviewer)[/dim]")
    name = Prompt.ask("  Skill name").strip()

    if not re.match(r"^[a-z][a-z0-9-]*$", name):
        console.print("[red]  Invalid name. Use lowercase letters, numbers, and dashes.[/red]")
        return None

    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        if not Confirm.ask(f"  Skill '{name}' already exists. Overwrite?", default=False):
            return None

    return name


def _ask_description() -> str:
    console.print()
    console.print("  [dim]Describe WHEN to use this skill, not just what it does.[/dim]")
    console.print("  [dim]Example: \"Use when generating API documentation from code.\"[/dim]")
    return Prompt.ask("  Description").strip()


def _ask_detailed_purpose() -> str:
    console.print()
    console.print("  [bold cyan]What should this skill do?[/bold cyan]")
    console.print("  [dim]Describe the workflow, input/output, and key steps. The AI will write the full skill.[/dim]")
    return Prompt.ask("  Purpose").strip()


def _ask_model_invocation() -> bool:
    console.print()
    console.print("  [bold cyan]Can the LLM invoke this skill proactively?[/bold cyan]")
    console.print("  [dim]Yes = agent can trigger it. No = only via /skill-name command.[/dim]")
    return Confirm.ask("  Model invocation", default=True)


def _ask_permissions() -> dict:
    console.print()
    console.print("  [bold cyan]Permissions:[/bold cyan]")

    fs_choices = {"1": "read", "2": "write", "3": "execute"}
    console.print("    [bold yellow]1[/bold yellow]  read      [dim]— read files only[/dim]")
    console.print("    [bold yellow]2[/bold yellow]  write     [dim]— read and write files[/dim]")
    console.print("    [bold yellow]3[/bold yellow]  execute   [dim]— full filesystem access[/dim]")
    fs_choice = Prompt.ask("  Filesystem", choices=["1", "2", "3"], default="2")
    filesystem = fs_choices[fs_choice]

    network = Confirm.ask("  Network access", default=False)
    shell = Confirm.ask("  Shell access", default=True)

    return {
        "filesystem": filesystem,
        "network": network,
        "shell": shell,
        "mcp_servers": [],
    }


def _ask_tags() -> list[str]:
    console.print()
    console.print("  [dim]Comma-separated tags (e.g. development, testing). Leave empty to skip.[/dim]")
    raw = Prompt.ask("  Tags", default="").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


# ── Phase 2: AI generation ──────────────────────────────────────────


def _load_example_skills() -> str:
    """Load 1-2 existing SKILL.md files as in-context examples."""
    if not SKILLS_DIR.is_dir():
        return ""

    # Prefer these as good examples
    preferred = ["quick", "verify"]
    examples: list[str] = []

    for name in preferred:
        skill_file = SKILLS_DIR / name / "SKILL.md"
        if skill_file.is_file():
            content = skill_file.read_text(encoding="utf-8")
            # Truncate very long skills
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            examples.append(content)
        if len(examples) >= 2:
            break

    # Fallback: take any skill
    if not examples:
        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                content = skill_file.read_text(encoding="utf-8")
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                examples.append(content)
                if len(examples) >= 2:
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
    detailed_purpose: str,
    model_invocation: bool,
    permissions: dict,
    tags: list[str],
    examples: str,
) -> str:
    """Construct the meta-prompt for skill body generation."""
    examples_section = ""
    if examples:
        examples_section = f"\n## Examples of existing skills\n\n{examples}\n"

    return f"""\
You are an expert at writing Octo skills. Generate the markdown body for a new skill \
with the following specifications.

## Skill Specifications

- **Name**: {name}
- **Description**: {description}
- **Purpose**: {detailed_purpose}
- **Model invocation**: {"yes — agent can trigger proactively" if model_invocation else "no — user-only via /{name}"}
- **Permissions**: filesystem={permissions.get("filesystem", "read")}, \
network={permissions.get("network", False)}, shell={permissions.get("shell", False)}
- **Tags**: {", ".join(tags) if tags else "(none)"}

## Skill Structure

A skill body is markdown that instructs the AI on HOW to execute a workflow. It should:

1. Start with a heading and brief overview of purpose
2. Have a **## Process** section with numbered phases
3. Have a **## Guidelines** section with dos/don'ts
4. Have a **## Guardrails** section with scope limits and error handling
5. Be concise — context window is a shared resource
6. Include anti-patterns (what NOT to do) to prevent generic output
7. Be between 150-400 words

## Format

Write ONLY the markdown body (everything AFTER the YAML frontmatter). \
Do NOT include the YAML frontmatter (---/name/version/etc.) — that will be added separately.
{examples_section}
Generate the skill body now. Output ONLY the skill text, no explanations or wrapper."""


def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        return "\n".join(lines[1:-1]).strip()
    return stripped


async def _generate_skill_body(
    *,
    name: str,
    description: str,
    detailed_purpose: str,
    model_invocation: bool,
    permissions: dict,
    tags: list[str],
) -> str:
    """Call LLM to generate the skill body."""
    from langchain_core.messages import HumanMessage

    from octo.models import make_model

    examples = _load_example_skills()
    meta_prompt = _build_meta_prompt(
        name=name,
        description=description,
        detailed_purpose=detailed_purpose,
        model_invocation=model_invocation,
        permissions=permissions,
        tags=tags,
        examples=examples,
    )

    model = make_model(tier="default")
    response = await model.ainvoke([HumanMessage(content=meta_prompt)])
    return _strip_code_fences(response.content)


# ── Phase 3: Preview & save ─────────────────────────────────────────


def _compose_skill_md(
    *,
    name: str,
    description: str,
    model_invocation: bool,
    permissions: dict,
    tags: list[str],
    body: str,
) -> str:
    """Compose the full SKILL.md file content."""
    # Quote description if it contains YAML-special characters
    safe_desc = description
    if any(c in description for c in ":{}[]&*?|>!%@`"):
        safe_desc = '"' + description.replace('"', '\\"') + '"'

    tags_str = "[" + ", ".join(tags) + "]" if tags else "[]"
    fs = permissions.get("filesystem", "read")
    net = "true" if permissions.get("network") else "false"
    sh = "true" if permissions.get("shell") else "false"

    return f"""\
---
name: {name}
version: 1.0.0
author: ""
description: {safe_desc}
tags: {tags_str}
model-invocation: {"true" if model_invocation else "false"}

dependencies:
  python: []
  npm: []
  mcp: []
  system: []

requires: []

permissions:
  filesystem: {fs}
  network: {net}
  shell: {sh}
  mcp_servers: []
---

{body}
"""


def _preview_and_save(content: str, name: str) -> bool:
    """Show preview and save if confirmed. Returns True on save."""
    console.print()
    console.print(Panel(
        content,
        title=f"[bold cyan]SKILL.md Preview: {name}[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(1, 2),
    ))

    console.print()
    choice = Prompt.ask("  Action", choices=["save", "cancel"], default="save")

    if choice == "cancel":
        console.print("  [dim]Cancelled.[/dim]")
        return False

    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")

    console.print(f"  [green]Created {skill_file}[/green]")
    console.print(f"  [dim]Edit anytime: {skill_file}[/dim]")
    return True


# ── Main entry point ────────────────────────────────────────────────


async def create_skill_wizard() -> str | None:
    """Guided wizard to create a new Octo skill.

    Returns the skill name on success, or None if cancelled.
    """
    _print_wizard_banner()

    # Phase 1: Collect inputs
    name = _ask_skill_name()
    if not name:
        return None

    description = _ask_description()
    if not description:
        console.print("  [red]Description is required.[/red]")
        return None

    detailed_purpose = _ask_detailed_purpose()
    if not detailed_purpose:
        console.print("  [red]Purpose is required.[/red]")
        return None

    model_invocation = _ask_model_invocation()
    permissions = _ask_permissions()
    tags = _ask_tags()

    # Phase 2: Generate skill body
    console.print()
    with console.status("[yellow]Generating skill body...[/yellow]", spinner="dots"):
        body = await _generate_skill_body(
            name=name,
            description=description,
            detailed_purpose=detailed_purpose,
            model_invocation=model_invocation,
            permissions=permissions,
            tags=tags,
        )

    # Phase 3: Preview and save
    content = _compose_skill_md(
        name=name,
        description=description,
        model_invocation=model_invocation,
        permissions=permissions,
        tags=tags,
        body=body,
    )

    if _preview_and_save(content, name):
        return name
    return None
