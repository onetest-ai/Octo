"""Graph assembly — supervisor + worker agents + middleware."""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from langchain.agents import create_agent
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from octo.config import AGENTS_DIR, DB_PATH, OCTO_DIR, PROJECTS, SUPERVISOR_MSG_CHAR_LIMIT, get_profile_tiers
from octo.context import build_system_prompt
from octo.loaders.agent_loader import AgentConfig, load_agents, load_octo_agents
from octo.loaders.skill_loader import check_missing_deps, load_skills, verify_skills_deps
from octo.middleware import ToolErrorMiddleware, ToolResultLimitMiddleware, build_summarization_middleware
from octo.models import make_model
from octo.tools import BUILTIN_TOOLS

# --- Extracted tool modules (Phase 2 of package split) --------------------
# Tools are now defined in octo.core.tools.* and imported here.
# Backward-compatible re-exports are provided for existing importers.

from octo.core.tools.planning import (
    write_todos, read_todos, update_state_md,
    _todos, _load_todos_from_disk, _save_todos_to_disk,
)
from octo.core.tools.memory import (
    write_memory, read_memories, update_long_term_memory,
)
from octo.core.tools.mcp_proxy import (
    find_tools, call_mcp_tool,
    register_mcp_tools as _register_mcp_tools,
    get_mcp_tool, set_session_pool,
    get_mcp_server_summaries,
)
from octo.core.tools.telegram_tools import (
    send_file, set_telegram_transport,
)


# --- Context window tracking ----------------------------------------------

_CONTEXT_LIMITS = {
    "claude": 200_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
    "o1": 128_000,
    "o3": 200_000,
}

# Module-level mutable state read by CLI for display
context_info: dict[str, int] = {"used": 0, "limit": 200_000}


def _get_context_limit(model_name: str) -> int:
    for prefix, limit in _CONTEXT_LIMITS.items():
        if prefix in model_name.lower():
            return limit
    return 200_000


def _build_pre_model_hook(model_name: str):
    """Build a pre_model_hook that tracks context and auto-trims when needed.

    Two-stage protection against context overflow:
    1. **Truncate oversized messages** — any single message over 30K chars
       is cut to prevent one huge tool result from filling the window.
    2. **Trim old messages** — when total tokens exceed 70% of the limit,
       drop the oldest messages (keeping the newest ~40%).
    """
    import logging
    from langchain_core.messages import SystemMessage

    _log = logging.getLogger("octo.graph")
    context_info["limit"] = _get_context_limit(model_name)

    _TRIM_THRESHOLD = 0.70   # start trimming at 70% usage
    _KEEP_RATIO = 0.40       # keep the newest 40% of messages
    _MIN_KEEP = 6            # never keep fewer than 6 messages
    _MAX_MSG_CHARS = SUPERVISOR_MSG_CHAR_LIMIT

    _IMAGE_BLOCK_TYPES = {"image_url", "image"}

    def _truncate_message(msg):
        """Truncate a single message's content if it's too large.

        Image blocks (image_url, image) are always preserved — they are
        handled natively by LLM provider APIs and should not be dropped
        or counted toward the text char limit.
        """
        content = msg.content
        if isinstance(content, str) and len(content) > _MAX_MSG_CHARS:
            truncated = content[:_MAX_MSG_CHARS] + (
                f"\n\n... [truncated — original was {len(content):,} chars]"
            )
            # Create a copy with truncated content
            return msg.model_copy(update={"content": truncated})
        if isinstance(content, list):
            # Separate image blocks (preserve as-is) from text blocks (truncate)
            image_parts = []
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_BLOCK_TYPES:
                    image_parts.append(part)
                else:
                    text_parts.append(part)

            # Only truncate text parts
            text_total = sum(len(str(p)) for p in text_parts)
            if text_total > _MAX_MSG_CHARS:
                truncated_parts = []
                running = 0
                for part in text_parts:
                    part_str = str(part)
                    if running + len(part_str) > _MAX_MSG_CHARS:
                        remaining = _MAX_MSG_CHARS - running
                        if remaining > 100 and isinstance(part, dict) and "text" in part:
                            truncated_part = dict(part)
                            truncated_part["text"] = part["text"][:remaining] + (
                                f"\n\n... [truncated — original was {text_total:,} chars total]"
                            )
                            truncated_parts.append(truncated_part)
                        break
                    truncated_parts.append(part)
                    running += len(part_str)
                # Re-attach image blocks after text
                return msg.model_copy(update={"content": truncated_parts + image_parts})
            elif image_parts:
                # No text truncation needed, but ensure order is preserved
                return msg
        return msg

    def _strip_images_for_counting(messages):
        """Create lightweight copies of messages with image data removed.

        Base64 image data inflates token counts by hundreds of thousands —
        but images are sent as binary to the API, not as tokens.  We replace
        image blocks with a small placeholder for accurate counting.
        """
        stripped = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in _IMAGE_BLOCK_TYPES
                for p in content
            ):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in _IMAGE_BLOCK_TYPES:
                        # ~85 tokens per image (Anthropic charges ~1600 tokens
                        # per image but that's separate from context counting)
                        new_parts.append({"type": "text", "text": "[image]"})
                    else:
                        new_parts.append(part)
                stripped.append(msg.model_copy(update={"content": new_parts}))
            else:
                stripped.append(msg)
        return stripped

    def pre_model_hook(state):
        messages = state.get("messages", [])

        # Stage 1: truncate any oversized individual messages
        messages = [_truncate_message(m) for m in messages]

        tokens = count_tokens_approximately(_strip_images_for_counting(messages))
        context_info["used"] = tokens
        limit = context_info["limit"]

        # Stage 2: trim old messages if context is too large
        if tokens > limit * _TRIM_THRESHOLD and len(messages) > _MIN_KEEP:
            keep = max(_MIN_KEEP, int(len(messages) * _KEEP_RATIO))
            trimmed = messages[-keep:]
            trimmed_tokens = count_tokens_approximately(_strip_images_for_counting(trimmed))
            _log.info(
                "Auto-trimming context: %d→%d tokens (%d→%d messages)",
                tokens, trimmed_tokens, len(messages), len(trimmed),
            )
            context_info["used"] = trimmed_tokens
            marker = SystemMessage(
                content=(
                    "[Earlier conversation was automatically trimmed to fit "
                    "the context window. Some history may be missing.]"
                )
            )
            return {"messages": [marker] + trimmed}

        return {"messages": messages}

    return pre_model_hook


# --- Model tier assignment (profile-aware) --------------------------------

_HIGH_TIER_KEYWORDS = {"architect", "planner", "rca-autofixer"}


def _agent_tier(name: str) -> str:
    """Pick model tier based on agent name and active profile."""
    profile = get_profile_tiers()
    for kw in _HIGH_TIER_KEYWORDS:
        if kw in name:
            return profile.get("worker_high_keywords", "high")
    return profile.get("worker_default", "low")


def _build_project_agents(project_agents: list[AgentConfig]) -> list:
    """Create one LangGraph worker per project in the registry.

    Each project worker wraps `claude_code` for that project.  The supervisor
    delegates to e.g. "elitea" or "onetest" and the worker runs tasks in
    the correct project directory with full codebase context.

    The worker can run general tasks (no agent flag) or delegate to a
    specific project sub-agent (e.g. architect, qa-engineer).
    """
    from octo.config import PROJECTS
    from octo.tools.claude_code import claude_code

    error_middleware = ToolErrorMiddleware()
    result_limit_middleware = ToolResultLimitMiddleware()
    summarization_middleware = build_summarization_middleware()
    workers = []

    for proj in PROJECTS.values():
        # Gather sub-agent names for this project
        sub_agents = [a.name for a in project_agents if a.source_project == proj.name]

        sub_agent_section = ""
        if sub_agents:
            lines = "\n".join(f"  - {name}" for name in sub_agents)
            sub_agent_section = (
                f"\n\nAvailable sub-agents (pass as `agent` parameter):\n{lines}\n\n"
                "Use a sub-agent when the task matches its specialty. "
                "Omit the `agent` parameter for general project work."
            )

        # Build metadata section from project config
        meta_lines: list[str] = []
        if proj.description:
            meta_lines.append(f"- **Description:** {proj.description}")
        if proj.repo_url:
            meta_lines.append(f"- **Repo:** {proj.repo_url}")
        if proj.issues_url:
            meta_lines.append(f"- **Issues:** {proj.issues_url}")
        if proj.tech_stack:
            meta_lines.append(f"- **Tech stack:** {', '.join(proj.tech_stack)}")
        if proj.default_branch:
            meta_lines.append(f"- **Default branch:** {proj.default_branch}")
        if proj.ci_url:
            meta_lines.append(f"- **CI:** {proj.ci_url}")
        if proj.docs_url:
            meta_lines.append(f"- **Docs:** {proj.docs_url}")
        meta_section = ("\n" + "\n".join(meta_lines) + "\n") if meta_lines else ""

        prompt = (
            f"You are the project worker for **{proj.name}** "
            f"(directory: `{proj.path}`).\n"
            f"{meta_section}\n"
            "Use the `claude_code` tool to execute tasks in this project. "
            "Claude Code has full access to the codebase, git history, and "
            "project-specific configuration.\n\n"
            f"Always pass `working_directory=\"{proj.path}\"` when calling claude_code."
            f"{sub_agent_section}"
        )

        tier = _agent_tier(proj.name)
        model = make_model(tier=tier)

        agent = create_agent(
            model=model,
            tools=[claude_code],
            name=proj.name,
            system_prompt=prompt,
            middleware=[error_middleware, result_limit_middleware, summarization_middleware],
        )
        workers.append(agent)

    return workers


def _build_worker_agents(
    agent_configs: list[AgentConfig],
    mcp_tools: list,
) -> list:
    """Create standard agents from AGENT.md configs using create_agent.

    Skips configs with type='deep_research' — those are built separately.
    """
    # Shared middleware
    error_middleware = ToolErrorMiddleware()
    result_limit_middleware = ToolResultLimitMiddleware()
    summarization_middleware = build_summarization_middleware()

    # Index builtin tools by name for agent-specific filtering
    builtin_by_name = {t.name: t for t in BUILTIN_TOOLS}

    workers = []
    for cfg in agent_configs:
        if cfg.type == "deep_research":
            continue

        tier = _agent_tier(cfg.name)
        model = make_model(cfg.model, tier=tier)

        if cfg.tools:
            # Agent specifies tool names — resolve from both MCP and builtin
            agent_tools = [t for t in mcp_tools if t.name in cfg.tools]
            agent_tools += [builtin_by_name[n] for n in cfg.tools if n in builtin_by_name]
        else:
            # No filter — give builtins + MCP proxy (deferred)
            agent_tools = list(BUILTIN_TOOLS) + [find_tools, call_mcp_tool]

        agent = create_agent(
            model=model,
            tools=agent_tools,
            name=cfg.name,
            system_prompt=cfg.system_prompt,
            middleware=[error_middleware, result_limit_middleware, summarization_middleware],
        )
        workers.append(agent)
    return workers


def _build_deep_agents(
    agent_configs: list[AgentConfig],
    mcp_tools: list,
) -> list:
    """Create deep research agents using deepagents' create_deep_agent.

    Deep agents come with built-in middleware (planning, filesystem,
    subagent spawning, summarization, prompt caching) — no need for
    ToolErrorMiddleware or BUILTIN_TOOLS.

    MCP tools are resolved from each agent's tools: list in AGENT.md,
    same as regular agents. If no tools specified, no extra MCP tools added.

    Each deep agent gets a shared FilesystemBackend workspace at
    .octo/workspace/<date>/ for research, notes, and temporary files.
    The supervisor's checkpointer handles state persistence for resume.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from octo.config import RESEARCH_WORKSPACE

    mcp_by_name = {t.name: t for t in mcp_tools}

    # Shared date-based workspace for all deep agents
    today = date.today().isoformat()
    workspace_dir = RESEARCH_WORKSPACE / today
    workspace_dir.mkdir(parents=True, exist_ok=True)

    backend = FilesystemBackend(
        root_dir=str(workspace_dir),
        virtual_mode=True,
    )

    deep_workers = []
    for cfg in agent_configs:
        if cfg.type != "deep_research":
            continue

        model = make_model(cfg.model) if cfg.model else make_model()

        # Resolve MCP tools from agent's tools: list, or provide proxy
        if cfg.tools:
            agent_tools = [mcp_by_name[n] for n in cfg.tools if n in mcp_by_name]
        else:
            agent_tools = [find_tools, call_mcp_tool]

        agent = create_deep_agent(
            model=model,
            tools=agent_tools or None,
            name=cfg.name,
            system_prompt=cfg.system_prompt,
            backend=backend,
            middleware=[ToolErrorMiddleware()],
        )
        deep_workers.append(agent)
    return deep_workers


def _build_supervisor_prompt(skills: list, octo_agents: list[AgentConfig] | None = None) -> str:
    """Compose the full supervisor system prompt."""
    base = build_system_prompt()
    parts = [base]

    # Only show LLM-invocable skills in the prompt
    llm_skills = [s for s in skills if s.model_invocation]
    if llm_skills:
        skill_lines = [f"- **/{s.name}**: {s.description}" for s in llm_skills]
        parts.append(
            "## Available Skills\n\n"
            + "\n".join(skill_lines)
            + "\n\n"
            "Skills are structured workflows you can invoke with the `use_skill` tool. "
            "Use them proactively when a user's request matches a skill — even if they "
            "don't type the slash command.\n\n"
            "The skill body contains detailed instructions — follow them as your working process."
        )

    parts.append(
        "## Memory\n\n"
        "You have persistent memory across sessions via files. Use it actively:\n\n"
        "- **`write_memory`**: Append to today's daily log. Use when:\n"
        "  - The user says 'remember this' or shares personal info/preferences\n"
        "  - You make a significant decision or learn a lesson\n"
        "  - Something important happens worth noting\n"
        "- **`read_memories`**: Read recent daily logs (last N days). Use to recall context.\n"
        "- **`update_long_term_memory`**: Rewrite your curated MEMORY.md. Use to:\n"
        "  - Add persistent facts about the user, projects, or yourself\n"
        "  - Distill patterns from daily logs into long-term knowledge\n"
        "  - Remove outdated information\n\n"
        "Daily logs are raw notes. Long-term memory is curated wisdom. "
        "Don't ask 'should I remember this?' — if it seems worth keeping, just save it."
    )

    parts.append(
        "## Task Planning\n\n"
        "For complex multi-step tasks, use `write_todos` to create a plan before executing. "
        "Update task statuses as you work through them. Use `read_todos` to review the current plan.\n\n"
        "**IMPORTANT**: Each new task gets a FRESH plan. When starting a new task, call "
        "`write_todos` with ONLY the new steps — do NOT include completed items from "
        "previous tasks. Old completed items are auto-archived."
    )

    parts.append(
        "## Deviation Handling\n\n"
        "When agents encounter unexpected issues during execution:\n\n"
        "### AUTO-FIX (no permission needed)\n"
        "- Bugs that block the current task\n"
        "- Missing imports, typos, syntax errors\n"
        "- Small fixes needed for the main task to succeed\n"
        "- Test failures caused by the current changes\n\n"
        "### STOP and ask the user\n"
        "- Architectural changes (new dependencies, structural refactors)\n"
        "- Changes to files outside the current task scope\n"
        "- Security-sensitive modifications\n"
        "- Deleting or renaming public APIs\n\n"
        "### ESCALATE to the user\n"
        "- Ambiguous requirements that could go multiple ways\n"
        "- Trade-offs that affect performance, cost, or UX\n"
        "- Anything you are uncertain about\n\n"
        "When in doubt, escalate. It is better to ask than to assume."
    )

    parts.append(
        "## Agent Handoff Response\n\n"
        "When a specialist agent finishes work and transfers back to you, "
        "you MUST summarize the results for the user in your own response. "
        "NEVER let 'Transferring back to supervisor' be your final answer.\n\n"
        "After receiving a handoff back:\n"
        "1. Review what the agent produced (files, research, analysis)\n"
        "2. Compose a clear summary of the results for the user\n"
        "3. Include key findings, not just 'the report is ready'\n"
        "4. The user may be on Telegram and cannot see files — always include "
        "the substance of the results in your message"
    )

    parts.append(
        "## Response Formatting\n\n"
        "Adapt your response format based on the channel the user is writing from.\n\n"
        "**Telegram** (messages prefixed with `[Channel: Telegram]`):\n"
        "- Keep responses concise and scannable\n"
        "- Use short paragraphs and bullet points\n"
        "- Avoid large code blocks — summarize code changes instead\n"
        "- No tables — use bullet lists\n"
        "- Don't reference local files — Telegram users can't see them\n"
        "- Include the substance of results directly in your message\n\n"
        "**CLI** (default, no channel tag):\n"
        "- Full Markdown formatting is supported\n"
        "- Code blocks, tables, and detailed output are fine"
    )

    parts.append(
        "## Project State (STATE.md)\n\n"
        "Use `update_state_md` to record project state after significant actions. "
        "This file persists across sessions and helps you (and the user) quickly "
        "orient when resuming work. Update it when:\n"
        "- Completing a major task or phase\n"
        "- Making an architectural decision\n"
        "- Before the session ends (capture 'stopped at' and 'next steps')\n"
        "- The user asks to pause or switch context"
    )

    parts.append(
        "## File Sharing\n\n"
        "Use `send_file` to send files to the user via Telegram. This is much better "
        "than pasting long content inline. Use it when:\n"
        "- The user asks for a research report, analysis, or document\n"
        "- An agent has produced a file (research workspace: `.octo/workspace/<date>/`)\n"
        "- The content is too long to include in a message\n\n"
        "If Telegram is not available, the tool returns the local file path instead."
    )

    parts.append(
        "## Task Scheduling\n\n"
        "You can schedule tasks to run later using `schedule_task`. Use this when:\n"
        "- The user says 'remind me in X hours/minutes'\n"
        "- The user wants periodic checks ('check every morning')\n"
        "- A task should run at a specific time\n\n"
        "Schedule types:\n"
        "- `at`: One-shot. Spec examples: 'in 2h', '15:00', '2024-02-11T15:00Z'\n"
        "- `every`: Recurring interval. Spec examples: '30m', '2h', '1d'\n"
        "- `cron`: Cron expression. Spec examples: '0 9 * * MON-FRI'\n\n"
        "Set `isolated=True` for tasks that don't need conversation context.\n"
        "Results are delivered to the user via Telegram and CLI."
    )

    # Project workers
    if PROJECTS:
        proj_lines = []
        for proj in PROJECTS.values():
            agents_str = ", ".join(proj.agents) if proj.agents else "(general only)"
            line = f"- **{proj.name}**"
            if proj.description:
                line += f" — {proj.description}"
            line += f": sub-agents: {agents_str}"
            extras = []
            if proj.issues_url:
                extras.append(f"issues: {proj.issues_url}")
            if proj.repo_url:
                extras.append(f"repo: {proj.repo_url}")
            if proj.tech_stack:
                extras.append(f"tech: {', '.join(proj.tech_stack)}")
            if extras:
                line += f" ({'; '.join(extras)})"
            proj_lines.append(line)
        parts.append(
            "## Project Workers\n\n"
            "Each project has a dedicated worker agent that runs Claude Code in "
            "the project directory with full codebase context. Delegate to the "
            "project worker by name (e.g. transfer to 'elitea' or 'onetest').\n\n"
            + "\n".join(proj_lines)
        )

    # Octo-native agents — auto-discovered from .octo/agents/*/AGENT.md
    if octo_agents:
        agent_lines = []
        for cfg in octo_agents:
            kind = " (deep research)" if cfg.type == "deep_research" else ""
            agent_lines.append(f"- **{cfg.name}**{kind}: {cfg.description}")
        parts.append(
            "## Specialist Agents\n\n"
            "These agents are available for delegation. Transfer to them by name "
            "when the user's request matches their specialty.\n"
            "They understand any language — delegate regardless of the language "
            "the user writes in.\n"
            "Each agent has a persistent workspace — it remembers previous sessions.\n\n"
            + "\n".join(agent_lines)
        )

    # MCP tool access — deferred via find_tools / call_mcp_tool
    _summaries = get_mcp_server_summaries()
    if _summaries:
        server_lines = []
        for s in _summaries:
            server_lines.append(f"- **{s['server']}** ({s['tools']} tools): {s['summary']}")
        parts.append(
            "## MCP Tool Access\n\n"
            "MCP tools are available via `find_tools(query)` and "
            "`call_mcp_tool(name, args)`.\n"
            "Workflow: search for tools first, then call them with the "
            "exact name and arguments returned by find_tools.\n\n"
            "Available servers:\n"
            + "\n".join(server_lines)
        )

    return "\n\n---\n\n".join(parts)


async def build_graph(
    mcp_tools: list | None = None,
    mcp_tools_by_server: dict[str, list] | None = None,
) -> Any:
    """Build and compile the full Octi supervisor graph.

    Returns:
        Tuple of (compiled app, all agent configs, skills).
    """
    mcp_tools = mcp_tools or []
    _register_mcp_tools(mcp_tools_by_server or {})

    # Load agents from both sources
    project_agents = load_agents()   # for display + project worker prompts
    octo_agents = load_octo_agents() # become LangGraph workers directly
    all_agents = project_agents + octo_agents
    skills = load_skills()
    verify_skills_deps(skills)  # warn about missing Python deps at startup

    # Build use_skill tool — closes over the loaded skills list
    skill_by_name = {s.name: s for s in skills}

    @tool
    def use_skill(skill_name: str, user_request: str = "") -> str:
        """Invoke a pre-built skill workflow by name.

        Use this when the user's request matches a skill's purpose — even if they
        didn't explicitly type the slash command.  Returns the full skill prompt
        which you should then follow as your working instructions.

        Args:
            skill_name: Name of the skill (e.g. "quick", "verify", "map-codebase").
            user_request: The user's original request or relevant context.
        """
        sk = skill_by_name.get(skill_name)
        if not sk:
            available = ", ".join(sorted(skill_by_name)) or "(none)"
            return f"Unknown skill '{skill_name}'. Available: {available}"

        # Check for missing Python dependencies and prepend install instructions
        missing = check_missing_deps(sk)
        dep_notice = ""
        if missing:
            pkgs = " ".join(missing)
            dep_notice = (
                f"[IMPORTANT: This skill requires Python packages that are not installed: "
                f"{', '.join(missing)}.\n"
                f"Install them first by running: {sys.executable} -m pip install {pkgs}\n"
                f"Then proceed with the task.]\n\n"
            )

        result = f"{dep_notice}[Skill: {skill_name}]\n\n{sk.body}"
        if user_request:
            result += f"\n\nUser request: {user_request}"
        # Progressive disclosure: list available references and scripts
        if sk.references or sk.scripts:
            result += "\n\n---\n**Bundled resources** (use Read/Bash tools to access on demand):"
            if sk.references:
                result += "\nReference docs:"
                for ref in sk.references:
                    result += f"\n- `{sk.skill_dir}/{ref}`"
            if sk.scripts:
                result += "\nScripts:"
                for scr in sk.scripts:
                    result += f"\n- `{sk.skill_dir}/{scr}`"
        return result

    # Build workers:
    # 1. Project workers — one per project, wraps claude_code (skip if claude CLI absent)
    # 2. Octo workers — native agents (verifier, etc.)
    # 3. Deep agents — deep research agents with persistent workspaces
    import shutil
    if shutil.which("claude"):
        project_workers = _build_project_agents(project_agents)
    else:
        project_workers = []
        if project_agents:
            import logging
            logging.getLogger("octo").warning(
                "Claude Code CLI not found — project workers disabled. "
                "Install: npm install -g @anthropic-ai/claude-code"
            )
    octo_workers = _build_worker_agents(octo_agents, mcp_tools)
    deep_workers = _build_deep_agents(octo_agents, mcp_tools)

    # Supervisor — no longer needs claude_code directly (project workers handle it)
    profile = get_profile_tiers()
    supervisor_model = make_model(tier=profile.get("supervisor", "default"))
    prompt = _build_supervisor_prompt(skills, octo_agents=octo_agents)

    # Build schedule_task tool (cron scheduling)
    from octo.heartbeat import make_schedule_task_tool
    schedule_task = make_schedule_task_tool()

    # Wrap supervisor tools in a TruncatingToolNode that:
    # 1. handle_tool_errors=True — MCP errors returned as messages, not crashes
    # 2. Truncates oversized results (e.g. search_code returning 73K chars)
    #    before they enter the graph state and blow the context window.
    #    create_supervisor does NOT accept middleware, so this is the only
    #    way to get both error handling and result truncation at supervisor level.
    from octo.middleware import TRUNCATION_NOTICE
    from octo.config import TOOL_RESULT_LIMIT
    from langgraph.prebuilt import ToolNode

    class TruncatingToolNode(ToolNode):
        """ToolNode that truncates oversized results before they enter state."""

        def _truncate_content(self, content):
            if isinstance(content, str) and len(content) > TOOL_RESULT_LIMIT:
                notice = TRUNCATION_NOTICE.format(
                    original=len(content), limit=TOOL_RESULT_LIMIT,
                )
                return content[:TOOL_RESULT_LIMIT] + notice
            if isinstance(content, list):
                total = sum(len(str(p)) for p in content)
                if total > TOOL_RESULT_LIMIT:
                    truncated = []
                    running = 0
                    for part in content:
                        part_len = len(str(part))
                        if running + part_len > TOOL_RESULT_LIMIT:
                            # Try to truncate the text inside a dict part
                            if isinstance(part, dict) and "text" in part:
                                remaining = TOOL_RESULT_LIMIT - running
                                if remaining > 200:
                                    trunc_part = dict(part)
                                    trunc_part["text"] = part["text"][:remaining] + TRUNCATION_NOTICE.format(
                                        original=total, limit=TOOL_RESULT_LIMIT,
                                    )
                                    truncated.append(trunc_part)
                            break
                        truncated.append(part)
                        running += part_len
                    if not truncated:
                        return str(content)[:TOOL_RESULT_LIMIT] + TRUNCATION_NOTICE.format(
                            original=total, limit=TOOL_RESULT_LIMIT,
                        )
                    return truncated
            return content

        def _run_one(self, call, input_type, tool_runtime):
            result = super()._run_one(call, input_type, tool_runtime)
            if hasattr(result, "content"):
                truncated = self._truncate_content(result.content)
                if truncated is not result.content:
                    result = result.model_copy(update={"content": truncated})
            return result

        async def _arun_one(self, call, input_type, tool_runtime):
            result = await super()._arun_one(call, input_type, tool_runtime)
            if hasattr(result, "content"):
                truncated = self._truncate_content(result.content)
                if truncated is not result.content:
                    result = result.model_copy(update={"content": truncated})
            return result

    supervisor_tool_list = (
        list(BUILTIN_TOOLS)
        + [find_tools, call_mcp_tool]
        + [write_todos, read_todos, update_state_md, use_skill,
           write_memory, read_memories, update_long_term_memory,
           schedule_task, send_file]
    )
    supervisor_tools = TruncatingToolNode(supervisor_tool_list, handle_tool_errors=True)

    from octo.models import resolve_model_name
    hook = _build_pre_model_hook(resolve_model_name())

    workflow = create_supervisor(
        agents=project_workers + octo_workers + deep_workers,
        model=supervisor_model,
        tools=supervisor_tools,
        prompt=prompt,
        pre_model_hook=hook,
    )

    # Compile with persistent async checkpointer
    conn = await aiosqlite.connect(str(DB_PATH))
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    app = workflow.compile(checkpointer=checkpointer)

    return app, all_agents, skills
