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
from octo.tools import BUILTIN_TOOLS, AGENT_LIFECYCLE_TOOLS

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
    "gpt-4o": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    "gemini": 1_000_000,
    "local/": 32_000,      # conservative default; most local models are 4K-128K
}

# Module-level mutable state read by CLI for display
context_info: dict[str, int] = {"used": 0, "limit": 200_000}


def _get_context_limit(model_name: str) -> int:
    for prefix, limit in _CONTEXT_LIMITS.items():
        if prefix in model_name.lower():
            return limit
    return 200_000


def _compute_tool_result_limit(model_name: str) -> int:
    """Proportional tool result limit: min(context_chars * 0.15, 40_000).

    If the user explicitly set TOOL_RESULT_LIMIT in .env, that value is
    used instead (backward-compatible override).
    """
    import os
    if os.getenv("TOOL_RESULT_LIMIT"):
        from octo.config import TOOL_RESULT_LIMIT
        return TOOL_RESULT_LIMIT
    context_limit = _get_context_limit(model_name)
    # 1 token ≈ 4 chars; limit tool results to 15% of context in chars
    return min(int(context_limit * 4 * 0.15), 40_000)


def _build_pre_model_hook(model_name: str, tool_count: int = 0):
    """Build a pre_model_hook that tracks context and auto-trims when needed.

    Two-stage protection against context overflow:
    1. **Truncate oversized messages** — any single message over 30K chars
       is cut to prevent one huge tool result from filling the window.
    2. **Trim old messages** — when total tokens exceed 70% of the limit,
       drop the oldest messages (keeping the newest ~40%).
    3. **Prompt caching** — inject provider-specific cache breakpoints into
       llm_input_messages so that system prompt and conversation prefix are
       cached across turns (Anthropic 90% savings, Bedrock ~90% savings,
       OpenAI/Azure automatic — no injection needed).

    Args:
        model_name: Model name for context limit detection.
        tool_count: Number of tools bound to the supervisor. Used to estimate
            tool schema token overhead (schemas are sent separately in
            Bedrock Converse API but still count toward context).
    """
    import logging
    from langchain_core.messages import HumanMessage, SystemMessage

    from octo.models import _detect_provider

    _log = logging.getLogger("octo.graph")
    context_info["limit"] = _get_context_limit(model_name)
    _provider = _detect_provider(model_name)

    _TRIM_THRESHOLD = 0.70   # start trimming at 70% usage
    _KEEP_RATIO = 0.40       # keep the newest 40% of messages
    _MIN_KEEP = 6            # never keep fewer than 6 messages
    _MAX_MSG_CHARS = SUPERVISOR_MSG_CHAR_LIMIT

    # Tool schemas are sent separately in the API request but still consume
    # context tokens.  Each tool schema averages ~800 tokens (name, description,
    # JSON schema for parameters).  This isn't counted by
    # count_tokens_approximately which only sees message content.
    _TOOL_SCHEMA_OVERHEAD = tool_count * 800

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

    def _inject_cache_breakpoints(msgs):
        """Inject provider-specific cache breakpoints into messages.

        Breakpoint 1 (Anthropic + Bedrock): System message — most stable,
        highest cache value.  Caches the full system prompt + tool schemas.

        Breakpoint 2 (Anthropic only): Second-to-last message — caches the
        growing conversation prefix so only the latest turn is re-processed.
        Bedrock cachePoint is system-level only, not per-message.

        OpenAI/Azure: Automatic prefix caching for identical prefixes ≥1024
        tokens — no injection needed, 50% savings are free.
        """
        if _provider not in ("anthropic", "bedrock"):
            return msgs

        result = list(msgs)

        # Breakpoint 1: system message
        for i, msg in enumerate(result):
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, str):
                    if _provider == "anthropic":
                        result[i] = msg.model_copy(update={
                            "content": [{"type": "text", "text": content,
                                         "cache_control": {"type": "ephemeral"}}],
                        })
                    elif _provider == "bedrock":
                        result[i] = msg.model_copy(update={
                            "content": [{"type": "text", "text": content},
                                        {"cachePoint": {"type": "default"}}],
                        })
                break  # only first system message

        # Breakpoint 2: turn boundary — Anthropic only
        if _provider == "anthropic" and len(result) >= 2:
            msg = result[-2]
            content = msg.content
            if isinstance(content, str) and content:
                result[-2] = msg.model_copy(update={
                    "content": [{"type": "text", "text": content,
                                 "cache_control": {"type": "ephemeral"}}],
                })

        return result

    def _stamp_date(msgs):
        """Prepend current date/time to the last HumanMessage.

        Injected into llm_input_messages (what the LLM sees), not stored state.
        Placed in the last message because it's always new — doesn't invalidate
        prompt caching (breakpoint 1 = system prompt, breakpoint 2 = convo prefix).
        """
        if not msgs:
            return msgs
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        local_now = _dt.now()
        date_note = (
            f"[Current: {now.strftime('%A, %B %d, %Y')} | "
            f"UTC {now.strftime('%H:%M')} | "
            f"Local {local_now.strftime('%H:%M')}]\n\n"
        )
        # Find last HumanMessage and prepend date
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage) and isinstance(msgs[i].content, str):
                msgs[i] = msgs[i].model_copy(update={
                    "content": date_note + msgs[i].content,
                })
                break
        return msgs

    def pre_model_hook(state):
        messages = state.get("messages", [])

        # Stage 1: truncate any oversized individual messages
        messages = [_truncate_message(m) for m in messages]

        msg_tokens = count_tokens_approximately(_strip_images_for_counting(messages))
        # Add tool schema overhead — schemas are sent in the API request
        # but not counted by count_tokens_approximately
        tokens = msg_tokens + _TOOL_SCHEMA_OVERHEAD
        context_info["used"] = tokens
        limit = context_info["limit"]

        # Stage 2: trim old messages if context is too large
        if tokens > limit * _TRIM_THRESHOLD and len(messages) > _MIN_KEEP:
            keep = max(_MIN_KEEP, int(len(messages) * _KEEP_RATIO))
            trimmed = messages[-keep:]
            trimmed_msg_tokens = count_tokens_approximately(_strip_images_for_counting(trimmed))
            trimmed_tokens = trimmed_msg_tokens + _TOOL_SCHEMA_OVERHEAD
            _log.info(
                "Auto-trimming context: %d→%d tokens (%d→%d messages, tool schema overhead: %d)",
                tokens, trimmed_tokens, len(messages), len(trimmed), _TOOL_SCHEMA_OVERHEAD,
            )
            context_info["used"] = trimmed_tokens
            marker = SystemMessage(
                content=(
                    "[Earlier conversation was automatically trimmed to fit "
                    "the context window. Some history may be missing.]"
                )
            )
            # Use llm_input_messages to send trimmed context to LLM
            # without accidentally growing the state via add_messages reducer.
            # State cleanup is handled by /compact and auto_compact in retry.py.
            return {"llm_input_messages": _inject_cache_breakpoints(_stamp_date([marker] + trimmed))}

        # Stage 3: inject cache breakpoints for prompt caching
        return {"llm_input_messages": _inject_cache_breakpoints(_stamp_date(messages))}

    return pre_model_hook


# --- Model tier assignment (profile-aware) --------------------------------

_HIGH_TIER_KEYWORDS = {"architect", "planner", "rca-autofixer"}

# Tier aliases recognized in AGENT.md `model:` field.
# These are treated as tier hints, not literal model names.
_TIER_ALIASES = {"high", "low", "default", "inherit", ""}


def _agent_tier(name: str) -> str:
    """Pick model tier based on agent name and active profile."""
    profile = get_profile_tiers()
    for kw in _HIGH_TIER_KEYWORDS:
        if kw in name:
            return profile.get("worker_high_keywords", "high")
    return profile.get("worker_default", "low")


def _resolve_agent_model(cfg_model: str, fallback_tier: str):
    """Resolve an agent's model from its AGENT.md `model:` field.

    If the value is a tier alias (high, low, default, inherit, or empty),
    use the corresponding tier.  Otherwise treat it as a literal model name
    after validating it looks like a real model identifier.
    """
    if cfg_model in _TIER_ALIASES:
        effective_tier = cfg_model if cfg_model in ("high", "low") else fallback_tier
        return make_model(tier=effective_tier)

    # Sanity check: model names should contain dots, dashes, or slashes
    # (e.g. "claude-sonnet-4-5", "eu.anthropic.claude-...", "gpt-4o")
    if not any(c in cfg_model for c in ".-/"):
        logger.warning(
            "Agent model '%s' looks like a tier alias, not a model ID. "
            "Use 'high', 'low', 'default', or 'inherit' for tier-based "
            "selection, or a full model identifier (e.g. "
            "'eu.anthropic.claude-sonnet-4-5-v1').",
            cfg_model,
        )
    return make_model(cfg_model, tier=fallback_tier)


def _caching_middleware():
    """Return (anthropic_cache, bedrock_cache) middleware instances.

    - Anthropic: uses built-in AnthropicPromptCachingMiddleware (90% savings).
    - Bedrock: uses custom BedrockCachingMiddleware (cachePoint blocks).
    - OpenAI/Azure: automatic prefix caching — no middleware needed.

    Both middlewares are no-ops for non-matching providers.
    """
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

    from octo.core.middleware import BedrockCachingMiddleware

    anthropic_cache = AnthropicPromptCachingMiddleware(
        unsupported_model_behavior="ignore",
    )
    bedrock_cache = BedrockCachingMiddleware()
    return anthropic_cache, bedrock_cache


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

    from octo.models import resolve_model_name

    error_middleware = ToolErrorMiddleware()
    summarization_middleware = build_summarization_middleware()
    anthropic_cache, bedrock_cache = _caching_middleware()
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
        worker_limit = _compute_tool_result_limit(resolve_model_name(tier=tier))
        result_limit_mw = ToolResultLimitMiddleware(max_chars=worker_limit)

        agent = create_agent(
            model=model,
            tools=[claude_code],
            name=proj.name,
            system_prompt=prompt,
            middleware=[error_middleware, result_limit_mw, summarization_middleware, anthropic_cache, bedrock_cache],
        )
        workers.append(agent)

    return workers


_AGENT_LIFECYCLE_FOOTER = """

---
## Reporting results

You have two lifecycle tools — use them, do NOT silently finish without reporting:

- **`task_complete(summary)`** — call this when you are done. Your summary is the ONLY
  thing the user will see. If you skip this, the user gets nothing.
- **`escalate_question(question)`** — call this when you need user input before continuing.

Never finish without calling one of these tools. Even if you only have partial results,
call `task_complete` with what you found so far.
"""


def _build_worker_agents(
    agent_configs: list[AgentConfig],
    mcp_tools: list,
) -> list:
    """Create standard agents from AGENT.md configs using create_agent.

    Skips configs with type='deep_research' — those are built separately.
    """
    # Shared middleware (result_limit is per-worker due to proportional caps)
    from octo.models import resolve_model_name

    error_middleware = ToolErrorMiddleware()
    summarization_middleware = build_summarization_middleware()
    anthropic_cache, bedrock_cache = _caching_middleware()

    # Index builtin + lifecycle tools by name for agent-specific filtering
    builtin_by_name = {t.name: t for t in BUILTIN_TOOLS}
    lifecycle_by_name = {t.name: t for t in AGENT_LIFECYCLE_TOOLS}
    all_named = {**builtin_by_name, **lifecycle_by_name}

    workers = []
    for cfg in agent_configs:
        if cfg.type == "deep_research":
            continue

        tier = _agent_tier(cfg.name)
        model = _resolve_agent_model(cfg.model, tier)
        worker_limit = _compute_tool_result_limit(resolve_model_name(tier=tier))
        result_limit_mw = ToolResultLimitMiddleware(max_chars=worker_limit)

        if cfg.tools:
            # Agent specifies tool names — resolve from builtin, lifecycle, and MCP
            mcp_by_name = {t.name: t for t in mcp_tools}
            agent_tools = []
            unresolved = []
            for name in cfg.tools:
                if name in all_named:
                    agent_tools.append(all_named[name])
                elif name in mcp_by_name:
                    agent_tools.append(mcp_by_name[name])
                else:
                    unresolved.append(name)
            if unresolved:
                logger.warning(
                    "Agent '%s': could not resolve tools: %s "
                    "(not in builtins, lifecycle, or MCP)",
                    cfg.name, ", ".join(unresolved),
                )
        else:
            # No filter — give builtins + lifecycle + MCP proxy (deferred)
            agent_tools = list(BUILTIN_TOOLS) + list(AGENT_LIFECYCLE_TOOLS) + [find_tools, call_mcp_tool]

        # Always ensure lifecycle tools are present even if not explicitly listed
        existing_names = {t.name for t in agent_tools}
        for lt in AGENT_LIFECYCLE_TOOLS:
            if lt.name not in existing_names:
                agent_tools.append(lt)

        # Append lifecycle instructions so the agent always reports results
        system_prompt = cfg.system_prompt + _AGENT_LIFECYCLE_FOOTER

        agent = create_agent(
            model=model,
            tools=agent_tools,
            name=cfg.name,
            system_prompt=system_prompt,
            middleware=[error_middleware, result_limit_mw, summarization_middleware, anthropic_cache, bedrock_cache],
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

        model = _resolve_agent_model(cfg.model, "default")

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
        "## Efficiency\n\n"
        "For short or ambiguous user messages (under ~15 words, no clear task):\n"
        "- Respond directly from your knowledge and memory first\n"
        "- Do NOT launch tool calls to investigate unless specifically asked\n"
        "- If you genuinely need more info, ask the user one clarifying question\n"
        "- A 2-sentence answer is better than 10 tool calls followed by a paragraph"
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
        "Results are delivered to the user via Telegram and CLI.\n\n"
        "Use `manage_scheduled_tasks` to list, cancel, pause, or resume scheduled tasks:\n"
        "- `action='list'` — show all scheduled tasks with their IDs and status\n"
        "- `action='cancel', job_id='<id>'` — permanently remove a scheduled task\n"
        "- `action='pause', job_id='<id>'` — temporarily pause a task\n"
        "- `action='resume', job_id='<id>'` — resume a paused task"
    )

    parts.append(
        "## Background Tasks\n\n"
        "For long-running tasks (>2 minutes), use `dispatch_background` to run them "
        "independently. The user can continue chatting while the task runs.\n\n"
        "Two modes:\n"
        "- **process**: Subprocess (e.g., `claude -p 'analyze the codebase'`, shell commands). "
        "Done when the process exits.\n"
        "- **agent**: Standalone LangGraph agent with tools. Can call `task_complete` when done "
        "or `escalate_question` to ask the user something.\n\n"
        "The user is notified automatically when a task completes. They can check status "
        "with `/tasks` and view details with `/task <id>`.\n\n"
        "Use background tasks when the user asks for large analysis, code generation, "
        "research, or anything that would take several minutes."
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

    # Swarm peers — show connected peer Octo instances
    swarm_servers = [s for s in _summaries if s["server"].startswith("swarm-")]
    if swarm_servers:
        peer_lines = [
            f"- **{s['server']}** ({s['tools']} tools): {s['summary']}"
            for s in swarm_servers
        ]
        parts.append(
            "## Swarm Peers\n\n"
            "Other Octo instances are available on the network. Their tools "
            "are accessible via `find_tools` / `call_mcp_tool` (prefixed with "
            "their server name).\n\n"
            "Each peer exposes:\n"
            "- `ask(question, context)` — get an answer synchronously\n"
            "- `dispatch_task(task, context, priority)` — queue a background task\n"
            "- `check_task(task_id)` — check task status\n"
            "- `get_info()` — peer capabilities and load\n\n"
            "Use peer delegation when:\n"
            "- A peer has capabilities matching the task\n"
            "- You want parallel execution across instances\n"
            "- The task can be self-contained\n\n"
            "Connected peers:\n"
            + "\n".join(peer_lines)
        )

    return "\n\n---\n\n".join(parts)


# --- Custom handoff tools with message truncation -------------------------

# Max messages to forward when handing off to a worker.
# Workers only need the recent context, not the full session history.
_HANDOFF_MAX_MESSAGES = 20
_HANDOFF_MAX_CHARS_PER_MSG = 30_000  # same as SUPERVISOR_MSG_CHAR_LIMIT


def _build_truncating_handoff_tools(agent_names: list[str]) -> list:
    """Create custom handoff tools that truncate messages before forwarding.

    The default ``create_handoff_tool`` passes ``state["messages"]`` — the
    ENTIRE supervisor state — to the worker.  Over a long session this can
    be 500K+ chars, causing the worker's LLM to choke.

    Our custom tools:
    1. Truncate individual message content to ``_HANDOFF_MAX_CHARS_PER_MSG``
    2. Keep only the last ``_HANDOFF_MAX_MESSAGES`` messages
    3. Always preserve the original user message (first HumanMessage)

    ``create_supervisor`` detects these tools via their metadata
    (``METADATA_KEY_HANDOFF_DESTINATION``) and skips auto-creating
    default handoff tools.
    """
    from langgraph_supervisor.handoff import (
        METADATA_KEY_HANDOFF_DESTINATION,
        _normalize_agent_name,
        create_handoff_tool,
    )

    tools = []
    for agent_name in agent_names:
        # Start with the standard handoff tool (correct annotations/metadata)
        base_tool = create_handoff_tool(agent_name=agent_name)

        # Wrap it: intercept the call, truncate state messages, then delegate
        original_func = base_tool.func  # sync inner function

        def _make_wrapper(orig_fn, a_name):
            """Build a wrapper that truncates messages before handoff."""
            from langchain_core.messages import AIMessage, HumanMessage

            def wrapper(state, tool_call_id):
                messages = state.get("messages", [])

                # --- Truncate individual messages ---
                truncated = []
                for msg in messages:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str) and len(content) > _HANDOFF_MAX_CHARS_PER_MSG:
                        msg = msg.model_copy(update={
                            "content": content[:_HANDOFF_MAX_CHARS_PER_MSG]
                            + f"\n\n... [truncated from {len(content):,} chars]"
                        })
                    elif isinstance(content, list):
                        total = sum(len(str(p)) for p in content)
                        if total > _HANDOFF_MAX_CHARS_PER_MSG:
                            parts, running = [], 0
                            for part in content:
                                plen = len(str(part))
                                if running + plen > _HANDOFF_MAX_CHARS_PER_MSG:
                                    break
                                parts.append(part)
                                running += plen
                            msg = msg.model_copy(update={"content": parts})
                    truncated.append(msg)

                # --- Keep only recent messages ---
                if len(truncated) > _HANDOFF_MAX_MESSAGES:
                    first_human = next(
                        (m for m in truncated if isinstance(m, HumanMessage)),
                        None,
                    )
                    recent = truncated[-_HANDOFF_MAX_MESSAGES:]
                    if first_human and first_human not in recent:
                        recent = [first_human] + recent[1:]
                    truncated = recent

                orig_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
                new_chars = sum(len(str(getattr(m, "content", ""))) for m in truncated)
                if new_chars < orig_chars:
                    logger.info(
                        "Handoff to %s: truncated %d→%d chars (%d→%d msgs)",
                        a_name, orig_chars, new_chars,
                        len(messages), len(truncated),
                    )
                    # Dump dropped messages so the agent can recover context
                    dropped = [m for m in messages if m not in truncated]
                    if dropped:
                        ctx_path = _dump_handoff_context(a_name, dropped)
                        if ctx_path:
                            # Prepend a hint message so the agent knows
                            from langchain_core.messages import SystemMessage
                            hint = SystemMessage(content=(
                                f"[Earlier conversation context was trimmed. "
                                f"Full context saved to: {ctx_path} — "
                                f"use the Read tool if you need it.]"
                            ))
                            truncated = [hint] + truncated

                # Call original handoff with truncated state
                patched_state = {**state, "messages": truncated}
                return orig_fn(patched_state, tool_call_id)

            return wrapper

        # Replace the inner function but keep all tool metadata intact
        base_tool.func = _make_wrapper(original_func, agent_name)
        tools.append(base_tool)

    return tools


def _dump_handoff_context(agent_name: str, messages: list) -> str | None:
    """Save trimmed conversation context to workspace for agent recovery.

    When the handoff tool truncates messages, the dropped content is saved
    to ``.octo/workspace/<date>/handoff-context-<agent>-<ts>.md`` so the
    agent can read it back via the Read tool if it needs prior context.

    Returns the file path on success, None on failure.
    """
    from octo.config import RESEARCH_WORKSPACE

    try:
        today = date.today().isoformat()
        workspace = RESEARCH_WORKSPACE / today
        workspace.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        path = workspace / f"handoff-context-{agent_name}-{ts}.md"

        lines = [f"# Handoff Context for {agent_name}", ""]
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in content
                )
            if not content or not str(content).strip():
                continue
            # Cap each message at 2K chars in the dump to keep it manageable
            text = str(content).strip()
            if len(text) > 2000:
                text = text[:2000] + "..."
            lines.append(f"## [{role}]")
            lines.append(text)
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved handoff context to %s", path)
        return str(path)
    except Exception:
        logger.warning("Failed to save handoff context", exc_info=True)
        return None


async def build_graph(
    mcp_tools: list | None = None,
    mcp_tools_by_server: dict[str, list] | None = None,
    checkpointer: Any = None,
    storage: Any = None,
) -> Any:
    """Build and compile the full Octi supervisor graph.

    Args:
        mcp_tools: Pre-loaded MCP tools list.
        mcp_tools_by_server: MCP tools grouped by server name.
        checkpointer: Optional LangGraph checkpointer. If None, creates
            a default AsyncSqliteSaver using DB_PATH.
        storage: Optional StorageBackend instance. When provided, memory
            and planning tools use it instead of hardcoded filesystem paths.

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
    from octo.heartbeat import make_schedule_task_tool, make_manage_scheduled_tasks_tool
    schedule_task = make_schedule_task_tool()
    manage_scheduled_tasks = make_manage_scheduled_tasks_tool()

    # Build dispatch_background tool (background workers)
    from octo.background import make_dispatch_background_tool
    dispatch_background = make_dispatch_background_tool()

    # Wrap supervisor tools in a TruncatingToolNode that:
    # 1. handle_tool_errors=True — MCP errors returned as messages, not crashes
    # 2. Truncates oversized results (e.g. search_code returning 73K chars)
    #    before they enter the graph state and blow the context window.
    #    create_supervisor does NOT accept middleware, so this is the only
    #    way to get both error handling and result truncation at supervisor level.
    from octo.middleware import TRUNCATION_NOTICE
    from langgraph.prebuilt import ToolNode

    # Proportional limit for supervisor tools (adapts to model context window)
    from octo.models import resolve_model_name
    _sup_tool_limit = _compute_tool_result_limit(resolve_model_name())

    class TruncatingToolNode(ToolNode):
        """ToolNode that truncates oversized results before they enter state."""

        def _truncate_content(self, content):
            limit = _sup_tool_limit
            if isinstance(content, str) and len(content) > limit:
                notice = TRUNCATION_NOTICE.format(
                    original=len(content), limit=limit,
                )
                return content[:limit] + notice
            if isinstance(content, list):
                total = sum(len(str(p)) for p in content)
                if total > limit:
                    truncated = []
                    running = 0
                    for part in content:
                        part_len = len(str(part))
                        if running + part_len > limit:
                            if isinstance(part, dict) and "text" in part:
                                remaining = limit - running
                                if remaining > 200:
                                    trunc_part = dict(part)
                                    trunc_part["text"] = part["text"][:remaining] + TRUNCATION_NOTICE.format(
                                        original=total, limit=limit,
                                    )
                                    truncated.append(trunc_part)
                            break
                        truncated.append(part)
                        running += part_len
                    if not truncated:
                        return str(content)[:limit] + TRUNCATION_NOTICE.format(
                            original=total, limit=limit,
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

    # Use StorageBackend-backed tools when storage is provided (engine mode),
    # otherwise use module-level tools with hardcoded paths (CLI mode).
    if storage:
        from octo.core.tools.memory import make_memory_tools
        from octo.core.tools.planning import make_planning_tools
        _mem_tools = make_memory_tools(storage)
        _plan_tools = make_planning_tools(storage)
    else:
        _mem_tools = [write_memory, read_memories, update_long_term_memory]
        _plan_tools = [write_todos, read_todos, update_state_md]

    supervisor_tool_list = (
        list(BUILTIN_TOOLS)
        + [find_tools, call_mcp_tool]
        + _plan_tools + [use_skill] + _mem_tools
        + [schedule_task, manage_scheduled_tasks, send_file, dispatch_background]
    )

    # --- Model healthcheck at startup -----------------------------------------
    # Validate configured model IDs before starting the graph to catch
    # typos and invalid identifiers early (instead of at first invocation).
    from octo.models import resolve_model_name, _detect_provider
    _model_issues = []
    for tier_label, tier_val in [("supervisor/default", "default"), ("high", "high"), ("low", "low")]:
        name = resolve_model_name(tier=tier_val)
        provider = _detect_provider(name)
        if provider == "bedrock" and not any(c in name for c in ".:"):
            _model_issues.append(f"  {tier_label}: '{name}' — missing region prefix or version suffix for Bedrock")
        elif not name:
            _model_issues.append(f"  {tier_label}: (empty) — no model configured")
    if _model_issues:
        logger.warning("Model healthcheck warnings:\n%s", "\n".join(_model_issues))

    # --- Custom handoff tools with message truncation -------------------------
    # The default create_handoff_tool passes the ENTIRE supervisor state to
    # workers.  Over a long session the state accumulates hundreds of thousands
    # of chars, which causes Bedrock's "model identifier is invalid" error
    # when the worker's LLM tries to process the oversized context.
    #
    # Fix: create custom handoff tools that truncate messages before
    # forwarding.  create_supervisor detects these (via metadata) and skips
    # auto-creation of default handoff tools.
    all_workers = project_workers + octo_workers + deep_workers
    handoff_tools = _build_truncating_handoff_tools(
        [w.name for w in all_workers],
    )

    supervisor_tools = TruncatingToolNode(
        supervisor_tool_list + handoff_tools,
        handle_tool_errors=True,
    )

    total_tools = len(supervisor_tool_list) + len(handoff_tools)
    hook = _build_pre_model_hook(resolve_model_name(), tool_count=total_tools)

    workflow = create_supervisor(
        agents=all_workers,
        model=supervisor_model,
        tools=supervisor_tools,
        prompt=prompt,
        pre_model_hook=hook,
    )

    # Compile with checkpointer (use provided or create default SQLite one)
    if checkpointer is None:
        conn = await aiosqlite.connect(str(DB_PATH))
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
    app = workflow.compile(checkpointer=checkpointer)

    return app, all_agents, skills
