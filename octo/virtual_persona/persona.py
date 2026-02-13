"""Persona generator — deep multi-conversation analysis and prompt generation.

Pipeline:
  1. Load conversations from local knowledge cache (requires prior /vp sync)
  2. Per-conversation analysis: extract behavioral patterns in context
  3. Web research: search public presence via Octo supervisor (articles, talks, social)
  4. Cross-conversation synthesis: delegate to Octo supervisor for deep analysis
  5. Generate personalized system prompt via Octo supervisor

Conversations are sorted by engagement index (how active the user was)
so the most representative chats are analyzed first. Only last 100 messages
per chat are used — older messages are already forgotten.

When octo_app is provided, phases 3-5 use the full Octo agent stack (subagents,
MCP tools including web search, skills). Without it, falls back to make_model(tier="low").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PER_CONVERSATION_PROMPT = """\
Analyze this person's messages in a {chat_type} conversation{topic_hint}.
Focus on HOW they communicate in this specific context.

Participants: {participants}
Chat type: {chat_type}

Their messages (with context of what they're replying to):
{messages}

Respond with ONLY valid JSON:
{{
  "context": "{chat_type}",
  "behavioral_notes": [
    "<specific observation about how they communicate HERE, e.g. 'uses profanity casually with close colleagues'>",
    "<observation 2>",
    "<observation 3>"
  ],
  "tone_in_context": "<how formal/casual they are in THIS conversation>",
  "language_used": "<primary language + any mixing observed>",
  "typical_message_length": "<very_short|short|medium|long>",
  "response_style": "<how they answer questions or react in this chat>",
  "notable_phrases": ["<characteristic phrase 1>", "<phrase 2>"],
  "expertise_shown": ["<topic they demonstrated knowledge of>"],
  "social_dynamics": "<how they interact with others here — leading, collaborative, observing, mentoring, etc.>"
}}"""

_WEB_RESEARCH_PROMPT = """\
I need you to research a person's public communication presence to help build \
a virtual persona that matches their real communication style.

Person: {user_name}
Email domain: {email_domain}
Known expertise topics: {topics}

Use web search to find:
1. Blog posts or articles they've written
2. Conference talks or presentations
3. GitHub/GitLab profile and contribution style (commit messages, PR reviews, issues)
4. Social media presence (LinkedIn, Twitter/X, etc.)
5. Professional profiles and bios
6. Any public technical discussions (forums, Stack Overflow, etc.)

For each source found, note:
- Their writing style (formal vs casual, technical depth, humor)
- Language(s) they use publicly
- How they explain technical concepts
- Their areas of expertise
- Any distinctive communication patterns

Return a structured summary of findings. If nothing is found, say so — that's fine."""

_SYNTHESIS_PROMPT = """\
You are creating a comprehensive communication profile by synthesizing data \
from multiple sources about one person.

## Per-conversation behavioral analyses ({conv_count} conversations):
{analyses}

## Public presence research:
{web_research}

Synthesize ALL of this into a unified profile. Identify:
- CONSISTENT patterns (appear across multiple conversations AND public presence)
- SITUATIONAL variations (behaves differently in different contexts)
- Core personality traits that persist regardless of context
- How their public voice compares to their private chat style

Respond with ONLY valid JSON:
{{
  "tone": {{
    "formality": "<formal|semi_formal|casual|very_casual>",
    "formality_variation": "<description of when they're more/less formal>",
    "directness": "<very_direct|direct|balanced|indirect>",
    "humor": "<frequent|occasional|rare|none>",
    "humor_style": "<type of humor they use>",
    "profanity": "<frequent|occasional|rare|never>",
    "emoticons_emoji": "<frequent|occasional|rare|never>"
  }},
  "language": {{
    "primary": "<main language>",
    "secondary": ["<other languages>"],
    "code_switching": "<frequent|occasional|rare|none>",
    "code_switching_pattern": "<when/how they switch languages>",
    "technical_jargon": "<heavy|moderate|light|none>"
  }},
  "message_style": {{
    "typical_length": "<very_short|short|medium|long>",
    "structure": "<telegraphic|conversational|structured|formal>",
    "greeting_habits": "<never|rarely|sometimes|always>",
    "capitalization": "<proper|mixed|minimal|none>",
    "punctuation": "<description of their punctuation habits>",
    "multi_message": "<whether they send multiple short messages vs one long one>"
  }},
  "personality_traits": [
    "<trait 1: 2-5 word description>",
    "<trait 2>",
    "<trait 3>",
    "<trait 4>",
    "<trait 5>"
  ],
  "behavioral_patterns": {{
    "in_group_chats": "<how they behave in groups>",
    "in_one_on_one": "<how they behave 1:1>",
    "when_asked_question": "<how they answer questions>",
    "when_explaining": "<how they explain technical concepts>",
    "when_uncertain": "<what they do when unsure>",
    "when_disagreeing": "<how they handle disagreement>",
    "when_sharing_work": "<how they present their work>",
    "when_problem_solving": "<their approach to solving problems collaboratively>"
  }},
  "public_vs_private": "<how their public communication style differs from private chats>",
  "expertise_topics": ["<topic 1>", "<topic 2>"],
  "common_phrases": ["<characteristic phrase 1>", "<phrase 2>"],
  "anti_patterns": [
    "<thing this person NEVER does>",
    "<anti-pattern 2>",
    "<anti-pattern 3>",
    "<anti-pattern 4>",
    "<anti-pattern 5>"
  ],
  "conversation_examples": [
    {{
      "context": "<situation description>",
      "their_message": "<actual message they sent>",
      "why_characteristic": "<why this is representative>"
    }}
  ],
  "summary": "<3-4 sentence characterization covering their core communication identity, \
how it shifts across contexts, and what makes their style distinctive>"
}}"""

_PROMPT_GENERATION = """\
You are building a system prompt for a virtual assistant that will respond on behalf \
of a real person in their work chat. The assistant must sound exactly like the actual \
person based on their deeply analyzed communication patterns.

Person: {user_name}

Comprehensive Communication Profile:
{analysis}

Generate a system prompt that:

1. VOICE & PERSONALITY — Capture their authentic tone. Include:
   - Their formality level and how it varies by context
   - Their humor style and when they use it
   - Their directness and how they handle uncertainty
   - Language mixing patterns (if any)
   - Specific phrases and expressions they use
   - What they NEVER do (anti-patterns)

2. BEHAVIORAL RULES — How they act in different situations:
   - Responding to technical questions
   - Explaining concepts
   - In group chats vs 1:1
   - When they're unsure
   - When sharing their work

3. EXPERTISE — What they can answer confidently vs must escalate

4. ESCALATION — Rules for when to stay silent:
   - Escalation is INVISIBLE — sender sees nothing
   - Personal decisions, scheduling, sensitive topics → silent escalation
   - Unknown topics → silent escalation

5. EXAMPLES — 5+ realistic Q&A pairs showing the expected style, drawn from \
actual patterns in the analysis

6. TRANSPARENCY — Add 🤖 at end of responses. Be honest about being virtual \
assistant if asked directly.

7. LANGUAGE — Match the sender's language. Include code-switching patterns if \
the person naturally mixes languages.

Write the complete system prompt as markdown. No code fences — just the raw prompt \
text that will be loaded directly into an LLM:"""


# ---------------------------------------------------------------------------
# Octo supervisor helper
# ---------------------------------------------------------------------------


async def _invoke_octo(
    octo_app: Any,
    prompt: str,
    thread_id: str = "vp:persona-gen",
    octo_config: dict[str, Any] | None = None,
) -> str:
    """Invoke the Octo supervisor graph and return the AI response text.

    Uses the full agent stack: subagents, MCP tools (including web search), skills.
    The thread_id is shared across calls so the supervisor maintains context.
    """
    cfg = {
        "configurable": {
            "thread_id": thread_id,
            **(octo_config or {}).get("configurable", {}),
        },
    }

    result = await octo_app.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=cfg,
    )

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
        if content and hasattr(msg, "type") and msg.type == "ai":
            return content
    # Fallback: last message
    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))
    return ""


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def fetch_from_cache(
    max_chats: int = 50,
    min_engagement: float = 0.05,
    min_messages: int = 3,
) -> list[dict[str, Any]]:
    """Load conversations from locally synced knowledge cache.

    Reads from ``knowledge/threads.json`` (index) and
    ``knowledge/messages/<hash>.json`` (cached messages).
    No Teams MCP calls — works fully offline after ``/vp sync``.

    Threads are sorted by engagement (most active first) and filtered
    by minimum engagement ratio and message count.

    Returns list of conversation dicts compatible with ``analyze_per_conversation``:
    ``{chat_id, chat_type, topic, participants, messages, own_message_count, total_messages}``
    """
    from octo.config import VP_DIR
    from octo.virtual_persona.knowledge import ConversationKnowledge

    knowledge = ConversationKnowledge(VP_DIR / "knowledge")

    threads = knowledge.get_active_threads(
        min_engagement=min_engagement,
        min_messages=min_messages,
    )

    if not threads:
        logger.warning("No active threads in knowledge cache — run /vp sync first")
        return []

    conversations: list[dict[str, Any]] = []
    for thread in threads[:max_chats]:
        chat_id = thread["chat_id"]
        cached_msgs = knowledge.get_cached_messages(chat_id)
        if not cached_msgs:
            continue

        # Convert cached format to persona analysis format
        processed_msgs: list[dict[str, Any]] = []
        own_count = 0
        for m in cached_msgs:
            content = m.get("content", "").strip()
            if not content or len(content) < 3:
                continue
            is_self = m.get("role") == "assistant"
            if is_self:
                own_count += 1
            processed_msgs.append({
                "sender": m.get("sender_name", "?"),
                "content": content[:500],
                "is_self": is_self,
            })

        if not processed_msgs:
            continue

        chat_type = thread.get("chat_type", "oneOnOne")
        conversations.append({
            "chat_id": chat_id,
            "chat_type": "group" if chat_type == "group" else "one_on_one",
            "topic": thread.get("topic", ""),
            "participants": thread.get("participants", []),
            "messages": processed_msgs,
            "own_message_count": own_count,
            "total_messages": len(processed_msgs),
            "engagement": thread.get("engagement", 0),
        })

    return conversations


async def analyze_per_conversation(
    conversations: list[dict[str, Any]],
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    """Run per-conversation behavioral analysis.

    Uses low-tier LLM directly for structured JSON output (needs fine control).
    """
    from octo.models import make_model

    model = make_model(tier="low")
    analyses: list[dict[str, Any]] = []

    for i, conv in enumerate(conversations):
        if on_progress:
            on_progress(i + 1, len(conversations), conv.get("chat_type", ""))

        # Format messages with context (show both sides for behavioral analysis)
        formatted = _format_conversation_messages(conv["messages"])
        if not formatted:
            continue

        topic_hint = f" about '{conv['topic']}'" if conv.get("topic") else ""
        participants_str = ", ".join(conv.get("participants", [])[:5]) or "unknown"

        prompt = _PER_CONVERSATION_PROMPT.format(
            chat_type=conv["chat_type"],
            topic_hint=topic_hint,
            participants=participants_str,
            messages=formatted,
        )

        try:
            response = await model.ainvoke(prompt)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            result["chat_id"] = conv["chat_id"]
            result["chat_type"] = conv["chat_type"]
            result["participant_count"] = len(conv.get("participants", []))
            result["message_count"] = conv["total_messages"]
            analyses.append(result)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Per-conversation analysis failed for chat %s: %s",
                conv["chat_id"][:8], exc,
            )
            continue

    return analyses


async def research_public_presence(
    user_name: str,
    user_email: str,
    topics: list[str],
    octo_app: Any | None = None,
    octo_config: dict[str, Any] | None = None,
) -> str:
    """Research the person's public communication presence via web search.

    When octo_app is provided, delegates to the Octo supervisor which has access
    to web search MCP tools (tavily_search, tavily_extract). Without it, returns
    empty string (web research skipped).
    """
    if octo_app is None:
        return ""

    email_domain = user_email.split("@")[-1] if "@" in user_email else ""
    topics_str = ", ".join(topics[:10]) if topics else "unknown"

    prompt = _WEB_RESEARCH_PROMPT.format(
        user_name=user_name or "unknown",
        email_domain=email_domain,
        topics=topics_str,
    )

    try:
        result = await _invoke_octo(
            octo_app, prompt,
            thread_id="vp:persona-gen",
            octo_config=octo_config,
        )
        return result
    except Exception as exc:
        logger.warning("Web research failed: %s", exc)
        return ""


async def synthesize_profile(
    per_conversation: list[dict[str, Any]],
    web_research: str = "",
    octo_app: Any | None = None,
    octo_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Cross-conversation synthesis — merge per-chat analyses into unified profile.

    When octo_app is provided, delegates to the Octo supervisor for deeper analysis.
    Falls back to low-tier LLM when octo_app is not available.
    """
    if not per_conversation:
        return None

    # Format all per-conversation analyses
    analyses_text = "\n\n---\n\n".join(
        f"Conversation {i+1} ({a.get('chat_type', '?')}, "
        f"{a.get('participant_count', '?')} participants, "
        f"{a.get('message_count', '?')} messages):\n"
        + json.dumps(
            {k: v for k, v in a.items()
             if k not in ("chat_id", "participant_count", "message_count")},
            indent=2, ensure_ascii=False,
        )
        for i, a in enumerate(per_conversation)
    )

    prompt = _SYNTHESIS_PROMPT.format(
        conv_count=len(per_conversation),
        analyses=analyses_text[:12000],
        web_research=web_research[:4000] if web_research else "(no public presence data)",
    )

    try:
        if octo_app is not None:
            raw = await _invoke_octo(
                octo_app, prompt,
                thread_id="vp:persona-gen",
                octo_config=octo_config,
            )
        else:
            from octo.models import make_model
            model = make_model(tier="low")
            response = await model.ainvoke(prompt)
            raw = response.content.strip()

        # Extract JSON from response
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Find first { and last } for JSON extraction
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Profile synthesis failed: %s", exc)
        return None


async def generate_persona_prompt(
    profile: dict[str, Any],
    user_name: str = "",
    octo_app: Any | None = None,
    octo_config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a VP system prompt from a synthesized communication profile.

    When octo_app is provided, delegates to the Octo supervisor for richer output.
    Falls back to low-tier LLM when octo_app is not available.
    """
    analysis_text = json.dumps(profile, indent=2, ensure_ascii=False)
    prompt = _PROMPT_GENERATION.format(
        user_name=user_name or "the user",
        analysis=analysis_text[:12000],
    )

    try:
        if octo_app is not None:
            text = await _invoke_octo(
                octo_app, prompt,
                thread_id="vp:persona-gen",
                octo_config=octo_config,
            )
        else:
            from octo.models import make_model
            model = make_model(tier="low")
            response = await model.ainvoke(prompt)
            text = response.content.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text if text else None
    except Exception as exc:
        logger.warning("Prompt generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def full_persona_pipeline(
    vp_dir: Path,
    user_name: str = "",
    octo_app: Any = None,
    octo_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Full pipeline: fetch → per-conv analysis → web research → synthesis → prompt.

    When octo_app is provided, the heavy phases (web research, synthesis, prompt
    generation) use the full Octo agent stack including web search, subagents,
    and skills. Without it, falls back to low-tier LLM for synthesis/generation
    and skips web research.

    Returns (profile_dict, generated_prompt) — either may be None on failure.
    """
    from octo import ui

    using_octo = octo_app is not None
    if using_octo:
        ui.print_info("Using Octo agent stack for deep analysis (web search + full agent stack)")
    else:
        ui.print_info("Running in lightweight mode (no web search). Pass octo_app for deeper analysis.")

    # Step 1: Load conversations from local cache (requires prior /vp sync)
    ui.print_info("Loading conversations from local knowledge cache...")
    conversations = fetch_from_cache(max_chats=50)
    if not conversations:
        ui.print_error("No conversations in cache. Run '/vp sync' first to fetch Teams data.")
        return None, None

    own_total = sum(c["own_message_count"] for c in conversations)
    ui.print_info(
        f"Loaded {len(conversations)} conversations with {own_total} of your messages (from cache)"
    )
    if len(conversations) < 5:
        ui.print_info("(More conversations = better analysis. Run '/vp sync' to refresh data)")

    # Step 2: Per-conversation analysis (low-tier LLM — structured, fast)
    ui.print_info("Analyzing behavioral patterns per conversation...")

    def _progress(current: int, total: int, chat_type: str) -> None:
        ui.print_info(f"  [{current}/{total}] analyzing {chat_type} chat...")

    per_conv = await analyze_per_conversation(conversations, on_progress=_progress)
    if not per_conv:
        ui.print_error("Per-conversation analysis failed — no results")
        return None, None
    ui.print_info(f"Analyzed {len(per_conv)} conversations successfully")

    # Save per-conversation analyses
    per_conv_path = vp_dir / "per-conversation-analyses.json"
    per_conv_path.write_text(
        json.dumps(per_conv, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Step 3: Web research (Octo supervisor — uses web search tools)
    web_research = ""
    if using_octo:
        # Extract user email from self-emails for web search
        from octo.virtual_persona.poller import _SELF_EMAILS
        user_email = next(iter(_SELF_EMAILS), "")

        # Collect expertise topics from per-conversation analyses
        all_topics: list[str] = []
        for a in per_conv:
            all_topics.extend(a.get("expertise_shown", []))
        unique_topics = list(dict.fromkeys(all_topics))[:15]

        ui.print_info("Researching public communication presence via web search...")
        web_research = await research_public_presence(
            user_name=user_name,
            user_email=user_email,
            topics=unique_topics,
            octo_app=octo_app,
            octo_config=octo_config,
        )
        if web_research:
            # Save web research
            web_path = vp_dir / "web-research.md"
            web_path.write_text(web_research + "\n", encoding="utf-8")
            ui.print_info("Public presence research completed")
        else:
            ui.print_info("No public presence data found (continuing without it)")
    else:
        ui.print_info("Skipping web research (no Octo supervisor available)")

    # Step 4: Cross-conversation synthesis (Octo supervisor or low-tier LLM)
    ui.print_info("Synthesizing communication profile across all conversations...")
    profile = await synthesize_profile(
        per_conv,
        web_research=web_research,
        octo_app=octo_app,
        octo_config=octo_config,
    )
    if profile is None:
        ui.print_error("Profile synthesis failed")
        return None, None

    # Save profile
    profile_path = vp_dir / "communication-analysis.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ui.print_status(f"Communication profile saved ({len(per_conv)} conversations analyzed)")

    # Step 5: Generate persona prompt (Octo supervisor or low-tier LLM)
    ui.print_info("Generating persona prompt from profile...")
    prompt_text = await generate_persona_prompt(
        profile,
        user_name=user_name,
        octo_app=octo_app,
        octo_config=octo_config,
    )
    if prompt_text is None:
        ui.print_error("Prompt generation failed — profile saved, retry with /vp persona generate")
        return profile, None

    # Save prompt
    prompt_path = vp_dir / "system-prompt.md"
    prompt_path.write_text(prompt_text + "\n", encoding="utf-8")

    # Invalidate cached prompt in VP graph
    import octo.virtual_persona.graph as _vp_graph_mod
    _vp_graph_mod._persona_prompt = None

    ui.print_status("Persona prompt generated!")

    # Print summary
    summary = profile.get("summary", "")
    if summary:
        ui.print_info(f"\nProfile summary: {summary}")

    traits = profile.get("personality_traits", [])
    if traits:
        ui.print_info(f"Traits: {', '.join(traits[:5])}")

    return profile, prompt_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_conversation_messages(
    messages: list[dict[str, Any]], max_messages: int = 30,
) -> str:
    """Format conversation messages showing both sides for behavioral analysis."""
    # Take a spread sample if too many
    if len(messages) > max_messages:
        step = len(messages) / max_messages
        messages = [messages[int(i * step)] for i in range(max_messages)]

    lines: list[str] = []
    for m in messages:
        sender = m.get("sender", "?")
        content = m.get("content", "")[:300]
        marker = " [SELF]" if m.get("is_self") else ""
        lines.append(f"[{sender}{marker}]: {content}")
    return "\n".join(lines)
