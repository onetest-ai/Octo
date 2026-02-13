"""VP decision graph — StateGraph pipeline for message classification and response."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded persona prompt (cached after first read)
_persona_prompt: str | None = None


def _load_persona_prompt() -> str:
    """Load the VP system prompt from .octo/virtual-persona/system-prompt.md."""
    global _persona_prompt
    if _persona_prompt is not None:
        return _persona_prompt

    from octo.config import VP_DIR

    prompt_path = VP_DIR / "system-prompt.md"
    if prompt_path.is_file():
        _persona_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        _persona_prompt = (
            "You are a virtual assistant representing the user. "
            "Rewrite the answer in a casual, direct, technical style. "
            "Keep it short. Add 🤖 at end."
        )
    return _persona_prompt


# ---------------------------------------------------------------------------
# Graph nodes — each takes VPState, returns partial VPState update
# ---------------------------------------------------------------------------


def check_delegation_lock(state: dict[str, Any]) -> dict[str, Any]:
    """Check if this chat is locked (delegated to real Artem).

    If locked, set decision='skip' so the graph terminates silently.
    """
    from octo.config import VP_DIR
    from octo.virtual_persona.access_control import AccessControl

    chat_id = state.get("chat_id", "")
    if not chat_id:
        return {}

    ac = AccessControl(VP_DIR / "access-control.yaml")
    if ac.is_delegated(chat_id):
        logger.debug("VP: chat %s is delegated, skipping", chat_id)
        return {"decision": "skip", "classification_reasoning": "Thread delegated to real Artem"}

    return {}


def access_check(state: dict[str, Any]) -> dict[str, Any]:
    """Check access control lists. Sets access_decision and related fields."""
    # Skip if already decided (delegation lock)
    if state.get("decision") == "skip":
        return {}

    from octo.config import VP_DIR
    from octo.virtual_persona.access_control import AccessControl

    ac = AccessControl(VP_DIR / "access-control.yaml")
    decision = ac.check_access(
        state.get("user_email", ""),
        state.get("chat_id", ""),
    )

    return {
        "access_decision": decision.action,
        "confidence_modifier": decision.confidence_modifier,
        "escalation_priority": decision.notify_priority,
    }


async def classify(state: dict[str, Any]) -> dict[str, Any]:
    """Run LLM-based confidence scoring. Sets confidence, category, decision.

    Access control routing:
    - allow_ai users: normal scoring → respond/disclaim/escalate/skip
    - always_user / not_authorized: still classify (for context), but route
      to "monitor" (if response needed) or "skip" (if not).
      Monitor = observe + suggest to Artem, completely invisible to sender.
    """
    # Skip if already decided
    if state.get("decision") == "skip":
        return {}

    is_monitored = state.get("access_decision") in ("always_user", "not_authorized")

    # Load thread context and user profile
    from octo.config import VP_DIR
    from octo.virtual_persona.knowledge import ConversationKnowledge
    from octo.virtual_persona.profiles import PeopleProfiles

    knowledge = ConversationKnowledge(VP_DIR / "knowledge")
    profiles = PeopleProfiles(VP_DIR / "profiles.json")

    thread_ctx = knowledge.get_thread_context(state.get("chat_id", ""))
    user_profile = profiles.get_profile(state.get("user_email", ""))

    from octo.virtual_persona.confidence import calculate_confidence

    result = await calculate_confidence(
        query=state.get("query", ""),
        context=state.get("context"),
        user_email=state.get("user_email", ""),
        user_name=state.get("user_name", ""),
        thread_context=thread_ctx,
        user_profile=user_profile,
        confidence_modifier=state.get("confidence_modifier", 0),
    )

    # Override decision for monitored users
    if is_monitored:
        if result.needs_response:
            decision = "monitor"
        else:
            decision = "skip"
    else:
        decision = result.decision

    return {
        "thread_context": thread_ctx or {},
        "user_profile": user_profile or {},
        "confidence": result.confidence,
        "category": result.category,
        "escalation_flags": result.escalation_flags,
        "classification_reasoning": result.reasoning,
        "decision": decision,
    }


def gather_context(state: dict[str, Any]) -> dict[str, Any]:
    """Gather cross-conversation context from the local knowledge base.

    Fast local search — no LLM calls, no API calls.
    Finds related threads by topic keywords and sender email.
    The deeper search (emails, Teams) is delegated to the context-gatherer
    agent via the Octo supervisor in delegate_to_octo.
    """
    # Skip if already decided to skip
    if state.get("decision") == "skip":
        return {}

    from octo.config import VP_DIR
    from octo.virtual_persona.knowledge import ConversationKnowledge

    knowledge = ConversationKnowledge(VP_DIR / "knowledge")
    current_chat_id = state.get("chat_id", "")
    user_email = state.get("user_email", "")
    thread_ctx = state.get("thread_context") or {}

    # --- 1. Search by topic keywords ---
    topic = thread_ctx.get("topic", "")
    key_points = thread_ctx.get("key_points") or []

    # Build search terms from topic + key points
    search_terms: list[str] = []
    if topic:
        # Split topic into meaningful words (skip short ones)
        search_terms.extend(
            w for w in topic.lower().split() if len(w) > 3
        )
    for kp in key_points[:3]:
        words = [w for w in kp.lower().split() if len(w) > 3]
        search_terms.extend(words[:2])

    # Deduplicate
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in search_terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)
    search_terms = unique_terms[:6]  # Cap to avoid too many searches

    topic_threads: list[dict] = []
    seen_ids: set[str] = set()
    for term in search_terms:
        for t in knowledge.search(term):
            tid = t.get("chat_id", "")
            if tid and tid != current_chat_id and tid not in seen_ids:
                seen_ids.add(tid)
                topic_threads.append(t)

    # --- 2. Find all threads with this person ---
    person_threads: list[dict] = []
    if user_email:
        all_threads = knowledge.list_threads(n=50)
        for t in all_threads:
            tid = t.get("chat_id", "")
            if tid == current_chat_id or tid in seen_ids:
                continue
            participants = t.get("participants") or []
            if user_email.lower() in [p.lower() for p in participants]:
                person_threads.append(t)
                seen_ids.add(tid)

    if not topic_threads and not person_threads:
        return {"related_context": ""}

    # --- 3. Build structured context string ---
    parts: list[str] = []
    user_name = state.get("user_name", user_email)

    if topic_threads:
        parts.append(f"Related conversations (by topic):")
        for t in topic_threads[:5]:
            t_topic = t.get("topic", "unknown")
            t_summary = t.get("summary", "")
            t_date = t.get("last_message_at", t.get("last_updated", ""))[:10]
            line = f"- [{t_date}] {t_topic}"
            if t_summary:
                line += f": {t_summary[:150]}"
            parts.append(line)

    if person_threads:
        parts.append(f"\nOther conversations with {user_name}:")
        for t in person_threads[:5]:
            t_topic = t.get("topic", "unknown")
            t_summary = t.get("summary", "")
            t_date = t.get("last_message_at", t.get("last_updated", ""))[:10]
            line = f"- [{t_date}] {t_topic}"
            if t_summary:
                line += f": {t_summary[:150]}"
            parts.append(line)

    related = "\n".join(parts)
    if related:
        logger.info(
            "VP: gathered %d topic + %d person related threads for %s",
            len(topic_threads), len(person_threads), user_email,
        )

    return {"related_context": related}


async def delegate_to_octo(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the Octo supervisor graph for knowledge work.

    The VP doesn't generate answers — it delegates to Octo's full agent stack
    (subagents, MCP tools, skills) and gets back a raw answer.

    Includes three layers of defense:
    1. Pre-filter the query via content_filter.sanitize()
    2. Catch content-policy / connection errors from the LLM
    3. Heal the checkpoint on error and silently escalate
    """
    # Skip expensive Octo call for low-confidence monitor messages.
    # The notification still fires (poller checks decision), just without
    # a suggested answer — saves tokens and latency.
    _MONITOR_DELEGATION_THRESHOLD = 40
    if (
        state.get("decision") == "monitor"
        and state.get("confidence", 0) < _MONITOR_DELEGATION_THRESHOLD
    ):
        logger.debug(
            "VP: skipping Octo delegation for low-confidence monitor (%.0f%%)",
            state.get("confidence", 0),
        )
        return {"raw_answer": ""}

    octo_app = state.get("_octo_app")
    octo_config = state.get("_octo_config", {})

    if octo_app is None:
        return {"raw_answer": "[VP Error: Octo supervisor not available]"}

    # Build VP-specific thread ID for conversation continuity
    chat_id = state.get("chat_id", "unknown")
    octo_thread_id = f"vp:{chat_id}"

    # --- Layer 1: Sanitize query before sending to LLM ---
    from octo.virtual_persona.content_filter import sanitize

    query = state.get("query", "")
    query, filter_actions = sanitize(query)
    content_filtered = bool(filter_actions)

    # Build Octo instruction prefix for constructive answers
    octo_instruction = (
        "You are helping prepare an answer for a colleague's message. "
        "Provide a constructive, actionable, and technically accurate response. "
        "Include specific details, code examples, links, or next steps where relevant. "
        "Be helpful and thorough — the answer will be reformatted into the person's voice later, "
        "so focus on substance and correctness, not style.\n\n"
    )

    # Add thread context for grounding
    thread_ctx = state.get("thread_context")
    if thread_ctx and thread_ctx.get("summary"):
        octo_instruction += (
            f"Context: This is from a conversation about '{thread_ctx.get('topic', 'unknown')}'. "
            f"Summary: {thread_ctx.get('summary', '')}.\n\n"
        )

    # Add cross-conversation context from local knowledge base
    related = state.get("related_context", "")
    if related:
        octo_instruction += (
            f"Related context from other conversations:\n{related}\n\n"
        )

    # For respond/disclaim: instruct Octo to search deeper via context-gatherer
    user_email = state.get("user_email", "")
    if state.get("decision") in ("respond", "disclaim") and user_email:
        octo_instruction += (
            "IMPORTANT: Before answering, use the context-gatherer agent to search "
            "for relevant emails and Teams conversations involving "
            f"{user_email} about similar topics. "
            "Incorporate any relevant findings into your answer.\n\n"
        )

    # Include recent conversation messages (critical for understanding context)
    context_msgs = state.get("context") or []
    if context_msgs:
        octo_instruction += "Recent conversation history (most recent last):\n"
        for cm in context_msgs:
            role = cm.get("role", "user")
            content = cm.get("content", "")
            label = "You" if role == "assistant" else "Colleague"
            octo_instruction += f"  {label}: {content}\n"
        octo_instruction += "\n"

    query = octo_instruction + f"Colleague's latest message: {query}"

    config = {
        "configurable": {
            "thread_id": octo_thread_id,
            **octo_config.get("configurable", {}),
        },
    }

    try:
        result = await octo_app.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
        # Extract the last AI message
        messages = result.get("messages", [])
        raw = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
            if content and hasattr(msg, "type") and msg.type == "ai":
                raw = content
                break
        if not raw and messages:
            last = messages[-1]
            raw = getattr(last, "content", str(last))

        return {
            "raw_answer": raw,
            "octo_thread_id": octo_thread_id,
            "content_filtered": content_filtered,
            "filter_actions": filter_actions,
        }
    except Exception as exc:
        # --- Layer 2 + 3: Detect content-policy errors, heal, silently escalate ---
        logger.error("VP delegate_to_octo failed: %s", exc)

        healed = False
        error_category = ""
        try:
            from octo.retry import (
                _classify_error,
                auto_clean_corrupted,
                auto_repair_orphaned_tools,
            )

            error_category = _classify_error(exc) or "unknown"

            if error_category in ("connection_closed", "orphaned_tools"):
                # Heal the VP's Octo thread checkpoint
                repaired = await auto_repair_orphaned_tools(octo_app, config)
                if not repaired:
                    await auto_clean_corrupted(octo_app, config)
                healed = True
                logger.info(
                    "VP healed checkpoint for thread %s after %s",
                    octo_thread_id,
                    error_category,
                )

                # Reset Bedrock client for fresh connection
                try:
                    from octo.models import reset_bedrock_client
                    reset_bedrock_client()
                except ImportError:
                    pass
        except Exception as heal_exc:
            logger.warning("VP checkpoint healing failed: %s", heal_exc)

        # Return error context for Artem's notification (sender never sees this)
        error_detail = (
            f"[VP Error: {error_category or 'unknown'} — {exc}]"
            f"{' (checkpoint healed)' if healed else ''}"
        )
        return {
            "raw_answer": error_detail,
            "octo_thread_id": octo_thread_id,
            "decision": "escalate",  # Override: silently escalate
            "content_filtered": content_filtered,
            "filter_actions": filter_actions + [f"error:{error_category}"],
        }


async def persona_format(state: dict[str, Any]) -> dict[str, Any]:
    """Reformat Octo's raw answer in Artem's persona style."""
    raw = state.get("raw_answer", "")
    if not raw or raw.startswith("[VP Error"):
        return {"response": raw}

    # Determine tone from profile
    profile = state.get("user_profile", {})
    tone = profile.get("tone", "casual_technical")

    from octo.virtual_persona.profiles import TONE_PRESETS

    tone_instruction = TONE_PRESETS.get(tone, TONE_PRESETS["casual_technical"])

    persona_prompt = _load_persona_prompt()

    format_prompt = (
        f"{persona_prompt}\n\n"
        f"---\n"
        f"Tone for this person: {tone_instruction}\n\n"
        f"Rewrite the following answer in the user's style. Keep it concise and natural. "
        f"Do NOT add any preamble — just the reformatted response text.\n\n"
        f"Raw answer:\n{raw[:3000]}"
    )

    try:
        from octo.models import make_model

        model = make_model(tier="low")
        response = await model.ainvoke(format_prompt)
        text = response.content.strip()
        # Ensure robot emoji marker
        if "🤖" not in text:
            text += " 🤖"
        return {"response": text}
    except Exception as exc:
        logger.warning("VP persona_format failed: %s", exc)
        # Fallback: return raw answer with emoji
        return {"response": f"{raw[:2000]} 🤖"}


async def persona_format_disclaim(state: dict[str, Any]) -> dict[str, Any]:
    """Reformat with a disclaimer caveat."""
    result = await persona_format(state)
    response = result.get("response", "")
    # Add disclaimer if not already present
    if "virtual assistant" not in response.lower() and "verify" not in response.lower():
        response += "\n\n(virtual assistant responding — verify with the real person if critical)"
    return {"response": response}


def escalate_finalize(state: dict[str, Any]) -> dict[str, Any]:
    """Lock the thread so VP stays away until Artem releases it.

    No response is sent to the sender — escalation is invisible.
    The poller handles private notification to Artem separately.
    """
    from octo.config import VP_DIR
    from octo.virtual_persona.access_control import AccessControl

    chat_id = state.get("chat_id", "")
    if chat_id:
        ac = AccessControl(VP_DIR / "access-control.yaml")
        ac.lock_thread(
            chat_id=chat_id,
            reason=f"Escalated: {state.get('category', 'unknown')}",
            user_email=state.get("user_email", ""),
            query_preview=state.get("query", "")[:200],
        )

    # No response — invisible to sender
    return {}


def log_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Write audit entry and update stats + profiles."""
    from octo.config import VP_DIR
    from octo.virtual_persona.stats import VPStats
    from octo.virtual_persona.profiles import PeopleProfiles

    decision = state.get("decision", "unknown")

    entry = {
        "decision": decision,
        "confidence": state.get("confidence", 0),
        "category": state.get("category", ""),
        "user_email": state.get("user_email", ""),
        "user_name": state.get("user_name", ""),
        "chat_id": state.get("chat_id", ""),
        "source": state.get("source", ""),
        "query_preview": (state.get("query", ""))[:200],
        "escalation_flags": state.get("escalation_flags", []),
        "reasoning": state.get("classification_reasoning", ""),
        "raw_answer_preview": (state.get("raw_answer", ""))[:200],
        "content_filtered": state.get("content_filtered", False),
        "filter_actions": state.get("filter_actions", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write audit + stats
    stats = VPStats(
        stats_path=VP_DIR / "stats.json",
        audit_path=VP_DIR / "audit.jsonl",
    )
    stats.record(entry)

    # Record interaction in profiles (unless skip)
    if decision != "skip":
        email = state.get("user_email", "")
        if email:
            profiles = PeopleProfiles(VP_DIR / "profiles.json")
            profiles.record_interaction(
                email=email,
                topic=state.get("thread_context", {}).get("topic", ""),
                category=state.get("category", ""),
            )

    return {"audit_entry": entry}


# ---------------------------------------------------------------------------
# Router — conditional edge after classify
# ---------------------------------------------------------------------------


def route_decision(state: dict[str, Any]) -> str:
    """Route to the correct node based on decision."""
    decision = state.get("decision", "escalate")
    return {
        "skip": "log_decision",
        "respond": "delegate_to_octo",
        "disclaim": "delegate_to_octo_disclaim",
        "escalate": "delegate_to_octo_escalate",
        "monitor": "delegate_to_octo_monitor",
    }.get(decision, "delegate_to_octo_monitor")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_vp_graph():
    """Build the VP decision StateGraph.

    Flow:
      check_delegation_lock → access_check → classify → gather_context → route:
        skip     → log → END                         (stay quiet)
        respond  → delegate → persona_format → log → END  (auto-respond)
        disclaim → delegate → persona_format_disclaim → log → END
        escalate → delegate → escalate_finalize → log → END  (silent, locks thread)
        monitor  → delegate → log → END              (silent, suggests to Artem)

    Escalate and monitor are invisible to sender — no reply sent.
    The poller sends private notifications to Artem for these paths.

    Returns a compiled LangGraph StateGraph ready for ainvoke().
    """
    from langgraph.graph import StateGraph, END

    from octo.virtual_persona.state import VPState

    graph = StateGraph(VPState)

    # Add nodes
    graph.add_node("check_delegation_lock", check_delegation_lock)
    graph.add_node("access_check", access_check)
    graph.add_node("classify", classify)
    graph.add_node("gather_context", gather_context)
    # Delegation nodes (same function, different routing after)
    graph.add_node("delegate_to_octo", delegate_to_octo)
    graph.add_node("delegate_to_octo_disclaim", delegate_to_octo)
    graph.add_node("delegate_to_octo_escalate", delegate_to_octo)
    graph.add_node("delegate_to_octo_monitor", delegate_to_octo)
    # Post-processing nodes
    graph.add_node("persona_format", persona_format)
    graph.add_node("persona_format_disclaim", persona_format_disclaim)
    graph.add_node("escalate_finalize", escalate_finalize)
    graph.add_node("log_decision", log_decision)

    # Edges: linear pipeline up to classify → gather_context
    graph.set_entry_point("check_delegation_lock")
    graph.add_edge("check_delegation_lock", "access_check")
    graph.add_edge("access_check", "classify")
    graph.add_edge("classify", "gather_context")

    # Conditional routing after gather_context
    graph.add_conditional_edges(
        "gather_context",
        route_decision,
        {
            "log_decision": "log_decision",
            "delegate_to_octo": "delegate_to_octo",
            "delegate_to_octo_disclaim": "delegate_to_octo_disclaim",
            "delegate_to_octo_escalate": "delegate_to_octo_escalate",
            "delegate_to_octo_monitor": "delegate_to_octo_monitor",
        },
    )

    # Respond path: delegate → persona_format → log
    graph.add_edge("delegate_to_octo", "persona_format")
    graph.add_edge("persona_format", "log_decision")

    # Disclaim path: delegate → persona_format_disclaim → log
    graph.add_edge("delegate_to_octo_disclaim", "persona_format_disclaim")
    graph.add_edge("persona_format_disclaim", "log_decision")

    # Escalate path: delegate → lock thread → log  (no response to sender)
    graph.add_edge("delegate_to_octo_escalate", "escalate_finalize")
    graph.add_edge("escalate_finalize", "log_decision")

    # Monitor path: delegate → log  (no response, no lock, poller notifies Artem)
    graph.add_edge("delegate_to_octo_monitor", "log_decision")

    # Log → END
    graph.add_edge("log_decision", END)

    return graph.compile()
