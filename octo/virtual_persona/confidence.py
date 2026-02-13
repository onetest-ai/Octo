"""Confidence scorer — LLM-based classification with hard escalation rules."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# --- Hard escalation keywords (always escalate regardless of LLM score) ---
_ESCALATION_KEYWORDS = [
    "urgent", "asap", "emergency", "critical issue", "production down",
    "p0", "p1", "incident", "outage",
]

# --- Categories that force escalation ---
_ALWAYS_ESCALATE_CATEGORIES = frozenset({
    "personal_decision",
    "sensitive_confidential",
    "social_relationship",
    "hr_legal",
    "financial",
})

# --- Categories that cap confidence at disclaim level ---
_CAP_DISCLAIM_CATEGORIES = frozenset({
    "realtime_context",
    "outside_expertise",
})

# Thresholds
RESPOND_THRESHOLD = 80
DISCLAIM_THRESHOLD = 60

_CLASSIFY_PROMPT = """\
You are a confidence scorer for a virtual assistant that responds on behalf of Artem Rozumenko.

Analyze the incoming message and determine:
1. Whether this message expects or needs a response from Artem (or is just acknowledgment/chatter)
2. How confident the virtual assistant should be in responding
3. The topic category

THREAD CONTEXT:
{thread_context}

SENDER PROFILE:
{user_profile}

RECENT MESSAGES:
{context}

NEW MESSAGE from {user_name} ({user_email}):
{query}

Respond with ONLY valid JSON, no other text:
{{
  "needs_response": true/false,
  "confidence": <0-100 integer>,
  "category": "<one of: technical_ai_ml, adjacent_technical, general_knowledge, personal_decision, sensitive_confidential, outside_expertise, realtime_context, social_relationship, hr_legal, financial, urgent_emergency, acknowledgment, social_chatter>",
  "escalation_flags": ["<flag1>", ...],
  "reasoning": "<1-2 sentence explanation>"
}}

SCORING GUIDELINES:
- technical_ai_ml (LangGraph, agents, Claude API, RAG, prompt engineering): 80-95
- adjacent_technical (general software, Python, architecture): 60-80
- general_knowledge (broad tech topics): 40-70
- personal_decision (meetings, commitments, assignments): 0-30
- sensitive_confidential (internal code, proprietary info): 0-20
- outside_expertise (non-tech domains): 0-40
- realtime_context ("what are you doing now?"): 0-30
- social_relationship (personal life, interpersonal): 0-20
- hr_legal / financial: 0-10
- urgent_emergency: 0-30 (flag for immediate escalation)
- acknowledgment ("ok", "thanks", "got it"): needs_response=false
- social_chatter (no question, just sharing): needs_response=false

ESCALATION FLAGS (include any that apply):
- "commitment_request" — asks for a promise/meeting/deadline
- "confidential_info" — references internal/proprietary systems
- "personal_judgment" — requires personal opinion about people/situations
- "urgent" — time-sensitive, needs immediate attention
- "unknown_person" — sender not in known contacts (be more conservative)

If the message is just an acknowledgment, social chatter, or doesn't expect a reply, \
set needs_response=false and confidence=0."""


@dataclass
class ClassificationResult:
    """Output of the confidence scoring pipeline."""

    needs_response: bool
    confidence: float  # 0-100
    category: str
    escalation_flags: list[str]
    reasoning: str
    decision: str  # "skip" | "respond" | "disclaim" | "escalate"


async def calculate_confidence(
    query: str,
    context: list[dict[str, str]] | None = None,
    user_email: str = "",
    user_name: str = "",
    thread_context: dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
    confidence_modifier: int = 0,
) -> ClassificationResult:
    """Score confidence for a VP response using low-tier LLM.

    Args:
        query: The incoming message text.
        context: Recent thread messages [{role, content}].
        user_email: Sender's email.
        user_name: Sender's display name.
        thread_context: From ConversationKnowledge (topic, summary, key_points).
        user_profile: From PeopleProfiles (title, tone, topics, interaction_count).
        confidence_modifier: Per-user adjustment from access control (-20 to +30).

    Returns:
        ClassificationResult with decision routing.
    """
    # Check for hard escalation keywords first
    hard_flags = _check_hard_escalation(query)

    # Format context for the prompt
    ctx_str = _format_context(context)
    thread_str = _format_thread_context(thread_context)
    profile_str = _format_user_profile(user_profile)

    prompt = _CLASSIFY_PROMPT.format(
        thread_context=thread_str,
        user_profile=profile_str,
        context=ctx_str,
        user_name=user_name or "Unknown",
        user_email=user_email or "unknown",
        query=query,
    )

    # Call low-tier LLM
    result = await _invoke_scorer(prompt)

    if result is None:
        # LLM failed — default to escalate for safety
        return ClassificationResult(
            needs_response=True,
            confidence=0.0,
            category="unknown",
            escalation_flags=hard_flags or ["llm_failure"],
            reasoning="Classification failed, escalating for safety",
            decision="escalate",
        )

    # Merge hard flags
    all_flags = list(set(result.get("escalation_flags", []) + hard_flags))

    # Extract fields
    needs_response = bool(result.get("needs_response", True))
    raw_confidence = float(result.get("confidence", 50))
    category = result.get("category", "unknown")
    reasoning = result.get("reasoning", "")

    # Apply hard rules
    if category in _ALWAYS_ESCALATE_CATEGORIES:
        raw_confidence = min(raw_confidence, DISCLAIM_THRESHOLD - 1)

    if category in _CAP_DISCLAIM_CATEGORIES:
        raw_confidence = min(raw_confidence, RESPOND_THRESHOLD - 1)

    if hard_flags:
        # Hard escalation keywords detected
        raw_confidence = min(raw_confidence, DISCLAIM_THRESHOLD - 1)
        if "urgent" in hard_flags:
            all_flags.append("urgent")

    # Apply per-user confidence modifier
    adjusted_confidence = max(0.0, min(100.0, raw_confidence + confidence_modifier))

    # Engagement-based adjustment by chat type.
    if thread_context:
        chat_type = thread_context.get("chat_type", "")
        engagement = thread_context.get("engagement", 0.0)
        msg_count = thread_context.get("message_count", 0)

        if chat_type == "oneOnOne":
            # 1-on-1: people expect a reply. Boost confidence by default.
            # Only penalize if Artem has actively ignored many messages
            # (engagement 0 with 5+ messages = deliberately not responding).
            if engagement == 0 and msg_count > 5:
                adjusted_confidence *= 0.7
            else:
                adjusted_confidence = min(100.0, adjusted_confidence * 1.15)
        elif chat_type not in ("group", "meeting"):
            # Unknown chat type — mild penalty for low engagement
            if engagement < 0.1:
                adjusted_confidence *= 0.7
        # Group/meeting: no engagement adjustment (0% is normal there,
        # poller already filters by @mention)

    # Unknown sender penalty
    if user_profile is None or user_profile.get("interaction_count", 0) == 0:
        if "unknown_person" not in all_flags:
            all_flags.append("unknown_person")
        # More conservative for unknown contacts
        adjusted_confidence = min(adjusted_confidence, adjusted_confidence * 0.85)

    # Route decision
    decision = _route_decision(needs_response, adjusted_confidence, all_flags)

    return ClassificationResult(
        needs_response=needs_response,
        confidence=round(adjusted_confidence, 1),
        category=category,
        escalation_flags=all_flags,
        reasoning=reasoning,
        decision=decision,
    )


def _route_decision(
    needs_response: bool, confidence: float, flags: list[str]
) -> str:
    """Map confidence score + flags to a routing decision."""
    if not needs_response:
        return "skip"

    # Any hard escalation flag forces escalate
    if "urgent" in flags and confidence < DISCLAIM_THRESHOLD:
        return "escalate"

    if confidence >= RESPOND_THRESHOLD:
        return "respond"
    elif confidence >= DISCLAIM_THRESHOLD:
        return "disclaim"
    else:
        return "escalate"


def _check_hard_escalation(query: str) -> list[str]:
    """Check for keywords that always trigger escalation."""
    q = query.lower()
    flags: list[str] = []
    for kw in _ESCALATION_KEYWORDS:
        if kw in q:
            flags.append("urgent")
            break
    return flags


def _format_context(context: list[dict[str, str]] | None) -> str:
    if not context:
        return "(no prior messages)"
    recent = context[-10:]
    return "\n".join(
        f"[{m.get('role', '?')}]: {m.get('content', '')[:200]}"
        for m in recent
    )


def _format_thread_context(ctx: dict[str, Any] | None) -> str:
    if not ctx:
        return "(no thread context available)"
    parts = []
    if ctx.get("topic"):
        parts.append(f"Topic: {ctx['topic']}")
    if ctx.get("summary"):
        parts.append(f"Summary: {ctx['summary']}")
    if ctx.get("key_points"):
        parts.append(f"Key points: {', '.join(ctx['key_points'][:5])}")
    if ctx.get("engagement") is not None:
        eng = ctx["engagement"]
        label = "high" if eng > 0.5 else "moderate" if eng > 0.2 else "low"
        parts.append(f"Artem's engagement in this thread: {label} ({eng:.0%})")
    if ctx.get("message_count"):
        parts.append(f"Thread messages: {ctx['message_count']}")
    return "\n".join(parts) if parts else "(no thread context available)"


def _format_user_profile(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "(unknown sender — be conservative)"
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("title"):
        parts.append(f"Title: {profile['title']}")
    if profile.get("department"):
        parts.append(f"Department: {profile['department']}")
    if profile.get("topics"):
        parts.append(f"Usual topics: {', '.join(profile['topics'][:5])}")
    if profile.get("interaction_count"):
        parts.append(f"Interactions: {profile['interaction_count']}")
    if profile.get("tone"):
        parts.append(f"Preferred tone: {profile['tone']}")
    return "\n".join(parts) if parts else "(unknown sender — be conservative)"


async def _invoke_scorer(prompt: str) -> dict[str, Any] | None:
    """Call low-tier LLM for classification. Returns parsed JSON or None."""
    try:
        from octo.models import make_model
    except ImportError:
        logger.warning("Cannot import make_model for confidence scoring")
        return None

    try:
        model = make_model(tier="low")
        response = await model.ainvoke(prompt)
        text = response.content.strip()

        # Strip markdown fencing if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(text)
        return data
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Confidence scoring LLM call failed: %s", exc)
        return None
