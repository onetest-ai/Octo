"""VP graph state definition."""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class VPState(TypedDict, total=False):
    """State passed through the VP decision graph.

    Fields are grouped by lifecycle stage:
    - Input: set by poller before graph invocation
    - Access control: set by access_check node
    - Classification: set by classify node
    - Output: set by responder/escalator nodes
    """

    # --- Input (set by poller) ---
    query: str                    # The sender's message text
    context: list[dict[str, str]] # Recent thread history [{role, content}]
    user_email: str               # Sender email / identifier
    user_name: str                # Sender display name
    chat_id: str                  # Teams/Telegram chat ID
    message_id: str               # Original message ID (for reply-to)
    source: str                   # "teams" or "telegram"

    # --- Thread knowledge (loaded by classify) ---
    thread_context: dict[str, Any]  # From ConversationKnowledge: topic, summary, key_points
    user_profile: dict[str, Any]    # From PeopleProfiles: title, tone, topics

    # --- Access control (set by access_check) ---
    access_decision: Literal["allow_ai", "always_user", "not_authorized"]
    confidence_modifier: int      # Per-user threshold adjustment (-20 to +30)
    escalation_priority: str      # "urgent" | "normal" | "low"

    # --- Classification (set by classify) ---
    confidence: float             # 0-100 score
    category: str                 # e.g. "technical_ai_ml", "personal_decision"
    escalation_flags: list[str]   # Hard escalation triggers detected
    classification_reasoning: str # LLM's reasoning for the score

    # --- Routing decision ---
    decision: Literal["skip", "respond", "disclaim", "escalate", "monitor"]

    # --- Cross-conversation context (set by gather_context) ---
    related_context: str          # Summary of related threads/emails for this person+topic

    # --- Delegation (set by delegate_to_octo) ---
    raw_answer: str               # Unformatted answer from Octo supervisor
    octo_thread_id: str           # VP-specific thread ID used for Octo invocation

    # --- Output (set by persona_format / escalate) ---
    response: str                 # Final text to send back
    audit_entry: dict[str, Any]   # Structured log entry for audit.jsonl

    # --- Content filter (set by poller / delegate_to_octo) ---
    content_filtered: bool        # True if content_filter sanitized the query
    filter_actions: list[str]     # List of applied filter actions

    # --- Runtime refs (set by poller, not persisted) ---
    _octo_app: Any                # Reference to Octo supervisor graph (for delegation)
    _octo_config: dict[str, Any]  # Config dict for Octo invocation
