"""Utility functions for processing Teams messages."""
from datetime import datetime, timedelta
from typing import Any


def aggregate_consecutive_messages(
    messages: list[dict[str, Any]],
    time_window_minutes: int = 2,
) -> list[dict[str, Any]]:
    """Aggregate consecutive messages from the same sender within a time window.
    
    Teams users often send multiple messages in quick succession instead of
    writing one long message. This function groups those into conversational turns.
    
    Args:
        messages: List of message dicts from Teams API (list-chat-messages format).
                  Expected keys: id, from (dict with displayName, userId, email),
                  body, createdDateTime, contentType, messageType, attachments, mentions
        time_window_minutes: Max minutes between messages to consider them part
                            of the same turn (default: 2)
    
    Returns:
        List of aggregated message dicts with the same structure, but with:
        - body: combined text from all messages in the turn
        - aggregatedMessageIds: list of original message IDs that were combined
        - aggregatedCount: number of messages in this turn
    """
    if not messages:
        return []
    
    # Sort by timestamp (oldest first for proper turn grouping)
    sorted_msgs = sorted(
        messages,
        key=lambda m: m.get("createdDateTime", ""),
    )
    
    aggregated = []
    current_turn: dict[str, Any] | None = None
    current_sender: str = ""
    current_time: datetime | None = None
    
    for msg in sorted_msgs:
        # Extract sender and timestamp
        from_dict = msg.get("from") or {}
        sender = from_dict.get("userId", "") or from_dict.get("displayName", "")
        timestamp_str = msg.get("createdDateTime", "")
        
        # Skip messages without sender (system messages, etc.)
        if not sender:
            if current_turn:
                aggregated.append(current_turn)
                current_turn = None
            continue
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = None
        
        # Decide if this message starts a new turn
        start_new_turn = (
            current_turn is None
            or sender != current_sender
            or current_time is None
            or timestamp is None
            or (timestamp - current_time) > timedelta(minutes=time_window_minutes)
        )
        
        if start_new_turn:
            # Finish previous turn
            if current_turn:
                aggregated.append(current_turn)
            
            # Start new turn
            current_turn = {
                "id": msg.get("id"),
                "from": msg.get("from"),
                "body": msg.get("body", ""),
                "contentType": msg.get("contentType"),
                "createdDateTime": msg.get("createdDateTime"),
                "messageType": msg.get("messageType"),
                "attachments": msg.get("attachments", []),
                "mentions": msg.get("mentions", []),
                "aggregatedMessageIds": [msg.get("id")],
                "aggregatedCount": 1,
            }
            current_sender = sender
            current_time = timestamp
        else:
            # Append to current turn
            if current_turn:
                current_body = current_turn.get("body", "")
                new_body = msg.get("body", "")
                # Join with newline if both are non-empty
                if current_body and new_body:
                    current_turn["body"] = f"{current_body}\n{new_body}"
                elif new_body:
                    current_turn["body"] = new_body
                
                # Merge attachments and mentions
                current_turn.setdefault("attachments", []).extend(msg.get("attachments", []))
                current_turn.setdefault("mentions", []).extend(msg.get("mentions", []))
                
                # Track aggregation metadata
                current_turn["aggregatedMessageIds"].append(msg.get("id"))
                current_turn["aggregatedCount"] += 1
                
                # Update timestamp to latest message in turn
                current_turn["createdDateTime"] = msg.get("createdDateTime")
                if timestamp:
                    current_time = timestamp
    
    # Don't forget the last turn
    if current_turn:
        aggregated.append(current_turn)
    
    return aggregated


def format_aggregated_conversation(
    messages: list[dict[str, Any]],
    max_body_length: int = 500,
) -> str:
    """Format aggregated Teams messages as a readable conversation transcript.
    
    Args:
        messages: List of (optionally aggregated) message dicts from Teams API
        max_body_length: Max characters per message body before truncating
    
    Returns:
        Formatted conversation string with timestamps and speakers
    """
    if not messages:
        return "(No messages)"
    
    lines = []
    for msg in messages:
        from_dict = msg.get("from") or {}
        sender = from_dict.get("displayName", "Unknown")
        timestamp = msg.get("createdDateTime", "")
        body = msg.get("body", "")
        
        # Format timestamp (extract date part only if available)
        time_label = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_label = dt.strftime("%b %d, %H:%M")
            except (ValueError, AttributeError):
                time_label = timestamp[:16]  # fallback: first 16 chars
        
        # Truncate long bodies
        if len(body) > max_body_length:
            body = body[:max_body_length] + "..."
        
        # Show aggregation indicator
        agg_count = msg.get("aggregatedCount", 1)
        agg_indicator = f" [{agg_count} msgs]" if agg_count > 1 else ""
        
        lines.append(f"[{time_label}] {sender}{agg_indicator}: {body}")
    
    return "\n".join(lines)
