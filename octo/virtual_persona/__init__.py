"""Virtual Persona — autonomous conversation handler for Teams/Telegram.

Two-stage pipeline:
  1. VP graph: access_check → classify → route decision
  2. Octo supervisor: query → subagents/tools/skills → raw answer
  3. VP graph: raw answer → persona formatting → send

Modules:
  state           VPState TypedDict
  access_control  YAML-backed allow/block lists + delegation locks
  confidence      LLM-based scoring with hard escalation rules
  graph           StateGraph pipeline (decision engine)
  poller          HeartbeatRunner-style async Teams polling loop
  cache           Message dedup (JSON-backed set with TTL)
  stats           Audit log (JSONL) + counters
  knowledge       Thread topic cache + LLM classification
  profiles        Contact map (roles, tone, topics)
  commands        /vp slash command handlers
"""
from __future__ import annotations
