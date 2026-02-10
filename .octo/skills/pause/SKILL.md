---
name: pause
description: Preserve cognitive state before stopping work. Creates a structured handoff note with current state, decisions, and exact next steps for seamless resume.
model-invocation: false
---

# Pause & Handoff

You are creating a structured handoff note so work can be resumed seamlessly — by you in a future session, or by another agent/human.

## Philosophy

- The goal is **cognitive state preservation** — capture not just what was done, but the thinking behind it
- A good handoff note lets someone resume without reading the full conversation history
- Be specific about the NEXT action — "continue working on X" is useless; "add error handling to `process_batch()` in `octo/graph.py:142`" is useful

## Process

### 1. Assess Current State

Read the current plan (`read_todos`), STATE.md, and review recent conversation context to understand:
- What was the original goal?
- What has been completed?
- What's in progress or partially done?
- What hasn't been started?

### 2. Create Handoff Note

Write a structured handoff to STATE.md:

```markdown
# Project State

_Last updated: [timestamp]_

## Current Position
- Session: [thread_id if available]
- Phase: [what phase of work: planning / executing / verifying / debugging]
- Status: paused

## Active Plan
[Summary of current plan from todos — what's done, what remains]

## Completed Work
- [Specific thing 1 that was completed]
- [Specific thing 2 that was completed]

## In-Progress Work
- [What was being worked on when paused]
- [Current state of that work — what's done, what's left]
- [Any partial changes in the codebase]

## Recent Decisions
- [Decision 1]: [rationale]
- [Decision 2]: [rationale]

## Blockers
- [Any blocking issues, if applicable]

## Session Continuity
- Stopped at: [specific point — file, function, step number]
- Next steps: [concrete, actionable next action]
- Resume notes: [anything the next session needs to know — gotchas, context, warnings]
```

### 3. Save State

- Update STATE.md with the handoff note (using `update_state_md`)
- Ensure plan.json is up to date (todos reflect current progress)
- Report to the user that state has been saved and what the next steps are

## Guidelines

- **Be concrete.** "Working on auth" is bad. "Implementing JWT refresh in `octo/auth.py`, token generation done, middleware not started" is good.
- **Include file paths.** The next session needs to know exactly where to look.
- **Note gotchas.** If you discovered something surprising, write it down. "The Bedrock API returns errors as 200s with error body" saves future debugging time.
- **Keep it under 50 lines.** This is a quick-reference note, not a novel.
