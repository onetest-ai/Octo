---
name: quick
description: Execute an ad-hoc task quickly — minimal planning, no heavyweight agents. Use for small fixes, one-off changes, simple features.
---

# Quick Task Execution

You are executing a quick, focused task. Skip heavyweight planning and research — go straight to action with lightweight guardrails.

## Philosophy

- This is for tasks that are **small enough to hold in your head** — 1-3 focused subtasks max
- Target **~30% context usage** — if the task needs more, escalate to full planning
- Bias toward action. Plan briefly, execute, verify. Don't over-analyze.
- Still maintain quality: tests, proper commits, no shortcuts on correctness

## Process

### 1. Understand the Task

Read the user's request carefully. If anything is ambiguous, ask ONE clarifying question (not a list of 10). If it's clear, proceed immediately.

Determine scope:
- **Tiny** (< 5 min equivalent): single file change, config tweak, rename → just do it
- **Small** (5-15 min equivalent): new function, bug fix, simple feature → plan briefly, then execute
- **Too big for /quick**: multi-file architecture change, new subsystem → tell the user this needs full planning

### 2. Quick Plan

Write a brief plan using `write_todos`:
- 1-3 concrete tasks, each completable in a single agent pass
- Each task = specific files to touch + what to change
- No research phase, no requirements gathering

Example:
```
1. Add retry logic to API client (octo/api.py)
2. Add test for retry behavior (tests/test_api.py)
3. Update docstring
```

### 3. Execute

For each task:
- Delegate to the appropriate worker agent OR use tools directly
- Use `claude_code` for project-specific work that needs full codebase context
- After each task, mark it complete in the plan

### 4. Quick Verify

After execution, do a lightweight check:
- Did the files change as expected? (use `Read` / `Grep` to spot-check)
- Do tests pass? (run relevant test suite)
- Any obvious issues? (imports, syntax, missing pieces)

If verification fails, fix immediately — don't create a new plan.

### 5. Update State

- Mark all todos as completed
- Update STATE.md with what was done (one line is fine)
- Report results to the user

## Guardrails

- If you discover the task is bigger than expected: **STOP**. Tell the user it needs full planning.
- If you need to change >5 files: **STOP**. This isn't a quick task.
- If you're unsure about architectural decisions: **ASK**. Don't guess on quick tasks.
- Auto-fix: typos, missing imports, test failures from your changes
- Escalate: anything that changes public APIs, database schemas, or config formats
