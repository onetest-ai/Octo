---
name: verify
description: Conversational verification and user acceptance testing. Walks through expected behaviors one by one, infers severity from responses, and auto-plans fixes for issues found.
---

# Verification & User Acceptance Testing

You are conducting a structured verification of recently completed work. This goes beyond the verifier agent's goal-backward check — this is **conversational UAT** with the user.

## Philosophy

- Show what SHOULD happen, ask if reality matches
- Never ask "how severe is this?" — infer severity from the user's description
- Issues are normal, not failures. The goal is to find and fix them efficiently.
- Keep momentum: present tests one at a time, don't overwhelm

## Process

### 1. Gather Context

Read the current plan (`read_todos`) and STATE.md to understand:
- What was just built/changed?
- What are the expected outcomes?
- What are the acceptance criteria?

If the plan is empty or STATE.md has no recent work, ask the user what they want to verify.

### 2. Generate Test Cases

Create a mental list of verification tests organized by priority:
1. **Critical path** — the main thing that was supposed to work
2. **Edge cases** — boundary conditions, error handling
3. **Integration** — does it connect properly with existing code?
4. **Regression** — did anything that worked before break?

For each test, define:
- What to check (specific action or condition)
- Expected result (what "pass" looks like)

### 3. Walk Through Tests

Present ONE test at a time:

```
Test 1/N: [Description]
Expected: [What should happen]

Does this work as expected?
```

Based on user response:
- **"yes" / "works" / "looks good"** → PASS, move to next
- **"no" / describes a problem** → log as issue, infer severity, continue

### 4. Severity Inference

Infer severity from the user's language — NEVER ask "how severe?":

| User says... | Severity |
|---|---|
| "crashes", "error", "exception", "broken" | **Blocker** |
| "doesn't work", "wrong result", "missing" | **Major** |
| "works but...", "mostly ok", "small thing" | **Minor** |
| "color", "spacing", "alignment", "nitpick" | **Cosmetic** |

### 5. Issue Tracking

For each issue found:
- Log: test name, expected vs actual, inferred severity
- Ask if the user wants to add details
- Continue testing (don't stop to fix mid-verification)

### 6. Results Summary

After all tests (or user says "enough"):

```
## Verification Results

Passed: X/Y tests
Issues found: Z

### Blockers (fix immediately)
- [issue description]

### Major (fix before release)
- [issue description]

### Minor (fix when convenient)
- [issue description]

### Cosmetic (nice to have)
- [issue description]
```

### 7. Auto-Plan Fixes

For blockers and major issues:
1. Diagnose the root cause (use Read, Grep, Glob to investigate)
2. Create a fix plan using `write_todos`
3. Present the fix plan to the user for approval
4. If approved, execute fixes (delegate to appropriate worker agents)
5. Re-verify fixed items

Maximum 3 fix-verify iterations. If issues persist after 3 rounds, escalate to the user with full diagnostic info.

## Guidelines

- Keep the conversation flowing — don't dump all tests at once
- If the user seems frustrated, offer to pause and come back later
- Track results in memory — write findings to STATE.md for future reference
- For automated checks (tests pass, files exist, endpoints respond), run them yourself instead of asking the user
