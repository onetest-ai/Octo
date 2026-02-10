---
name: map-codebase
description: Analyze an existing codebase in parallel — produces structured docs on tech stack, architecture, quality, and concerns. Use when onboarding to a new project.
---

# Codebase Mapping

You are performing a comprehensive analysis of an existing codebase. The goal is to produce structured documentation that helps you and the user understand the project deeply.

## Philosophy

- Agents write docs directly — the orchestrator only coordinates and summarizes
- Always include file paths with backticks — these docs are reference material
- Be honest about quality issues — this is an assessment, not a sales pitch
- Focus on what matters for working in this codebase day-to-day

## Process

### 1. Identify Target

Ask the user which project/directory to map. If they specify a project name, resolve it via the project registry. If they give a path, use that directly.

Confirm the target directory exists and has code in it.

### 2. Parallel Analysis

Delegate 4 focused analysis tasks to worker agents simultaneously. Each agent should use Read, Grep, Glob, and Bash tools to explore the codebase.

**Focus 1: Technology & Stack**
- Languages, frameworks, major libraries
- Package manager, dependency count, version constraints
- Build system, bundler, transpiler
- Runtime requirements (Node version, Python version, etc.)
- Output: tech stack summary with version info

**Focus 2: Architecture & Structure**
- Directory layout and organization pattern
- Entry points (main files, CLI commands, API routes)
- Core abstractions (base classes, interfaces, protocols)
- Data flow: how does data enter, transform, and exit?
- Key patterns: MVC, event-driven, microservices, monolith, etc.
- Output: architecture overview with dependency graph

**Focus 3: Code Quality**
- Test coverage and testing strategy (unit, integration, e2e)
- Linting/formatting config (eslint, prettier, ruff, black, etc.)
- Type safety (TypeScript strict, Python type hints, mypy)
- Error handling patterns (try/catch, Result types, error boundaries)
- Documentation level (docstrings, README, API docs)
- Output: quality assessment with specific findings

**Focus 4: Concerns & Risks**
- Hardcoded values, magic numbers, TODO/FIXME/HACK comments
- Security patterns (auth, input validation, secrets handling)
- Performance concerns (N+1 queries, missing indexes, large bundles)
- Technical debt indicators (deprecated APIs, version conflicts)
- Missing: what SHOULD exist but doesn't (tests, docs, CI, etc.)
- Output: risk register with severity ratings

### 3. Synthesize Results

After all 4 analyses complete, write a synthesis document:

```markdown
# Codebase Map: [Project Name]

## Quick Facts
- Language: [primary] + [secondary]
- Framework: [main framework]
- Size: [file count, LOC estimate]
- Test coverage: [level/percentage if available]
- Overall health: [Good / Needs Work / Concerning]

## Architecture
[1-2 paragraph summary from Focus 2]

## Key Entry Points
- [file:line] — [what it does]

## Tech Stack
[Table from Focus 1]

## Quality Assessment
[Summary from Focus 3]

## Top Concerns
1. [Most critical concern]
2. [Second concern]
3. [Third concern]

## Recommendations
- [Actionable suggestion 1]
- [Actionable suggestion 2]
- [Actionable suggestion 3]
```

### 4. Save Output

Save the full analysis to `.octo/codebase-maps/[project-name]/`:
- `MAP.md` — the synthesis document
- `STACK.md` — detailed tech stack
- `ARCHITECTURE.md` — detailed architecture analysis
- `QUALITY.md` — detailed quality assessment
- `CONCERNS.md` — detailed concerns and risks

Update STATE.md to record that codebase mapping was completed.

## Guidelines

- Respect `.gitignore` patterns — don't analyze `node_modules`, `.venv`, `dist`, etc.
- For monorepos, map the top level first, then ask which sub-project to deep-dive
- If the codebase is very large (>1000 files), sample representative files rather than reading everything
- Include concrete file paths and line numbers — vague observations aren't useful
- Don't make assumptions about intent — describe what the code DOES, not what it SHOULD do
