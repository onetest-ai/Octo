# Octo

[![PyPI](https://img.shields.io/pypi/v/octo-agent)](https://pypi.org/project/octo-agent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

LangGraph multi-agent CLI with Rich console UI, Telegram transport, and proactive AI.

Octo orchestrates AI agents from multiple projects through a single chat interface. It loads AGENT.md files, connects to MCP servers, routes tasks to the right agent via a supervisor pattern, and proactively reaches out when something needs attention.

## Prerequisites

**Required:**
- Python 3.11 or higher
- Node.js 18+ (most MCP servers use `npx`)
- At least one LLM provider configured (Anthropic, AWS Bedrock, OpenAI, Azure OpenAI, or GitHub Models)

**Optional:**
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — enables project workers that delegate tasks via `claude -p`
- [skills.sh](https://skills.sh) (`npm install -g skills`) — enables `/skills import` and `/skills find` from chat

## Installation

> **Do not install globally.** Octo has many dependencies that can conflict with
> other packages. Always use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install octo-agent
```

Or for development (editable install from source):

```bash
git clone https://github.com/onetest-ai/Octo.git
cd Octo
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
octo init          # interactive setup wizard — creates .env + scaffolds .octo/
octo               # start chatting
```

`octo init` walks you through provider selection, credential entry, and workspace setup. It validates your credentials with a real API call before saving.

**QuickStart mode** (3 prompts — pick provider, paste key, done):

```bash
octo init --quick
```

**Non-interactive** (for CI / Docker):

```bash
ANTHROPIC_API_KEY=sk-ant-... octo init --quick --provider anthropic --no-validate --force
```

## Health Check

```bash
octo doctor        # verify configuration — 8 checks with PASS/FAIL
octo doctor --fix  # re-run setup wizard on failures
octo doctor --json # machine-readable output
```

## Features

### Multi-Agent Supervisor

A supervisor agent routes user requests to the right specialist. Three types of workers:

- **Project workers** — one per registered project, wraps `claude -p` for full codebase access
- **Standard agents** — loaded from AGENT.md files with MCP + builtin tools
- **Deep research agents** — powered by `deepagents` with persistent workspaces, planning, and summarization middleware

### Agent Creation Wizard

Create new agents interactively from chat:

```
/create-agent
```

The wizard guides you through:
1. **Name & description** — validated format, collision detection
2. **Agent type** — standard (tools + prompt) or deep research (persistent workspace)
3. **Tool selection** — numbered table of all available tools (built-in + MCP), with shortcuts (`builtin`, `all`, `none`)
4. **Purpose description** — free-text description of what the agent should do
5. **AI generation** — LLM generates a full system prompt based on your description, selected tools, and examples from existing agents

The generated AGENT.md is previewed before saving. Agents are immediately available after graph rebuild.

### Proactive AI

Inspired by OpenClaw's heartbeat mechanism. Octo can reach out first — no user prompt needed.

**Heartbeat** — periodic timer (default 30m) that reads `.octo/persona/HEARTBEAT.md` for standing instructions. Two-phase design: Phase 1 uses a cheap model to decide if action is needed; Phase 2 invokes the full graph only when there's something to say. `HEARTBEAT_OK` sentinel suppresses delivery (no spam).

**Cron scheduler** — persistent job scheduler with three types:
- `at` — one-shot (e.g. "in 2h", "15:00")
- `every` — recurring interval (e.g. "30m", "1d")
- `cron` — 5-field cron expression (e.g. "0 9 * * MON-FRI")

Jobs stored in `.octo/cron.json`. Agents can self-schedule via the `schedule_task` tool.

**Background workers** — dispatch long-running tasks that run independently while you keep chatting:
- **Process mode**: fire-and-forget subprocess (`claude -p`, shell commands) — done when process exits
- **Agent mode**: standalone LangGraph agent with `task_complete`/`escalate_question` tools
- Tasks persist as JSON in `.octo/tasks/`. Semaphore-capped concurrency (`BG_MAX_CONCURRENT`).
- Results delivered via proactive notification (CLI + Telegram). In Telegram, swipe-reply to a task notification to resume a paused task.
- Supervisor can auto-dispatch via `dispatch_background` tool, or use `/bg <command>` manually.

### Telegram Transport

Full bidirectional Telegram bot that shares the same conversation thread as the CLI. Features:
- Text and voice messages (transcription via Whisper, TTS via ElevenLabs)
- Markdown-to-HTML conversion for rich formatting
- User authorization (`/authorize`, `/revoke`)
- Proactive message delivery (heartbeat + cron results)
- File attachments — `send_file` tool sends research reports as Telegram documents
- Reply routing — swipe-reply to VP or background task notifications to respond in-context
- Shared `asyncio.Lock` prevents races between CLI, Telegram, heartbeat, and cron

### Context Window Management

Three layers of protection against context overflow:

1. **TruncatingToolNode** — supervisor-level tool result truncation at source (40K char limit)
2. **ToolResultLimitMiddleware** — worker-level truncation via `create_agent` middleware
3. **Pre-model hook** — auto-trims old messages when context exceeds 70% capacity

Manual controls: `/compact` (LLM-summarized compaction), `/context` (visual usage bar).

### ESC to Abort

Press ESC during agent execution to cancel the running graph invocation and return to the input prompt. Uses raw terminal mode (`termios`) to detect bare ESC keypresses without interfering with prompt_toolkit input. Ctrl+C also works during execution.

### Persistent Memory

- **Daily logs** — `write_memory` appends timestamped entries to `.octo/memory/YYYY-MM-DD.md`
- **Long-term memory** — curated `MEMORY.md` updated via `update_long_term_memory`
- **Project state** — `STATE.md` captures current position, active plan, decisions, and next steps

### Task Planning

`write_todos` / `read_todos` tools let agents break work into steps. Plans persist in `.octo/plans/plan_<datetime>.json` (timestamped, never overwritten). View progress with `/plan`.

### Research Workspace

Deep research agents share a date-based workspace at `.octo/workspace/<date>/`. Files persist across sessions. Agents use `write_file` with simple filenames for research notes and reports. When users need files delivered, the supervisor uses `send_file` to attach them via Telegram.

### Model Profiles

Three built-in profiles control cost vs quality tradeoffs:

| Profile | Supervisor | Workers | High-tier agents |
|---|---|---|---|
| `quality` | high | default | high |
| `balanced` | default | low | high |
| `budget` | low | low | default |

Switch with `/profile <name>`.

### MCP Server Management

Live management without restart:

```
/mcp              # show status
/mcp reload       # reload all servers
/mcp add          # interactive wizard
/mcp disable X    # disable a server
/mcp enable X     # re-enable a server
/mcp remove X     # remove a server
/call [srv] tool  # call any MCP tool directly
```

### OAuth Authentication

Browser-based OAuth flow for MCP servers that require it:

```bash
octo auth login <server>    # open browser for OAuth
octo auth status            # check token status
octo auth logout <server>   # revoke tokens
```

### Session Management

Sessions persist in `.octo/sessions.json`. Resume previous conversations:

```bash
octo --resume              # resume last session
octo --thread <id>         # resume specific thread
```

`/sessions` lists recent sessions, `/clear` starts fresh.

### Tool Error Handling

`ToolErrorMiddleware` catches tool execution errors, calls a cheap LLM to explain what went wrong, and returns a helpful `[Tool error]` message instead of crashing the agent loop.

### Conversation Compression

- **Workers** — `SummarizationMiddleware` triggers at 70% context or 100 messages
- **Supervisor** — `pre_model_hook` auto-trims at 70% threshold
- **Manual** — `/compact` LLM-summarizes old messages

### Skills Marketplace

Skills are reusable prompt modules that extend Octo's capabilities. Each skill is a `SKILL.md` with YAML frontmatter declaring dependencies, requirements, and permissions.

**From chat:**

```
/skills              # list installed skills
/skills search pdf   # search marketplace
/skills install pdf  # install + auto-install deps + reload graph
/skills remove pdf   # uninstall + reload graph
```

**From CLI:**

```bash
octo skills search pdf          # search marketplace
octo skills info pdf            # detailed info + deps
octo skills install pdf         # install with auto dependency resolution
octo skills install pdf --no-deps   # skip dependency installation
octo skills remove pdf          # uninstall
octo skills update --all        # update all installed skills
octo skills list                # list installed
```

Dependencies declared in `SKILL.md` frontmatter are installed automatically:
- **Python** — `pip install` into the active venv
- **npm** — `npm install --prefix .octo/` (local node_modules)
- **MCP** — added to `.mcp.json` (restart or `/mcp reload` to activate)
- **System** — displayed for manual installation (e.g. `brew install`)

At startup, Octo checks installed skills for missing Python deps and logs warnings. When a skill is invoked at runtime, missing deps are detected and the agent is instructed to install them before proceeding.

### Built-in Tools

Available to all agents (configurable per agent via `tools:` in AGENT.md):

| Tool | Description |
|---|---|
| `Read` | Read file contents |
| `Grep` | Search file contents with regex |
| `Glob` | Find files by pattern |
| `Edit` | Edit files with string replacement |
| `Bash` | Execute shell commands |
| `claude_code` | Delegate to Claude Code CLI (`claude -p`) |

### Virtual Persona

AI-powered digital twin that monitors Teams conversations and responds on your behalf.

**How it works:** The VP poller checks Teams chats every N seconds (configurable). For each chat with new messages, it aggregates all unprocessed messages into one batch, classifies confidence, and routes:

| Decision | Confidence | Action |
|---|---|---|
| **respond** | >=80% | Auto-reply in your voice via Teams |
| **disclaim** | 60-79% | Reply with disclaimer caveat |
| **escalate** | <60% | Silent notification to you (thread locked) |
| **monitor** | any (non-allowed users) | Silent notification, no reply |
| **skip** | n/a | Acknowledgments, chatter — ignored |

**Smart behaviors:**
- **Message aggregation** — Multiple consecutive messages are batched into one response (no spam)
- **1-on-1 boost** — Direct messages get +15% confidence (people expect replies in DMs)
- **Group chat filtering** — Only processes messages that @mention you
- **Already-answered detection** — Skips messages before your last reply in a thread
- **Inactive chat skip** — No API calls for chats without new messages since last poll
- **Engagement tracking** — Threads where you've never engaged get lower confidence
- **Persona formatting** — Raw answers are rewritten in your communication style with language-appropriate tone

**Telegram integration:**
- Escalation/monitor notifications arrive with emoji categorization and confidence bars
- Reply to a notification to send your response to the Teams chat
- Reply "ignore" to mute a chat permanently
- Thread delegation auto-releases after you reply

**Data directory:** `.octo/virtual-persona/` — system-prompt.md, access-control.yaml, profiles.json, knowledge/, audit.jsonl, stats.json

### Voice

ElevenLabs TTS integration. Enable with `/voice on` or `--voice` flag. Telegram voice messages are transcribed via Whisper and replied to with voice.

## Project Structure

```
.env                    # credentials, model config (generated by octo init)
.mcp.json               # MCP server definitions — optional
.octo/                  # workspace state
├── persona/            # SOUL.md, IDENTITY.md, USER.md, MEMORY.md, HEARTBEAT.md, ...
├── agents/             # Octo-native agent definitions (AGENT.md per folder)
├── skills/             # skill definitions (SKILL.md per folder)
├── memory/             # daily memory logs (YYYY-MM-DD.md)
├── plans/              # task plans (plan_<datetime>.json)
├── workspace/          # research workspace (date-based subdirs)
├── projects/           # project registry (auto-generated JSON)
├── STATE.md            # human-readable project state
├── cron.json           # scheduled tasks
├── sessions.json       # session registry
└── octo.db             # conversation checkpoints (SQLite)
octo/                   # Python package
├── abort.py            # ESC-to-abort raw terminal listener
├── callbacks.py        # LangChain callback handler (tool panels, spinner)
├── cli.py              # Click CLI + async chat loop
├── config.py           # .env loading, workspace discovery, constants
├── context.py          # system prompt composition
├── graph.py            # supervisor graph assembly + tools
├── heartbeat.py        # proactive AI: heartbeat timer + cron scheduler
├── mcp_manager.py      # live MCP server management
├── middleware.py        # tool error handling, result truncation, summarization
├── models.py           # model factory (5 providers, auto-detection)
├── sessions.py         # session registry
├── telegram.py         # Telegram bot transport
├── ui.py               # Rich console UI
├── voice.py            # ElevenLabs TTS + Whisper STT
├── tools/              # built-in tools (filesystem, shell, claude_code)
├── loaders/            # agent, MCP, and skill loaders
├── wizard/             # setup wizard + health check
└── oauth/              # browser-based OAuth for MCP servers
```

## Configuration

All config lives in `.env` (generated by `octo init`, or create manually). See [`.env.example`](.env.example) for a full template.

You only need to configure **one** LLM provider:

```env
# --- Option A: Anthropic (simplest) ---
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=claude-sonnet-4-5-20250929

# --- Option B: AWS Bedrock ---
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
DEFAULT_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# --- Option C: OpenAI ---
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o

# --- Option D: GitHub Models (free tier available) ---
GITHUB_TOKEN=ghp_...
DEFAULT_MODEL=github/openai/gpt-4.1
```

Additional configuration (all optional):

```env
# Model tiers — different agents use different tiers to balance cost vs quality
HIGH_TIER_MODEL=...    # complex reasoning, architecture
LOW_TIER_MODEL=...     # summarization, cheap tasks

# Model profile — quality | balanced | budget
MODEL_PROFILE=balanced

# Agent directories — load AGENT.md files from external projects (colon-separated)
AGENT_DIRS=/path/to/project-a/.claude/agents:/path/to/project-b/.claude/agents

# Telegram bot (shared thread with console)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_OWNER_ID=...

# Heartbeat — proactive check-ins
HEARTBEAT_INTERVAL=30m        # supports: 30s, 2m, 1h, or bare 1800 (seconds)
HEARTBEAT_ACTIVE_HOURS_START=08:00
HEARTBEAT_ACTIVE_HOURS_END=22:00

# Virtual Persona — Teams digital twin
VP_ENABLED=true
VP_POLL_INTERVAL=2m           # supports: 30s, 2m, 1h, or bare 120 (seconds)
VP_ACTIVE_HOURS_START=08:00
VP_ACTIVE_HOURS_END=22:00

# Claude Code — extra args injected into all `claude -p` calls
ADDITIONAL_CLAUDE_ARGS=--dangerously-skip-permissions
CLAUDE_CODE_TIMEOUT=2400       # subprocess timeout in seconds (default 2400)

# Background workers
BG_MAX_CONCURRENT=3            # max parallel background tasks (default 3)

# Voice (ElevenLabs TTS)
ELEVENLABS_API_KEY=...
```

## Model Factory

The model factory (`octo/models.py`) auto-detects the provider from the model name:

| Model name pattern | Provider |
|---|---|
| `github/*` | GitHub Models |
| `eu.anthropic.*`, `us.anthropic.*` | AWS Bedrock |
| `claude-*` | Anthropic direct |
| `gpt-*`, `o1-*`, `o3-*` | OpenAI |
| `gpt-*` + `AZURE_OPENAI_ENDPOINT` set | Azure OpenAI |

Override with `LLM_PROVIDER` env var if needed.

GitHub Models auto-routes to the right LangChain class based on the model name:
- `github/claude-*` or `github/anthropic/claude-*` → `ChatAnthropic`
- Everything else (`github/openai/gpt-4.1`, `github/mistral-large`, etc.) → `ChatOpenAI`

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show commands |
| `/clear` | Reset conversation (new thread) |
| `/compact` | Summarize older messages to free context |
| `/context` | Show context window usage |
| `/agents` | List loaded agents |
| `/skills [cmd]` | Skills (list/search/install/remove) |
| `/tools` | List MCP tools by server |
| `/call [srv] <tool>` | Call MCP tool directly |
| `/mcp [cmd]` | MCP servers (add/remove/disable/enable/reload) |
| `/projects` | Show project registry |
| `/sessions [id]` | List sessions or switch to one |
| `/plan` | Show current task plan with progress |
| `/profile [name]` | Show/switch model profile |
| `/heartbeat [test]` | Heartbeat status or force a tick |
| `/cron [cmd]` | Scheduled tasks (list/add/remove/pause/resume) |
| `/bg <command>` | Run command in background |
| `/tasks` | List background tasks |
| `/task <id> [cmd]` | Task details / cancel / resume |
| `/vp [cmd]` | Virtual Persona (status/allow/block/ignore/release/sync/persona/stats) |
| `/create-agent` | AI-assisted agent creation wizard |
| `/voice on\|off` | Toggle TTS |
| `/model <name>` | Switch model |
| `/<agent> <prompt>` | Send prompt directly to a specific agent |
| `/<skill>` | Invoke a skill |
| ESC | Abort running agent |
| `exit` | End session |

## CLI Commands

| Command | Description |
|---|---|
| `octo` | Start interactive chat (default) |
| `octo init` | Run setup wizard |
| `octo doctor` | Check configuration health |
| `octo skills` | Skills marketplace (search/install/update/remove) |
| `octo auth` | Manage MCP OAuth tokens |

## Architecture

```
                        ┌──────────────────────┐
Console (Rich) ←──────→│                      │←───→ Project Workers (claude -p)
                        │    Supervisor        │←───→ Standard Agents (AGENT.md)
Telegram Bot   ←──────→│  (create_supervisor) │←───→ Deep Research Agents
                        │                      │
Heartbeat      ────────→│    asyncio.Lock      │←───→ MCP Tools (.mcp.json)
Cron Scheduler ────────→│                      │←───→ Built-in Tools
VP Poller      ────────→│                      │      Todo / State / Memory / File tools
                        └──────────────────────┘
                               ↑
                        ┌──────┴───────┐
                        │  VP Graph    │←───→ Teams (via MCP)
                        │ (StateGraph) │
                        └──────────────┘
```

All transports share the same conversation thread and graph lock. The supervisor routes to specialist agents based on the request, manages task plans, writes memories, schedules tasks, and sends files. The VP graph runs independently — it classifies incoming Teams messages, delegates to the supervisor for knowledge work, then reformats answers in the user's persona.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
