# Octo

LangGraph multi-agent CLI with Rich console UI, Telegram transport, and proactive AI.

Octo orchestrates AI agents from multiple projects through a single chat interface. It loads AGENT.md files, connects to MCP servers, routes tasks to the right agent via a supervisor pattern, and proactively reaches out when something needs attention.

## Quick Start

```bash
pip install -e .
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

### Proactive AI

Inspired by OpenClaw's heartbeat mechanism. Octo can reach out first — no user prompt needed.

**Heartbeat** — periodic timer (default 30m) that reads `.octo/persona/HEARTBEAT.md` for standing instructions. Two-phase design: Phase 1 uses a cheap model to decide if action is needed; Phase 2 invokes the full graph only when there's something to say. `HEARTBEAT_OK` sentinel suppresses delivery (no spam).

**Cron scheduler** — persistent job scheduler with three types:
- `at` — one-shot (e.g. "in 2h", "15:00")
- `every` — recurring interval (e.g. "30m", "1d")
- `cron` — 5-field cron expression (e.g. "0 9 * * MON-FRI")

Jobs stored in `.octo/cron.json`. Agents can self-schedule via the `schedule_task` tool.

### Telegram Transport

Full bidirectional Telegram bot that shares the same conversation thread as the CLI. Features:
- Text and voice messages (transcription via Whisper, TTS via ElevenLabs)
- Markdown-to-HTML conversion for rich formatting
- User authorization (`/authorize`, `/revoke`)
- Proactive message delivery (heartbeat + cron results)
- File attachments — `send_file` tool sends research reports as Telegram documents
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
├── models.py           # model factory (4 providers, auto-detection)
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

All config lives in `.env` (generated by `octo init`, or create manually):

```env
# LLM Provider — auto-detected from model name, or set explicitly
# LLM_PROVIDER=bedrock  # anthropic | bedrock | openai | azure | github

# Model tiers (different agents use different tiers to save costs)
DEFAULT_MODEL=eu.anthropic.claude-sonnet-4-5-20250929-v1:0
HIGH_TIER_MODEL=eu.anthropic.claude-sonnet-4-5-20250929-v1:0
LOW_TIER_MODEL=eu.anthropic.claude-haiku-4-5-20251001-v1:0

# AWS Bedrock
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Anthropic direct (used when model name starts with "claude-")
# ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (used when model name starts with "gpt-" or "o1-"/"o3-")
# OPENAI_API_KEY=sk-...

# Azure OpenAI
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-12-01-preview

# GitHub Models (GPT, Claude, Mistral, Llama via GitHub PAT)
# GITHUB_TOKEN=ghp_...
# GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com        # OpenAI-compat
# GITHUB_MODELS_ANTHROPIC_BASE_URL=https://models.inference.ai.azure.com  # Claude

# Agent directories — external AGENT.md sources (colon-separated)
AGENT_DIRS=/path/to/project-a/.claude/agents:/path/to/project-b/.claude/agents

# Model profile — quality | balanced | budget
MODEL_PROFILE=balanced

# Telegram (shared thread with console)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_OWNER_ID=...

# Heartbeat
HEARTBEAT_INTERVAL=30m                  # how often (30m, 1h, 60s)
HEARTBEAT_ACTIVE_HOURS_START=08:00      # local time
HEARTBEAT_ACTIVE_HOURS_END=22:00        # local time

# Voice (ElevenLabs TTS)
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
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
| `/voice on\|off` | Toggle TTS |
| `/model <name>` | Switch model |
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
                        └──────────────────────┘      Todo / State / Memory / File tools
```

All transports share the same conversation thread and graph lock. The supervisor routes to specialist agents based on the request, manages task plans, writes memories, schedules tasks, and sends files.
