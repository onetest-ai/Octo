# Octo Project Rules

## Project Overview
Octo is a LangGraph multi-agent CLI with Rich console UI, Telegram transport, and proactive AI.
Python 3.13, installed via `pip install -e .`, virtualenv at `.venv/`.

## Architecture Quick Reference
- **Entry point**: `octo/cli.py` (Click CLI + async chat loop)
- **Graph assembly**: `octo/graph.py` (supervisor + workers + deep agents + tools)
- **Model factory**: `octo/models.py` (5 providers: anthropic, bedrock, openai, azure, github)
- **Config**: `octo/config.py` (loads .env, all constants)
- **UI**: `octo/ui.py` (Rich console — input, banners, panels, help)
- **Telegram**: `octo/telegram.py` (bidirectional bot transport)
- **Wizard**: `octo/wizard/` (onboarding, doctor, validators, templates)
- **State dir**: `.octo/` (persona, agents, skills, memory, plans, workspace, sessions, cron, DB)

## Critical Conventions

### Python
- Always use `.venv/bin/python` for running/compiling — `python` is not on PATH.
- After editing ANY Python file, **always run** `.venv/bin/python -m py_compile <file>` to verify syntax.
- Lazy-import heavy dependencies (boto3, langchain_*, etc.) inside factory functions.
- Use `from __future__ import annotations` in all modules.

### Multi-File Changes
Adding a new **LLM provider** requires updating these files (in order):
1. `octo/config.py` — env vars and constants
2. `octo/models.py` — factory function, `_detect_provider()`, `_PROVIDERS` dict
3. `octo/wizard/onboarding.py` — `_DEFAULT_MODELS`, `_PROVIDERS` list, `_collect_credentials()`, `_write_env_file()` sections
4. `octo/wizard/validators.py` — `_validate_<provider>()` function and dispatch dict
5. `octo/wizard/doctor.py` — credential check imports and `creds` dict
6. `README.md` — Configuration section + Model Factory table

Adding a new **slash command** requires:
1. `octo/cli.py` — handler in the command dispatch block + `slash_cmds` list
2. `octo/ui.py` — help table entry + optional toolbar hint

Adding a new **built-in tool** requires:
1. Tool function in `octo/tools/` (or `octo/graph.py` for supervisor-only tools)
2. `octo/graph.py` — add to `BUILTIN_TOOLS` or `supervisor_tool_list`
3. Supervisor prompt update if the tool needs usage instructions

### Memory & Documentation
- After significant changes, update `MEMORY.md` at `.claude/projects/<project-slug>/memory/MEMORY.md`
- Keep MEMORY.md under 200 lines (it's loaded into system prompt)
- Update `README.md` when user-visible features change

### Error Handling
- `create_supervisor` does NOT accept middleware — use `pre_model_hook` or custom tools instead
- `create_agent` accepts middleware — use `ToolErrorMiddleware` + `SummarizationMiddleware`
- `ChatBedrockConverse.bind_tools()` stores Pydantic objects, not dicts — patched in models.py
- Bedrock client uses `read_timeout=300` and `retries={"max_attempts": 0}` (retries handled by `octo/retry.py`)

### What NOT to Do
- Do NOT use `create_react_agent` — it's deprecated. Use `create_agent` from `langchain.agents`.
- Do NOT use `AsyncSqliteSaver.from_conn_string()` — it returns a context manager. Use `aiosqlite.connect()` + `AsyncSqliteSaver(conn)`.
- Do NOT catch `asyncio.CancelledError` in loops — it must propagate for task cancellation to work.
- Do NOT use `MultiServerMCPClient` as a context manager — just instantiate and call `get_tools()`.

## Testing
- Syntax check: `.venv/bin/python -m py_compile octo/<file>.py`
- Quick import check: `.venv/bin/python -c "from octo.<module> import <symbol>"`
- Run CLI: `.venv/bin/python -m octo`
