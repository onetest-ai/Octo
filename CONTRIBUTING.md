# Contributing to Octo

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/<org>/octo.git
cd octo

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Run the setup wizard
octo init
```

## Making Changes

1. **Fork** the repo and create a feature branch from `main`
2. **Read** the relevant code before modifying it — start with `CLAUDE.md` for architecture reference
3. **Verify syntax** after editing any Python file:
   ```bash
   .venv/bin/python -m py_compile octo/<file>.py
   ```
4. **Test** your changes by running `octo` and exercising the affected feature
5. **Keep it focused** — one logical change per PR

## Code Conventions

- Use `from __future__ import annotations` in all modules
- Lazy-import heavy dependencies (boto3, langchain_*, etc.) inside factory functions
- Follow existing patterns — the codebase is consistent, match what's already there
- No auto-generated docstrings or type stubs — only add comments where logic isn't self-evident

## Key Architecture Rules

These come from hard-won experience — please don't skip them:

- **`create_supervisor`** does NOT accept middleware — use `pre_model_hook` or custom tools
- **`create_agent`** accepts middleware — use `ToolErrorMiddleware` + `SummarizationMiddleware`
- **`AsyncSqliteSaver.from_conn_string()`** returns a context manager, not a saver — use `aiosqlite.connect()` + `AsyncSqliteSaver(conn)`
- **`MultiServerMCPClient`** is NOT a context manager — just instantiate and call `get_tools()`
- **`create_react_agent`** is deprecated — use `create_agent`

See `CLAUDE.md` for the full list of conventions and multi-file change checklists.

## Adding Features

Common changes have checklists in `CLAUDE.md`:

- **New LLM provider** — 6 files (config, models, wizard, validators, doctor, README)
- **New slash command** — 2 files (cli.py, ui.py)
- **New built-in tool** — 2-3 files (tools/, graph.py, optionally supervisor prompt)

## Pull Requests

- Keep PRs small and focused
- Describe **what** changed and **why**
- Include steps to test the change manually
- Ensure all Python files compile without errors

## Reporting Issues

- Check existing issues first
- Include: Python version, OS, error traceback, steps to reproduce
- For model/provider issues: include the model name and provider being used

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
