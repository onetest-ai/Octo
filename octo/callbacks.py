"""Rich console callback handler for LangGraph agent execution.

Styled after Alita CLI — tool call panels, thinking spinners,
smart output formatting, and user-friendly error display.
"""
from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

logger = logging.getLogger(__name__)

# Share the single Console instance from ui.py — ensures consistent
# terminal width detection after resize (SIGWINCH handled there).
from octo.ui import console

# Box styles for different block types
TOOL_BOX = box.ROUNDED
OUTPUT_BOX = box.ROUNDED
ERROR_BOX = box.HEAVY


class OctiCallbackHandler(BaseCallbackHandler):
    """Displays tool calls, LLM activity, and errors with rich formatting."""

    def __init__(
        self,
        verbose: bool = True,
        show_tool_outputs: bool = True,
        show_thinking: bool = True,
        show_llm_calls: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.show_tool_outputs = show_tool_outputs
        self.show_thinking = show_thinking
        self.show_llm_calls = show_llm_calls

        # State tracking
        self.tool_runs: Dict[str, Dict[str, Any]] = {}
        self.llm_runs: Dict[str, Dict[str, Any]] = {}
        self.pending_tokens: Dict[str, List[str]] = defaultdict(list)
        self.current_model: str = ""
        self.step_counter: int = 0

        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

        # Active task name (from todos) for spinner display
        self.active_task: str = ""

        # External status spinner (set by chat loop, stopped on first tool call)
        self.status = None

    def _stop_status(self):
        """Stop the external status spinner if set."""
        if self.status is not None:
            try:
                self.status.stop()
                self.status = None
            except Exception:
                pass

    # ── Formatting helpers ──────────────────────────────────────────────

    def _format_json_content(self, data: Any, max_length: int = 1500) -> str:
        """Format data as pretty JSON string."""
        try:
            if isinstance(data, str):
                if data.strip().startswith(("{", "[")):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        return data[:max_length] + ("..." if len(data) > max_length else "")
            formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            if len(formatted) > max_length:
                formatted = formatted[:max_length] + "\n... (truncated)"
            return formatted
        except Exception:
            return str(data)[:max_length]

    def _format_tool_output(self, output: Any) -> Any:
        """Smart-format tool output: JSON → syntax, markdown → rendered, else text."""
        if output is None:
            return Text("(no output)", style="dim italic")
        try:
            # Extract .content from ToolMessage / AIMessage objects
            # to avoid displaying repr with literal \n and \t escapes
            if hasattr(output, "content") and isinstance(output.content, str):
                output_str = output.content
            else:
                output_str = str(output)
            max_length = 2000
            # JSON
            if output_str.strip().startswith(("{", "[")):
                try:
                    parsed = json.loads(output_str)
                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                    if len(formatted) > max_length:
                        formatted = formatted[:max_length] + f"\n... (truncated, {len(output_str)} chars total)"
                    return Syntax(formatted, "json", theme="monokai", word_wrap=True, line_numbers=False)
                except json.JSONDecodeError:
                    pass
            # Truncate
            if len(output_str) > max_length:
                output_str = output_str[:max_length] + f"\n... (truncated, {len(str(output))} chars total)"
            # Markdown
            if any(m in output_str for m in ["```", "**", "##", "- ", "* ", "\n\n"]):
                return Markdown(output_str)
            return Text(output_str, style="white")
        except Exception:
            return Text(str(output)[:500], style="white")

    # ── Tool callbacks ──────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        self._stop_status()

        # Update active task from current plan
        try:
            from octo.graph import _todos
            active = next((t["task"] for t in _todos if t.get("status") == "in_progress"), "")
            if active:
                self.active_task = active[:60]
        except Exception:
            pass

        if not self.show_tool_outputs:
            return

        tool_name = serialized.get("name", "Unknown Tool")
        tool_run_id = str(run_id)
        self.step_counter += 1

        self.tool_runs[tool_run_id] = {
            "name": tool_name,
            "start_time": datetime.now(tz=timezone.utc),
            "inputs": inputs or input_str,
            "step": self.step_counter,
        }

        # Format inputs
        tool_inputs = inputs if inputs else input_str
        content_parts = []
        if tool_inputs:
            if isinstance(tool_inputs, dict):
                formatted = self._format_json_content(tool_inputs, max_length=1200)
                content_parts.append(
                    Syntax(formatted, "json", theme="monokai", word_wrap=True, line_numbers=False)
                )
            elif isinstance(tool_inputs, str) and tool_inputs:
                display = tool_inputs[:800] + "..." if len(tool_inputs) > 800 else tool_inputs
                content_parts.append(Text(display, style="white"))

        panel_content = Group(*content_parts) if content_parts else Text("(no input)", style="dim italic")

        console.print()
        console.print(Panel(
            panel_content,
            title=f"[bold yellow]🔧 Tool Call[/bold yellow] [dim]│[/dim] [bold cyan]{tool_name}[/bold cyan]",
            title_align="left",
            subtitle=f"[dim]Step {self.step_counter}[/dim]",
            subtitle_align="right",
            border_style="yellow",
            box=TOOL_BOX,
            padding=(0, 1),
        ))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes running."""
        if not self.show_tool_outputs:
            return

        tool_run_id = str(run_id)
        tool_info = self.tool_runs.pop(tool_run_id, {})
        tool_name = tool_info.get("name", kwargs.get("name", "Unknown"))
        step_num = tool_info.get("step", "?")

        duration_str = ""
        start_time = tool_info.get("start_time")
        if start_time:
            elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
            duration_str = f" │ {elapsed:.2f}s"

        console.print(Panel(
            self._format_tool_output(output),
            title=f"[bold green]✓ Result[/bold green] [dim]│[/dim] [dim]{tool_name}[/dim]",
            title_align="left",
            subtitle=f"[dim]Step {step_num}{duration_str}[/dim]",
            subtitle_align="right",
            border_style="green",
            box=OUTPUT_BOX,
            padding=(0, 1),
        ))
        console.print()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        tool_run_id = str(run_id)
        tool_info = self.tool_runs.pop(tool_run_id, {})
        tool_name = tool_info.get("name", kwargs.get("name", "Unknown"))
        step_num = tool_info.get("step", "?")

        duration_str = ""
        start_time = tool_info.get("start_time")
        if start_time:
            elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
            duration_str = f" │ {elapsed:.2f}s"

        content_parts = []
        content_parts.append(Text(str(error), style="red bold"))

        tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        if tb_str and tb_str.strip():
            content_parts.append(Text(""))
            content_parts.append(Text("Exception Traceback:", style="dim bold"))
            if len(tb_str) > 1500:
                tb_str = tb_str[:1500] + f"\n... (truncated, {len(tb_str)} chars total)"
            content_parts.append(
                Syntax(tb_str, "python", theme="monokai", word_wrap=True, line_numbers=False)
            )

        panel_content = Group(*content_parts) if len(content_parts) > 1 else content_parts[0]

        console.print()
        console.print(Panel(
            panel_content,
            title=f"[bold red]✗ Error[/bold red] [dim]│[/dim] [bold]{tool_name}[/bold]",
            title_align="left",
            subtitle=f"[dim]Step {step_num}{duration_str}[/dim]",
            subtitle_align="right",
            border_style="red",
            box=ERROR_BOX,
            padding=(0, 1),
        ))
        console.print()

    # ── LLM callbacks ───────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.show_llm_calls:
            return
        model_name = metadata.get("ls_model_name", "") if metadata else ""
        self.current_model = model_name
        self.llm_runs[str(run_id)] = {
            "model": model_name,
            "start_time": datetime.now(tz=timezone.utc),
        }
        console.print()
        console.print(Panel(
            Text("Processing...", style="italic"),
            title=f"[bold blue]🤔 LLM[/bold blue] [dim]│[/dim] [dim]{model_name or 'model'}[/dim]",
            title_align="left",
            border_style="blue",
            box=box.SIMPLE,
            padding=(0, 1),
        ))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.show_llm_calls:
            return
        model_name = metadata.get("ls_model_name", "") if metadata else ""
        self.current_model = model_name
        self.llm_runs[str(run_id)] = {
            "model": model_name,
            "start_time": datetime.now(tz=timezone.utc),
        }
        console.print()
        console.print(Panel(
            Text("Processing...", style="italic"),
            title=f"[bold blue]🤔 LLM[/bold blue] [dim]│[/dim] [dim]{model_name or 'model'}[/dim]",
            title_align="left",
            border_style="blue",
            box=box.SIMPLE,
            padding=(0, 1),
        ))

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[ChatGenerationChunk] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.show_thinking and token:
            self.pending_tokens[str(run_id)].append(token)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        llm_run_id = str(run_id)
        llm_info = self.llm_runs.pop(llm_run_id, {})
        tokens = self.pending_tokens.pop(llm_run_id, [])

        # Show thinking only during intermediate steps (when tools are active)
        if self.show_thinking and tokens and len(self.tool_runs) > 0:
            thinking_text = "".join(tokens)
            if thinking_text.strip():
                max_len = 600
                display = thinking_text[:max_len] + ("..." if len(thinking_text) > max_len else "")
                console.print(Panel(
                    Text(display, style="dim italic"),
                    title="[dim]💭 Thinking[/dim]",
                    title_align="left",
                    border_style="dim",
                    box=box.SIMPLE,
                    padding=(0, 1),
                ))

        # Extract token usage from LLMResult
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if not usage:
                usage = response.llm_output.get("usage", {})
            input_t = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_t = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            if input_t or output_t:
                self.last_input_tokens = input_t
                self.last_output_tokens = output_t
                self.total_input_tokens += input_t
                self.total_output_tokens += output_t

        if self.show_llm_calls and llm_info:
            start_time = llm_info.get("start_time")
            model = llm_info.get("model", "model")
            if start_time:
                elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
                console.print(f"[dim]✓ LLM complete ({model}, {elapsed:.2f}s)[/dim]")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """User-friendly LLM error display with hints."""
        error_str = str(error)
        user_message = None
        hint = None

        error_lower = error_str.lower()

        if "read timeout" in error_lower or "connect timeout" in error_lower or "timed out" in error_lower:
            user_message = "Request timed out"
            hint = "The LLM provider took too long to respond. Try again — it may be a transient issue."
        elif "model identifier is invalid" in error_lower or "BedrockException" in error_str:
            user_message = "Invalid model identifier"
            hint = "The model may not be available in your region or the model ID is incorrect.\nUse /model to check the current model."
        elif "rate limit" in error_lower or "too many requests" in error_lower:
            user_message = "Rate limit exceeded"
            hint = "Wait a moment and try again, or switch to a different model."
        elif "context length" in error_lower or "too long" in error_lower:
            user_message = "Context length exceeded"
            hint = "The conversation is too long. Use /compact to summarize history, or /clear to reset."
        elif "authentication" in error_lower or "unauthorized" in error_lower or "api key" in error_lower:
            user_message = "Authentication failed"
            hint = "Check your API credentials in .env."
        elif "model not found" in error_lower or "does not exist" in error_lower:
            user_message = "Model not found"
            hint = "The requested model is not available. Check DEFAULT_MODEL in .env."
        elif "throttling" in error_lower or "serviceunav" in error_lower:
            user_message = "Service temporarily unavailable"
            hint = "The provider is overloaded. Wait a moment and try again."

        console.print()
        if user_message:
            content = Text()
            content.append(f"❌ {user_message}\n\n", style="bold red")
            if hint:
                content.append(f"💡 {hint}\n\n", style="yellow")
            content.append("Technical details:\n", style="dim")
            content.append(error_str[:300] + ("..." if len(error_str) > 300 else ""), style="dim")
            console.print(Panel(
                content,
                title="[bold red]✗ LLM Error[/bold red]",
                title_align="left",
                border_style="red",
                box=ERROR_BOX,
                padding=(0, 1),
            ))
        else:
            console.print(Panel(
                Text(error_str, style="red"),
                title="[bold red]✗ LLM Error[/bold red]",
                title_align="left",
                border_style="red",
                box=ERROR_BOX,
                padding=(0, 1),
            ))

    # ── Chain callbacks ──────────────────────────────────────────────────

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.verbose:
            return

        # langgraph-supervisor uses Command(goto=...) for agent handoffs,
        # which surface as "chain errors" in callbacks — suppress these
        error_str = str(error)
        if "Command(graph=" in error_str or "goto=" in error_str:
            return

        # Suppress errors that are already handled by on_llm_error —
        # these propagate up through every graph layer as chain errors,
        # producing duplicate panels for the same root cause.
        error_lower = error_str.lower()
        _SUPPRESSED = [
            "too long", "context length",       # context overflow
            "read timeout", "timed out",        # timeouts
            "connect timeout",                  # connection timeouts
            "rate limit", "too many requests",  # throttling
            "throttling", "serviceunav",        # service issues
        ]
        if any(phrase in error_lower for phrase in _SUPPRESSED):
            return

        console.print()
        console.print(Panel(
            Text(error_str, style="red"),
            title="[bold red]✗ Chain Error[/bold red]",
            title_align="left",
            border_style="red",
            box=ERROR_BOX,
            padding=(0, 1),
        ))

    # ── Custom LangGraph events ─────────────────────────────────────────

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.verbose:
            return
        if name == "on_conditional_edge" and self.show_llm_calls:
            condition = data.get("condition", "")
            if condition:
                console.print(f"[dim]📍 Conditional: {condition[:100]}[/dim]")
        elif name == "on_transitional_edge" and self.show_llm_calls:
            next_step = data.get("next_step", "")
            if next_step and next_step != "__end__":
                console.print(f"[dim]→ Transition: {next_step}[/dim]")

    # ── Utility ──────────────────────────────────────────────────────────

    def reset_step_counter(self) -> None:
        """Reset the step counter and token tracking for a new conversation."""
        self.step_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.active_task = ""

    def get_context_usage(self) -> dict:
        """Return current context usage metrics."""
        return {
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
            "last_input": self.last_input_tokens,
            "last_output": self.last_output_tokens,
        }


def create_cli_callback(verbose: bool = True, debug: bool = False) -> OctiCallbackHandler:
    """Create callback handler with appropriate settings."""
    return OctiCallbackHandler(
        verbose=verbose,
        show_tool_outputs=verbose,
        show_thinking=verbose,
        show_llm_calls=debug,
    )
