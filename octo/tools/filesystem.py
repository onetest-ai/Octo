"""Filesystem tools — Read, Grep, Glob, Edit.

Named to match Claude Code's built-in tools so agents loaded from
AGENT.md files can reference them by name.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from langchain_core.tools import tool


@tool
def Read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file from the filesystem.

    Args:
        file_path: Absolute or relative path to the file.
        offset: Line number to start reading from (0-based).
        limit: Maximum number of lines to read.

    Returns:
        File contents with line numbers, or an error message.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            return f"Error: File not found: {file_path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)]
        result = "\n".join(numbered)
        if len(result) > 50000:
            result = result[:50000] + "\n... (truncated)"
        return result
    except Exception as e:
        return f"Error reading {file_path}: {e}"


@tool
def Grep(
    pattern: str,
    path: str = ".",
    glob: str = "",
    include: str = "",
    context: int = 0,
) -> str:
    """Search file contents using regex.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search in.
        glob: Glob pattern to filter files (e.g. '*.py').
        include: Alias for glob.
        context: Number of context lines before/after each match.

    Returns:
        Matching lines with file paths and line numbers.
    """
    file_glob = glob or include
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    root = Path(path).expanduser().resolve()
    results: list[str] = []
    max_results = 200

    def _search_file(fp: Path):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if regex.search(line):
                # Context lines
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                for j in range(start, end):
                    prefix = ">" if j == i else " "
                    results.append(f"{fp}:{j + 1}:{prefix} {lines[j]}")
                if context > 0:
                    results.append("")  # separator
                if len(results) >= max_results:
                    return

    if root.is_file():
        _search_file(root)
    else:
        for dirpath, _, filenames in os.walk(root):
            # Skip hidden dirs and common noise
            if any(part.startswith(".") for part in Path(dirpath).parts if part != "."):
                continue
            if "node_modules" in dirpath or "__pycache__" in dirpath:
                continue
            for fname in filenames:
                if file_glob and not fnmatch.fnmatch(fname, file_glob):
                    continue
                fp = Path(dirpath) / fname
                _search_file(fp)
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

    if not results:
        return f"No matches for '{pattern}' in {path}"
    output = "\n".join(results)
    if len(output) > 50000:
        output = output[:50000] + "\n... (truncated)"
    return output


@tool
def Glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. '**/*.py', 'src/**/*.ts').
        path: Root directory to search from.

    Returns:
        List of matching file paths, one per line.
    """
    root = Path(path).expanduser().resolve()
    try:
        matches = sorted(root.glob(pattern))
        # Filter out hidden directories and common noise
        filtered = [
            str(m)
            for m in matches
            if not any(
                part.startswith(".") for part in m.relative_to(root).parts
            )
            and "node_modules" not in str(m)
            and "__pycache__" not in str(m)
        ]
        if not filtered:
            return f"No files matching '{pattern}' in {path}"
        if len(filtered) > 500:
            filtered = filtered[:500]
            filtered.append(f"... ({len(matches)} total, showing first 500)")
        return "\n".join(filtered)
    except Exception as e:
        return f"Error: {e}"


@tool
def Edit(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing a string.

    Args:
        file_path: Path to the file to modify.
        old_string: The exact text to find and replace.
        new_string: The replacement text.

    Returns:
        Confirmation message or error.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            return f"Error: File not found: {file_path}"
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1:
            return f"Error: old_string found {count} times — provide more context to make it unique"
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except Exception as e:
        return f"Error editing {file_path}: {e}"


# Export with Claude Code-compatible names
read_tool = Read
grep_tool = Grep
glob_tool = Glob
edit_tool = Edit
