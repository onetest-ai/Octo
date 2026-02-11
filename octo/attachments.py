"""Attachment processing — detect file paths, copy to uploads, build messages.

File content extraction is NOT done here — that's handled by agent tools
and skills (e.g. pdf, pptx skills). This module only:
1. Detects file paths in user input
2. Copies files to .octo/workspace/uploads/
3. Builds message content with file references (text inline, images as base64)
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
import shutil
from pathlib import Path

from octo.config import RESEARCH_WORKSPACE

logger = logging.getLogger(__name__)

UPLOADS_DIR = RESEARCH_WORKSPACE / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Extensions we read inline as text (code, config, plain text)
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml",
    ".yml", ".toml", ".xml", ".html", ".css", ".scss", ".csv", ".sql",
    ".sh", ".bash", ".zsh", ".fish", ".rb", ".go", ".rs", ".java",
    ".kt", ".c", ".cpp", ".h", ".hpp", ".swift", ".m", ".r", ".lua",
    ".pl", ".php", ".env", ".ini", ".cfg", ".conf", ".log", ".diff",
    ".patch", ".dockerfile", ".tf", ".hcl", ".proto", ".graphql",
    ".vue", ".svelte", ".astro", ".mdx", ".rst", ".tex", ".bib",
}

# Extensions we handle as images (base64 for multimodal models)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}

# Binary extensions we reject (archives, executables, media)
_BINARY_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
    ".dmg", ".iso", ".img", ".pkg", ".deb", ".rpm",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".class", ".jar", ".war",
}

# Max text file size to inline (100KB)
_MAX_INLINE_SIZE = 100_000


def _detect_file_paths(text: str) -> list[tuple[str, str]]:
    """Detect file paths in user input.

    Returns list of (matched_text, resolved_path) tuples.
    """
    results = []
    seen: set[str] = set()

    # Pattern 1: Quoted paths — "path" or 'path'
    for match in re.finditer(r"""['"]([/~][^'"]+)['"]""", text):
        candidate = match.group(1)
        resolved = _resolve_path(candidate)
        if resolved and resolved not in seen:
            results.append((match.group(0), resolved))
            seen.add(resolved)

    # Pattern 2: Unquoted absolute or home-relative paths
    for match in re.finditer(r'(?:^|\s)([/~][\S]+)', text):
        candidate = match.group(1)
        resolved = _resolve_path(candidate)
        if resolved and resolved not in seen:
            results.append((candidate, resolved))
            seen.add(resolved)

    return results


def _resolve_path(candidate: str) -> str | None:
    """Resolve a candidate string to an existing file path, or None."""
    try:
        p = Path(os.path.expanduser(candidate)).resolve()
        if p.is_file():
            return str(p)
    except (OSError, ValueError):
        pass
    return None


def copy_to_uploads(src_path: str, filename: str | None = None) -> str:
    """Copy a file to the uploads directory. Returns the destination path."""
    src = Path(src_path)
    name = filename or src.name

    dest = UPLOADS_DIR / name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = UPLOADS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(str(src), str(dest))
    return str(dest)


def process_user_input(text: str) -> tuple[str | list, list[str]]:
    """Process user input, detecting and handling file attachments.

    Returns:
        (content, uploaded_paths) where content is either a string or
        a list of content blocks (for multimodal messages with images).
    """
    detected = _detect_file_paths(text)
    if not detected:
        return text, []

    uploaded_paths = []
    text_attachments = []
    image_blocks = []
    clean_text = text

    for matched, resolved in detected:
        ext = Path(resolved).suffix.lower()
        filename = Path(resolved).name
        size = os.path.getsize(resolved)

        # Reject known binary formats
        if ext in _BINARY_EXTENSIONS:
            text_attachments.append(
                f"\n[Skipped binary file: {filename} — "
                f"archives and executables are not supported]"
            )
            clean_text = clean_text.replace(matched, "").strip()
            continue

        dest = copy_to_uploads(resolved)
        uploaded_paths.append(dest)

        if ext in _IMAGE_EXTENSIONS:
            # Base64 image for multimodal models
            try:
                with open(resolved, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                mime = mimetypes.guess_type(resolved)[0] or "image/png"
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })
                clean_text = clean_text.replace(matched, "").strip()
                logger.info("Attached image: %s → %s", filename, dest)
            except Exception as e:
                logger.warning("Failed to read image %s: %s", resolved, e)
                text_attachments.append(f"\n[Attached image: `{dest}`]")

        elif ext in _TEXT_EXTENSIONS:
            # Inline text content
            try:
                if size <= _MAX_INLINE_SIZE:
                    content = Path(resolved).read_text(encoding="utf-8", errors="replace")
                    text_attachments.append(
                        f"\n---\n**{filename}** ({size:,} bytes, saved to `{dest}`):\n"
                        f"```{ext.lstrip('.')}\n{content}\n```"
                    )
                else:
                    text_attachments.append(
                        f"\n[Attached: `{dest}` ({size:,} bytes — "
                        f"too large to inline, use Read tool)]"
                    )
                clean_text = clean_text.replace(matched, "").strip()
                logger.info("Attached text: %s → %s (%d bytes)", filename, dest, size)
            except Exception as e:
                logger.warning("Failed to read %s: %s", resolved, e)
                text_attachments.append(f"\n[Attached: `{dest}`]")

        else:
            # Documents (PDF, Office, etc.) — reference only, let skills handle
            clean_text = clean_text.replace(matched, "").strip()
            text_attachments.append(
                f"\n[Attached: `{dest}` ({filename}, {size:,} bytes) — "
                f"use appropriate tools or skills to read this file]"
            )
            logger.info("Attached document: %s → %s", filename, dest)

    # Build final content
    if not clean_text:
        clean_text = "Process the attached file(s)."

    final_text = clean_text + "\n".join(text_attachments)

    if image_blocks:
        # Multimodal: text + images
        content_blocks: list = [{"type": "text", "text": final_text}]
        content_blocks.extend(image_blocks)
        return content_blocks, uploaded_paths

    return final_text, uploaded_paths


def process_pasted_attachments(
    text: str, pasted_paths: list[str],
) -> tuple[str | list, list[str]]:
    """Process attachments that were pasted via Ctrl+V.

    These files are already in the uploads directory. We just need to
    build the message content with references / base64 image blocks.

    The text may contain [filename.ext] tags from the paste — strip them
    to get the clean user prompt.
    """
    import re as _re

    # Strip [filename] tags inserted by the paste handler
    clean_text = _re.sub(r"\[[\w._-]+\]\s*", "", text).strip()
    if not clean_text:
        clean_text = "Process the attached file(s)."

    image_blocks = []
    text_attachments = []

    for path in pasted_paths:
        ext = Path(path).suffix.lower()
        filename = Path(path).name
        size = os.path.getsize(path)

        if ext in _IMAGE_EXTENSIONS:
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                mime = mimetypes.guess_type(path)[0] or "image/png"
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })
            except Exception as e:
                logger.warning("Failed to read pasted image %s: %s", path, e)
                text_attachments.append(f"\n[Attached image: `{path}`]")

        elif ext in _TEXT_EXTENSIONS:
            try:
                if size <= _MAX_INLINE_SIZE:
                    content = Path(path).read_text(encoding="utf-8", errors="replace")
                    text_attachments.append(
                        f"\n---\n**{filename}** ({size:,} bytes):\n"
                        f"```{ext.lstrip('.')}\n{content}\n```"
                    )
                else:
                    text_attachments.append(
                        f"\n[Attached: `{path}` ({size:,} bytes — use Read tool)]"
                    )
            except Exception:
                text_attachments.append(f"\n[Attached: `{path}`]")

        else:
            text_attachments.append(
                f"\n[Attached: `{path}` ({filename}, {size:,} bytes) — "
                f"use appropriate tools or skills to read this file]"
            )

    final_text = clean_text + "\n".join(text_attachments)

    if image_blocks:
        content_blocks: list = [{"type": "text", "text": final_text}]
        content_blocks.extend(image_blocks)
        return content_blocks, pasted_paths

    return final_text, pasted_paths
