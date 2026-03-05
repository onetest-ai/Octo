"""Voice tools — TTS and STT for agents.

Requires the [voice] optional extra. Tools are no-ops when deps are missing.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from langchain_core.tools import tool


def _resolve_output_path(output_path: str | None, prefix: str) -> str:
    """Resolve where to save audio: explicit path > workspace > temp."""
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # Save to .octo/workspace/<date>/ by default
    try:
        from octo.config import RESEARCH_WORKSPACE
        today = date.today().isoformat()
        ws = RESEARCH_WORKSPACE / today
        ws.mkdir(parents=True, exist_ok=True)
        # Generate unique name
        import uuid
        return str(ws / f"{prefix}_{uuid.uuid4().hex[:8]}.wav")
    except Exception:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"{prefix}_")
        os.close(fd)
        return path


def _audio_info(path: str) -> str:
    """Return human/AI-readable audio file summary."""
    size = os.path.getsize(path)
    if size < 1024:
        size_str = f"{size} B"
    elif size < 1024 * 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size / (1024 * 1024):.1f} MB"

    # Try to get duration
    duration_str = ""
    try:
        import soundfile as sf
        info = sf.info(path)
        secs = info.duration
        mins = int(secs // 60)
        remaining = secs - mins * 60
        duration_str = f"{mins}m{remaining:.1f}s" if mins else f"{remaining:.1f}s"
    except Exception:
        pass

    parts = [f"path: {path}", f"size: {size_str}"]
    if duration_str:
        parts.append(f"duration: {duration_str}")
    return "Audio generated. " + ", ".join(parts)


@tool
async def generate_speech(
    text: str,
    voice: str = "Jon",
    instruct: str | None = None,
    language: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate speech audio from text (TTS).

    Uses local ParlerTTS model with named speakers for consistent voice.
    Automatically chunks long text.
    Output is saved to .octo/workspace/<date>/ by default,
    or to the specified output_path.

    Available voices: Jon (default male), Laura (warm female),
    Gary (deep authoritative male), Lea (bright cheerful female).
    Old names work: Ryan→Jon, Vivian→Laura.
    OpenAI aliases: alloy/fable→Jon, echo/onyx→Gary, nova→Laura, shimmer→Lea.

    Args:
        text: The text to convert to speech.
        voice: Voice name or alias.
        instruct: Emotion preset name OR full voice description override.
                  Presets: calm, explaining, surprised, laughing, serious, whispering.
                  Or pass a full description (e.g. "A calm male voice speaks slowly").
        language: Reserved for future use. Currently English only.
        output_path: Where to save the WAV file. If not provided,
                     saves to .octo/workspace/<today>/.

    Returns:
        Audio file info: path, size, duration.
    """
    try:
        from octo.core.voice import is_available, local_synthesize
    except ImportError:
        return "Error: voice module not available. Install with: pip install octo-agent[voice]"

    if not is_available():
        return "Error: voice dependencies not installed. Install with: pip install octo-agent[voice]"

    try:
        audio = await local_synthesize(text, voice=voice, instruct=instruct, language=language)
        path = _resolve_output_path(output_path, "speech")
        with open(path, "wb") as f:
            f.write(audio)
        return _audio_info(path)
    except Exception as e:
        return f"TTS error: {e}"


@tool
async def generate_multi_voice_speech(
    segments: list | str,
    pause_ms: int = 300,
    output_path: str | None = None,
) -> str:
    """Generate multi-voice speech with different voices for different parts.

    Each segment has its own voice and optional emotion/style.
    Long segments are auto-chunked. Segments are concatenated with pauses.
    Output is saved to .octo/workspace/<date>/ by default.

    Args:
        segments: List of objects, each with "text", "voice", and optional
                  "instruct" fields. instruct can be an emotion preset
                  (calm, explaining, surprised, laughing, serious, whispering)
                  or a full description. Example:
                  [
                    {"text": "Hello!", "voice": "Ryan", "instruct": "laughing"},
                    {"text": "Interesting...", "voice": "Vivian", "instruct": "skeptical"}
                  ]
                  Currently English only.
        pause_ms: Milliseconds of silence between segments (default 300).
        output_path: Where to save the WAV file. If not provided,
                     saves to .octo/workspace/<today>/.

    Returns:
        Audio file info: path, size, duration.
    """
    try:
        from octo.core.voice import is_available, local_synthesize_multi
    except ImportError:
        return "Error: voice module not available. Install with: pip install octo-agent[voice]"

    if not is_available():
        return "Error: voice dependencies not installed. Install with: pip install octo-agent[voice]"

    # Accept JSON string
    if isinstance(segments, str):
        try:
            segments = json.loads(segments)
        except json.JSONDecodeError as e:
            return f"Error: segments must be valid JSON — {e}"

    if not isinstance(segments, list) or not segments:
        return "Error: segments must be a non-empty list of {text, voice} objects"

    try:
        audio = await local_synthesize_multi(segments, pause_ms=pause_ms)
        path = _resolve_output_path(output_path, "multi_speech")
        with open(path, "wb") as f:
            f.write(audio)
        return _audio_info(path)
    except Exception as e:
        return f"Multi-voice TTS error: {e}"


@tool
async def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text (STT).

    Uses local Whisper model (MLX on Apple Silicon, faster-whisper on CUDA/CPU).

    Args:
        file_path: Path to the audio file (WAV, MP3, OGG, etc.).

    Returns:
        Transcribed text, or error message.
    """
    if not os.path.isfile(file_path):
        return f"File not found: {file_path}"

    try:
        from octo.core.voice import local_transcribe
    except ImportError:
        return "Error: voice module not available. Install with: pip install octo-agent[voice]"

    try:
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return await local_transcribe(audio_data)
    except ImportError:
        return "Error: STT backend not installed. Install mlx-whisper or faster-whisper."
    except Exception as e:
        return f"STT error: {e}"


VOICE_TOOLS = [generate_speech, generate_multi_voice_speech, transcribe_audio]
