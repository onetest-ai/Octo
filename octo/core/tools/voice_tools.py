"""Voice tools — TTS and STT for agents.

Requires the [voice] optional extra. Tools are no-ops when deps are missing.
"""
from __future__ import annotations

import json
import os
import tempfile

from langchain_core.tools import tool


@tool
async def generate_speech(
    text: str,
    voice: str = "Aiden",
    instruct: str | None = None,
) -> str:
    """Generate speech audio from text (TTS).

    Uses local Qwen3-TTS model. Automatically chunks long text.

    Available voices: Aiden (English male), Ryan (English male),
    Vivian (Chinese female), Serena (Chinese female), Dylan (Chinese male),
    Eric (Chinese male), Uncle_Fu (Chinese male),
    Ono_Anna (Japanese female), Sohee (Korean female).

    OpenAI aliases also work: alloy→Aiden, echo→Ryan, nova→Vivian,
    shimmer→Serena, fable→Dylan, onyx→Uncle_Fu.

    Args:
        text: The text to convert to speech.
        voice: Voice name or alias.
        instruct: Emotion/style instruction (e.g. "Say it warmly",
                  "Say it with excitement"). Qwen3-TTS specific.

    Returns:
        Path to the generated WAV audio file, or error message.
    """
    try:
        from octo.core.voice import is_available, local_synthesize
    except ImportError:
        return "Error: voice module not available. Install with: pip install octo-agent[voice]"

    if not is_available():
        return "Error: voice dependencies not installed. Install with: pip install octo-agent[voice]"

    try:
        audio = await local_synthesize(text, voice=voice, instruct=instruct)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="speech_")
        os.write(fd, audio)
        os.close(fd)
        return path
    except Exception as e:
        return f"TTS error: {e}"


@tool
async def generate_multi_voice_speech(
    segments: list | str,
    pause_ms: int = 300,
) -> str:
    """Generate multi-voice speech with different voices for different parts.

    Each segment has its own voice and optional emotion instruction.
    Long segments are auto-chunked. Segments are concatenated with pauses.

    Args:
        segments: List of objects, each with "text", "voice", and optional
                  "instruct" fields. Example:
                  [
                    {"text": "Hello!", "voice": "Ryan", "instruct": "Say it energetically"},
                    {"text": "Hi there!", "voice": "Vivian", "instruct": "Say it warmly"}
                  ]
        pause_ms: Milliseconds of silence between segments (default 300).

    Returns:
        Path to the generated WAV audio file, or error message.
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
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="multi_speech_")
        os.write(fd, audio)
        os.close(fd)
        return path
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
