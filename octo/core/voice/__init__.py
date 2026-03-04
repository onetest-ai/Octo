"""Local voice engine — in-process TTS/STT without HTTP overhead.

Requires the [voice] optional extra: pip install octo-agent[voice]
"""
from __future__ import annotations


def is_available() -> bool:
    """Return True if local voice deps (qwen_tts, soundfile) are importable."""
    try:
        import qwen_tts  # noqa: F401
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


async def local_synthesize(
    text: str, voice: str = "Aiden", instruct: str | None = None,
) -> bytes:
    """Synthesize text to WAV bytes using local Qwen3-TTS."""
    from octo.core.voice.tts import synthesize
    return await synthesize(text, voice, instruct=instruct)


async def local_synthesize_multi(
    segments: list[dict], pause_ms: int = 300,
) -> bytes:
    """Multi-voice synthesis. Each segment: {text, voice, instruct?}."""
    from octo.core.voice.tts import synthesize_multi
    return await synthesize_multi(segments, pause_ms=pause_ms)


async def local_transcribe(audio_data: bytes) -> str:
    """Transcribe audio bytes to text using local Whisper."""
    from octo.core.voice.stt import transcribe
    return await transcribe(audio_data)
