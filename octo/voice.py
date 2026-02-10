"""ElevenLabs voice — TTS, STT, and audio synthesis."""
from __future__ import annotations

import io
import logging

from octo.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

logger = logging.getLogger(__name__)

_enabled = False

_DEFAULT_VOICE_ID = "yl2ZDV1MzN4HbQJbMihG"
_TTS_MODEL = "eleven_flash_v2_5"
_STT_MODEL = "scribe_v1"
_OUTPUT_FORMAT = "mp3_44100_128"


def toggle_voice(on: bool | None = None) -> bool:
    """Toggle or set voice on/off. Returns new state."""
    global _enabled
    if on is None:
        _enabled = not _enabled
    else:
        _enabled = on
    return _enabled


def is_enabled() -> bool:
    return _enabled


async def transcribe(audio_data: bytes) -> str:
    """Transcribe audio bytes to text via ElevenLabs STT.

    Args:
        audio_data: Raw audio bytes (ogg/opus, mp3, wav, etc.)

    Returns:
        Transcribed text, or empty string on failure.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("No ELEVENLABS_API_KEY — cannot transcribe")
        return ""

    try:
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        result = await client.speech_to_text.convert(
            file=io.BytesIO(audio_data),
            model_id=_STT_MODEL,
        )
        return result.text or ""

    except ImportError:
        logger.warning("elevenlabs package not installed")
        return ""
    except Exception:
        logger.exception("STT error")
        return ""


async def synthesize(text: str) -> bytes | None:
    """Convert text to audio bytes via ElevenLabs TTS.

    Returns mp3 audio bytes, or None if disabled / no API key / error.
    """
    if not ELEVENLABS_API_KEY:
        return None

    try:
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
        voice_id = ELEVENLABS_VOICE_ID or _DEFAULT_VOICE_ID

        chunks = []
        async for chunk in client.text_to_speech.convert(
            text=text[:5000],
            voice_id=voice_id,
            model_id=_TTS_MODEL,
            output_format=_OUTPUT_FORMAT,
        ):
            chunks.append(chunk)

        return b"".join(chunks)

    except ImportError:
        logger.warning("elevenlabs package not installed")
        return None
    except Exception:
        logger.exception("TTS synthesis error")
        return None


async def speak(text: str) -> None:
    """Speak text via ElevenLabs TTS (local playback). No-op if disabled."""
    if not _enabled or not ELEVENLABS_API_KEY:
        return

    audio = await synthesize(text)
    if not audio:
        return

    try:
        from elevenlabs.play import play
        play(io.BytesIO(audio))
    except ImportError:
        logger.warning("elevenlabs package not installed")
    except Exception:
        logger.exception("TTS playback error")
