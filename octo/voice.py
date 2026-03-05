"""Voice module — local TTS/STT (ParlerTTS + Whisper) with ElevenLabs fallback.

Engines (auto-detected):
  1. Local — requires octo-agent[voice] extra (torch, parler_tts, mlx-whisper)
  2. ElevenLabs — requires ELEVENLABS_API_KEY
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_enabled = False

# ── ElevenLabs defaults ──────────────────────────────────────────────
_DEFAULT_VOICE_ID = "yl2ZDV1MzN4HbQJbMihG"
_TTS_MODEL = "eleven_flash_v2_5"
_STT_MODEL = "scribe_v1"
_OUTPUT_FORMAT = "mp3_44100_128"

# ── Local engine availability (lazy) ─────────────────────────────────
_local_available: bool | None = None


def _has_local_voice() -> bool:
    global _local_available
    if _local_available is None:
        try:
            from octo.core.voice import is_available
            _local_available = is_available()
        except ImportError:
            _local_available = False
    return _local_available


def _active_engine() -> str:
    """Return 'local', 'elevenlabs', or 'none'."""
    if _has_local_voice():
        return "local"
    from octo.config import ELEVENLABS_API_KEY
    if ELEVENLABS_API_KEY:
        return "elevenlabs"
    return "none"


# ── Public API ───────────────────────────────────────────────────────

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


def engine_info() -> dict[str, str]:
    """Return current engine configuration for /voice status."""
    engine = _active_engine()
    if engine == "local":
        # Check STT separately (may not have mlx-whisper installed)
        try:
            from octo.core.voice.stt import _detect_backend
            stt_backend = _detect_backend()
            stt_ready = True
        except ImportError:
            stt_backend = "none"
            stt_ready = False
        return {
            "tts": "local (ParlerTTS)",
            "stt": f"local ({stt_backend})" if stt_ready else "unavailable",
            "tts_ready": "True",
            "stt_ready": str(stt_ready),
        }
    if engine == "elevenlabs":
        return {
            "tts": "elevenlabs",
            "stt": "elevenlabs",
            "tts_ready": "True",
            "stt_ready": "True",
        }
    return {
        "tts": "unavailable",
        "stt": "unavailable",
        "tts_ready": "False",
        "stt_ready": "False",
    }


async def transcribe(audio_data: bytes) -> str:
    """Transcribe audio bytes to text.

    Returns transcribed text, or empty string on failure.
    """
    engine = _active_engine()

    if engine == "local":
        try:
            from octo.core.voice import local_transcribe
            return await local_transcribe(audio_data)
        except ImportError:
            logger.warning("Local STT deps not installed (mlx-whisper/faster-whisper)")
        except Exception:
            logger.exception("Local STT error")

    if engine == "elevenlabs" or _active_engine() != "none":
        return await _transcribe_elevenlabs(audio_data)

    return ""


async def synthesize(
    text: str,
    voice: str = "Jon",
    instruct: str | None = None,
    language: str | None = None,
    prep: bool = True,
) -> bytes | None:
    """Convert text to audio bytes.

    Args:
        text: Text to synthesize.
        voice: Voice name (Qwen3-TTS speaker or OpenAI alias).
        instruct: Emotion/style instruction (e.g. "Say it warmly").
        language: "English" or "Russian". Auto-detected if not provided.
        prep: If True, run voiceover text preparation before synthesis.

    Returns audio bytes (WAV or mp3), or None on failure.
    """
    if prep:
        text = await prepare_for_voice(text)

    engine = _active_engine()

    if engine == "local":
        try:
            from octo.core.voice import local_synthesize
            return await local_synthesize(text, voice=voice, instruct=instruct, language=language)
        except Exception:
            logger.exception("Local TTS error")
            return None

    if engine == "elevenlabs":
        return await _synthesize_elevenlabs(text)

    return None


async def synthesize_multi(
    segments: list[dict],
    pause_ms: int = 300,
) -> bytes | None:
    """Multi-voice synthesis. Each segment: {text, voice, instruct?}.

    Only available with local engine (Qwen3-TTS).
    """
    if not _has_local_voice():
        logger.warning("Multi-voice requires local voice engine (octo-agent[voice])")
        return None

    try:
        from octo.core.voice import local_synthesize_multi
        return await local_synthesize_multi(segments, pause_ms=pause_ms)
    except Exception:
        logger.exception("Multi-voice TTS error")
        return None


async def speak(text: str) -> None:
    """Speak text via TTS (local playback). No-op if disabled."""
    if not _enabled:
        return

    engine = _active_engine()
    if engine == "none":
        return

    audio = await synthesize(text, prep=True)
    if not audio:
        return

    fmt = "mp3" if engine == "elevenlabs" else "wav"
    await _play_audio(audio, fmt)


async def prepare_for_voice(text: str) -> str:
    """Rewrite text as a natural voiceover script via LLM.

    Strips markdown, file paths, code blocks, URLs and converts
    to conversational speech. Falls back to raw text on error.
    """
    if not text or len(text.strip()) < 20:
        return text

    try:
        from octo.models import make_model

        model = make_model(tier="low")
        prompt = (
            "Rewrite the following text as a natural voiceover script.\n"
            "Rules:\n"
            "- Remove file paths, code blocks, URLs, and technical formatting\n"
            "- Convert markdown to natural speech\n"
            "- Keep it concise — summarize long technical details\n"
            "- Maintain the key message and tone\n"
            "- Output ONLY the voiceover text, nothing else\n\n"
            + text[:3000]
        )
        response = await model.ainvoke(prompt)
        result = response.content.strip()
        if result:
            logger.debug("Voiceover prep: %d chars -> %d chars", len(text), len(result))
            return result
        return text
    except Exception:
        logger.debug("Voiceover prep failed, using raw text", exc_info=True)
        return text


# ── ElevenLabs engines ───────────────────────────────────────────────

async def _transcribe_elevenlabs(audio_data: bytes) -> str:
    """Transcribe via ElevenLabs STT API."""
    from octo.config import ELEVENLABS_API_KEY

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
        logger.exception("ElevenLabs STT error")
        return ""


async def _synthesize_elevenlabs(text: str) -> bytes | None:
    """Synthesize via ElevenLabs TTS API."""
    from octo.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

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
        logger.exception("ElevenLabs TTS error")
        return None


# ── Audio playback ───────────────────────────────────────────────────

async def _play_audio(audio: bytes, fmt: str) -> None:
    """Play audio bytes locally (cross-platform)."""
    if fmt == "mp3":
        try:
            from elevenlabs.play import play
            play(io.BytesIO(audio))
            return
        except ImportError:
            logger.warning("elevenlabs package not installed for mp3 playback")
            return
        except Exception:
            logger.exception("MP3 playback error")
            return

    # WAV/other — use system audio player
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=f".{fmt}")
        os.write(fd, audio)
        os.close(fd)

        if sys.platform == "darwin":
            cmd = ["afplay", tmp_path]
        elif _IS_WINDOWS:
            cmd = [
                "powershell", "-c",
                f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()",
            ]
        else:
            cmd = ["aplay", "-q", tmp_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)
    except Exception:
        logger.exception("Audio playback error")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
