"""Voice module — multi-engine STT/TTS with voiceover text preparation.

Engines:
  STT: ElevenLabs (cloud) or configurable command (e.g. Whisper)
  TTS: ElevenLabs (cloud) or configurable command (e.g. Kokoro)

Subprocess protocol for local engines:
  STT: {WHISPER_COMMAND} <input_audio_file>  →  text on stdout
  TTS: {KOKORO_COMMAND} <output_audio_file>  ←  text on stdin, audio to file
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_enabled = False

# ── ElevenLabs defaults ──────────────────────────────────────────────
_DEFAULT_VOICE_ID = "yl2ZDV1MzN4HbQJbMihG"
_TTS_MODEL = "eleven_flash_v2_5"
_STT_MODEL = "scribe_v1"
_OUTPUT_FORMAT = "mp3_44100_128"


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
    from octo.config import (
        VOICE_STT_ENGINE, VOICE_TTS_ENGINE,
        ELEVENLABS_API_KEY, WHISPER_COMMAND, KOKORO_COMMAND,
    )
    stt = VOICE_STT_ENGINE.lower()
    tts = VOICE_TTS_ENGINE.lower()
    stt_ready = (
        bool(ELEVENLABS_API_KEY) if stt == "elevenlabs"
        else bool(WHISPER_COMMAND)
    )
    tts_ready = (
        bool(ELEVENLABS_API_KEY) if tts == "elevenlabs"
        else bool(KOKORO_COMMAND)
    )
    return {
        "stt": stt,
        "tts": tts,
        "stt_ready": str(stt_ready),
        "tts_ready": str(tts_ready),
    }


async def transcribe(audio_data: bytes) -> str:
    """Transcribe audio bytes to text using the configured STT engine.

    Returns transcribed text, or empty string on failure.
    """
    from octo.config import VOICE_STT_ENGINE
    engine = VOICE_STT_ENGINE.lower()

    if engine == "elevenlabs":
        return await _transcribe_elevenlabs(audio_data)
    return await _transcribe_command(audio_data)


async def synthesize(text: str, prep: bool = True) -> bytes | None:
    """Convert text to audio bytes using the configured TTS engine.

    Args:
        text: Text to synthesize.
        prep: If True, run voiceover text preparation (LLM rewrite)
              before synthesis. Set False for pre-prepared text
              (e.g. video production scripts).

    Returns audio bytes (mp3 or wav), or None on failure.
    """
    if prep:
        text = await prepare_for_voice(text)

    from octo.config import VOICE_TTS_ENGINE
    engine = VOICE_TTS_ENGINE.lower()

    if engine == "elevenlabs":
        return await _synthesize_elevenlabs(text)
    return await _synthesize_command(text)


async def speak(text: str) -> None:
    """Speak text via TTS (local playback). No-op if disabled."""
    if not _enabled:
        return

    from octo.config import VOICE_TTS_ENGINE
    engine = VOICE_TTS_ENGINE.lower()

    if engine == "elevenlabs":
        from octo.config import ELEVENLABS_API_KEY
        if not ELEVENLABS_API_KEY:
            return
    else:
        from octo.config import KOKORO_COMMAND
        if not KOKORO_COMMAND:
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


# ── Configurable command engines ─────────────────────────────────────

def _parse_command(command_str: str) -> list[str]:
    """Parse a command string into args list (cross-platform)."""
    if _IS_WINDOWS:
        try:
            return shlex.split(command_str, posix=False)
        except ValueError:
            return command_str.split()
    return shlex.split(command_str)


async def _transcribe_command(audio_data: bytes) -> str:
    """Transcribe audio via configurable STT command (e.g. Whisper).

    Protocol: {WHISPER_COMMAND} <input_audio_file> -> text on stdout
    """
    from octo.config import WHISPER_COMMAND

    if not WHISPER_COMMAND:
        logger.warning("WHISPER_COMMAND not set — cannot transcribe")
        return ""

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".ogg")
        os.write(fd, audio_data)
        os.close(fd)

        cmd = _parse_command(WHISPER_COMMAND) + [tmp_path]
        logger.debug("STT command: %s", cmd)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120,
        )

        if proc.returncode != 0:
            logger.warning(
                "STT command failed (exit %d): %s",
                proc.returncode, stderr.decode(errors="replace")[:500],
            )
            return ""

        return stdout.decode(errors="replace").strip()
    except asyncio.TimeoutError:
        logger.warning("STT command timed out after 120s")
        return ""
    except FileNotFoundError as e:
        logger.warning("STT command not found: %s", e)
        return ""
    except Exception:
        logger.exception("STT command error")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _synthesize_command(text: str) -> bytes | None:
    """Synthesize audio via configurable TTS command (e.g. Kokoro).

    Protocol: {KOKORO_COMMAND} <output_audio_file> <- text on stdin
    """
    from octo.config import KOKORO_COMMAND

    if not KOKORO_COMMAND:
        logger.warning("KOKORO_COMMAND not set — cannot synthesize")
        return None

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = _parse_command(KOKORO_COMMAND) + [tmp_path]
        logger.debug("TTS command: %s", cmd)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=text[:5000].encode()),
            timeout=180,
        )

        if proc.returncode != 0:
            logger.warning(
                "TTS command failed (exit %d): %s",
                proc.returncode, stderr.decode(errors="replace")[:500],
            )
            return None

        output = Path(tmp_path)
        if not output.exists() or output.stat().st_size == 0:
            logger.warning("TTS command produced no output file")
            return None

        return output.read_bytes()
    except asyncio.TimeoutError:
        logger.warning("TTS command timed out after 180s")
        return None
    except FileNotFoundError as e:
        logger.warning("TTS command not found: %s", e)
        return None
    except Exception:
        logger.exception("TTS command error")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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
