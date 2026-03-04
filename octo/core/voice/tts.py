"""Local Qwen3-TTS engine — in-process inference with chunking and multi-voice.

All heavy imports (torch, qwen_tts, soundfile, numpy) are lazy.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Model singleton ──────────────────────────────────────────────────
_model = None
_model_id: str | None = None

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# ── Voice mapping ────────────────────────────────────────────────────
_VOICE_MAP = {
    "alloy": "Aiden",
    "echo": "Ryan",
    "fable": "Dylan",
    "onyx": "Uncle_Fu",
    "nova": "Vivian",
    "shimmer": "Serena",
}

_SPEAKERS = {
    "Vivian": {"lang": "Chinese", "desc": "Bright, slightly edgy young female"},
    "Serena": {"lang": "Chinese", "desc": "Warm, gentle young female"},
    "Uncle_Fu": {"lang": "Chinese", "desc": "Seasoned male, low mellow timbre"},
    "Dylan": {"lang": "Chinese", "desc": "Youthful Beijing male, clear natural"},
    "Eric": {"lang": "Chinese", "desc": "Lively Chengdu male, slightly husky"},
    "Ryan": {"lang": "English", "desc": "Dynamic male, strong rhythmic drive"},
    "Aiden": {"lang": "English", "desc": "Sunny American male, clear midrange"},
    "Ono_Anna": {"lang": "Japanese", "desc": "Playful Japanese female, light timbre"},
    "Sohee": {"lang": "Korean", "desc": "Warm Korean female, rich emotion"},
}

_LANG_DEFAULTS = {
    "Vivian": "Chinese",
    "Serena": "Chinese",
    "Uncle_Fu": "Chinese",
    "Dylan": "Chinese",
    "Eric": "Chinese",
    "Ryan": "English",
    "Aiden": "English",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}

# ── Generation defaults (deterministic for voice consistency) ────────
_GEN_KWARGS = {
    "do_sample": False,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 10,
    "repetition_penalty": 1.1,
}


# ── Model loading ────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    """Resolve model path: VOICE_MODEL_DIR env > .octo/models/ > HF hub."""
    from pathlib import Path

    env_dir = os.environ.get("VOICE_MODEL_DIR")
    if env_dir:
        return env_dir

    # Check .octo/models/qwen3-tts-1.7b (standard local install)
    for candidate in [
        Path.home() / ".octo" / "models" / "qwen3-tts-1.7b",
        Path(".octo") / "models" / "qwen3-tts-1.7b",
    ]:
        if candidate.exists() and any(candidate.iterdir()):
            return str(candidate)

    return _DEFAULT_MODEL


def _get_model():
    """Lazy-load Qwen3-TTS model (singleton)."""
    global _model, _model_id

    model_id = _resolve_model_path()

    if _model is not None and _model_id == model_id:
        return _model

    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS: float16 causes inf/nan in softmax
    else:
        device = "cpu"
        dtype = torch.float32

    load_kwargs: dict = {
        "device_map": "auto" if device in ("cuda", "mps") else "cpu",
        "dtype": dtype,
    }

    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    logger.info("Loading Qwen3-TTS: %s on %s (%s)", model_id, device, dtype)
    _model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
    _model_id = model_id
    return _model


# ── Language detection ───────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Simple language detection based on Unicode character ranges."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "Chinese"
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            return "Japanese"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            return "Korean"
        if 0x0400 <= cp <= 0x04FF:
            return "Russian"
        if 0x00C0 <= cp <= 0x00FF:
            return "Auto"
    return "Auto"


# ── Synchronous synthesis ────────────────────────────────────────────

def _synthesize_sync(
    text: str,
    voice: str,
    instruct: str | None = None,
) -> bytes:
    """Run TTS inference synchronously. Returns WAV bytes."""
    import soundfile as sf

    model = _get_model()

    speaker = _VOICE_MAP.get(voice, voice)
    if speaker not in _SPEAKERS:
        speaker = "Aiden"

    lang = _detect_language(text)
    if lang == "Auto":
        lang = _LANG_DEFAULTS.get(speaker, "English")

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=lang,
        speaker=speaker,
        instruct=instruct or "",
        **_GEN_KWARGS,
    )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Text chunking ────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[\u3002\uff01\uff1f])")
_CLAUSE_RE = re.compile(r"(?<=[,;\u3001\uff0c\uff1b])\s*")


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            clauses = _CLAUSE_RE.split(sent)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if len(clause) > max_chars:
                    for i in range(0, len(clause), max_chars):
                        chunks.append(clause[i : i + max_chars])
                elif current and len(current) + 1 + len(clause) > max_chars:
                    chunks.append(current)
                    current = clause
                else:
                    current = f"{current} {clause}".strip()
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


# ── Audio concatenation ──────────────────────────────────────────────

def concat_audio_chunks(chunks: list[bytes], pause_ms: int = 300) -> bytes:
    """Concatenate WAV audio chunks with silence between them."""
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    import numpy as np
    import soundfile as sf

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None

    for idx, chunk in enumerate(chunks):
        data, sr = sf.read(io.BytesIO(chunk), dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch in chunk {idx}: {sr} vs {sample_rate}")
        arrays.append(data)

    assert sample_rate is not None
    silence = np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)

    parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        parts.append(arr)
        if i < len(arrays) - 1:
            parts.append(silence)

    combined = np.concatenate(parts)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Public async API ─────────────────────────────────────────────────

async def synthesize(
    text: str,
    voice: str = "Aiden",
    instruct: str | None = None,
    max_chars: int = 1000,
) -> bytes:
    """Synthesize text to WAV bytes. Auto-chunks long text."""
    chunks = chunk_text(text, max_chars)
    if not chunks:
        return b""

    if len(chunks) == 1:
        return await asyncio.to_thread(_synthesize_sync, chunks[0], voice, instruct)

    logger.info("Chunked TTS: %d chunks (%d chars)", len(chunks), len(text))
    audio_chunks: list[bytes] = []
    for i, chunk in enumerate(chunks):
        logger.debug("Synthesizing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        audio = await asyncio.to_thread(_synthesize_sync, chunk, voice, instruct)
        audio_chunks.append(audio)

    return concat_audio_chunks(audio_chunks)


async def synthesize_multi(
    segments: list[dict],
    pause_ms: int = 300,
) -> bytes:
    """Multi-voice synthesis. Each segment: {text, voice, instruct?}.

    Auto-chunks long segments. Concatenates with silence between segments.
    """
    if not segments:
        return b""

    wav_parts: list[bytes] = []
    for seg in segments:
        wav = await synthesize(
            text=seg["text"],
            voice=seg.get("voice", "Aiden"),
            instruct=seg.get("instruct"),
        )
        wav_parts.append(wav)

    return concat_audio_chunks(wav_parts, pause_ms)
