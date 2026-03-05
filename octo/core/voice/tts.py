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

# Native language of each voice — used to detect cross-lingual scenarios
_NATIVE_LANG = {
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

# ── Generation parameters ────────────────────────────────────────────
# Default: natural sound with moderate sampling
_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.4,
    "top_p": 0.92,
    "top_k": 40,
    "repetition_penalty": 1.15,
}

# Cross-lingual: tighter params when voice speaks non-native language
# Prevents phoneme artifacts (e.g. "thzen" from Chinese voices on English)
_GEN_KWARGS_CROSS_LINGUAL = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 20,
    "repetition_penalty": 1.3,
    # Stabilize sub-talker (12Hz tokenizer) for cross-lingual first tokens
    "subtalker_temperature": 0.3,
    "subtalker_top_k": 15,
    "subtalker_top_p": 0.8,
}

# Sacrificial prefix per language: absorbs the CJK phoneme bleed.
# Must end with sentence boundary (period) so there's a detectable pause.
_SACRIFICE_PREFIX = {
    "English": "One moment please.",
    "Russian": "Одну секунду пожалуйста.",
}

_VOICE_SEED = int(os.environ.get("VOICE_SEED", "42"))


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


# ── Language detection (Russian vs English only) ─────────────────────

def _detect_language(text: str) -> str:
    """Detect language from text: Russian or English.

    Only two languages supported. Cyrillic → Russian, everything else → English.
    No autodetect for Chinese/Japanese/Korean — those are voice properties,
    not text properties. The caller must pass language explicitly if needed.
    """
    for ch in text:
        if 0x0400 <= ord(ch) <= 0x04FF:
            return "Russian"
    return "English"


# ── Synchronous synthesis ────────────────────────────────────────────

def _find_sacrifice_boundary(audio, sr: int) -> int:
    """Find the silence gap after the sacrificial prefix.

    Scans the audio for a silence region (energy below threshold) that
    marks the pause between the sacrifice phrase and the real text.
    Starts searching after 300ms (prefix is at least that long) and
    looks for consecutive low-energy windows.

    Returns sample index to trim at, or 0 if no boundary found.
    """
    import numpy as np

    window_ms = 10
    window_samples = int(sr * window_ms / 1000)
    # Search between 300ms and 3s (sacrifice is ~1s speech)
    start_sample = int(sr * 0.3)
    end_sample = min(int(sr * 3.0), len(audio) // 2)

    if end_sample <= start_sample:
        return 0

    region = audio[start_sample:end_sample]
    n_windows = len(region) // window_samples
    if n_windows < 5:
        return 0

    energy = np.array([
        np.sqrt(np.mean(region[i * window_samples:(i + 1) * window_samples] ** 2))
        for i in range(n_windows)
    ])

    # Threshold: 5% of median energy (silence is well below speech)
    median_e = np.median(energy[energy > 0]) if np.any(energy > 0) else 0
    if median_e < 1e-6:
        return 0
    silence_threshold = median_e * 0.05

    # Find first run of 3+ consecutive silent windows (≥30ms silence)
    consecutive = 0
    for i in range(n_windows):
        if energy[i] < silence_threshold:
            consecutive += 1
            if consecutive >= 3:
                # Found silence gap — trim to end of silence
                silence_end = i + 1
                # Skip any remaining silence
                while silence_end < n_windows and energy[silence_end] < silence_threshold:
                    silence_end += 1
                trim_sample = start_sample + silence_end * window_samples
                return trim_sample
        else:
            consecutive = 0

    return 0


def _estimate_max_tokens(text: str) -> int:
    """Estimate max_new_tokens to prevent runaway generation.

    Qwen3-TTS 12Hz: ~12 audio tokens per second of speech.
    Speaking rate: ~4-6 chars/sec (Russian/CJK), ~10-12 chars/sec (English).
    Normal: 1 char ≈ 2-3 audio tokens. We allow 5 tokens/char (2x safety).
    Minimum 256 tokens (~21s audio) — enough for any short phrase.
    """
    return max(256, len(text) * 5)


def _synthesize_sync(
    text: str,
    voice: str,
    instruct: str | None = None,
    language: str | None = None,
) -> bytes:
    """Run TTS inference synchronously. Returns WAV bytes."""
    import torch
    import soundfile as sf

    # Fixed seed for voice consistency across chunks
    torch.manual_seed(_VOICE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_VOICE_SEED)

    model = _get_model()

    speaker = _VOICE_MAP.get(voice, voice)
    if speaker not in _SPEAKERS:
        speaker = "Aiden"

    # Always autodetect language from text content — LLMs often pass wrong value
    lang = _detect_language(text)

    # Select gen kwargs: cross-lingual needs tighter params
    native = _NATIVE_LANG.get(speaker, "English")
    is_cross_lingual = native != lang
    if is_cross_lingual:
        gen_kwargs = dict(_GEN_KWARGS_CROSS_LINGUAL)
        logger.debug(
            "Cross-lingual: %s (native %s) speaking %s — using tight params",
            speaker, native, lang,
        )
    else:
        gen_kwargs = dict(_GEN_KWARGS)

    # Cross-lingual: prepend sacrificial phrase to absorb CJK phoneme bleed.
    # The artifact bleeds into the sacrifice instead of the real text.
    synth_text = text
    if is_cross_lingual:
        prefix = _SACRIFICE_PREFIX.get(lang, _SACRIFICE_PREFIX["English"])
        synth_text = f"{prefix} {text}"

    gen_kwargs["max_new_tokens"] = _estimate_max_tokens(synth_text)

    wavs, sr = model.generate_custom_voice(
        text=synth_text,
        language=lang,
        speaker=speaker,
        instruct=instruct or "",
        **gen_kwargs,
    )

    audio = wavs[0]

    # Trim the sacrificial prefix by finding the silence gap after it
    if is_cross_lingual:
        boundary = _find_sacrifice_boundary(audio, sr)
        if boundary > 0:
            trim_ms = int(boundary / sr * 1000)
            logger.info(
                "Trimmed %dms sacrifice prefix from %s speaking %s",
                trim_ms, speaker, lang,
            )
            audio = audio[boundary:]
        else:
            # Boundary not found — regenerate without sacrifice prefix
            logger.warning("Sacrifice boundary not found, regenerating without prefix")
            torch.manual_seed(_VOICE_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(_VOICE_SEED)
            gen_kwargs["max_new_tokens"] = _estimate_max_tokens(text)
            wavs2, sr = model.generate_custom_voice(
                text=text,
                language=lang,
                speaker=speaker,
                instruct=instruct or "",
                **gen_kwargs,
            )
            audio = wavs2[0]

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Text chunking ────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?<=[\u3002\uff01\uff1f])")
_CLAUSE_RE = re.compile(r"(?<=[,;\u3001\uff0c\uff1b])\s*")


def _split_by_words(text: str, max_chars: int) -> list[str]:
    """Split text by word boundaries. Never cuts a word in half."""
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Split text into chunks at natural boundaries.

    Priority: sentences → clauses → words. Never splits mid-word.
    """
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
            # Try clause boundaries
            clauses = _CLAUSE_RE.split(sent)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if len(clause) > max_chars:
                    # Split by words — never mid-word
                    chunks.extend(_split_by_words(clause, max_chars))
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
    language: str | None = None,
    max_chars: int = 300,
) -> bytes:
    """Synthesize text to WAV bytes. Auto-chunks long text."""
    chunks = chunk_text(text, max_chars)
    if not chunks:
        return b""

    if len(chunks) == 1:
        return await asyncio.to_thread(
            _synthesize_sync, chunks[0], voice, instruct, language,
        )

    logger.info("Chunked TTS: %d chunks (%d chars)", len(chunks), len(text))
    audio_chunks: list[bytes] = []
    for i, chunk in enumerate(chunks):
        logger.debug("Synthesizing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        audio = await asyncio.to_thread(
            _synthesize_sync, chunk, voice, instruct, language,
        )
        audio_chunks.append(audio)

    return concat_audio_chunks(audio_chunks)


async def synthesize_multi(
    segments: list[dict],
    pause_ms: int = 300,
) -> bytes:
    """Multi-voice synthesis. Each segment: {text, voice, instruct?, language?}.

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
            language=seg.get("language"),
        )
        wav_parts.append(wav)

    return concat_audio_chunks(wav_parts, pause_ms)
