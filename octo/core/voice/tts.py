"""Local ParlerTTS engine — in-process inference with chunking and multi-voice.

All heavy imports (torch, parler_tts, soundfile, numpy) are lazy.
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
_tokenizer = None
_desc_tokenizer = None
_model_id: str | None = None

_DEFAULT_MODEL = "parler-tts/parler-tts-mini-v1.1"
_SAMPLE_RATE = 44100

# ── Voice profiles (description + per-voice seed) ────────────────────
# Each voice is defined by a natural language description and a fixed seed.
# The seed controls timbre — different seeds produce different voices
# from the same description.

_VOICES: dict[str, dict] = {
    "Ryan": {
        "description": "Ryan speaks with excitement and energy in a slightly high pitch",
        "seed": 42,
    },
    "Vivian": {
        "description": "A female speaker speaks with laughter and curiosity",
        "seed": 87,
    },
}

# OpenAI alias → voice name (backward compat)
_VOICE_MAP = {
    "alloy": "Ryan",
    "echo": "Ryan",
    "fable": "Ryan",
    "onyx": "Ryan",
    "nova": "Vivian",
    "shimmer": "Vivian",
}

_DEFAULT_VOICE = "Ryan"


# ── Model loading ────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    """Resolve model path: VOICE_MODEL_DIR env > HF hub."""
    env_dir = os.environ.get("VOICE_MODEL_DIR")
    if env_dir:
        return env_dir
    return _DEFAULT_MODEL


def _get_model():
    """Lazy-load ParlerTTS model + tokenizers (singleton)."""
    global _model, _tokenizer, _desc_tokenizer, _model_id

    model_id = _resolve_model_path()
    if _model is not None and _model_id == model_id:
        return _model, _tokenizer, _desc_tokenizer

    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    logger.info("Loading ParlerTTS: %s on %s (%s)", model_id, device, dtype)
    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _desc_tokenizer = AutoTokenizer.from_pretrained(
        _model.config.text_encoder._name_or_path,
    )
    _model_id = model_id
    return _model, _tokenizer, _desc_tokenizer


# ── Voice resolution ─────────────────────────────────────────────────

def _resolve_voice(voice: str, instruct: str | None = None) -> tuple[str, int]:
    """Resolve voice name/alias to (description, seed).

    If instruct is provided, it replaces the default description
    (allowing full control over voice characteristics). Seed stays.
    """
    # Map OpenAI alias to voice name
    name = _VOICE_MAP.get(voice, voice)

    # Look up voice profile
    profile = _VOICES.get(name)
    if not profile:
        profile = _VOICES[_DEFAULT_VOICE]

    description = instruct if instruct else profile["description"]
    return description, profile["seed"]


# ── Synchronous synthesis ────────────────────────────────────────────

def _synthesize_sync(
    text: str,
    voice: str,
    instruct: str | None = None,
    language: str | None = None,
) -> bytes:
    """Run TTS inference synchronously. Returns WAV bytes."""
    import torch
    import soundfile as sf

    description, seed = _resolve_voice(voice, instruct)

    # Per-voice seed for consistent timbre
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model, tokenizer, desc_tokenizer = _get_model()
    device = next(model.parameters()).device

    # Tokenize description and text
    desc_inputs = desc_tokenizer(description, return_tensors="pt").to(device)
    text_inputs = tokenizer(text, return_tensors="pt").to(device)

    generation = model.generate(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=text_inputs.input_ids,
        prompt_attention_mask=text_inputs.attention_mask,
        do_sample=True,
        temperature=1.0,
    )

    audio = generation.cpu().numpy().squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio, _SAMPLE_RATE, format="WAV", subtype="PCM_16")
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
    voice: str = "Ryan",
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
    """Multi-voice synthesis. Each segment: {text, voice, instruct?}.

    Auto-chunks long segments. Groups by voice for consistent timbre
    (all segments for one voice generated together before switching).
    Concatenates in original order with silence between segments.
    """
    if not segments:
        return b""

    # Group segment indices by resolved voice key (name + instruct)
    from collections import defaultdict

    voice_groups: dict[tuple[str, str | None], list[int]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        voice = seg.get("voice", "Ryan")
        instruct = seg.get("instruct")
        key = (voice, instruct)
        voice_groups[key].append(idx)

    # Generate all segments grouped by voice — keeps timbre consistent
    wav_by_idx: dict[int, bytes] = {}
    for (voice, instruct), indices in voice_groups.items():
        logger.info("Generating %d segments for voice=%s", len(indices), voice)
        for idx in indices:
            seg = segments[idx]
            wav = await synthesize(
                text=seg["text"],
                voice=voice,
                instruct=instruct,
                language=seg.get("language"),
            )
            wav_by_idx[idx] = wav

    # Reassemble in original order
    wav_parts = [wav_by_idx[i] for i in range(len(segments))]
    return concat_audio_chunks(wav_parts, pause_ms)
