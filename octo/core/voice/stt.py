"""Local Whisper STT engine — MLX (Apple Silicon) or faster-whisper (CUDA/CPU).

All heavy imports are lazy.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_MLX = "mlx-community/whisper-large-v3-mlx"
_DEFAULT_MODEL_FASTER = "large-v3"


# ── Backend detection ────────────────────────────────────────────────

def _detect_backend() -> str:
    """Return 'mlx' or 'faster' based on available packages."""
    try:
        import mlx_whisper  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import faster_whisper  # noqa: F401
        return "faster"
    except ImportError:
        pass
    raise ImportError(
        "No STT backend available. Install mlx-whisper (Apple Silicon) "
        "or faster-whisper (CUDA/CPU)."
    )


# ── MLX Whisper ──────────────────────────────────────────────────────

def _resolve_stt_model_mlx() -> str:
    """Resolve MLX Whisper model: VOICE_MODEL_DIR > .octo/models/ > HF hub."""
    from pathlib import Path

    env_dir = os.environ.get("VOICE_MODEL_DIR")
    if env_dir:
        return env_dir

    for candidate in [
        Path.home() / ".octo" / "models" / "whisper-large-v3-mlx",
        Path(".octo") / "models" / "whisper-large-v3-mlx",
    ]:
        if candidate.exists() and any(candidate.iterdir()):
            return str(candidate)

    return _DEFAULT_MODEL_MLX


def _transcribe_mlx_sync(audio_data: bytes, language: str | None = None) -> str:
    """Transcribe using mlx-whisper (requires temp file)."""
    import mlx_whisper

    hf_repo = _resolve_stt_model_mlx()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_data)
        tmp.flush()

        kwargs: dict = {"path_or_hf_repo": hf_repo}
        if language:
            kwargs["language"] = language

        result = mlx_whisper.transcribe(tmp.name, **kwargs)

    return result.get("text", "").strip()


# ── faster-whisper ───────────────────────────────────────────────────

_fw_model = None
_fw_model_size: str | None = None


def _transcribe_faster_sync(audio_data: bytes, language: str | None = None) -> str:
    """Transcribe using faster-whisper."""
    global _fw_model, _fw_model_size
    from faster_whisper import WhisperModel

    model_dir = os.environ.get("VOICE_MODEL_DIR")
    size = model_dir if model_dir else _DEFAULT_MODEL_FASTER

    if _fw_model is None or _fw_model_size != size:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("Loading faster-whisper: %s (device=%s)", size, device)
        _fw_model = WhisperModel(size, device=device, compute_type=compute_type)
        _fw_model_size = size

    buf = io.BytesIO(audio_data)
    kwargs: dict = {"vad_filter": True}
    if language:
        kwargs["language"] = language

    segments, _info = _fw_model.transcribe(buf, **kwargs)
    return " ".join(s.text.strip() for s in segments)


# ── Public async API ─────────────────────────────────────────────────

async def transcribe(audio_data: bytes, language: str | None = None) -> str:
    """Transcribe audio bytes to text. Auto-detects MLX or faster-whisper."""
    backend = _detect_backend()
    if backend == "mlx":
        return await asyncio.to_thread(_transcribe_mlx_sync, audio_data, language)
    return await asyncio.to_thread(_transcribe_faster_sync, audio_data, language)
