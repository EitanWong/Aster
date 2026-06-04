from __future__ import annotations

from pathlib import Path

from aster.core.config import ASRSettings, EmbeddingsSettings, TTSSettings
from aster.core.errors import AsterError

_EMBEDDING_MODELS = frozenset(
    {
        "mlx-community/ModernBERT-base-mlx",
        "mlx-community/all-MiniLM-L6-v2-4bit",
        "mlx-community/bert-base-uncased-mlx",
        "mlx-community/bge-large-en-v1.5-4bit",
        "mlx-community/embeddinggemma-300m-6bit",
        "mlx-community/multilingual-e5-large-mlx",
        "mlx-community/multilingual-e5-small-mlx",
    }
)

_STT_MODEL_ALIASES = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
    "asr": "__configured__",
    "Qwen3-ASR-0.6B": "models/qwen3-asr-0.6b",
}

_TTS_MODEL_ALIASES = {
    "kokoro": "mlx-community/Kokoro-82M-bf16",
    "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
    "chatterbox": "mlx-community/chatterbox-turbo-fp16",
    "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
    "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
    "voxcpm": "mlx-community/VoxCPM1.5",
    "tts": "__configured__",
    "Qwen3-TTS-0.6B": "models/qwen3-tts-0.6b-base",
    "Qwen3-TTS-0.6B-Base": "models/qwen3-tts-0.6b-base",
}


def validate_embedding_model(requested_model: str | None, settings: EmbeddingsSettings) -> str | None:
    if requested_model is None:
        return None
    if requested_model in _configured_model_ids(settings.model, settings.model_path):
        return None
    supported = ", ".join(sorted(_configured_model_ids(settings.model, settings.model_path)))
    allowlist = ", ".join(sorted(_EMBEDDING_MODELS))
    raise AsterError(
        code="embedding_model_not_available",
        message=(
            f"Embedding model '{requested_model}' is not available on this Aster instance. "
            f"Request-time embedding model loading is disabled; use the configured model: {supported}. "
            f"Known vllm-mlx embedding models for server configuration are: {allowlist}."
        ),
        status_code=400,
    )


def validate_stt_model(requested_model: str | None, settings: ASRSettings) -> str:
    return _validate_audio_model(
        endpoint="Transcription",
        requested_model=requested_model,
        configured_model=settings.model,
        configured_model_path=settings.model_path,
        aliases=_STT_MODEL_ALIASES,
    )


def validate_tts_model(requested_model: str | None, settings: TTSSettings) -> str:
    return _validate_audio_model(
        endpoint="Speech",
        requested_model=requested_model,
        configured_model=settings.model,
        configured_model_path=settings.model_path,
        aliases=_TTS_MODEL_ALIASES,
    )


def _validate_audio_model(
    *,
    endpoint: str,
    requested_model: str | None,
    configured_model: str,
    configured_model_path: str | None,
    aliases: dict[str, str],
) -> str:
    configured = _configured_model_ids(configured_model, configured_model_path)
    if requested_model is None:
        return configured_model_path or configured_model

    resolved = aliases.get(requested_model, requested_model)
    if resolved == "__configured__":
        return configured_model_path or configured_model
    if requested_model in configured or resolved in configured:
        return resolved

    supported = sorted(set(configured) | {alias for alias, target in aliases.items() if target == "__configured__" or target in configured})
    raise AsterError(
        code="audio_model_not_available",
        message=(
            f"{endpoint} model '{requested_model}' is not available on this Aster instance. "
            "Request-time audio model loading is disabled. "
            f"Supported request models for the configured backend are: {', '.join(supported)}."
        ),
        status_code=400,
    )


def _configured_model_ids(model: str | None, model_path: str | None) -> set[str]:
    ids: set[str] = set()
    for item in (model, model_path):
        if not item:
            continue
        ids.add(item)
        ids.add(Path(item).name)
    return ids
