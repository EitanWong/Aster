from __future__ import annotations

from aster.audio.service import ASRService, TTSService
from aster.core.config import ASRSettings, TTSSettings
from aster.core.errors import ConfigurationError


def create_asr_service(settings: ASRSettings) -> ASRService | None:
    if not settings.enabled:
        return None
    if settings.backend != "mlx":
        raise ConfigurationError(
            code="unsupported_asr_backend",
            message=f"Unsupported ASR backend: {settings.backend}",
            status_code=400,
        )
    from aster.audio.mlx_asr import MLXASRRuntime

    return MLXASRRuntime(settings)


def create_tts_service(settings: TTSSettings) -> TTSService | None:
    if not settings.enabled:
        return None
    if settings.backend != "mlx":
        raise ConfigurationError(
            code="unsupported_tts_backend",
            message=f"Unsupported TTS backend: {settings.backend}",
            status_code=400,
        )
    from aster.audio.mlx_tts import MLXTTSRuntime

    return MLXTTSRuntime(settings)
