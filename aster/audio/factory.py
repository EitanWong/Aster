from __future__ import annotations

from aster.audio.mlx_asr import MLXASRRuntime
from aster.audio.mlx_tts import MLXTTSRuntime
from aster.audio.service import ASRService, TTSService
from aster.audio.vllm_mlx_asr import VLLMMLXASRRuntime
from aster.audio.vllm_mlx_tts import VLLMMLXTTSRuntime
from aster.core.config import ASRSettings, TTSSettings
from aster.core.errors import ConfigurationError


def create_asr_service(settings: ASRSettings) -> ASRService | None:
    if not settings.enabled:
        return None
    if settings.backend == "mlx":
        return MLXASRRuntime(settings)
    if settings.backend == "vllm_mlx":
        return VLLMMLXASRRuntime(settings)
    raise ConfigurationError(
        code="unsupported_asr_backend",
        message=f"Unsupported ASR backend: {settings.backend}",
        status_code=400,
    )


def create_tts_service(settings: TTSSettings) -> TTSService | None:
    if not settings.enabled:
        return None
    if settings.backend == "mlx":
        return MLXTTSRuntime(settings)
    if settings.backend == "vllm_mlx":
        return VLLMMLXTTSRuntime(settings)
    raise ConfigurationError(
        code="unsupported_tts_backend",
        message=f"Unsupported TTS backend: {settings.backend}",
        status_code=400,
    )
