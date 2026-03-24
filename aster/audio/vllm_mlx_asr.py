"""vLLM-MLX backed ASR runtime."""

from __future__ import annotations

import os
import tempfile

from aster.audio.service import ASRResult, ASRService
from aster.core.config import ASRSettings
from aster.telemetry.logging import get_logger


class VLLMMLXASRRuntime(ASRService):
    """ASR service using vllm_mlx.audio.stt."""

    def __init__(self, settings: ASRSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._engine = None
        self._healthy = True

    def _load_model(self) -> None:
        try:
            from vllm_mlx.audio.stt import STTEngine

            model_ref = self.settings.model_path or self.settings.model
            self.logger.info(f"Loading vLLM-MLX ASR model from {model_ref}")
            self._engine = STTEngine(model_ref)
            self._engine.load()
            self._healthy = True
            self.logger.info("vLLM-MLX ASR model loaded successfully")
        except Exception as exc:
            self.logger.error(f"Failed to load vLLM-MLX ASR model: {exc}")
            self._healthy = False

    def _ensure_loaded(self) -> None:
        if self._engine is None:
            self._load_model()
        if not self._healthy or self._engine is None:
            raise RuntimeError("ASR model not loaded")

    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRResult:
        self._ensure_loaded()

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name

            result = self._engine.transcribe(tmp_path, language=language)
            return ASRResult(
                text=result.text,
                language=result.language or language,
                duration=result.duration,
                confidence=None,
            )
        except Exception:
            self.logger.exception("vllm_mlx_transcribe_failed")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def health(self) -> bool:
        return self._healthy
