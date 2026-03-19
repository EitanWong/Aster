"""MLX-based ASR runtime for Qwen3-ASR models."""

from __future__ import annotations

import time
from typing import Any

from aster.audio.service import ASRResult, ASRService
from aster.core.config import AudioSettings
from aster.telemetry.logging import get_logger


class MLXASRRuntime(ASRService):
    """ASR service using mlx-audio and Qwen3-ASR models."""

    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._model: Any = None
        self._healthy = False
        self._load_model()

    def _load_model(self) -> None:
        """Load ASR model from mlx-audio."""
        try:
            from mlx_audio.stt.utils import load_model

            self.logger.info(f"Loading ASR model from {self.settings.asr_model_path}")
            self._model = load_model(self.settings.asr_model_path or self.settings.asr_model)
            self._healthy = True
            self.logger.info("ASR model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {e}")
            self._healthy = False

    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRResult:
        """Transcribe audio bytes to text."""
        if not self._healthy or self._model is None:
            raise RuntimeError("ASR model not loaded")

        try:
            from mlx_audio.stt.generate import generate_transcription
            import tempfile

            # Write audio bytes to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name

            started = time.perf_counter()
            result = await generate_transcription(
                model=self._model,
                audio_path=tmp_path,
                language=language,
            )
            duration = time.perf_counter() - started

            self.logger.info(
                "transcribe_complete",
                extra={
                    "duration_s": round(duration, 4),
                    "text_length": len(result.text) if hasattr(result, "text") else 0,
                },
            )

            return ASRResult(
                text=result.text if hasattr(result, "text") else str(result),
                language=language,
                duration=duration,
                confidence=None,
            )
        except Exception as e:
            self.logger.exception("transcribe_failed", extra={"error": str(e)})
            raise

    def health(self) -> bool:
        """Check if ASR service is healthy."""
        return self._healthy
