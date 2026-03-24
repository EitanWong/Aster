"""MLX-based ASR runtime for Qwen3-ASR models."""

from __future__ import annotations

import time
from typing import Any

from aster.audio.service import ASRResult, ASRService
from aster.core.config import ASRSettings
from aster.telemetry.logging import get_logger


class MLXASRRuntime(ASRService):
    """ASR service using mlx-audio and Qwen3-ASR models."""

    def __init__(self, settings: ASRSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._model: Any = None
        self._healthy = False
        self._load_model()

    def _load_model(self) -> None:
        """Load ASR model from mlx-audio."""
        try:
            from mlx_audio.stt.utils import load_model

            self.logger.info(f"Loading ASR model from {self.settings.model_path}")
            self._model = load_model(self.settings.model_path or self.settings.model)
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
            import tempfile
            import soundfile as sf
            import numpy as np

            # Write audio bytes to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name

            # Load audio using soundfile
            audio_data, sample_rate = sf.read(tmp_path)
            
            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            started = time.perf_counter()
            
            # Use model.generate directly instead of generate_transcription
            segments = self._model.generate(
                audio_data,
                verbose=False,
            )
            
            duration = time.perf_counter() - started

            # Extract text from segments
            text = ""
            if isinstance(segments, list):
                text = " ".join([seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in segments])
            elif hasattr(segments, "text"):
                text = segments.text
            else:
                text = str(segments)

            self.logger.info(
                "transcribe_complete",
                extra={
                    "duration_s": round(duration, 4),
                    "text_length": len(text),
                },
            )

            return ASRResult(
                text=text,
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
