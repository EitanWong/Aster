"""MLX-based TTS runtime for Qwen3-TTS models."""

from __future__ import annotations

import time
from typing import Any

from aster.audio.service import TTSResult, TTSService
from aster.core.config import AudioSettings
from aster.telemetry.logging import get_logger


class MLXTTSRuntime(TTSService):
    """TTS service using mlx-audio and Qwen3-TTS models."""

    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._base_model: Any = None
        self._custom_voice_model: Any = None
        self._healthy = False
        self._load_models()

    def _load_models(self) -> None:
        """Load TTS models from mlx-audio."""
        try:
            from mlx_audio.tts.utils import load_model

            # Load Base model
            self.logger.info(f"Loading TTS Base model from {self.settings.tts_model_path}")
            self._base_model = load_model(self.settings.tts_model_path or self.settings.tts_model)

            # Load CustomVoice model if configured
            if self.settings.tts_custom_voice_model:
                self.logger.info(f"Loading TTS CustomVoice model from {self.settings.tts_custom_voice_path}")
                self._custom_voice_model = load_model(
                    self.settings.tts_custom_voice_path or self.settings.tts_custom_voice_model
                )

            self._healthy = True
            self.logger.info("TTS models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load TTS models: {e}")
            self._healthy = False

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str | None = None,
        speed: float = 1.0,
        reference_audio: bytes | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
    ) -> TTSResult:
        """Synthesize text to speech."""
        if not self._healthy or self._base_model is None:
            raise RuntimeError("TTS models not loaded")

        try:
            import tempfile
            import os
            import numpy as np
            import soundfile as sf

            started = time.perf_counter()

            # Use base model (Qwen3-TTS)
            model = self._base_model

            # model.generate() returns a generator of result objects
            # Each result has .audio attribute (mx.array)
            results = list(model.generate(
                text=text,
                voice=voice or "default",
                language=language or "English",
            ))

            if not results:
                raise RuntimeError("No audio generated from TTS model")

            # Collect audio from all results
            audio_chunks = []
            for result in results:
                if hasattr(result, 'audio'):
                    # result.audio is mx.array, convert to numpy
                    audio_chunk = np.array(result.audio, dtype=np.float32)
                    audio_chunks.append(audio_chunk)
                else:
                    # Fallback if result is directly audio data
                    audio_chunks.append(np.array(result, dtype=np.float32))

            if not audio_chunks:
                raise RuntimeError("No audio data extracted from results")

            # Concatenate all chunks
            audio_output = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

            # Ensure float32
            if audio_output.dtype != np.float32:
                audio_output = audio_output.astype(np.float32)

            # Normalize to [-1, 1] range if needed
            max_val = np.abs(audio_output).max()
            if max_val > 1.0:
                audio_output = audio_output / max_val

            # Convert to int16 for WAV format
            audio_int16 = (audio_output * 32767).astype(np.int16)

            # Write to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_int16, 22050)
                with open(tmp.name, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(tmp.name)

            duration = time.perf_counter() - started

            self.logger.info(
                "synthesize_complete",
                extra={
                    "duration_s": round(duration, 4),
                    "text_length": len(text),
                    "voice": voice,
                    "audio_samples": len(audio_int16),
                },
            )

            return TTSResult(
                audio=audio_bytes,
                duration=duration,
                sample_rate=22050,
            )
        except Exception as e:
            self.logger.exception("synthesize_failed", extra={"error": str(e)})
            raise

    def health(self) -> bool:
        """Check if TTS service is healthy."""
        return self._healthy
