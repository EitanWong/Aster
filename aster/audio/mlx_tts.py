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
        if not self._healthy:
            raise RuntimeError("TTS models not loaded")

        try:
            from mlx_audio.tts.generate import generate_audio
            import tempfile
            import os

            started = time.perf_counter()

            # Determine which model to use
            use_custom_voice = speaker is not None or instruct is not None
            model = self._custom_voice_model if use_custom_voice else self._base_model

            if model is None:
                raise RuntimeError(f"{'CustomVoice' if use_custom_voice else 'Base'} TTS model not available")

            # Prepare output directory
            with tempfile.TemporaryDirectory() as tmpdir:
                output_prefix = os.path.join(tmpdir, "output")

                if use_custom_voice:
                    # CustomVoice mode: use speaker + instruct
                    result = await generate_audio(
                        model=model,
                        text=text,
                        speaker=speaker or self.settings.tts_default_voice,
                        instruct=instruct or "",
                        file_prefix=output_prefix,
                    )
                else:
                    # Base mode: use reference audio for voice cloning
                    ref_audio_path = None
                    if reference_audio:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(reference_audio)
                            ref_audio_path = tmp.name

                    result = await generate_audio(
                        model=model,
                        text=text,
                        ref_audio=ref_audio_path,
                        file_prefix=output_prefix,
                    )

                    if ref_audio_path and os.path.exists(ref_audio_path):
                        os.unlink(ref_audio_path)

                # Read generated audio
                audio_files = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
                if not audio_files:
                    raise RuntimeError("No audio file generated")

                audio_path = os.path.join(tmpdir, audio_files[0])
                with open(audio_path, "rb") as f:
                    audio_data = f.read()

            duration = time.perf_counter() - started

            self.logger.info(
                "synthesize_complete",
                extra={
                    "duration_s": round(duration, 4),
                    "text_length": len(text),
                    "mode": "custom_voice" if use_custom_voice else "base",
                },
            )

            return TTSResult(
                audio=audio_data,
                duration=duration,
                sample_rate=22050,
            )
        except Exception as e:
            self.logger.exception("synthesize_failed", extra={"error": str(e)})
            raise

    def health(self) -> bool:
        """Check if TTS service is healthy."""
        return self._healthy
