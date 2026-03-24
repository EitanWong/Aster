"""vLLM-MLX backed TTS runtime."""

from __future__ import annotations

import io
import wave

import numpy as np

from aster.audio.service import TTSResult, TTSService
from aster.core.config import TTSSettings
from aster.telemetry.logging import get_logger


_LANGUAGE_CODES = {
    "en": "a",
    "english": "a",
    "es": "e",
    "spanish": "e",
    "fr": "f",
    "french": "f",
    "ja": "j",
    "japanese": "j",
    "zh": "z",
    "chinese": "z",
    "it": "i",
    "italian": "i",
    "pt": "p",
    "portuguese": "p",
    "hi": "h",
    "hindi": "h",
}


class VLLMMLXTTSRuntime(TTSService):
    """TTS service using vllm_mlx.audio.tts."""

    def __init__(self, settings: TTSSettings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__)
        self._engine = None
        self._healthy = True

    def _load_model(self) -> None:
        try:
            from vllm_mlx.audio.tts import TTSEngine

            model_ref = self.settings.model_path or self.settings.model
            self.logger.info(f"Loading vLLM-MLX TTS model from {model_ref}")
            self._engine = TTSEngine(model_ref)
            self._engine.load()
            self._healthy = True
            self.logger.info("vLLM-MLX TTS model loaded successfully")
        except Exception as exc:
            self.logger.error(f"Failed to load vLLM-MLX TTS model: {exc}")
            self._healthy = False

    def _ensure_loaded(self) -> None:
        if self._engine is None:
            self._load_model()
        if not self._healthy or self._engine is None:
            raise RuntimeError("TTS model not loaded")

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
        self._ensure_loaded()

        if reference_audio is not None or speaker is not None or instruct is not None:
            self.logger.info("vllm_mlx_tts_extra_controls_ignored")

        try:
            result = self._engine.generate(
                text=text,
                voice=voice or self.settings.default_voice,
                speed=speed,
                lang_code=self._lang_code(language),
            )
            return TTSResult(
                audio=self._to_wav_bytes(result.audio, result.sample_rate),
                duration=result.duration,
                sample_rate=result.sample_rate,
            )
        except Exception:
            self.logger.exception("vllm_mlx_synthesize_failed")
            raise

    def health(self) -> bool:
        return self._healthy

    def _lang_code(self, language: str | None) -> str:
        if not language:
            return "a"
        return _LANGUAGE_CODES.get(language.strip().lower(), "a")

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        pcm = np.asarray(audio, dtype=np.float32)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_int16.tobytes())
            return buffer.getvalue()
