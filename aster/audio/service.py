"""Audio service abstraction for ASR and TTS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from aster.telemetry.logging import get_logger


@dataclass(slots=True)
class ASRResult:
    text: str
    language: str | None = None
    duration: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class TTSResult:
    audio: bytes
    duration: float | None = None
    sample_rate: int = 22050


class ASRService(ABC):
    """Abstract ASR service interface."""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ASRResult:
        """Transcribe audio to text."""
        pass

    @abstractmethod
    def health(self) -> bool:
        """Check service health."""
        pass


class TTSService(ABC):
    """Abstract TTS service interface."""

    @abstractmethod
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
        pass

    @abstractmethod
    def health(self) -> bool:
        """Check service health."""
        pass


class AudioServiceContainer:
    """Container for audio services."""

    def __init__(self, asr: ASRService | None = None, tts: TTSService | None = None) -> None:
        self.asr = asr
        self.tts = tts
        self.logger = get_logger(__name__)

    def health(self) -> dict[str, bool]:
        """Check health of all audio services."""
        return {
            "asr": self.asr.health() if self.asr else False,
            "tts": self.tts.health() if self.tts else False,
        }
