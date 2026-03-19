"""Audio API schemas for OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request."""

    file: bytes = Field(..., description="Audio file content")
    model: str = Field(default="Qwen3-ASR-0.6B", description="ASR model to use")
    language: str | None = Field(default=None, description="Language code (e.g., 'en', 'zh')")
    prompt: str | None = Field(default=None, description="Optional prompt to guide transcription")
    response_format: Literal["json", "text", "srt", "vtt"] = Field(default="json")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str
    language: str | None = None
    duration: float | None = None
    model: str = "Qwen3-ASR-0.6B"
    aster: dict[str, Any] = Field(default_factory=dict)


class SpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request."""

    input: str = Field(..., description="Text to synthesize", max_length=4096)
    model: str = Field(default="Qwen3-TTS-0.6B-Base", description="TTS model to use")
    voice: str = Field(default="default", description="Voice ID or preset")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str | None = Field(default=None, description="Language code")
    reference_audio: bytes | None = Field(default=None, description="Reference audio for voice cloning (Base model)")
    speaker: str | None = Field(default=None, description="Preset speaker ID (CustomVoice model)")
    instruct: str | None = Field(default=None, description="Style/emotion instruction (CustomVoice model)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechResponse(BaseModel):
    """OpenAI-compatible speech response metadata."""

    model: str
    duration: float | None = None
    aster: dict[str, Any] = Field(default_factory=dict)
