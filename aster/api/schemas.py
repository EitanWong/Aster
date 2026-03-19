from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "aster"


class HealthResponse(BaseModel):
    status: str
    degraded: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "default"
    language: str | None = None
    speed: float = 1.0
    reference_audio: str | None = None
    speaker: str | None = None
    instruct: str | None = None


class ASRResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    confidence: float | None = None

