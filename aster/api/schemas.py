from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContentPart(BaseModel):
    type: str | None = None
    text: str | None = None
    input_text: str | None = None
    content: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[ContentPart] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    n: int | None = None
    stop: str | list[str] | None = None
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    enable_thinking: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None

    model_config = ConfigDict(extra="allow")


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
