from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from aster.inference.media import MediaRef


def _empty_chat_template_kwargs() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class InferenceRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    media: tuple[MediaRef, ...] = field(default_factory=tuple)
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: str | list[str] | None = None
    stop_token_ids: tuple[int, ...] = ()
    parser_stop_sequences: tuple[str, ...] = ()
    request_class: str = "default"
    trace_id: str | None = None
    request_aliases: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    enable_thinking: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=_empty_chat_template_kwargs)
    thinking_token_budget: int | None = None
    structured_output_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class InferenceResponse:
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    prefill_cache_hit: bool
    generation_cache_reuse: bool
    speculative_enabled: bool
    speculative_path_mode: Literal["disabled", "target_reuse", "full_prompt_no_cache"]
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    finish_reason: str = "stop"
