from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class InferenceRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    request_class: str = "default"
    trace_id: str | None = None
    enable_thinking: bool = False


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
