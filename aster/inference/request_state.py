from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aster.inference.contracts import InferenceRequest, InferenceResponse
from aster.inference.stream_collector import StreamCollector


class RequestPhase(StrEnum):
    SUBMITTED = "submitted"
    ADMITTED = "admitted"
    PREFIX_LOOKUP = "prefix_lookup"
    PREFILL_WAIT = "prefill_wait"
    PREFILLING = "prefilling"
    DECODE_READY = "decode_ready"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class RequestState:
    request_id: str
    request: InferenceRequest
    created_at: float = field(default_factory=time.monotonic)
    phase: RequestPhase = RequestPhase.SUBMITTED
    prompt_tokens: list[int] = field(default_factory=list)
    reuse_points: tuple[int, ...] = ()
    model_fingerprint: str | None = None
    admission_prepared: bool = False
    prompt_cache: Any | None = None
    matched_prefix_tokens: int = 0
    cache_token_count: int = 0
    attached_snapshot_key: str | None = None
    estimated_bytes: int = 0
    prefill_transient_profile: Any | None = None
    admission_retries: int = 0
    checkpoints_created: set[int] = field(default_factory=set)
    decode_sampler: Any | None = None
    decode_detokenizer: Any | None = None
    decode_stop_token_ids: frozenset[int] = field(default_factory=frozenset)
    decode_logits_processors: tuple[Any, ...] = ()
    decode_logits_processor_context_size: int | None = None
    next_input_token: int | None = None
    response_future: asyncio.Future[InferenceResponse] | None = None
    stream_collector: StreamCollector | None = None
    output_parts: list[str] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    stop_sequences: tuple[str, ...] = ()
    pending_stop_text: str = ""
    completion_tokens: int = 0
    finish_reason: str | None = None
    enqueued_at: float | None = None
    admission_started_at: float | None = None
    admitted_at: float | None = None
    prefill_started_at: float | None = None
    prefill_finished_at: float | None = None
    decode_init_started_at: float | None = None
    decode_ready_at: float | None = None
    prefill_seconds: float = 0.0
    prefill_steps: int = 0
    prefill_active_memory_gb: float | None = None
    prefill_transient_bytes_per_token: float = 0.0
    decode_started_at: float | None = None
    first_token_at: float | None = None
    last_decode_step_at: float | None = None
    decode_steps: int = 0
    completed_at: float | None = None
    response_ready_at: float | None = None
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    cancel_requested: bool = False
    terminal_accounted: bool = False

    @property
    def target_cache_token_count(self) -> int:
        return max(len(self.prompt_tokens) - 1, 0)

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_tokens)

    def mark_enqueued(self) -> None:
        if self.enqueued_at is None:
            self.enqueued_at = time.monotonic()

    def mark_admission_started(self) -> None:
        if self.admission_started_at is None:
            self.admission_started_at = time.monotonic()

    def mark_admitted(self) -> None:
        if self.admitted_at is None:
            self.admitted_at = time.monotonic()

    def mark_prefill_started(self) -> None:
        if self.prefill_started_at is None:
            self.prefill_started_at = time.monotonic()

    def mark_prefill_finished(self) -> None:
        if self.prefill_finished_at is None:
            self.prefill_finished_at = time.monotonic()

    def mark_decode_init_started(self) -> None:
        if self.decode_init_started_at is None:
            self.decode_init_started_at = time.monotonic()

    def mark_decode_ready(self) -> None:
        if self.decode_ready_at is None:
            self.decode_ready_at = time.monotonic()

    def mark_decode_started(self) -> None:
        if self.decode_started_at is None:
            self.decode_started_at = time.monotonic()

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()

    def mark_decode_step(self) -> None:
        self.decode_steps += 1
        self.last_decode_step_at = time.monotonic()

    def mark_terminal(self, phase: RequestPhase) -> None:
        self.phase = phase
        self.completed_at = time.monotonic()

    def mark_response_ready(self) -> None:
        if self.response_ready_at is None:
            self.response_ready_at = time.monotonic()
