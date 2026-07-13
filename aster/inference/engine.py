from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from aster.core.config import RuntimeSettings
from aster.core.errors import AsterError, OverloadedError
from aster.inference.contracts import InferenceRequest, InferenceResponse
from aster.inference.decode_engine import DecodeChunk
from aster.inference.embedding_backends import build_embedding_backend
from aster.inference.mlx_streams import bind_generation_streams
from aster.inference.model_runner import (
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
    PrefillChunkResult,
)
from aster.inference.prefix_store import PrefixStore
from aster.inference.prompt_warmup import ensure_user_terminator, load_warmup_file
from aster.inference.request_state import RequestPhase, RequestState
from aster.inference.runtime_kernel import RuntimeKernel, build_runtime_kernel
from aster.inference.stream_collector import StreamCollector
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry

RunnerResult = TypeVar("RunnerResult")
PrepareResult = str

_PREPARE_ADMITTED: PrepareResult = "admitted"
_PREPARE_DEFERRED: PrepareResult = "deferred"
_PREPARE_TERMINAL: PrepareResult = "terminal"

_WAITING_PHASES = {
    RequestPhase.SUBMITTED,
    RequestPhase.ADMITTED,
    RequestPhase.PREFIX_LOOKUP,
}
_RUNNING_PHASES = {
    RequestPhase.PREFILL_WAIT,
    RequestPhase.PREFILLING,
    RequestPhase.DECODE_READY,
    RequestPhase.DECODING,
}
_RUNNER_IN_FLIGHT_PHASES = {
    RequestPhase.PREFILLING,
    RequestPhase.DECODING,
}


@dataclass(slots=True)
class EngineStatus:
    engine_running: bool
    pending_requests: int
    prefill_requests: int
    decode_requests: int
    snapshot_entries: int
    snapshot_bytes: int
    prefix_reuse_attempts: int
    prefix_reuse_hits: int
    prefix_tokens_reused: int
    prefill_steps: int
    decode_steps: int
    completed_requests: int
    failed_requests: int
    cancelled_requests: int
    admission_rejections: int
    timed_out_requests: int
    runtime_cache_clear_attempts: int
    runtime_cache_clear_failures: int
    cancelled_prefill_checkpoints: int
    prefill_yield_rotations: int


class InferenceEngine:
    def __init__(self, settings: RuntimeSettings, metrics: MetricsRegistry) -> None:
        self.settings = settings
        self.metrics = metrics
        self.logger = get_logger(__name__)
        self.model_runner = ModelRunner(settings)
        self.runtime_kernel: RuntimeKernel = build_runtime_kernel(settings, self.model_runner)
        self.prefix_store = PrefixStore(
            budget_bytes=settings.engine.snapshot_budget_bytes,
            max_entries=settings.engine.snapshot_max_entries,
            min_prefix_tokens=settings.engine.snapshot_min_prefix_tokens,
            enabled=settings.engine.prefix_cache_enabled,
        )
        self.embedding_backend = build_embedding_backend(settings)
        self._submission_queue: asyncio.Queue[RequestState] = asyncio.Queue(
            maxsize=settings.api.max_queue_depth
        )
        self._runner_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="aster-mlx-runner",
        )
        self._prefill_queue: deque[str] = deque()
        self._decode_queue: deque[str] = deque()
        self._prefill_yield_request_ids: set[str] = set()
        self._requests: dict[str, RequestState] = {}
        self._request_aliases: dict[str, str] = {}
        self._recent_request_timelines: deque[dict[str, object]] = deque(maxlen=256)
        self._active_estimated_bytes = 0
        self._task: asyncio.Task[None] | None = None
        self._idle_event = asyncio.Event()
        self._running = False
        self._prefix_reuse_attempts = 0
        self._prefix_reuse_hits = 0
        self._prefix_tokens_reused = 0
        self._prefill_steps = 0
        self._decode_steps = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._cancelled_requests = 0
        self._admission_rejections = 0
        self._timed_out_requests = 0
        self._total_requests_processed = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._runtime_cache_clear_attempts = 0
        self._runtime_cache_clear_failures = 0
        self._cancelled_prefill_checkpoints = 0
        self._prefill_yield_rotations = 0
        self._prefill_model_seconds = 0.0
        self._prefill_model_tokens = 0
        self._max_prefill_step_seconds = 0.0
        self._decode_runner_seconds = 0.0
        self._decode_runner_batches = 0
        self._decode_runner_items = 0
        self._decode_runner_tokens = 0
        self._max_decode_batch_seconds = 0.0
        self._prefix_cache_loaded_from_disk = False
        self._warm_prompts_completed = False
        self._started_at: float | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._engine_loop(), name="aster-inference-engine")

    async def stop(self) -> None:
        self._running = False
        self._idle_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def health(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        requests = [self._request_status_snapshot(state, now=now) for state in self._requests.values()]
        num_waiting = sum(1 for state in self._requests.values() if state.phase in _WAITING_PHASES)
        num_running = sum(1 for state in self._requests.values() if state.phase in _RUNNING_PHASES)
        status = EngineStatus(
            engine_running=self.health(),
            pending_requests=self._submission_queue.qsize(),
            prefill_requests=len(self._prefill_queue),
            decode_requests=len(self._decode_queue),
            snapshot_entries=self.prefix_store.entry_count,
            snapshot_bytes=self.prefix_store.current_bytes,
            prefix_reuse_attempts=self._prefix_reuse_attempts,
            prefix_reuse_hits=self._prefix_reuse_hits,
            prefix_tokens_reused=self._prefix_tokens_reused,
            prefill_steps=self._prefill_steps,
            decode_steps=self._decode_steps,
            completed_requests=self._completed_requests,
            failed_requests=self._failed_requests,
            cancelled_requests=self._cancelled_requests,
            admission_rejections=self._admission_rejections,
            timed_out_requests=self._timed_out_requests,
            runtime_cache_clear_attempts=self._runtime_cache_clear_attempts,
            runtime_cache_clear_failures=self._runtime_cache_clear_failures,
            cancelled_prefill_checkpoints=self._cancelled_prefill_checkpoints,
            prefill_yield_rotations=self._prefill_yield_rotations,
        )
        return {
            "status": "running" if status.engine_running else "stopped",
            "model": self.settings.model.name,
            "uptime_s": round(now - self._started_at, 1) if self._started_at is not None else 0.0,
            "steps_executed": status.prefill_steps + status.decode_steps,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "total_requests_processed": self._total_requests_processed,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "requests": requests,
            "recent_request_timelines": list(self._recent_request_timelines),
            "engine_running": status.engine_running,
            "pending_requests": status.pending_requests,
            "prefill_requests": status.prefill_requests,
            "decode_requests": status.decode_requests,
            "snapshot_entries": status.snapshot_entries,
            "snapshot_bytes": status.snapshot_bytes,
            "prefix_reuse_attempts": status.prefix_reuse_attempts,
            "prefix_reuse_hits": status.prefix_reuse_hits,
            "prefix_tokens_reused": status.prefix_tokens_reused,
            "prefix_cache_stats": self.prefix_store.stats_snapshot(),
            "prefill_steps": status.prefill_steps,
            "decode_steps": status.decode_steps,
            "completed_requests": status.completed_requests,
            "failed_requests": status.failed_requests,
            "cancelled_requests": status.cancelled_requests,
            "admission_rejections": status.admission_rejections,
            "timed_out_requests": status.timed_out_requests,
            "runtime_cache_clear_attempts": status.runtime_cache_clear_attempts,
            "runtime_cache_clear_failures": status.runtime_cache_clear_failures,
            "cancelled_prefill_checkpoints": status.cancelled_prefill_checkpoints,
            "prefill_yield_rotations": status.prefill_yield_rotations,
            "active_estimated_bytes": self._active_estimated_bytes,
            "runtime_kernel": self.runtime_kernel.capabilities.name,
            "runtime_kernel_continuous_batching": self.runtime_kernel.capabilities.continuous_batching,
            "runtime_kernel_available": self.runtime_kernel.capabilities.available,
            "decode_batch_diagnostics": self.runtime_kernel.decode_diagnostics(),
            "engine_timing": self._engine_timing_status(),
            "scheduler": {
                "mode": "decode_first_chunked_prefill",
                "max_active_requests": self.settings.engine.max_active_requests,
                "max_decode_batch": self.settings.engine.max_decode_batch,
                "prefill_batch_size": self.settings.batch.prefill_batch_size,
                "admission_policy": "fill_available_active_slots",
                "prefill_token_budget": self.settings.engine.prefill_token_budget,
            },
        }

    def _request_status_snapshot(self, state: RequestState, *, now: float) -> dict[str, object]:
        ttft_s = None
        tokens_per_second = None
        if state.first_token_at is not None:
            ttft_s = round(state.first_token_at - state.created_at, 3)
            generation_seconds = now - state.first_token_at
            if generation_seconds > 0 and state.completion_tokens > 0:
                tokens_per_second = round(state.completion_tokens / generation_seconds, 1)

        max_tokens = max(state.request.max_tokens, 0)
        progress = min(round(state.completion_tokens / max_tokens, 3), 1.0) if max_tokens else 0.0
        cache_hit_type = None
        if state.matched_prefix_tokens > 0:
            cache_hit_type = "prefix"
        elif state.admission_prepared and self.settings.engine.prefix_cache_enabled:
            cache_hit_type = "miss"
        next_reuse_point = self._next_checkpoint_reuse_point(state)
        checkpointed_reuse_points = [
            point for point in sorted(state.checkpoints_created) if point in state.reuse_points
        ]

        return {
            "request_id": state.request_id,
            "status": self._public_request_status(state),
            "phase": self._public_request_phase(state),
            "elapsed_s": round(max(now - state.created_at, 0.0), 2),
            "prompt_tokens": state.prompt_token_count,
            "completion_tokens": state.completion_tokens,
            "max_tokens": max_tokens,
            "progress": progress,
            "tokens_per_second": tokens_per_second,
            "ttft_s": ttft_s,
            "cache_hit_type": cache_hit_type,
            "cached_tokens": state.matched_prefix_tokens,
            "cache_token_count": state.cache_token_count,
            "target_cache_token_count": state.target_cache_token_count,
            "prefill_remaining_tokens": max(
                state.target_cache_token_count - state.cache_token_count,
                0,
            ),
            "reuse_points": list(state.reuse_points),
            "next_reuse_point": next_reuse_point,
            "checkpointed_reuse_points": checkpointed_reuse_points,
            "checkpoints_created_count": len(state.checkpoints_created),
            "admission_retries": state.admission_retries,
            "estimated_bytes": state.estimated_bytes,
            "cancel_requested": state.cancel_requested,
            "timeline": self._request_timeline_snapshot(state, now=now),
        }

    def _request_timeline_snapshot(
        self,
        state: RequestState,
        *,
        now: float | None = None,
    ) -> dict[str, object]:
        now = time.monotonic() if now is None else now

        def duration(start: float | None, end: float | None) -> float | None:
            if start is None or end is None:
                return None
            return round(max(end - start, 0.0), 6)

        admission_prepare_s = duration(state.admission_started_at, state.admitted_at)
        prefill_wall_s = duration(state.prefill_started_at, state.prefill_finished_at)
        if prefill_wall_s is None and state.prefill_started_at is not None:
            prefill_wall_s = duration(state.prefill_started_at, now)
        decode_duration_s = duration(state.decode_started_at, state.completed_at)
        if decode_duration_s is None and state.decode_started_at is not None:
            decode_duration_s = duration(state.decode_started_at, now)

        queue_start = state.enqueued_at or state.created_at
        queue_end = state.admission_started_at or state.admitted_at
        return {
            "request_id": state.request_id,
            "status": self._public_request_status(state),
            "phase": self._public_request_phase(state),
            "prompt_tokens": state.prompt_token_count,
            "completion_tokens": state.completion_tokens,
            "max_tokens": state.request.max_tokens,
            "prefill_steps": state.prefill_steps,
            "decode_steps": state.decode_steps,
            "cached_tokens": state.matched_prefix_tokens,
            "cache_token_count": state.cache_token_count,
            "target_cache_token_count": state.target_cache_token_count,
            "created_to_enqueued_s": duration(state.created_at, state.enqueued_at),
            "queue_wait_s": duration(queue_start, queue_end),
            "admission_prepare_s": admission_prepare_s,
            "admission_to_prefill_s": duration(state.admitted_at, state.prefill_started_at),
            "prefill_wall_s": prefill_wall_s,
            "prefill_model_s": round(state.prefill_seconds, 6),
            "prefill_to_decode_init_s": duration(
                state.prefill_finished_at,
                state.decode_init_started_at,
            ),
            "decode_init_s": duration(state.decode_init_started_at, state.decode_ready_at),
            "decode_wait_s": duration(state.decode_ready_at, state.decode_started_at),
            "ttft_s": duration(state.created_at, state.first_token_at),
            "decode_to_first_token_s": duration(state.decode_started_at, state.first_token_at),
            "decode_duration_s": decode_duration_s,
            "total_latency_s": duration(state.created_at, state.completed_at),
            "terminal_to_response_ready_s": duration(state.completed_at, state.response_ready_at),
            "last_decode_step_age_s": duration(state.last_decode_step_at, now),
        }

    @staticmethod
    def _public_request_status(state: RequestState) -> str:
        if state.phase in _WAITING_PHASES:
            return "waiting"
        if state.phase in _RUNNING_PHASES:
            return "running"
        if state.phase is RequestPhase.CANCELLED:
            return "cancelled"
        if state.phase is RequestPhase.FAILED:
            return "failed"
        if state.phase is RequestPhase.COMPLETED:
            return "finished"
        return state.phase.value

    @staticmethod
    def _public_request_phase(state: RequestState) -> str:
        if state.phase is RequestPhase.SUBMITTED:
            return "queued"
        if state.phase is RequestPhase.ADMITTED:
            return "admission"
        if state.phase is RequestPhase.PREFIX_LOOKUP:
            return "prefix_lookup"
        if state.phase in {RequestPhase.PREFILL_WAIT, RequestPhase.PREFILLING}:
            return "prefill"
        if state.phase in {RequestPhase.DECODE_READY, RequestPhase.DECODING}:
            return "generation"
        return state.phase.value

    async def warmup(self) -> None:
        try:
            await self._runner_call(self.runtime_kernel.warmup)
        except Exception:
            self.logger.warning("engine_warmup_failed", exc_info=True)
            return
        await self._load_prefix_cache_from_disk()
        await self._warm_prefix_cache()

    async def aclose(self) -> None:
        await self.stop()
        self._save_prefix_cache_to_disk()
        await self.embedding_backend.aclose()
        self._runner_executor.shutdown(wait=True, cancel_futures=True)

    def supports_embeddings(self) -> bool:
        return self.embedding_backend.supports_embeddings()

    def configured_embedding_model(self) -> str | None:
        return self.embedding_backend.configured_model()

    def get_cache_stats(self) -> dict[str, object]:
        return {
            "prefix_cache": self.prefix_store.stats_snapshot(),
            "runtime_kernel": self.runtime_kernel.capabilities.name,
            "active_estimated_bytes": self._active_estimated_bytes,
        }

    def clear_prefix_cache(self) -> dict[str, object]:
        result = self.prefix_store.clear(include_pinned=False)
        self.metrics.snapshot_bytes.set(self.prefix_store.current_bytes)
        self.metrics.snapshot_entries.set(self.prefix_store.entry_count)
        return result

    async def clear_runtime_caches(self) -> dict[str, object]:
        return {
            "prefix_cache": self.clear_prefix_cache(),
            "runtime": await self._runner_call(self.runtime_kernel.clear_runtime_caches),
        }

    async def embeddings(self, *, model: str | None, input_data: str | list[str]) -> dict[str, object]:
        return await self.embedding_backend.embeddings(model=model, input_data=input_data)

    async def count_text_tokens(self, texts: tuple[str, ...]) -> int:
        return await self._runner_call(self.runtime_kernel.count_text_tokens, texts)

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        state = self._new_state(request, stream=False)
        future = asyncio.get_running_loop().create_future()
        state.response_future = future
        await self._enqueue(state)
        timeout_seconds = self._request_timeout_seconds(request)
        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            await self.cancel(state.request_id)
            raise self._request_timeout_error(timeout_seconds) from exc
        except asyncio.CancelledError:
            await self.cancel(state.request_id)
            raise

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        return await self.submit(request)

    def _request_timeout_seconds(self, request: InferenceRequest) -> float:
        if request.timeout_seconds is not None:
            return max(float(request.timeout_seconds), 1e-3)
        return self.settings.api.request_timeout_seconds

    def _request_timeout_error(self, timeout_seconds: float) -> AsterError:
        self._timed_out_requests += 1
        exc = AsterError(
            code="request_timeout",
            message="Inference request timed out",
            status_code=504,
            details={"timeout_seconds": timeout_seconds},
        )
        self.metrics.errors.labels(code=exc.code).inc()
        return exc

    async def stream(self, request: InferenceRequest) -> AsyncIterator[DecodeChunk]:
        state = self._new_state(request, stream=True)
        await self._enqueue(state)
        collector = state.stream_collector
        assert collector is not None
        timeout_seconds = self._request_timeout_seconds(request)
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    await self.cancel(state.request_id)
                    raise self._request_timeout_error(timeout_seconds)
                item = collector.get_nowait()
                if item is None and not collector.closed:
                    item = await asyncio.wait_for(collector.get(), timeout=remaining)
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        except TimeoutError as exc:
            await self.cancel(state.request_id)
            raise self._request_timeout_error(timeout_seconds) from exc
        finally:
            if state.request_id in self._requests and self._requests[state.request_id].phase not in {
                RequestPhase.COMPLETED,
                RequestPhase.CANCELLED,
                RequestPhase.FAILED,
            }:
                await self.cancel(state.request_id)

    async def cancel(self, request_id: str) -> bool:
        canonical_request_id = self._request_aliases.get(request_id, request_id)
        state = self._requests.get(canonical_request_id)
        if state is None:
            return False
        state.cancel_requested = True
        if state.phase in _RUNNER_IN_FLIGHT_PHASES:
            self._idle_event.set()
            return True
        await self._cancel_request(state)
        self._idle_event.set()
        return True

    def _is_live_request(self, state: RequestState) -> bool:
        return self._requests.get(state.request_id) is state

    def _new_state(self, request: InferenceRequest, *, stream: bool) -> RequestState:
        request_id = request.trace_id or str(uuid.uuid4())
        return RequestState(
            request_id=request_id,
            request=request,
            stream_collector=(
                StreamCollector(
                    stream_interval_tokens=self.settings.engine.stream_interval_tokens,
                )
                if stream
                else None
            ),
            stop_sequences=self._normalize_stop_sequences(
                request.stop,
                request.parser_stop_sequences,
            ),
        )

    async def _enqueue(self, state: RequestState) -> None:
        if self._submission_queue.full():
            self._admission_rejections += 1
            self.metrics.admission_rejections.inc()
            self.metrics.queue_depth.set(self._submission_queue.qsize())
            exc = OverloadedError(
                code="queue_full",
                message="Inference engine queue is full",
                status_code=503,
            )
            self.metrics.errors.labels(code=exc.code).inc()
            raise exc
        duplicate_id = self._active_request_id_collision(state)
        if duplicate_id is not None:
            raise AsterError(
                code="duplicate_request_id",
                message="Inference request id is already active",
                status_code=409,
                details={"request_id": duplicate_id},
            )
        if self._task is None or self._task.done():
            await self.start()
        state.mark_enqueued()
        self._requests[state.request_id] = state
        for alias in state.request.request_aliases:
            if alias and alias != state.request_id:
                self._request_aliases[alias] = state.request_id
        await self._submission_queue.put(state)
        self.metrics.queue_depth.set(self._submission_queue.qsize())
        self._idle_event.set()

    def _active_request_id_collision(self, state: RequestState) -> str | None:
        candidate_ids = [state.request_id, *state.request.request_aliases]
        for candidate_id in candidate_ids:
            if not candidate_id:
                continue
            if candidate_id in self._requests or candidate_id in self._request_aliases:
                return candidate_id
        return None

    async def _engine_loop(self) -> None:
        try:
            while self._running:
                did_work = await self._scheduler_step()
                self.metrics.queue_depth.set(self._submission_queue.qsize())
                self.metrics.active_requests.set(len(self._prefill_queue) + len(self._decode_queue))
                self.metrics.prefill_active.set(len(self._prefill_queue))
                self.metrics.decode_active.set(len(self._decode_queue))
                self.metrics.snapshot_bytes.set(self.prefix_store.current_bytes)
                self.metrics.snapshot_entries.set(self.prefix_store.entry_count)
                if did_work:
                    await asyncio.sleep(0)
                    continue
                self._idle_event.clear()
                await self._idle_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("engine_loop_failed")
            raise

    async def _scheduler_step(self) -> bool:
        did_work = await self._process_cancellations()
        did_work = await self._step_decode() or did_work
        did_work = await self._step_prefill() or did_work
        drained = await self._drain_submissions()
        if drained:
            self._rotate_yielding_prefill_continuations()
        did_work = drained or did_work
        return did_work

    async def _drain_submissions(self) -> bool:
        drained = False
        while (
            not self._submission_queue.empty()
            and (len(self._prefill_queue) + len(self._decode_queue)) < self.settings.engine.max_active_requests
        ):
            state = self._submission_queue.get_nowait()
            result = await self._prepare_request(state)
            drained = True
            if result == _PREPARE_DEFERRED:
                break
        return drained

    async def _prepare_request(self, state: RequestState) -> PrepareResult:
        if not self._is_live_request(state):
            return _PREPARE_TERMINAL
        try:
            state.mark_admission_started()
            if not state.admission_prepared:
                state.phase = RequestPhase.PREFIX_LOOKUP
                prepared = await self._runner_call(self.runtime_kernel.encode_request, state.request)
                if not self._is_live_request(state):
                    return _PREPARE_TERMINAL
                state.prompt_tokens = prepared.prompt_tokens
                state.reuse_points = prepared.reuse_points
                self._validate_context_budget(state)
                state.model_fingerprint = await self._runner_call(self.runtime_kernel.model_fingerprint)
                if not self._is_live_request(state):
                    return _PREPARE_TERMINAL
                state.estimated_bytes = await self._runner_call(
                    self.runtime_kernel.estimate_request_bytes,
                    len(state.prompt_tokens),
                    state.request.max_tokens,
                )
                if not self._is_live_request(state) or state.cancel_requested:
                    return _PREPARE_TERMINAL
                state.admission_prepared = True
            if not self._is_live_request(state) or state.cancel_requested:
                return _PREPARE_TERMINAL
            if not self._ensure_admission_budget(state.estimated_bytes):
                if self._defer_admission(state):
                    return _PREPARE_DEFERRED
                raise self._admission_memory_error(state)

            state.phase = RequestPhase.ADMITTED
            observed_queue_wait = state.admitted_at is not None
            state.mark_admitted()
            if not observed_queue_wait:
                self.metrics.queue_wait_latency.observe(
                    max((state.admitted_at or state.created_at) - state.created_at, 0.0)
                )

            if self.settings.engine.prefix_cache_enabled:
                self._prefix_reuse_attempts += 1
                self.metrics.prefix_reuse_attempts.inc()
                matched = self.prefix_store.lookup(
                    self.settings.model.name,
                    state.prompt_tokens,
                    model_fingerprint=state.model_fingerprint,
                )
            else:
                matched = None
            if matched is not None:
                state.attached_snapshot_key = matched.key
                self.prefix_store.pin(matched.key)
                state.prompt_cache = await self._runner_call(
                    self.runtime_kernel.clone_cache,
                    matched.prompt_cache,
                    matched.cache_token_count,
                )
                if not self._is_live_request(state):
                    return _PREPARE_TERMINAL
                state.matched_prefix_tokens = matched.prefix_token_count
                state.cache_token_count = matched.cache_token_count
                self._prefix_reuse_hits += 1
                self._prefix_tokens_reused += matched.cache_token_count
                self.metrics.prefix_cache_hits.inc()
                self.metrics.prefix_tokens_reused.inc(matched.cache_token_count)
            else:
                self.metrics.prefix_cache_misses.inc()

            self._active_estimated_bytes += state.estimated_bytes
            state.admission_retries = 0
            state.phase = RequestPhase.PREFILL_WAIT
            self._prefill_queue.append(state.request_id)
            return _PREPARE_ADMITTED
        except Exception as exc:
            await self._fail_request(state, exc)
            return _PREPARE_TERMINAL

    def _defer_admission(self, state: RequestState) -> bool:
        has_active_requests = self._active_estimated_bytes > 0
        if not has_active_requests:
            if state.admission_retries >= self.settings.engine.admission_retry_limit:
                return False
            state.admission_retries += 1
        state.phase = RequestPhase.SUBMITTED
        self._submission_queue.put_nowait(state)
        self.metrics.queue_depth.set(self._submission_queue.qsize())
        self._idle_event.set()
        return True

    def _admission_memory_error(self, state: RequestState) -> OverloadedError:
        self._admission_rejections += 1
        self.metrics.admission_rejections.inc()
        return OverloadedError(
            code="memory_pressure",
            message="Request rejected due to memory pressure",
            status_code=503,
            details={
                "admission_retries": state.admission_retries,
                "estimated_bytes": state.estimated_bytes,
            },
        )

    async def _step_prefill(self) -> bool:
        if not self._prefill_queue:
            return False
        request_id = self._prefill_queue.popleft()
        self._prefill_yield_request_ids.discard(request_id)
        state = self._requests.get(request_id)
        if state is None:
            return True
        if state.cancel_requested:
            await self._cancel_request(state)
            return True

        state.phase = RequestPhase.PREFILLING
        remaining = state.target_cache_token_count - state.cache_token_count
        if remaining <= 0:
            state.mark_prefill_finished()
            await self._activate_decode(state)
            return True

        state.mark_prefill_started()
        target = self._prefill_chunk_target(state, remaining=remaining)
        next_reuse_point = self._next_checkpoint_reuse_point(
            state,
            target_cache_token_count=target,
        )
        if next_reuse_point is not None:
            target = next_reuse_point - 1

        try:
            result = await self._runner_call(
                self.runtime_kernel.prefill_to,
                prompt_tokens=state.prompt_tokens,
                prompt_cache=state.prompt_cache,
                cache_token_count=state.cache_token_count,
                target_cache_token_count=target,
            )
            if not self._is_live_request(state):
                await self._store_cancelled_prefill_checkpoint(state, result)
                return True
            if state.cancel_requested:
                await self._store_cancelled_prefill_checkpoint(state, result)
                await self._cancel_request(state)
                return True
            processed_tokens = max(result.cache_token_count - state.cache_token_count, 0)
            state.prompt_cache = result.prompt_cache
            state.cache_token_count = result.cache_token_count
            state.prefill_seconds += result.elapsed_seconds
            state.prefill_steps += 1
            self._prefill_model_tokens += processed_tokens
            self._prefill_model_seconds += result.elapsed_seconds
            self._max_prefill_step_seconds = max(
                self._max_prefill_step_seconds,
                result.elapsed_seconds,
            )
            self._prefill_steps += 1
            self.metrics.prefill_steps.inc()
            self.metrics.prefill_latency.observe(result.elapsed_seconds)
            await self._maybe_checkpoint(state)
            if state.cache_token_count >= state.target_cache_token_count:
                state.mark_prefill_finished()
                await self._activate_decode(state)
            else:
                state.phase = RequestPhase.PREFILL_WAIT
                self._prefill_queue.append(state.request_id)
                self._prefill_yield_request_ids.add(state.request_id)
        except MemoryError as exc:
            await self._recover_from_memory_pressure(state, exc)
        except Exception as exc:
            await self._fail_request(state, exc)
        return True

    def _rotate_yielding_prefill_continuations(self) -> None:
        if not self._prefill_yield_request_ids or len(self._prefill_queue) < 2:
            return
        ready_now: list[str] = []
        yielding: list[str] = []
        for request_id in self._prefill_queue:
            if request_id in self._prefill_yield_request_ids:
                yielding.append(request_id)
            else:
                ready_now.append(request_id)
        if not yielding:
            self._prefill_yield_request_ids.intersection_update(self._prefill_queue)
            return
        self._prefill_queue = deque([*ready_now, *yielding])
        self._prefill_yield_rotations += len(yielding)
        self._prefill_yield_request_ids.difference_update(yielding)

    async def _step_decode(self) -> bool:
        if not self._decode_queue:
            return False
        batch: list[RequestState] = []
        max_batch_size = min(self.settings.engine.max_decode_batch, len(self._decode_queue))
        while self._decode_queue and len(batch) < max_batch_size:
            request_id = self._decode_queue.popleft()
            state = self._requests.get(request_id)
            if state is None:
                continue
            if state.cancel_requested:
                await self._cancel_request(state)
                continue
            batch.append(state)
        if not batch:
            return True

        self.metrics.decode_batch.observe(len(batch))
        for state in batch:
            state.phase = RequestPhase.DECODING
            state.mark_decode_started()

        try:
            decode_started = time.perf_counter()
            results = await self._runner_call(
                self.runtime_kernel.decode_batch_step,
                [self._decode_work_item(state) for state in batch],
            )
            decode_elapsed = time.perf_counter() - decode_started
            self._decode_runner_seconds += decode_elapsed
            self._decode_runner_batches += 1
            self._decode_runner_items += len(batch)
            self._max_decode_batch_seconds = max(
                self._max_decode_batch_seconds,
                decode_elapsed,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Decode batch result mismatch: expected {len(batch)}, received {len(results)}"
                )
        except MemoryError as exc:
            await self._recover_decode_batch(batch, exc)
            return True
        except Exception as exc:
            await self._clear_runtime_caches_for_recovery("decode_error")
            for state in batch:
                await self._fail_request(state, exc)
            return True

        self._decode_steps += 1
        self.metrics.decode_steps.inc()
        self._decode_runner_tokens += sum(
            1 for result in results if isinstance(result, DecodeResult) and result.token_id is not None
        )
        for state, result in zip(batch, results, strict=False):
            try:
                if not self._is_live_request(state):
                    continue
                if state.cancel_requested:
                    await self._cancel_request(state)
                    continue
                if isinstance(result, MemoryError):
                    await self._recover_from_memory_pressure(state, result)
                    continue
                if isinstance(result, BaseException):
                    await self._fail_request(state, result)
                    continue
                state.mark_decode_step()
                await self._handle_decode_step(state, result)
                if result.finish_reason is not None:
                    await self._complete_request(state, finish_reason=result.finish_reason or "stop")
                    continue
                state.phase = RequestPhase.DECODE_READY
                self._decode_queue.append(state.request_id)
            except MemoryError as exc:
                await self._recover_from_memory_pressure(state, exc)
            except Exception as exc:
                await self._fail_request(state, exc)
        return True

    def _decode_work_item(self, state: RequestState) -> DecodeWorkItem:
        if state.next_input_token is None or state.decode_sampler is None or state.decode_detokenizer is None:
            raise RuntimeError(f"Request {state.request_id} is not decode-initialized")
        logits_processor_tokens = state.prompt_tokens + state.output_token_ids
        if logits_processor_tokens and logits_processor_tokens[-1] == state.next_input_token:
            logits_processor_tokens = logits_processor_tokens[:-1]
        return DecodeWorkItem(
            prompt_cache=state.prompt_cache,
            input_token=state.next_input_token,
            sampler=state.decode_sampler,
            detokenizer=state.decode_detokenizer,
            stop_token_ids=state.decode_stop_token_ids,
            logits_processors=state.decode_logits_processors,
            logits_processor_tokens=logits_processor_tokens,
            completion_tokens=state.completion_tokens,
            max_tokens=state.request.max_tokens,
        )

    def _validate_context_budget(self, state: RequestState) -> None:
        context_length = self.settings.model.context_length
        requested_tokens = len(state.prompt_tokens) + max(state.request.max_tokens, 0)
        if requested_tokens <= context_length:
            return
        raise AsterError(
            code="context_length_exceeded",
            message="Request exceeds the configured model context length",
            status_code=400,
            details={
                "context_length": context_length,
                "prompt_tokens": len(state.prompt_tokens),
                "max_tokens": state.request.max_tokens,
                "requested_tokens": requested_tokens,
            },
        )

    async def _handle_decode_step(self, state: RequestState, step: DecodeResult) -> None:
        state.prompt_cache = step.prompt_cache
        if step.token_id is not None:
            state.next_input_token = step.token_id
        if step.completion_tokens > state.completion_tokens:
            state.mark_first_token()
        if step.text:
            stop_matched = await self._append_decoded_text(
                state,
                step.text,
                index=max(step.completion_tokens - 1, 0),
            )
            if stop_matched:
                step.finish_reason = "stop"
        state.completion_tokens = step.completion_tokens
        if step.token_id is not None:
            state.output_token_ids.append(step.token_id)
        if state.decode_started_at is not None and state.completion_tokens > 0:
            elapsed = max(time.monotonic() - state.decode_started_at, 1e-6)
            state.generation_tps = state.completion_tokens / elapsed
        state.peak_memory_gb = max(state.peak_memory_gb, step.peak_memory_gb)
        if (
            state.completion_tokens == 1
            and state.first_token_at is not None
            and state.decode_started_at is not None
        ):
            self.metrics.first_token_latency.observe(state.first_token_at - state.decode_started_at)
        if step.token_id is not None:
            self.metrics.decode_tokens.inc()

    async def _append_decoded_text(self, state: RequestState, text: str, *, index: int) -> bool:
        if not text:
            return False
        if not state.stop_sequences:
            await self._publish_text(state, text, index=index)
            return False

        candidate = state.pending_stop_text + text
        stop_index = self._first_stop_index(candidate, state.stop_sequences)
        if stop_index is not None:
            await self._publish_text(state, candidate[:stop_index], index=index)
            state.pending_stop_text = ""
            return True

        hold_length = self._stop_prefix_suffix_length(candidate, state.stop_sequences)
        visible_text = candidate[: len(candidate) - hold_length] if hold_length else candidate
        state.pending_stop_text = candidate[len(candidate) - hold_length :] if hold_length else ""
        await self._publish_text(state, visible_text, index=index)
        return False

    async def _flush_pending_stop_text(self, state: RequestState, *, index: int) -> None:
        if not state.pending_stop_text:
            return
        pending = state.pending_stop_text
        state.pending_stop_text = ""
        await self._publish_text(state, pending, index=index)

    async def _publish_text(self, state: RequestState, text: str, *, index: int) -> None:
        if not text:
            return
        state.output_parts.append(text)
        if state.stream_collector is not None:
            await state.stream_collector.add_text(text, index=index)

    @staticmethod
    def _normalize_stop_sequences(
        stop: str | list[str] | None,
        extra_stop_sequences: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        sequences: list[str] = []
        if isinstance(stop, str):
            if stop:
                sequences.append(stop)
        elif isinstance(stop, list):
            sequences.extend(item for item in stop if item)
        sequences.extend(item for item in extra_stop_sequences if item)
        return tuple(dict.fromkeys(sequences))

    @staticmethod
    def _first_stop_index(text: str, stops: tuple[str, ...]) -> int | None:
        indexes = [index for stop in stops if (index := text.find(stop)) >= 0]
        return min(indexes) if indexes else None

    @staticmethod
    def _stop_prefix_suffix_length(text: str, stops: tuple[str, ...]) -> int:
        max_length = 0
        for stop in stops:
            for length in range(1, min(len(stop), len(text) + 1)):
                if text.endswith(stop[:length]):
                    max_length = max(max_length, length)
        return max_length

    async def _activate_decode(self, state: RequestState) -> None:
        full_cache_tokens = state.target_cache_token_count
        await self._store_checkpoint(state, logical_prefix_tokens=len(state.prompt_tokens), cache_token_count=full_cache_tokens)
        if not self._is_live_request(state):
            return
        if state.request.max_tokens <= 0:
            await self._complete_request(state, finish_reason="length")
            return
        state.mark_decode_init_started()
        decode_init = await self._runner_call(
            self.runtime_kernel.initialize_decode,
            prompt_tokens=state.prompt_tokens,
            cache_token_count=state.cache_token_count,
            prompt_cache=state.prompt_cache,
            request=state.request,
        )
        if not self._is_live_request(state) or state.cancel_requested:
            return
        state.prompt_cache = decode_init.prompt_cache
        state.decode_sampler = decode_init.sampler
        state.decode_detokenizer = decode_init.detokenizer
        state.decode_stop_token_ids = decode_init.stop_token_ids
        state.decode_logits_processors = decode_init.logits_processors
        state.next_input_token = decode_init.next_input_token
        state.mark_decode_ready()
        state.phase = RequestPhase.DECODE_READY
        self._decode_queue.append(state.request_id)

    async def _maybe_checkpoint(self, state: RequestState) -> None:
        logical_prefix_tokens = state.cache_token_count + 1
        if logical_prefix_tokens < self.settings.engine.snapshot_min_prefix_tokens:
            return

        checkpoints: set[int] = set(state.reuse_points)
        if logical_prefix_tokens == len(state.prompt_tokens):
            checkpoints.add(logical_prefix_tokens)
        prefill_budget = max(self.settings.engine.prefill_token_budget, 1)
        chunk_checkpoint_max_tokens = self.settings.engine.snapshot_chunk_checkpoint_max_tokens
        if (
            state.cache_token_count > 0
            and state.cache_token_count % prefill_budget == 0
            and logical_prefix_tokens <= chunk_checkpoint_max_tokens
        ):
            checkpoints.add(logical_prefix_tokens)

        for prefix_tokens in sorted(checkpoints):
            if prefix_tokens in state.checkpoints_created:
                continue
            if prefix_tokens - 1 != state.cache_token_count:
                continue
            await self._store_checkpoint(
                state,
                logical_prefix_tokens=prefix_tokens,
                cache_token_count=prefix_tokens - 1,
            )

    async def _store_checkpoint(
        self,
        state: RequestState,
        *,
        logical_prefix_tokens: int,
        cache_token_count: int,
    ) -> None:
        if not self.settings.engine.prefix_cache_enabled:
            return
        if not self._is_live_request(state):
            return
        if logical_prefix_tokens in state.checkpoints_created:
            return
        if state.prompt_cache is None:
            return
        snapshot_cache = await self._runner_call(
            self.runtime_kernel.clone_cache,
            state.prompt_cache,
            cache_token_count,
        )
        if not self._is_live_request(state):
            return
        approx_bytes = await self._runner_call(self.runtime_kernel.estimate_cache_bytes, snapshot_cache)
        if not self._is_live_request(state):
            return
        entry = self.prefix_store.store(
            model_name=self.settings.model.name,
            model_fingerprint=state.model_fingerprint,
            prefix_tokens=state.prompt_tokens[:logical_prefix_tokens],
            cache_token_count=cache_token_count,
            prompt_cache=snapshot_cache,
            approx_bytes=approx_bytes,
        )
        if entry is not None:
            state.checkpoints_created.add(logical_prefix_tokens)

    async def _store_cancelled_prefill_checkpoint(
        self,
        state: RequestState,
        result: PrefillChunkResult,
    ) -> None:
        if not self.settings.engine.prefix_cache_enabled:
            return
        if result.prompt_cache is None:
            return
        logical_prefix_tokens = min(result.cache_token_count + 1, len(state.prompt_tokens))
        if logical_prefix_tokens < self.settings.engine.snapshot_min_prefix_tokens:
            return
        if logical_prefix_tokens in state.checkpoints_created:
            return
        try:
            snapshot_cache = await self._runner_call(
                self.runtime_kernel.clone_cache,
                result.prompt_cache,
                result.cache_token_count,
            )
            approx_bytes = await self._runner_call(
                self.runtime_kernel.estimate_cache_bytes,
                snapshot_cache,
            )
            entry = self.prefix_store.store(
                model_name=self.settings.model.name,
                model_fingerprint=state.model_fingerprint,
                prefix_tokens=state.prompt_tokens[:logical_prefix_tokens],
                cache_token_count=result.cache_token_count,
                prompt_cache=snapshot_cache,
                approx_bytes=approx_bytes,
            )
        except Exception:
            self.logger.warning(
                "cancelled_prefill_checkpoint_failed",
                exc_info=True,
                extra={"request_id": state.request_id},
            )
            return
        if entry is not None:
            state.checkpoints_created.add(logical_prefix_tokens)
            self._cancelled_prefill_checkpoints += 1
            self.logger.info(
                "cancelled_prefill_checkpoint_stored",
                extra={
                    "request_id": state.request_id,
                    "prefix_tokens": logical_prefix_tokens,
                    "cache_token_count": result.cache_token_count,
                },
            )

    async def _complete_request(self, state: RequestState, *, finish_reason: str = "stop") -> None:
        tail_text = await self._runner_call(self.runtime_kernel.finalize_detokenizer, state.decode_detokenizer)
        if tail_text:
            await self._append_decoded_text(
                state,
                tail_text,
                index=max(state.completion_tokens - 1, 0),
            )
        await self._flush_pending_stop_text(
            state,
            index=max(state.completion_tokens - 1, 0),
        )

        prompt_tps = (state.prompt_token_count / state.prefill_seconds) if state.prefill_seconds > 0 else 0.0
        response = InferenceResponse(
            request_id=state.request_id,
            text="".join(state.output_parts),
            prompt_tokens=state.prompt_token_count,
            completion_tokens=state.completion_tokens,
            cache_hit=state.matched_prefix_tokens > 0,
            prefill_cache_hit=state.matched_prefix_tokens > 0,
            generation_cache_reuse=state.matched_prefix_tokens > 0,
            speculative_enabled=False,
            speculative_path_mode="disabled",
            prompt_tps=prompt_tps,
            generation_tps=state.generation_tps,
            peak_memory_gb=state.peak_memory_gb,
            finish_reason=finish_reason,
        )
        self.metrics.request_latency.observe(max(time.monotonic() - state.created_at, 0.0))
        if state.decode_started_at is not None:
            self.metrics.decode_latency.observe(max(time.monotonic() - state.decode_started_at, 0.0))
        state.mark_terminal(RequestPhase.COMPLETED)
        self._credit_terminal_request(state)
        self._completed_requests += 1
        if state.stream_collector is not None:
            await state.stream_collector.finish(
                index=state.completion_tokens,
                stats={
                    "request_id": response.request_id,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "cache_hit": response.cache_hit,
                    "prefill_cache_hit": response.prefill_cache_hit,
                    "generation_cache_reuse": response.generation_cache_reuse,
                    "speculative_enabled": response.speculative_enabled,
                    "speculative_path_mode": response.speculative_path_mode,
                    "prompt_tps": response.prompt_tps,
                    "generation_tps": response.generation_tps,
                    "peak_memory_gb": response.peak_memory_gb,
                    "finish_reason": response.finish_reason,
                },
            )
        if state.response_future is not None and not state.response_future.done():
            state.mark_response_ready()
            state.response_future.set_result(response)
        elif state.stream_collector is not None:
            state.mark_response_ready()
        self._record_recent_timeline(state)
        self._cleanup_request(state)

    async def _cancel_request(self, state: RequestState) -> None:
        state.mark_terminal(RequestPhase.CANCELLED)
        self._credit_terminal_request(state)
        self._cancelled_requests += 1
        if state.stream_collector is not None:
            await state.stream_collector.close()
        if state.response_future is not None and not state.response_future.done():
            state.mark_response_ready()
            state.response_future.set_exception(
                AsterError(code="request_cancelled", message="Request cancelled", status_code=499)
            )
        elif state.stream_collector is not None:
            state.mark_response_ready()
        self.metrics.cancellations.inc()
        self._record_recent_timeline(state)
        self._cleanup_request(state)

    async def _fail_request(self, state: RequestState, exc: Exception) -> None:
        state.mark_terminal(RequestPhase.FAILED)
        self._credit_terminal_request(state)
        self._failed_requests += 1
        if state.stream_collector is not None:
            await state.stream_collector.fail(exc)
        if state.response_future is not None and not state.response_future.done():
            state.mark_response_ready()
            state.response_future.set_exception(exc)
        elif state.stream_collector is not None:
            state.mark_response_ready()
        self.metrics.errors.labels(code=getattr(exc, "code", exc.__class__.__name__)).inc()
        self._record_recent_timeline(state)
        self._cleanup_request(state)

    async def _process_cancellations(self) -> bool:
        cancelled = False
        for state in list(self._requests.values()):
            if state.cancel_requested and state.phase not in {
                RequestPhase.COMPLETED,
                RequestPhase.CANCELLED,
                RequestPhase.FAILED,
            }:
                await self._cancel_request(state)
                cancelled = True
        return cancelled

    def _prefill_chunk_target(self, state: RequestState, *, remaining: int) -> int:
        budget = self._prefill_budget()
        if self._can_finish_prefill_in_idle_step(remaining=remaining, budget=budget):
            budget = remaining
        return min(state.target_cache_token_count, state.cache_token_count + max(budget, 1))

    def _next_checkpoint_reuse_point(
        self,
        state: RequestState,
        *,
        target_cache_token_count: int | None = None,
    ) -> int | None:
        min_checkpoint_tokens = self.settings.engine.snapshot_min_prefix_tokens
        return min(
            (
                point
                for point in state.reuse_points
                if (
                    point >= min_checkpoint_tokens
                    and point not in state.checkpoints_created
                    and state.cache_token_count < point - 1
                    and (
                        target_cache_token_count is None
                        or point - 1 <= target_cache_token_count
                    )
                )
            ),
            default=None,
        )

    def _can_finish_prefill_in_idle_step(self, *, remaining: int, budget: int) -> bool:
        if remaining > self.settings.engine.idle_prefill_token_limit:
            return False
        if budget < self.settings.engine.prefill_token_budget:
            return False
        return not self._decode_queue and not self._prefill_queue and self._submission_queue.empty()

    async def _recover_from_memory_pressure(self, state: RequestState, exc: Exception) -> None:
        self.prefix_store.evict_until_below(max(self.prefix_store.current_bytes // 2, 0))
        await self._clear_runtime_caches_for_recovery("memory_pressure")
        state.cancel_requested = True
        await self._fail_request(
            state,
            OverloadedError(
                code="memory_pressure",
                message="Request failed under memory pressure",
                status_code=503,
                details={"error": str(exc)},
            ),
        )

    async def _recover_decode_batch(self, batch: list[RequestState], exc: Exception) -> None:
        self.prefix_store.evict_until_below(max(self.prefix_store.current_bytes // 2, 0))
        for state in reversed(batch[:-1]):
            self._decode_queue.appendleft(state.request_id)
        if batch:
            await self._recover_from_memory_pressure(batch[-1], exc)

    async def _clear_runtime_caches_for_recovery(self, reason: str) -> None:
        self._runtime_cache_clear_attempts += 1
        try:
            result = await self._runner_call(self.runtime_kernel.clear_runtime_caches)
        except Exception:
            self._runtime_cache_clear_failures += 1
            self.logger.warning(
                "runtime_cache_recovery_clear_failed",
                exc_info=True,
                extra={"reason": reason},
            )
            return
        self.logger.info(
            "runtime_cache_recovery_cleared",
            extra={"reason": reason, "result": result},
        )

    def _record_recent_timeline(self, state: RequestState) -> None:
        self._recent_request_timelines.append(self._request_timeline_snapshot(state))

    def _credit_terminal_request(self, state: RequestState) -> None:
        if state.terminal_accounted:
            return
        state.terminal_accounted = True
        self._total_requests_processed += 1
        self._total_prompt_tokens += state.prompt_token_count
        self._total_completion_tokens += state.completion_tokens

    def _cleanup_request(self, state: RequestState) -> None:
        self.prefix_store.unpin(state.attached_snapshot_key)
        self._active_estimated_bytes = max(self._active_estimated_bytes - state.estimated_bytes, 0)
        self._requests.pop(state.request_id, None)
        self._prefill_yield_request_ids.discard(state.request_id)
        for alias, canonical_request_id in list(self._request_aliases.items()):
            if canonical_request_id == state.request_id:
                self._request_aliases.pop(alias, None)
        self._prefill_queue = deque(rid for rid in self._prefill_queue if rid != state.request_id)
        self._decode_queue = deque(rid for rid in self._decode_queue if rid != state.request_id)

    def _ensure_admission_budget(self, estimated_bytes: int) -> bool:
        available = self.runtime_kernel.available_memory_bytes()
        budget = int(available * max(1.0 - self.settings.engine.memory_headroom_ratio, 0.1))
        projected = self._active_estimated_bytes + self.prefix_store.current_bytes + estimated_bytes
        if projected <= budget:
            return True
        self.prefix_store.evict_until_below(max(budget - self._active_estimated_bytes - estimated_bytes, 0))
        projected = self._active_estimated_bytes + self.prefix_store.current_bytes + estimated_bytes
        if projected > budget:
            return False
        return True

    def _prefill_budget(self) -> int:
        available = self.runtime_kernel.available_memory_bytes()
        budget = int(available * max(1.0 - self.settings.engine.memory_headroom_ratio, 0.1))
        in_use = self._active_estimated_bytes + self.prefix_store.current_bytes
        if in_use >= budget:
            return self.settings.engine.pressure_prefill_token_budget
        return self.settings.engine.prefill_token_budget

    def _engine_timing_status(self) -> dict[str, object]:
        prefill_tps = (
            self._prefill_model_tokens / self._prefill_model_seconds
            if self._prefill_model_seconds > 0
            else 0.0
        )
        decode_tps = (
            self._decode_runner_tokens / self._decode_runner_seconds
            if self._decode_runner_seconds > 0
            else 0.0
        )
        avg_decode_batch_size = (
            self._decode_runner_items / self._decode_runner_batches
            if self._decode_runner_batches > 0
            else 0.0
        )
        avg_decode_batch_seconds = (
            self._decode_runner_seconds / self._decode_runner_batches
            if self._decode_runner_batches > 0
            else 0.0
        )
        return {
            "prefill_model_seconds": round(self._prefill_model_seconds, 6),
            "prefill_model_tokens": self._prefill_model_tokens,
            "prompt_tps": round(prefill_tps, 3),
            "max_prefill_step_seconds": round(self._max_prefill_step_seconds, 6),
            "decode_runner_seconds": round(self._decode_runner_seconds, 6),
            "decode_runner_batches": self._decode_runner_batches,
            "decode_runner_items": self._decode_runner_items,
            "decode_runner_tokens": self._decode_runner_tokens,
            "generation_tps": round(decode_tps, 3),
            "avg_decode_batch_size": round(avg_decode_batch_size, 3),
            "avg_decode_batch_seconds": round(avg_decode_batch_seconds, 6),
            "max_decode_batch_seconds": round(self._max_decode_batch_seconds, 6),
        }

    async def _runner_call(
        self,
        fn: Callable[..., RunnerResult],
        *args: object,
        **kwargs: object,
    ) -> RunnerResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._runner_executor,
            partial(self._runner_entrypoint, fn, *args, **kwargs),
        )

    @staticmethod
    def _runner_entrypoint(
        fn: Callable[..., RunnerResult],
        *args: object,
        **kwargs: object,
    ) -> RunnerResult:
        bind_generation_streams()
        return fn(*args, **kwargs)

    async def _load_prefix_cache_from_disk(self) -> None:
        path = self.settings.engine.prefix_cache_persist_path
        if (
            self._prefix_cache_loaded_from_disk
            or not path
            or not self.settings.engine.prefix_cache_enabled
            or not self.settings.engine.prefix_cache_load_on_warmup
        ):
            return
        try:
            fingerprint = await self._runner_call(self.runtime_kernel.model_fingerprint)
            loaded = self.prefix_store.load_from_disk(
                path,
                model_name=self.settings.model.name,
                model_fingerprint=fingerprint,
            )
            self._prefix_cache_loaded_from_disk = True
            if loaded:
                self.logger.info(
                    "prefix_cache_loaded",
                    extra={"path": path, "entries": loaded},
                )
        except Exception:
            self.logger.warning("prefix_cache_load_failed", exc_info=True, extra={"path": path})

    def _save_prefix_cache_to_disk(self) -> None:
        path = self.settings.engine.prefix_cache_persist_path
        if (
            not path
            or not self.settings.engine.prefix_cache_enabled
            or not self.settings.engine.prefix_cache_save_on_shutdown
        ):
            return
        try:
            saved = self.prefix_store.save_to_disk(path)
            if saved:
                self.logger.info(
                    "prefix_cache_saved",
                    extra={"path": path, "entries": saved},
                )
        except Exception:
            self.logger.warning("prefix_cache_save_failed", exc_info=True, extra={"path": path})

    async def _warm_prefix_cache(self) -> None:
        path = self.settings.engine.warm_prompts_path
        if self._warm_prompts_completed or not path or not self.settings.engine.prefix_cache_enabled:
            return
        try:
            prompts = load_warmup_file(path)
        except Exception:
            self.logger.warning("warm_prompts_load_failed", exc_info=True, extra={"path": path})
            return

        semaphore = asyncio.Semaphore(max(self.settings.engine.warm_prompts_concurrency, 1))
        results = await asyncio.gather(
            *(self._warm_one_prompt(messages, semaphore=semaphore) for messages in prompts),
            return_exceptions=True,
        )
        warmed = sum(1 for result in results if result is True)
        failed = sum(1 for result in results if result is not True)
        self._warm_prompts_completed = True
        self.logger.info(
            "warm_prompts_completed",
            extra={"path": path, "warmed": warmed, "failed": failed},
        )

    async def _warm_one_prompt(
        self,
        messages: list[dict[str, object]],
        *,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        async with semaphore:
            normalized = [
                {
                    "role": str(message.get("role") or "user"),
                    "content": str(message.get("content") or ""),
                }
                for message in messages
            ]
            try:
                strict_prefix = await self._runner_call(
                    self.runtime_kernel.strict_chat_prefix_prompt,
                    normalized,
                    enable_thinking=self.settings.model.enable_thinking,
                )
                if strict_prefix:
                    await self.submit(
                        InferenceRequest(
                            prompt=strict_prefix,
                            max_tokens=self.settings.engine.warm_prompts_max_tokens,
                            temperature=0.0,
                        )
                    )
                else:
                    await self.submit(
                        InferenceRequest(
                            messages=ensure_user_terminator(normalized),
                            max_tokens=self.settings.engine.warm_prompts_max_tokens,
                            temperature=0.0,
                            enable_thinking=self.settings.model.enable_thinking,
                        )
                    )
                return True
            except Exception:
                self.logger.warning("warm_prompt_failed", exc_info=True)
                return False


__all__ = ["InferenceEngine", "InferenceRequest", "InferenceResponse"]
