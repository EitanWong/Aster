from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from aster.core.config import RuntimeSettings
from aster.core.errors import AsterError, ConfigurationError
from aster.inference.constrained import (
    ThinkingAwareJsonLogitsProcessor,
    build_json_logits_processor,
)
from aster.inference.contracts import InferenceRequest, InferenceResponse
from aster.inference.decode_engine import DecodeChunk
from aster.inference.embedding_backends import build_embedding_backend
from aster.inference.model_runner import ModelRunner
from aster.inference.prefix_store import PrefixStore, SnapshotEntry
from aster.inference.thinking_processor import ThinkingAwareLogitsProcessor
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry

_BatchProfile = tuple[int, bool, int]


@dataclass(slots=True)
class _BatchLane:
    profile: _BatchProfile
    generator: Any
    request_ids: set[str] = field(default_factory=set)
    uid_to_rid: dict[int, str] = field(default_factory=dict)
    rid_to_uid: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    admission_window_ms: float = 0.0
    target_request_count: int = 1
    sealed: bool = False


@dataclass(slots=True)
class _RequestState:
    request_id: str
    request: InferenceRequest
    prompt_tokens: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    generated_tokens: int = 0
    created_at: float = field(default_factory=time.monotonic)
    enqueued_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: str = "stop"
    cancel_requested: bool = False
    matched_prefix_tokens: int = 0
    prefix_cache_key: str | None = None
    effective_stop_token_ids: frozenset[int] = frozenset()
    batch_profile: _BatchProfile | None = None
    # Streaming support
    stream_event: asyncio.Event | None = None
    stream_chunks: deque[DecodeChunk] = field(default_factory=deque)
    stream_error: BaseException | None = None
    # Completion future for non-streaming
    response_future: asyncio.Future[InferenceResponse] | None = None

    @property
    def max_tokens(self) -> int:
        return self.request.max_tokens

    @property
    def stop_token_ids(self) -> frozenset[int]:
        if self.effective_stop_token_ids:
            return self.effective_stop_token_ids
        return frozenset(self.request.stop_token_ids)

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_tokens)


class BatchedEngine:
    """High-performance continuous batching engine wrapping mlx_lm.BatchGenerator."""

    def __init__(self, settings: RuntimeSettings, metrics: MetricsRegistry) -> None:
        if (
            settings.engine.batch_generator_max_lanes > 1
            and settings.engine.batch_generator_lane_admission_window_ms <= 0
        ):
            raise ConfigurationError(
                code="unsafe_batch_generator_lane_config",
                message=(
                    "batch_generator_lane_admission_window_ms must be positive "
                    "when batch_generator_max_lanes is greater than one"
                ),
                status_code=400,
            )
        self.settings = settings
        self.metrics = metrics
        self.logger = get_logger(__name__)
        self.model_runner = ModelRunner(settings)
        self.prefix_store = PrefixStore(
            budget_bytes=settings.engine.snapshot_budget_bytes,
            max_entries=settings.engine.snapshot_max_entries,
            min_prefix_tokens=settings.engine.snapshot_min_prefix_tokens,
            enabled=settings.engine.prefix_cache_enabled,
        )
        self.embedding_backend = build_embedding_backend(settings)

        # Internal state
        self._state: dict[str, _RequestState] = {}
        self._waiting: deque[str] = deque()
        self._running: set[str] = set()
        self._finished: dict[str, InferenceResponse] = {}
        self._pending_aborts: set[str] = set()
        self._stream_events: dict[str, asyncio.Event] = {}

        # BatchGenerator (created during start)
        self._batch_lanes: dict[_BatchProfile, _BatchLane] = {}
        self._batch_generator_max_lanes = settings.engine.batch_generator_max_lanes
        self._batch_generator_lane_admission_window_ms = (
            settings.engine.batch_generator_lane_admission_window_ms
        )
        self._batch_generator_longest_lane_step_quanta = (
            settings.engine.batch_generator_longest_lane_step_quanta
        )
        self._mx: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._make_sampler: Any | None = None
        self._make_logits_processors: Any | None = None

        # Paged KV cache (created after model load)
        self._paged_cache: Any | None = None

        # Engine loop
        self._engine_loop_task: asyncio.Task[None] | None = None
        self._running_flag: bool = False
        self._started_at: float | None = None

        # Counters
        self._completed_requests: int = 0
        self._failed_requests: int = 0
        self._cancelled_requests: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        # Prefix cache tracking
        self._model_fingerprint: str | None = None
        self._prefix_pin_count: int = 0

    # ------------------------------------------------------------------
    # Public API (mirrors InferenceEngine)
    # ------------------------------------------------------------------

    def health(self) -> bool:
        return self._engine_loop_task is not None and not self._engine_loop_task.done()

    def status(self) -> dict[str, object]:
        return {
            "engine_running": self.health(),
            "engine_type": "batched",
            "pending_requests": len(self._waiting),
            "running_requests": len(self._running),
            "active_requests": len(self._running),
            "completed_requests": self._completed_requests,
            "failed_requests": self._failed_requests,
            "cancelled_requests": self._cancelled_requests,
            "snapshot_entries": self.prefix_store.entry_count,
            "snapshot_bytes": self.prefix_store.current_bytes,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }

    async def start(self) -> None:
        if self._engine_loop_task is not None and not self._engine_loop_task.done():
            return
        self._running_flag = True
        self._started_at = time.monotonic()
        self._engine_loop_task = asyncio.create_task(
            self._engine_loop(), name="aster-batched-engine"
        )

    async def stop(self) -> None:
        self._running_flag = False
        if self._engine_loop_task is not None:
            self._engine_loop_task.cancel()
            try:
                await self._engine_loop_task
            except asyncio.CancelledError:
                pass
            self._engine_loop_task = None

    async def aclose(self) -> None:
        await self.stop()
        for lane in self._batch_lanes.values():
            try:
                lane.generator.close()
            except Exception:
                pass
        self._batch_lanes.clear()
        await self.embedding_backend.aclose()

    async def warmup(self) -> None:
        self._ensure_loaded()

    def supports_embeddings(self) -> bool:
        return self.embedding_backend.supports_embeddings()

    def configured_embedding_model(self) -> str | None:
        return self.embedding_backend.configured_model()

    def get_cache_stats(self) -> dict[str, object]:
        return {
            "prefix_cache": self.prefix_store.stats_snapshot(),
            "runtime_kernel": "batch_generator",
            "active_estimated_bytes": 0,
        }

    def clear_prefix_cache(self) -> dict[str, object]:
        result = self.prefix_store.clear(include_pinned=False)
        return result

    async def clear_runtime_caches(self) -> dict[str, object]:
        return {
            "prefix_cache": self.clear_prefix_cache(),
            "runtime": {"cleared": False, "reason": "not_supported_in_batched_engine"},
        }

    async def embeddings(
        self, *, model: str | None, input_data: str | list[str]
    ) -> dict[str, object]:
        return await self.embedding_backend.embeddings(model=model, input_data=input_data)

    async def count_text_tokens(self, texts: tuple[str, ...]) -> int:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        assert tokenizer is not None
        total = 0
        for text in texts:
            if not text:
                continue
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(text)
            total += len(token_ids)
        return total

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        state = self._new_request_state(request, stream=False)
        future: asyncio.Future[InferenceResponse] = asyncio.get_running_loop().create_future()
        state.response_future = future
        self._enqueue(state)
        timeout_seconds = request.timeout_seconds or self.settings.api.request_timeout_seconds
        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError:
            await self.cancel(state.request_id)
            raise AsterError(
                code="request_timeout",
                message=f"Request {state.request_id} timed out after {timeout_seconds:.1f}s",
                status_code=504,
            ) from None
        except asyncio.CancelledError:
            await self.cancel(state.request_id)
            raise

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        return await self.submit(request)

    async def stream(self, request: InferenceRequest) -> AsyncIterator[DecodeChunk]:
        state = self._new_request_state(request, stream=True)
        event = asyncio.Event()
        state.stream_event = event
        self._stream_events[state.request_id] = event
        self._enqueue(state)

        timeout_seconds = request.timeout_seconds or self.settings.api.request_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                # Drain any available chunks
                while state.stream_chunks:
                    chunk = state.stream_chunks.popleft()
                    if chunk.finished:
                        yield chunk
                        return
                    yield chunk

                # Check for errors
                if state.stream_error is not None:
                    raise state.stream_error

                # Check if request finished
                if state.request_id not in self._state:
                    return

                # Wait for next chunk
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    await self.cancel(state.request_id)
                    raise AsterError(
                        code="request_timeout",
                        message=f"Stream request {state.request_id} timed out",
                        status_code=504,
                    ) from None

                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except TimeoutError:
                    await self.cancel(state.request_id)
                    raise AsterError(
                        code="request_timeout",
                        message=f"Stream request {state.request_id} timed out",
                        status_code=504,
                    ) from None
        except (TimeoutError, AsterError):
            raise
        except Exception:
            await self.cancel(state.request_id)
            raise
        finally:
            self._stream_events.pop(state.request_id, None)

    async def cancel(self, request_id: str) -> bool:
        if request_id not in self._state:
            # Check if already finished
            if request_id in self._finished:
                return False
            return False

        state = self._state[request_id]
        state.cancel_requested = True
        self._pending_aborts.add(request_id)
        self._cancelled_requests += 1
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_request_state(self, request: InferenceRequest, *, stream: bool) -> _RequestState:
        request_id = request.trace_id or str(uuid.uuid4())
        return _RequestState(request_id=request_id, request=request)

    def _enqueue(self, state: _RequestState) -> None:
        state.enqueued_at = time.monotonic()
        self._waiting.append(state.request_id)
        self._state[state.request_id] = state

    @staticmethod
    def _cache_offset(prompt_cache: Any) -> int:
        if not isinstance(prompt_cache, list) or not prompt_cache:
            return 0
        offsets = [
            int(offset)
            for layer in prompt_cache
            if isinstance((offset := getattr(layer, "offset", None)), int)
        ]
        return min(offsets) if offsets else 0

    @staticmethod
    def _estimate_cache_bytes(prompt_cache: Any) -> int:
        if not isinstance(prompt_cache, list):
            return 0
        total = 0
        for layer in prompt_cache:
            if hasattr(layer, "nbytes"):
                total += int(layer.nbytes)
                continue
            state = getattr(layer, "state", None)
            values = state if isinstance(state, (list, tuple)) else [state]
            for value in values:
                if hasattr(value, "nbytes"):
                    total += int(value.nbytes)
        return total

    def _decode_output_text(self, token_ids: list[int]) -> str:
        try:
            decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            decoded = self._tokenizer.decode(token_ids)
        except Exception:
            return ""
        return decoded if isinstance(decoded, str) else ""

    def _prepare_prefix_cache_insert(
        self,
        entry: SnapshotEntry,
        prompt_tokens: list[int],
    ) -> tuple[Any, list[int], int] | None:
        prefix_token_count = entry.prefix_token_count
        if prefix_token_count <= 0 or prefix_token_count > len(prompt_tokens):
            return None

        required_cache_tokens = max(prefix_token_count - 1, 0)
        current_cache_tokens = self._cache_offset(entry.prompt_cache)
        if current_cache_tokens < required_cache_tokens:
            return None

        try:
            if current_cache_tokens == required_cache_tokens:
                # BatchGenerator owns and mutates its cache objects after insert.
                cached = copy.deepcopy(entry.prompt_cache)
            else:
                # Older snapshots may contain generated tokens. Only trim cache
                # types that explicitly support safe rewind; hybrid caches are
                # rejected rather than silently corrupting recurrent state.
                cached = self.model_runner.clone_cache(
                    entry.prompt_cache,
                    required_cache_tokens,
                )
        except Exception:
            return None

        if self._cache_offset(cached) != required_cache_tokens:
            return None
        return cached, prompt_tokens[required_cache_tokens:], prefix_token_count

    def _release_prefix_pin(self, state: _RequestState) -> None:
        if state.prefix_cache_key is None:
            return
        self.prefix_store.unpin(state.prefix_cache_key)
        self._prefix_pin_count = max(self._prefix_pin_count - 1, 0)
        state.prefix_cache_key = None

    def _cache_restore_compatible(
        self,
        use_cache: bool,
        cache_token_count: int,
        prompt_token_count: int,
        *,
        lane: _BatchLane | None = None,
    ) -> bool:
        profile = (
            prompt_token_count,
            use_cache,
            cache_token_count if use_cache else 0,
        )
        active_request_ids = lane.request_ids if lane is not None else self._running
        for rid in active_request_ids:
            state = self._state.get(rid)
            if state is None:
                continue
            other_profile = (
                len(state.prompt_tokens),
                state.prefix_cache_key is not None,
                max(len(state.prompt_tokens) - 1, 0)
                if state.prefix_cache_key is not None
                else 0,
            )
            if other_profile != profile:
                return False
        return True

    def _prompt_length_compatible(self, prompt_token_count: int) -> bool:
        return all(
            len(state.prompt_tokens) == prompt_token_count
            for rid in self._running
            if (state := self._state.get(rid)) is not None
        )

    def _process_prompt_responses(
        self,
        lane: _BatchLane,
        responses: list[Any],
    ) -> None:
        if not responses or not self.prefix_store.enabled or self._model_fingerprint is None:
            return
        for response in responses:
            progress = getattr(response, "progress", ())
            if (
                not getattr(response, "end_of_segment", False)
                or getattr(response, "end_of_prompt", False)
                or not isinstance(progress, tuple)
                or len(progress) < 2
                or progress[0] + 1 != progress[1]
            ):
                continue
            uid = response.uid
            rid = lane.uid_to_rid.get(uid)
            if rid is None:
                continue
            state = self._state.get(rid)
            if state is None or state.cancel_requested:
                continue

            try:
                extracted = lane.generator.extract_cache([uid])
                cache_data = extracted.get(uid)
                prompt_cache = cache_data[0] if isinstance(cache_data, tuple) else cache_data
                target_cache_tokens = max(len(state.prompt_tokens) - 1, 0)
                if self._cache_offset(prompt_cache) != target_cache_tokens:
                    continue
                self.prefix_store.store(
                    model_name=self.settings.model.name,
                    model_fingerprint=self._model_fingerprint,
                    prefix_tokens=state.prompt_tokens,
                    cache_token_count=target_cache_tokens,
                    prompt_cache=prompt_cache,
                    approx_bytes=self._estimate_cache_bytes(prompt_cache),
                )
            except Exception:
                self.logger.debug(
                    "batch_generator_prompt_cache_store_failed",
                    exc_info=True,
                    extra={"request_id": rid},
                )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import mlx.core as mx  # noqa: F811
            from mlx_lm import load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except Exception as exc:
            raise ConfigurationError(
                code="mlx_runtime_unavailable",
                message="MLX runtime dependencies are not installed or importable.",
                status_code=500,
                details={"error": str(exc)},
            ) from exc

        result = load(self.settings.model.path, lazy=False, return_config=True)
        result_len = len(result)  # type: ignore[arg-type]
        if result_len == 4:  # type: ignore[arg-type]
            model, tokenizer, config, _ = result  # type: ignore[misc]
        elif result_len == 3:  # type: ignore[arg-type]
            model, tokenizer, config = result  # type: ignore[misc]
        elif result_len == 2:  # type: ignore[arg-type]
            model, tokenizer = result  # type: ignore[misc]
        else:
            raise ConfigurationError(
                code="invalid_model_load_result",
                message=f"Unexpected MLX load() return shape (got {result_len} elements).",
                status_code=500,
            )

        self._mx = mx
        self._model = model
        self._tokenizer = tokenizer
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors

        # Generate model fingerprint for prefix cache namespace
        import hashlib
        fingerprint_src = str(self.settings.model.path) + getattr(tokenizer, "name_or_path", "")
        self._model_fingerprint = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:16]

        # Initialize paged KV cache
        try:
            layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", None)
            num_layers = len(layers) if layers else 24
            from aster.inference.paged_cache import PagedCacheManager
            self._paged_cache = PagedCacheManager(
                num_layers=num_layers,
                block_size=self.settings.engine.paged_cache_block_size or 64,
                max_blocks=self.settings.engine.paged_cache_max_blocks or 1000,
            )
        except Exception:
            self._paged_cache = None

    def _create_batch_generator(self) -> Any:
        from mlx_lm.generate import BatchGenerator

        engine = self.settings.engine
        return BatchGenerator(
            self._model,
            max_tokens=engine.max_active_requests * 512,
            completion_batch_size=engine.max_active_requests,
            prefill_batch_size=engine.max_active_requests,
            prefill_step_size=engine.prefill_token_budget,
        )

    def _get_or_create_lane(
        self,
        profile: _BatchProfile,
        *,
        wait_for_cohort: bool = False,
    ) -> _BatchLane | None:
        lane = self._batch_lanes.get(profile)
        if lane is not None:
            if not lane.sealed:
                if (
                    lane.admission_window_ms > 0
                    and len(lane.request_ids) >= lane.target_request_count
                ):
                    return None
                return lane
            if lane.request_ids:
                return None
            try:
                lane.generator.close()
            except Exception:
                pass
            del self._batch_lanes[profile]
        if len(self._batch_lanes) >= self._batch_generator_max_lanes:
            for old_profile, old_lane in tuple(self._batch_lanes.items()):
                if old_lane.request_ids:
                    continue
                try:
                    old_lane.generator.close()
                except Exception:
                    pass
                del self._batch_lanes[old_profile]
                break
            else:
                return None
        lane = _BatchLane(
            profile=profile,
            generator=self._create_batch_generator(),
            admission_window_ms=(
                self._batch_generator_lane_admission_window_ms if wait_for_cohort else 0.0
            ),
            target_request_count=(
                self._cohort_target_request_count() if wait_for_cohort else 1
            ),
        )
        self._batch_lanes[profile] = lane
        return lane

    def _cohort_target_request_count(self) -> int:
        settings = getattr(self, "settings", None)
        if settings is None:
            return 1
        configured_target = settings.engine.batch_generator_lane_target_size
        if configured_target > 0:
            return configured_target
        remaining_slots = settings.engine.max_active_requests - len(self._running)
        return max(remaining_slots, 1)

    def _lane_ready_for_step(self, lane: _BatchLane, *, now: float | None = None) -> bool:
        if lane.sealed:
            return True
        window_ms = lane.admission_window_ms
        if window_ms <= 0:
            return True
        if len(lane.request_ids) >= lane.target_request_count:
            return True
        current = time.monotonic() if now is None else now
        return current - lane.created_at + 1e-9 >= window_ms / 1000.0

    def _lane_step_quanta(
        self,
        lane: _BatchLane,
        active_lanes: list[_BatchLane] | tuple[_BatchLane, ...],
    ) -> int:
        quantum = getattr(self, "_batch_generator_longest_lane_step_quanta", 1)
        if quantum <= 1 or not active_lanes:
            return 1
        longest_prompt_tokens = max(item.profile[0] for item in active_lanes)
        return quantum if lane.profile[0] == longest_prompt_tokens else 1

    def _prepare_prompt_tokens(self, state: _RequestState) -> list[int]:
        request = state.request
        if request.messages:
            try:
                if hasattr(self._tokenizer, "apply_chat_template"):
                    prompt_text = self._tokenizer.apply_chat_template(
                        request.messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=request.enable_thinking,
                        **request.chat_template_kwargs,
                    )
                    if isinstance(prompt_text, str):
                        return self._tokenizer.encode(prompt_text)
                    return list(prompt_text)
            except Exception:
                pass
            # Fallback: manually build
            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **request.chat_template_kwargs,
                )
                return self._tokenizer.encode(prompt_text)
            except Exception:
                text = "".join(
                    f"{msg['role']}: {msg['content']}\n" for msg in request.messages
                )
                return self._tokenizer.encode(text)

        if request.prompt is not None:
            return self._tokenizer.encode(request.prompt)

        raise ConfigurationError(
            code="missing_prompt",
            message="InferenceRequest requires either prompt or messages",
            status_code=400,
        )

    def _make_sampler_for_request(self, request: InferenceRequest) -> Any:
        temp = request.temperature
        top_p = request.top_p
        # top_k=0 in sample_utils means disabled
        top_k = request.top_k
        min_p = request.min_p
        return self._make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)

    def _make_logits_processors_for_request(self, request: InferenceRequest) -> list[Any]:
        processors: list[Any] = []

        if request.repetition_penalty != 1.0:
            try:
                from mlx_lm.sample_utils import make_repetition_penalty
                processors.append(make_repetition_penalty(request.repetition_penalty))
            except ImportError:
                pass

        # Thinking-aware processor
        if request.enable_thinking:
            try:
                processors.append(
                    ThinkingAwareLogitsProcessor(
                        self._tokenizer,
                        thinking_token_budget=request.thinking_token_budget,
                    )
                )
            except Exception:
                pass

        # JSON structured output
        if request.structured_output_schema is not None:
            try:
                json_processor = build_json_logits_processor(
                    request.structured_output_schema,
                    self._tokenizer,
                )
                if json_processor is not None:
                    processors.append(json_processor)
                if request.enable_thinking:
                    processors.append(
                        ThinkingAwareJsonLogitsProcessor(
                            self._tokenizer,
                            request.structured_output_schema,
                            thinking_token_budget=request.thinking_token_budget,
                        )
                    )
            except Exception:
                pass

        return processors

    def _normalize_stop_token_ids(self, request: InferenceRequest) -> set[int]:
        stop_ids: set[int] = set()

        # Stop token IDs from request
        stop_ids.update(request.stop_token_ids)

        # Tokenize string stop sequences
        if request.stop:
            stops = [request.stop] if isinstance(request.stop, str) else request.stop
            for stop in stops:
                if stop:
                    try:
                        encoded = self._tokenizer.encode(stop)
                        stop_ids.update(encoded)
                    except Exception:
                        pass

        # Add EOS token
        if hasattr(self._tokenizer, "eos_token_id") and self._tokenizer.eos_token_id is not None:
            stop_ids.add(self._tokenizer.eos_token_id)

        return stop_ids

    # ------------------------------------------------------------------
    # Engine loop
    # ------------------------------------------------------------------

    async def _engine_loop(self) -> None:
        self._ensure_loaded()

        # Install chunked prefill for fairness on long prompts
        # NOTE: disabled because mlx-lm internal API differs by version
        # See aster/inference/chunked_prefill.py for the implementation reference
        self.logger.info("chunked_prefill_disabled_mlx_lm_api_mismatch")

        try:
            while self._running_flag:
                # 1. Process pending aborts
                self._process_aborts()

                # 2. Schedule waiting requests
                self._schedule_waiting()

                # 3. Run generation steps per profile lane. Calls remain
                # sequential so MLX stream ownership stays with this loop.
                active_lanes = tuple(
                    lane for lane in self._batch_lanes.values() if lane.request_ids
                )
                for lane in active_lanes:
                    if not lane.request_ids:
                        continue
                    if not self._lane_ready_for_step(lane):
                        continue
                    lane.sealed = True
                    for _ in range(self._lane_step_quanta(lane, active_lanes)):
                        if not lane.request_ids:
                            break
                        try:
                            prompt_responses, responses = lane.generator.next()
                            self._process_prompt_responses(lane, prompt_responses)
                            if responses:
                                self._process_responses(lane, responses)
                        except IndexError:
                            # mlx-lm _next() fails on empty segments — recover by
                            # aborting only this lane's requests.
                            self.logger.warning(
                                "batch_generator_empty_segments_recovering",
                                extra={"profile": lane.profile},
                            )
                            for rid in list(lane.request_ids):
                                self._pending_aborts.add(rid)
                            break
                        except Exception as exc:
                            self.logger.warning(
                                "batch_generator_step_failed",
                                exc_info=True,
                                extra={"error": str(exc), "profile": lane.profile},
                            )
                            break

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.exception(
                "batched_engine_loop_crashed",
                extra={"error": str(exc)},
            )
            for state in list(self._state.values()):
                self._release_prefix_pin(state)
                if state.response_future is not None and not state.response_future.done():
                    state.response_future.set_exception(exc)
                if state.stream_event is not None:
                    state.stream_error = exc
                    state.stream_event.set()
        finally:
            # Drain all remaining futures to prevent stuck callers
            for state in list(self._state.values()):
                if state.response_future is not None and not state.response_future.done():
                    state.response_future.set_exception(
                        AsterError(
                            code="engine_stopped",
                            message="Engine stopped before request completed",
                            status_code=503,
                        )
                    )
                if state.stream_event is not None:
                    state.stream_error = AsterError(
                        code="engine_stopped",
                        message="Engine stopped before request completed",
                        status_code=503,
                    )
                    state.stream_event.set()
            self._state.clear()
            self._finished.clear()
            for lane in self._batch_lanes.values():
                try:
                    lane.generator.close()
                except Exception:
                    pass
            self._batch_lanes.clear()

    def _process_aborts(self) -> None:
        for rid in list(self._pending_aborts):
            self._pending_aborts.discard(rid)

            state = self._state.get(rid)
            lane = (
                self._batch_lanes.get(state.batch_profile)
                if state is not None and state.batch_profile is not None
                else None
            )
            if lane is not None:
                uid = lane.rid_to_uid.pop(rid, None)
                if uid is not None:
                    try:
                        lane.generator.remove([uid])
                    except Exception:
                        pass
                    lane.uid_to_rid.pop(uid, None)
                lane.request_ids.discard(rid)

            if rid in self._running:
                self._running.discard(rid)

            state = self._state.pop(rid, None)
            if state is None:
                continue
            self._release_prefix_pin(state)

            if state.response_future is not None and not state.response_future.done():
                state.response_future.set_exception(
                    AsterError(
                        code="request_cancelled",
                        message=f"Request {rid} was cancelled",
                        status_code=499,
                    )
                )
            if state.stream_event is not None:
                state.stream_error = AsterError(
                    code="request_cancelled",
                    message=f"Request {rid} was cancelled",
                    status_code=499,
                )
                state.stream_event.set()

    def _schedule_waiting(self) -> None:
        engine = self.settings.engine
        attempts = len(self._waiting)
        while (
            len(self._running) < engine.max_active_requests
            and self._waiting
            and attempts > 0
        ):
            attempts -= 1
            rid = self._waiting.popleft()
            state = self._state.get(rid)
            if state is None:
                continue

            try:
                # Tokenization is retained while a request waits for a free
                # compatible lane, avoiding repeated tokenizer work.
                tokens = state.prompt_tokens or self._prepare_prompt_tokens(state)
                state.prompt_tokens = tokens
                if state.started_at is None:
                    state.started_at = time.monotonic()

                # Sampler
                sampler = self._make_sampler_for_request(state.request)

                # Logits processors
                logits_processors = self._make_logits_processors_for_request(state.request)
                # Ensure it's a list of lists for BatchGenerator
                wrapped_lps = (
                    [logits_processors] if logits_processors else []
                )
                state.effective_stop_token_ids = frozenset(
                    self._normalize_stop_token_ids(state.request)
                )

                # Create paged cache block table for this request
                if self._paged_cache is not None:
                    try:
                        self._paged_cache.create_table(rid)
                    except Exception:
                        pass

                # Prefix cache lookup
                prompt_cache = None
                prompt_tokens_for_insert = tokens
                if self.prefix_store.enabled and self._model_fingerprint:
                    matched = self.prefix_store.lookup(
                        self.settings.model.name,
                        tokens,
                        model_fingerprint=self._model_fingerprint,
                    )
                    if matched is not None:
                        self.prefix_store.pin(matched.key)
                        self._prefix_pin_count += 1
                        prepared = self._prepare_prefix_cache_insert(matched, tokens)
                        if prepared is not None:
                            prompt_cache, prompt_tokens_for_insert, cached_tokens = prepared
                            state.matched_prefix_tokens = cached_tokens
                            state.prefix_cache_key = matched.key
                        else:
                            self._release_prefix_pin(state)

                cached_prefix_tokens = max(len(tokens) - len(prompt_tokens_for_insert), 0)
                profile: _BatchProfile = (
                    len(tokens),
                    prompt_cache is not None,
                    cached_prefix_tokens if prompt_cache is not None else 0,
                )
                lane = self._get_or_create_lane(
                    profile,
                    wait_for_cohort=bool(self._running) and not self._waiting,
                )
                if lane is None:
                    self._release_prefix_pin(state)
                    state.matched_prefix_tokens = 0
                    if self._paged_cache is not None:
                        try:
                            self._paged_cache.remove_table(rid)
                        except Exception:
                            pass
                    self._waiting.append(rid)
                    continue

                if not self._cache_restore_compatible(
                    prompt_cache is not None,
                    cached_prefix_tokens,
                    len(tokens),
                    lane=lane,
                ):
                    self._release_prefix_pin(state)
                    state.matched_prefix_tokens = 0
                    if self._paged_cache is not None:
                        try:
                            self._paged_cache.remove_table(rid)
                        except Exception:
                            pass
                    self._waiting.append(rid)
                    continue

                # Insert into the compatible profile lane.
                kwargs = dict(
                    max_tokens=[state.max_tokens],
                    samplers=[sampler],
                    logits_processors=wrapped_lps,
                )
                kwargs["prompts"] = [prompt_tokens_for_insert]
                kwargs["all_tokens"] = [list(tokens[:cached_prefix_tokens])]
                if prompt_cache is not None:
                    kwargs["caches"] = [prompt_cache]
                uid_list = lane.generator.insert(**kwargs)
                uid = uid_list[0]
                state.batch_profile = profile
                lane.request_ids.add(rid)
                lane.uid_to_rid[uid] = rid
                lane.rid_to_uid[rid] = uid
                self._running.add(rid)
            except Exception as exc:
                self.logger.warning(
                    "schedule_request_failed",
                    exc_info=True,
                    extra={"request_id": rid, "error": str(exc)},
                )
                state = self._state.pop(rid, None)
                if state is not None:
                    self._release_prefix_pin(state)
                    if state.response_future is not None and not state.response_future.done():
                        state.response_future.set_exception(exc)
                    if state.stream_event is not None:
                        state.stream_error = exc
                        state.stream_event.set()

    def _process_responses(self, lane: _BatchLane, responses: list[Any]) -> None:
        if not responses:
            return

        for response in responses:
            uid = response.uid
            rid = lane.uid_to_rid.get(uid)
            if rid is None:
                continue

            state = self._state.get(rid)
            if state is None or state.cancel_requested:
                continue

            token = response.token
            state.output_token_ids.append(token)
            state.generated_tokens += 1

            # Extract KV cache for finishing requests (uid is still in generation batch)
            finish_reason = ""
            if response.finish_reason is not None:
                finish_reason = response.finish_reason
            elif state.generated_tokens >= state.max_tokens:
                finish_reason = "length"
            elif token in state.stop_token_ids:
                finish_reason = "stop"

            if finish_reason:
                self._finish_request(state, lane=lane, finish_reason=finish_reason)
            else:
                # Stream the token
                try:
                    text = self._decode_output_text([token])
                except Exception:
                    text = ""
                if state.stream_event is not None:
                    chunk = DecodeChunk(token=text, index=state.generated_tokens - 1, finished=False)
                    state.stream_chunks.append(chunk)
                    state.stream_event.set()

    def _finish_request(
        self,
        state: _RequestState,
        *,
        lane: _BatchLane,
        finish_reason: str,
    ) -> None:
        state.completed_at = time.monotonic()
        state.finish_reason = finish_reason

        # Release paged cache blocks
        if self._paged_cache is not None:
            try:
                self._paged_cache.remove_table(state.request_id)
            except Exception:
                pass

        prompt_seconds = (state.started_at or state.created_at) - state.created_at
        gen_seconds = state.completed_at - (state.started_at or state.created_at)

        state.prompt_tps = len(state.prompt_tokens) / max(prompt_seconds, 0.001)
        state.generation_tps = state.generated_tokens / max(gen_seconds, 0.001)

        # Decode full text
        state.output_text = self._decode_output_text(state.output_token_ids)

        response = InferenceResponse(
            request_id=state.request_id,
            text=state.output_text,
            prompt_tokens=len(state.prompt_tokens),
            completion_tokens=state.generated_tokens,
            cache_hit=state.matched_prefix_tokens > 0,
            prefill_cache_hit=state.matched_prefix_tokens > 0,
            generation_cache_reuse=state.matched_prefix_tokens > 0,
            speculative_enabled=False,
            speculative_path_mode="disabled",
            prompt_tps=state.prompt_tps,
            generation_tps=state.generation_tps,
            peak_memory_gb=state.peak_memory_gb,
            finish_reason=finish_reason,
        )

        self._finished[state.request_id] = response
        # Cap finished dict to prevent unbounded memory growth
        if len(self._finished) > 1000:
            oldest = next(iter(self._finished))
            self._finished.pop(oldest, None)
        self._running.discard(state.request_id)
        self._completed_requests += 1
        self._total_prompt_tokens += len(state.prompt_tokens)
        self._total_completion_tokens += state.generated_tokens

        # Remove from the owning lane and release any pinned prefix snapshot.
        uid = lane.rid_to_uid.pop(state.request_id, None)
        if uid is not None:
            lane.uid_to_rid.pop(uid, None)
            try:
                lane.generator.remove([uid])
            except Exception:
                pass
        lane.request_ids.discard(state.request_id)
        self._release_prefix_pin(state)

        # Signal completion
        if state.response_future is not None and not state.response_future.done():
            state.response_future.set_result(response)

        if state.stream_event is not None:
            chunk = DecodeChunk(token="", index=state.generated_tokens, finished=True)
            state.stream_chunks.append(chunk)
            state.stream_event.set()

        # Clean state
        self._state.pop(state.request_id, None)


__all__ = ["BatchedEngine"]
