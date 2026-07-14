from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aster.core.config import RuntimeSettings
from aster.core.errors import AsterError
from aster.inference.contracts import InferenceRequest
from aster.inference.engine import InferenceEngine
from aster.inference.model_runner import (
    DecodeInit,
    DecodeResult,
    PrefillChunkResult,
    PrefillTransientProfile,
    PreparedPrompt,
)
from aster.inference.request_state import RequestPhase, RequestState
from aster.inference.runtime_kernel import ManualRuntimeKernel


class _MetricValue:
    def __init__(self) -> None:
        self.value = 0.0
        self.samples: list[float] = []

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def set(self, value: float) -> None:
        self.value = value

    def observe(self, value: float) -> None:
        self.samples.append(value)

    def labels(self, **_: object) -> _MetricValue:
        return self


class DummyMetrics:
    def __init__(self) -> None:
        self.request_latency = _MetricValue()
        self.first_token_latency = _MetricValue()
        self.prefill_latency = _MetricValue()
        self.decode_latency = _MetricValue()
        self.queue_depth = _MetricValue()
        self.active_requests = _MetricValue()
        self.prefill_active = _MetricValue()
        self.decode_active = _MetricValue()
        self.decode_batch = _MetricValue()
        self.prefill_steps = _MetricValue()
        self.decode_steps = _MetricValue()
        self.decode_tokens = _MetricValue()
        self.prefix_reuse_attempts = _MetricValue()
        self.prefix_cache_hits = _MetricValue()
        self.prefix_cache_misses = _MetricValue()
        self.prefix_tokens_reused = _MetricValue()
        self.snapshot_bytes = _MetricValue()
        self.snapshot_entries = _MetricValue()
        self.cancellations = _MetricValue()
        self.admission_rejections = _MetricValue()
        self.queue_wait_latency = _MetricValue()
        self.worker_restarts = _MetricValue()
        self.errors = _MetricValue()


@dataclass
class FakeDetokenizer:
    last_segment: str = ""
    fail_decode: bool = False

    def add_token(self, token: int) -> None:
        self.last_segment = chr(token)

    def finalize(self) -> None:
        self.last_segment = ""


class FakeRunner:
    def __init__(self) -> None:
        self.prefill_calls = 0
        self.decode_batch_sizes: list[int] = []
        self.decode_request_ids: list[tuple[str | None, ...]] = []
        self.prefill_delay_seconds = 0.0
        self.decode_delay_seconds = 0.0
        self.decode_error: BaseException | None = None
        self.clear_runtime_cache_calls = 0
        self.prefill_started = threading.Event()
        self.prompt_map: dict[str, list[int]] = {}
        self.encode_calls = 0
        self.thread_ids: list[int] = []
        self.strict_prefix_calls: list[list[dict[str, str]]] = []
        self.clone_cache_token_counts: list[int | None] = []
        self.available_memory_bytes_value = 1 << 30
        self.transient_profile: PrefillTransientProfile | None = None
        self.initialize_requests: list[InferenceRequest] = []
        self.decode_diagnostics_payload: dict[str, object] = {"batch_attempts": 0}

    def _record_thread(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def warmup(self) -> None:
        self._record_thread()
        return None

    def encode_request(self, request: InferenceRequest) -> PreparedPrompt:
        self._record_thread()
        self.encode_calls += 1
        prompt = request.prompt or "empty"
        if prompt not in self.prompt_map:
            size = max(len(prompt.split()), 1)
            self.prompt_map[prompt] = list(range(1, size + 2))
        tokens = self.prompt_map[prompt]
        reuse_points = (len(tokens) - 1,) if len(tokens) > 2 else ()
        return PreparedPrompt(prompt_tokens=tokens, reuse_points=reuse_points)

    def estimate_request_bytes(self, prompt_tokens: int, max_tokens: int) -> int:
        self._record_thread()
        return max(prompt_tokens + max_tokens, 1) * 8

    def prefill_transient_profile(self) -> PrefillTransientProfile | None:
        self._record_thread()
        return self.transient_profile

    def model_fingerprint(self) -> str:
        self._record_thread()
        return "fake-model"

    def strict_chat_prefix_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        self._record_thread()
        del enable_thinking, chat_template_kwargs
        self.strict_prefix_calls.append(messages)
        return "warm strict prefix"

    def available_memory_bytes(self) -> int:
        return self.available_memory_bytes_value

    def clone_cache(
        self, prompt_cache: Any | None, cache_token_count: int | None = None
    ) -> Any | None:
        self._record_thread()
        self.clone_cache_token_counts.append(cache_token_count)
        return copy.deepcopy(prompt_cache)

    def prefill_to(
        self,
        *,
        prompt_tokens: list[int],
        prompt_cache: Any | None,
        cache_token_count: int,
        target_cache_token_count: int,
    ) -> PrefillChunkResult:
        self._record_thread()
        del prompt_tokens
        self.prefill_started.set()
        if self.prefill_delay_seconds > 0:
            time.sleep(self.prefill_delay_seconds)
        self.prefill_calls += 1
        live_cache = dict(prompt_cache or {})
        live_cache["cache_tokens"] = target_cache_token_count
        return PrefillChunkResult(
            prompt_cache=live_cache,
            cache_token_count=target_cache_token_count,
            elapsed_seconds=0.001,
            peak_memory_gb=1.5,
            active_memory_gb=1.0,
        )

    def initialize_decode(
        self,
        *,
        prompt_tokens: list[int],
        cache_token_count: int,
        prompt_cache: Any | None,
        request: InferenceRequest,
    ) -> DecodeInit:
        self._record_thread()
        self.initialize_requests.append(request)
        del cache_token_count
        return DecodeInit(
            prompt_cache=dict(prompt_cache or {}),
            next_input_token=prompt_tokens[-1],
            sampler=lambda x: x,
            detokenizer=FakeDetokenizer(fail_decode=request.trace_id == "decode-fails"),
            stop_token_ids=frozenset(request.stop_token_ids),
        )

    def decode_batch_step(self, items: list[Any]) -> list[DecodeResult | BaseException]:
        self._record_thread()
        if self.decode_error is not None:
            raise self.decode_error
        if self.decode_delay_seconds > 0:
            time.sleep(self.decode_delay_seconds)
        self.decode_batch_sizes.append(len(items))
        self.decode_request_ids.append(tuple(getattr(item, "request_id", None) for item in items))
        results: list[DecodeResult | BaseException] = []
        for item in items:
            if getattr(item.detokenizer, "fail_decode", False):
                results.append(RuntimeError("per-request decode failed"))
                continue
            next_count = item.completion_tokens + 1
            token_id = ord("a") + item.completion_tokens
            if token_id in item.stop_token_ids:
                results.append(
                    DecodeResult(
                        prompt_cache=dict(item.prompt_cache or {}),
                        token_id=token_id,
                        text="",
                        completion_tokens=next_count,
                        peak_memory_gb=0.25,
                        finish_reason="stop",
                    )
                )
                continue
            text = chr(token_id)
            finish_reason = "length" if next_count >= min(item.max_tokens, 2) else None
            prompt_cache = dict(item.prompt_cache or {})
            prompt_cache["decoded"] = next_count
            results.append(
                DecodeResult(
                    prompt_cache=prompt_cache,
                    token_id=token_id,
                    text=text,
                    completion_tokens=next_count,
                    peak_memory_gb=0.25,
                    finish_reason=finish_reason,
                )
            )
        return results

    def decode_diagnostics(self) -> dict[str, object]:
        return dict(self.decode_diagnostics_payload)

    def finalize_detokenizer(self, detokenizer: Any | None) -> str:
        self._record_thread()
        if detokenizer is None:
            return ""
        detokenizer.finalize()
        return ""

    def estimate_cache_bytes(self, prompt_cache: Any | None) -> int:
        self._record_thread()
        return 64 if prompt_cache else 0

    def clear_runtime_caches(self) -> dict[str, object]:
        self._record_thread()
        self.clear_runtime_cache_calls += 1
        return {"mlx_cache_cleared": True}


def _make_engine(
    engine_overrides: dict[str, Any] | None = None,
    api_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
    batch_overrides: dict[str, Any] | None = None,
) -> tuple[InferenceEngine, FakeRunner]:
    engine_settings = {
        "max_active_requests": 8,
        "max_decode_batch": 4,
        "prefill_token_budget": 2,
        "snapshot_budget_bytes": 4096,
        "snapshot_min_prefix_tokens": 2,
        "snapshot_max_entries": 32,
        "prefix_cache_enabled": True,
        "stream_interval_tokens": 1,
    }
    if engine_overrides:
        engine_settings.update(engine_overrides)
    batch_settings = {
        "prefill_batch_size": 4,
        "decode_batch_size": engine_settings["max_decode_batch"],
    }
    if batch_overrides:
        batch_settings.update(batch_overrides)
    settings = RuntimeSettings.model_validate(
        {
            "api": api_overrides or {},
            "model": model_overrides or {},
            "embeddings": {"enabled": False},
            "engine": engine_settings,
            "batch": batch_settings,
        }
    )
    engine = InferenceEngine(settings, DummyMetrics())  # type: ignore[arg-type]
    runner = FakeRunner()
    engine.model_runner = runner  # type: ignore[assignment]
    engine.runtime_kernel = ManualRuntimeKernel(runner)  # type: ignore[arg-type]
    return engine, runner


def test_engine_cleanup_releases_owned_prompt_cache() -> None:
    class ReleasableCache(list[Any]):
        released = 0

        def release(self) -> None:
            self.released += 1

    engine, _runner = _make_engine()
    cache = ReleasableCache()
    state = RequestState(
        request_id="release-cache",
        request=InferenceRequest(prompt="ignored"),
        prompt_cache=cache,
    )
    engine._requests[state.request_id] = state

    engine._cleanup_request(state)

    assert cache.released == 1
    assert state.request_id not in engine._requests


def test_prefill_chunk_target_is_clamped_by_transient_memory_budget() -> None:
    engine, runner = _make_engine(engine_overrides={"prefill_token_budget": 16})
    runner.available_memory_bytes_value = 1000
    state = RequestState(
        request_id="transient-budget",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=list(range(21)),
        cache_token_count=0,
        prefill_transient_profile=PrefillTransientProfile(
            n_q_heads=1,
            head_dim=1,
            score_dtype_size=10,
        ),
    )

    assert engine._prefill_chunk_target(state, remaining=20) == 9


def test_prefill_chunk_target_rejects_unaffordable_transient_activation() -> None:
    engine, runner = _make_engine(engine_overrides={"prefill_token_budget": 16})
    runner.available_memory_bytes_value = 200
    state = RequestState(
        request_id="unaffordable-transient-budget",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=list(range(22)),
        cache_token_count=20,
        prefill_transient_profile=PrefillTransientProfile(
            n_q_heads=1,
            head_dim=1,
            score_dtype_size=10,
        ),
    )

    assert engine._prefill_chunk_target(state, remaining=1) is None


def test_prefill_rejects_unaffordable_transient_before_model_execution() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(engine_overrides={"prefill_token_budget": 16})
        runner.available_memory_bytes_value = 200
        state = RequestState(
            request_id="unaffordable-transient-prefill",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=list(range(22)),
            cache_token_count=20,
            prefill_transient_profile=PrefillTransientProfile(
                n_q_heads=1,
                head_dim=1,
                score_dtype_size=10,
            ),
        )
        engine._requests[state.request_id] = state
        engine._prefill_queue.append(state.request_id)
        try:
            assert await engine._step_prefill()
        finally:
            await engine.aclose()

        assert runner.prefill_calls == 0
        assert state.request_id not in engine._requests
        assert engine.status()["failed_requests"] == 1

    asyncio.run(scenario())


def test_transient_prefill_guard_completes_with_chunked_prefill() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(engine_overrides={"prefill_token_budget": 16})
        runner.available_memory_bytes_value = 1000
        runner.transient_profile = PrefillTransientProfile(
            n_q_heads=1,
            head_dim=1,
            score_dtype_size=10,
        )
        await engine.start()
        try:
            result = await engine.submit(
                InferenceRequest(prompt=" ".join("token" for _ in range(20)), max_tokens=2)
            )
            status = engine.status()
        finally:
            await engine.stop()

        assert result.text == "ab"
        assert runner.prefill_calls >= 3
        assert len(set(runner.thread_ids)) == 1
        assert status["completed_requests"] == 1
        assert status["failed_requests"] == 0

    asyncio.run(scenario())


def test_activate_decode_skips_full_checkpoint_after_prefix_hit() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        state = RequestState(
            request_id="prefix-branch-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            prompt_cache={"cache_tokens": 5},
            matched_prefix_tokens=3,
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._activate_decode(state)
            assert engine.prefix_store.entry_count == 0
        finally:
            await engine.aclose()

    asyncio.run(scenario())


def test_activate_decode_keeps_full_checkpoint_for_exact_hit() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        state = RequestState(
            request_id="exact-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            prompt_cache={"cache_tokens": 5},
            matched_prefix_tokens=6,
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._activate_decode(state)
            assert engine.prefix_store.entry_count == 1
        finally:
            await engine.aclose()


    asyncio.run(scenario())


def test_maybe_checkpoint_skips_full_prompt_after_prefix_hit() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        state = RequestState(
            request_id="prefill-prefix-branch-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            cache_token_count=5,
            prompt_cache={"cache_tokens": 5},
            matched_prefix_tokens=3,
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._maybe_checkpoint(state)
            assert engine.prefix_store.entry_count == 0
        finally:
            await engine.aclose()

    asyncio.run(scenario())


def test_decode_work_item_does_not_duplicate_current_token_for_logits_processors() -> None:
    engine, _runner = _make_engine()
    state = RequestState(
        request_id="decode-history",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=[1, 2, 3],
    )
    state.next_input_token = 65
    state.output_token_ids = [65]
    state.decode_sampler = lambda value: value
    state.decode_detokenizer = FakeDetokenizer()

    item = engine._decode_work_item(state)

    assert item.logits_processor_tokens == [1, 2, 3]


def test_engine_passes_structured_schema_to_runtime_decode() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        await engine.start()
        try:
            result = await engine.submit(
                InferenceRequest(
                    prompt="structured output",
                    max_tokens=1,
                    structured_output_schema=schema,
                )
            )
        finally:
            await engine.stop()

        assert result.finish_reason == "length"
        assert runner.initialize_requests
        assert runner.initialize_requests[0].structured_output_schema == schema

    asyncio.run(scenario())


def test_engine_batches_decode_steps_for_concurrent_requests() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            first = asyncio.create_task(
                engine.submit(InferenceRequest(prompt="one two", max_tokens=2, trace_id="batch-one"))
            )
            second = asyncio.create_task(
                engine.submit(
                    InferenceRequest(prompt="three four", max_tokens=2, trace_id="batch-two")
                )
            )
            first_result, second_result = await asyncio.gather(first, second)
        finally:
            await engine.stop()

        assert first_result.text == "ab"
        assert second_result.text == "ab"
        assert first_result.finish_reason == "length"
        assert second_result.finish_reason == "length"
        assert first_result.peak_memory_gb == 1.5
        assert second_result.peak_memory_gb == 1.5
        assert any(size >= 2 for size in runner.decode_batch_sizes)
        assert any(set(ids) >= {"batch-one", "batch-two"} for ids in runner.decode_request_ids)
        status = engine.status()
        assert status["decode_steps"] >= 1
        timing = status["engine_timing"]
        assert timing["prefill_model_tokens"] > 0
        assert timing["prompt_tps"] > 0
        assert timing["max_prefill_peak_memory_gb"] == 1.5
        assert timing["max_prefill_active_memory_gb"] == 1.0
        assert timing["decode_runner_batches"] >= 1
        assert timing["decode_runner_tokens"] >= 4
        assert timing["generation_tps"] > 0
        assert timing["avg_decode_batch_size"] > 1
        assert status["completed_requests"] == 2
        assert status["runtime_kernel"] == "manual"
        recent = status["recent_request_timelines"]
        assert {entry["request_id"] for entry in recent} >= {"batch-one", "batch-two"}
        for entry in recent:
            if entry["request_id"] not in {"batch-one", "batch-two"}:
                continue
            assert entry["status"] == "finished"
            assert entry["prefill_steps"] >= 1
            assert entry["decode_steps"] >= 2
            assert entry["total_latency_s"] is not None
            assert entry["queue_wait_s"] is not None
            assert entry["prefill_model_s"] > 0
            assert entry["decode_duration_s"] is not None

    asyncio.run(scenario())


def test_scheduler_step_prioritizes_decode_before_new_admissions(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        calls: list[str] = []

        async def process_cancellations() -> bool:
            calls.append("cancellations")
            return False

        async def step_decode() -> bool:
            calls.append("decode")
            return True

        async def step_prefill() -> bool:
            calls.append("prefill")
            return False

        async def drain_submissions() -> bool:
            calls.append("admission")
            return True

        monkeypatch.setattr(engine, "_process_cancellations", process_cancellations)
        monkeypatch.setattr(engine, "_step_decode", step_decode)
        monkeypatch.setattr(engine, "_step_prefill", step_prefill)
        monkeypatch.setattr(engine, "_drain_submissions", drain_submissions)

        try:
            did_work = await engine._scheduler_step()
        finally:
            await engine.aclose()

        assert did_work is True
        assert calls == ["cancellations", "decode", "prefill", "admission"]

    asyncio.run(scenario())


def test_drain_submissions_fills_available_active_slots() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(batch_overrides={"prefill_batch_size": 2})
        states = [
            RequestState(
                request_id=f"admission-{index}",
                request=InferenceRequest(prompt=f"prompt {index}", max_tokens=1),
            )
            for index in range(5)
        ]
        for state in states:
            engine._requests[state.request_id] = state
            engine._submission_queue.put_nowait(state)

        try:
            did_work = await engine._drain_submissions()
            status = engine.status()
        finally:
            await engine.aclose()

        assert did_work is True
        assert runner.encode_calls == 5
        assert len(engine._prefill_queue) == 5
        assert engine._submission_queue.qsize() == 0
        assert status["scheduler"] == {
            "mode": "decode_first_chunked_prefill",
            "max_active_requests": 8,
            "max_decode_batch": 4,
            "prefill_batch_size": 2,
            "admission_policy": "fill_available_active_slots",
            "prefill_token_budget": 2,
        }

    asyncio.run(scenario())


def test_prefill_continuation_yields_to_new_admissions() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(batch_overrides={"prefill_batch_size": 1})
        long_state = RequestState(
            request_id="long-prefill",
            request=InferenceRequest(prompt="ignored", max_tokens=1),
            prompt_tokens=list(range(1, 10)),
            model_fingerprint="fake-model",
            admission_prepared=True,
            estimated_bytes=80,
            phase=RequestPhase.PREFILL_WAIT,
        )
        short_state = RequestState(
            request_id="short-new",
            request=InferenceRequest(prompt="short", max_tokens=1),
        )
        engine._requests[long_state.request_id] = long_state
        engine._requests[short_state.request_id] = short_state
        engine._active_estimated_bytes = long_state.estimated_bytes
        engine._prefill_queue.append(long_state.request_id)
        engine._submission_queue.put_nowait(short_state)

        try:
            did_work = await engine._scheduler_step()
        finally:
            await engine.aclose()

        assert did_work is True
        assert runner.prefill_calls == 1
        assert runner.encode_calls == 1
        assert long_state.cache_token_count == 2
        assert list(engine._prefill_queue)[:2] == ["short-new", "long-prefill"]
        assert "long-prefill" not in engine._prefill_yield_request_ids
        assert engine.status()["prefill_yield_rotations"] == 1

    asyncio.run(scenario())


def test_engine_defers_admission_under_transient_memory_pressure() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(
            engine_overrides={
                "max_active_requests": 2,
                "admission_retry_limit": 8,
            }
        )
        runner.available_memory_bytes_value = 64
        await engine.start()
        try:
            first = asyncio.create_task(
                engine.submit(InferenceRequest(prompt="one two", max_tokens=1))
            )
            second = asyncio.create_task(
                engine.submit(InferenceRequest(prompt="three four", max_tokens=1))
            )
            first_result, second_result = await asyncio.gather(first, second)
        finally:
            await engine.stop()

        assert first_result.text == "a"
        assert second_result.text == "a"
        assert runner.encode_calls == 2
        assert engine.metrics.admission_rejections.value == 0
        status = engine.status()
        assert status["completed_requests"] == 2
        assert status["failed_requests"] == 0

    asyncio.run(scenario())


def test_engine_does_not_exhaust_admission_retries_while_active_request_can_finish() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(
            engine_overrides={
                "max_active_requests": 2,
                "admission_retry_limit": 1,
            }
        )
        runner.available_memory_bytes_value = 64
        await engine.start()
        try:
            first = asyncio.create_task(
                engine.submit(InferenceRequest(prompt="one two", max_tokens=2))
            )
            second = asyncio.create_task(
                engine.submit(InferenceRequest(prompt="three four", max_tokens=1))
            )
            first_result, second_result = await asyncio.gather(first, second)
        finally:
            await engine.stop()

        assert first_result.text == "ab"
        assert second_result.text == "a"
        assert runner.encode_calls == 2
        assert engine.metrics.admission_rejections.value == 0
        status = engine.status()
        assert status["completed_requests"] == 2
        assert status["failed_requests"] == 0

    asyncio.run(scenario())


def test_prefill_idle_fast_path_only_when_no_other_scheduler_work() -> None:
    engine, _runner = _make_engine()
    state = RequestState(
        request_id="prefill-target",
        request=InferenceRequest(prompt="ignored"),
        prompt_tokens=list(range(8)),
    )

    assert engine._prefill_chunk_target(state, remaining=7) == 7

    engine._prefill_queue.append("other-prefill")
    assert engine._prefill_chunk_target(state, remaining=7) == 2
    engine._prefill_queue.clear()

    engine._decode_queue.append("other-decode")
    assert engine._prefill_chunk_target(state, remaining=7) == 2
    engine._decode_queue.clear()

    engine._submission_queue.put_nowait(
        RequestState(request_id="other-submission", request=InferenceRequest(prompt="ignored"))
    )
    assert engine._prefill_chunk_target(state, remaining=7) == 2


def test_prefill_idle_fast_path_is_disabled_under_pressure_budget() -> None:
    engine, _runner = _make_engine()

    assert engine._can_finish_prefill_in_idle_step(remaining=7, budget=2) is True
    assert engine._can_finish_prefill_in_idle_step(remaining=7, budget=1) is False


def test_engine_rejects_admission_after_retry_limit() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(engine_overrides={"admission_retry_limit": 1})
        runner.available_memory_bytes_value = 16
        await engine.start()
        try:
            with pytest.raises(AsterError) as exc_info:
                await engine.submit(InferenceRequest(prompt="one", max_tokens=1))
        finally:
            await engine.stop()

        assert exc_info.value.code == "memory_pressure"
        assert exc_info.value.details == {
            "admission_retries": 1,
            "estimated_bytes": 24,
        }
        assert engine.metrics.admission_rejections.value == 1
        status = engine.status()
        assert status["failed_requests"] == 1
        assert status["admission_rejections"] == 1

    asyncio.run(scenario())


def test_engine_queue_full_rejection_is_counted() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine(api_overrides={"max_queue_depth": 1})
        engine._submission_queue.put_nowait(
            RequestState(request_id="already-queued", request=InferenceRequest(prompt="queued"))
        )

        with pytest.raises(AsterError) as exc_info:
            await engine.submit(InferenceRequest(prompt="overflow", trace_id="queue-overflow"))

        assert exc_info.value.code == "queue_full"
        assert exc_info.value.status_code == 503
        assert engine.metrics.admission_rejections.value == 1
        assert engine.metrics.errors.value == 1
        assert engine.metrics.queue_depth.value == 1
        assert engine.status()["admission_rejections"] == 1
        assert "queue-overflow" not in engine._requests

    asyncio.run(scenario())


def test_engine_rejects_duplicate_active_request_id_without_corrupting_original() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            first = asyncio.create_task(
                engine.submit(
                    InferenceRequest(
                        prompt="first duplicate",
                        max_tokens=1,
                        trace_id="duplicate-id",
                    )
                )
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and "duplicate-id" not in engine._requests:
                await asyncio.sleep(0.01)
            assert "duplicate-id" in engine._requests

            with pytest.raises(AsterError) as exc_info:
                await engine.submit(
                    InferenceRequest(
                        prompt="second duplicate",
                        max_tokens=1,
                        trace_id="duplicate-id",
                    )
                )

            first_result = await first
        finally:
            await engine.stop()

        assert exc_info.value.code == "duplicate_request_id"
        assert exc_info.value.status_code == 409
        assert exc_info.value.details == {"request_id": "duplicate-id"}
        assert first_result.request_id == "duplicate-id"
        assert first_result.text == "a"
        status = engine.status()
        assert status["completed_requests"] == 1
        assert status["failed_requests"] == 0
        assert status["cancelled_requests"] == 0
        assert "duplicate-id" not in engine._requests

    asyncio.run(scenario())


def test_engine_cancel_accepts_active_request_alias() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.decode_delay_seconds = 0.02
        await engine.start()
        try:
            stream = engine.stream(
                InferenceRequest(
                    prompt="cancel through response id",
                    max_tokens=8,
                    trace_id="trace-cancel-alias",
                    request_aliases=("chatcmpl-cancel-alias",),
                )
            )
            first = await anext(stream)
            assert first.token == "a"
            assert await engine.cancel("chatcmpl-cancel-alias") is True
            await stream.aclose()

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if (
                    "trace-cancel-alias" not in engine._requests
                    and "chatcmpl-cancel-alias" not in engine._request_aliases
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            await engine.stop()

        status = engine.status()
        assert status["cancelled_requests"] == 1
        assert "trace-cancel-alias" not in engine._requests
        assert "chatcmpl-cancel-alias" not in engine._request_aliases

    asyncio.run(scenario())


def test_engine_isolates_per_request_decode_failure_in_batch() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            ok = asyncio.create_task(
                engine.submit(
                    InferenceRequest(
                        prompt="one two",
                        max_tokens=2,
                        trace_id="decode-ok",
                    )
                )
            )
            failed = asyncio.create_task(
                engine.submit(
                    InferenceRequest(
                        prompt="three four",
                        max_tokens=2,
                        trace_id="decode-fails",
                    )
                )
            )
            ok_result, failed_result = await asyncio.gather(ok, failed, return_exceptions=True)
        finally:
            await engine.stop()

        assert not isinstance(ok_result, BaseException)
        assert ok_result.text == "ab"
        assert isinstance(failed_result, RuntimeError)
        assert str(failed_result) == "per-request decode failed"
        assert any(size >= 2 for size in runner.decode_batch_sizes)
        status = engine.status()
        assert status["completed_requests"] == 1
        assert status["failed_requests"] == 1

    asyncio.run(scenario())


def test_engine_applies_text_stop_sequences() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        await engine.start()
        try:
            result = await engine.submit(InferenceRequest(prompt="one two", max_tokens=4, stop="b"))
        finally:
            await engine.stop()

        assert result.text == "a"
        assert result.completion_tokens == 2

    asyncio.run(scenario())


def test_engine_applies_stop_token_ids() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        await engine.start()
        try:
            result = await engine.submit(
                InferenceRequest(prompt="one two", max_tokens=4, stop_token_ids=(ord("b"),))
            )
        finally:
            await engine.stop()

        assert result.text == "a"
        assert result.completion_tokens == 2
        assert result.finish_reason == "stop"

    asyncio.run(scenario())


def test_decode_memory_error_clears_runtime_cache_before_failing_request() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.decode_error = MemoryError("simulated oom")
        await engine.start()
        try:
            with pytest.raises(AsterError) as exc_info:
                await engine.submit(
                    InferenceRequest(prompt="oom during decode", max_tokens=2)
                )
        finally:
            await engine.stop()

        assert exc_info.value.code == "memory_pressure"
        assert runner.clear_runtime_cache_calls == 1
        status = engine.status()
        assert status["failed_requests"] == 1
        assert status["runtime_cache_clear_attempts"] == 1
        assert status["runtime_cache_clear_failures"] == 0

    asyncio.run(scenario())


def test_engine_rejects_requests_over_context_length_before_prefill() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(model_overrides={"context_length": 4})
        await engine.start()
        try:
            with pytest.raises(AsterError) as exc_info:
                await engine.submit(InferenceRequest(prompt="one two three", max_tokens=1))
        finally:
            await engine.stop()

        assert exc_info.value.code == "context_length_exceeded"
        assert exc_info.value.status_code == 400
        assert exc_info.value.details == {
            "context_length": 4,
            "prompt_tokens": 4,
            "max_tokens": 1,
            "requested_tokens": 5,
        }
        assert runner.prefill_calls == 0
        assert engine.status()["failed_requests"] == 1

    asyncio.run(scenario())


def test_engine_reuses_prefix_checkpoints_and_skips_prefill_work() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            first = await engine.submit(InferenceRequest(prompt="alpha beta gamma", max_tokens=1))
            first_prefill_calls = runner.prefill_calls
            second = await engine.submit(InferenceRequest(prompt="alpha beta gamma", max_tokens=1))
        finally:
            await engine.stop()

        assert first.text == "a"
        assert second.text == "a"
        assert runner.prefill_calls == first_prefill_calls
        status = engine.status()
        assert status["prefix_reuse_hits"] == 1
        assert status["prefix_tokens_reused"] > 0
        assert status["prefix_cache_stats"]["exact_hits"] == 1

    asyncio.run(scenario())


def test_engine_checkpoints_completed_prefill_chunks_at_cache_budget() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(
            engine_overrides={"snapshot_chunk_checkpoint_max_tokens": 4096}
        )
        state = RequestState(
            request_id="chunk-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5],
            cache_token_count=2,
            prompt_cache={"cache_tokens": 2},
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._maybe_checkpoint(state)
        finally:
            await engine.aclose()

        assert state.checkpoints_created == {3}
        assert runner.clone_cache_token_counts == [2]
        matched = engine.prefix_store.lookup(
            engine.settings.model.name,
            [1, 2, 3],
            model_fingerprint="fake-model",
        )
        assert matched is not None
        assert matched.prefix_token_count == 3
        assert matched.cache_token_count == 2

    asyncio.run(scenario())


def test_engine_disables_periodic_checkpoints_by_default() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        state = RequestState(
            request_id="default-no-chunk-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            cache_token_count=2,
            prompt_cache={"cache_tokens": 2},
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._maybe_checkpoint(state)
        finally:
            await engine.aclose()

        assert state.checkpoints_created == set()
        assert runner.clone_cache_token_counts == []

    asyncio.run(scenario())


def test_engine_limits_periodic_checkpoints_but_keeps_explicit_reuse_points() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(
            engine_overrides={"snapshot_chunk_checkpoint_max_tokens": 4}
        )
        state = RequestState(
            request_id="bounded-chunk-checkpoint",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6, 7],
            reuse_points=(6,),
            cache_token_count=5,
            prompt_cache={"cache_tokens": 5},
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        try:
            await engine._maybe_checkpoint(state)
        finally:
            await engine.aclose()

        assert state.checkpoints_created == {6}
        assert runner.clone_cache_token_counts == [5]

    asyncio.run(scenario())


def test_engine_prefill_targets_earliest_pending_reuse_point() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        state = RequestState(
            request_id="reuse-boundary",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            reuse_points=(5, 3),
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        engine._prefill_queue.append(state.request_id)
        try:
            await engine._step_prefill()
        finally:
            await engine.aclose()

        assert state.cache_token_count == 2
        assert state.checkpoints_created == {3}
        assert runner.clone_cache_token_counts == [2]
        matched = engine.prefix_store.lookup(
            engine.settings.model.name,
            [1, 2, 3],
            model_fingerprint="fake-model",
        )
        assert matched is not None
        assert matched.prefix_token_count == 3
        assert matched.cache_token_count == 2

    asyncio.run(scenario())


def test_engine_prefill_skips_reuse_points_below_snapshot_minimum() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(
            engine_overrides={"snapshot_min_prefix_tokens": 4}
        )
        state = RequestState(
            request_id="reuse-boundary-min",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            reuse_points=(3, 5),
            model_fingerprint="fake-model",
        )
        engine._requests[state.request_id] = state
        engine._prefill_queue.append(state.request_id)
        try:
            await engine._step_prefill()
        finally:
            await engine.aclose()

        assert state.cache_token_count == 4
        assert state.checkpoints_created == {5}
        assert runner.clone_cache_token_counts == [4]
        assert engine.prefix_store.lookup(
            engine.settings.model.name,
            [1, 2, 3],
            model_fingerprint="fake-model",
        ) is None
        matched = engine.prefix_store.lookup(
            engine.settings.model.name,
            [1, 2, 3, 4, 5],
            model_fingerprint="fake-model",
        )
        assert matched is not None
        assert matched.prefix_token_count == 5
        assert matched.cache_token_count == 4

    asyncio.run(scenario())


def test_engine_status_reports_live_request_details() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        engine._started_at = time.monotonic()
        state = RequestState(
            request_id="status-live",
            request=InferenceRequest(prompt="status live request", max_tokens=2),
            phase=RequestPhase.PREFILL_WAIT,
            prompt_tokens=[1, 2, 3, 4, 5, 6],
            reuse_points=(3, 5),
            cache_token_count=3,
            checkpoints_created={3},
            admission_prepared=True,
            estimated_bytes=48,
        )
        engine._requests[state.request_id] = state

        status = engine.status()
        assert status["model"] == "Qwen3.5-9B"
        assert status["num_running"] == 1
        assert status["num_waiting"] == 0
        assert status["steps_executed"] == status["prefill_steps"] + status["decode_steps"]
        assert status["total_requests_processed"] == 0
        assert status["recent_request_timelines"] == []
        assert status["decode_batch_diagnostics"] == {"batch_attempts": 0}
        assert status["engine_timing"]["prefill_model_tokens"] == 0
        assert status["engine_timing"]["decode_runner_batches"] == 0
        request_info = status["requests"][0]
        assert request_info["request_id"] == "status-live"
        assert request_info["status"] == "running"
        assert request_info["phase"] == "prefill"
        assert request_info["prompt_tokens"] == 6
        assert request_info["completion_tokens"] == 0
        assert request_info["max_tokens"] == 2
        assert request_info["progress"] == 0.0
        assert request_info["cache_hit_type"] == "miss"
        assert request_info["cached_tokens"] == 0
        assert request_info["cache_token_count"] == 3
        assert request_info["target_cache_token_count"] == 5
        assert request_info["prefill_remaining_tokens"] == 2
        assert request_info["reuse_points"] == [3, 5]
        assert request_info["next_reuse_point"] == 5
        assert request_info["checkpointed_reuse_points"] == [3]
        assert request_info["checkpoints_created_count"] == 1
        assert request_info["admission_retries"] == 0
        assert request_info["estimated_bytes"] == 48
        timeline = request_info["timeline"]
        assert timeline["request_id"] == "status-live"
        assert timeline["phase"] == "prefill"
        assert timeline["prefill_steps"] == 0
        assert timeline["decode_steps"] == 0

        assert await engine.cancel("status-live") is True

        final_status = engine.status()
        assert final_status["requests"] == []
        assert final_status["recent_request_timelines"][-1]["request_id"] == "status-live"
        assert final_status["recent_request_timelines"][-1]["status"] == "cancelled"
        assert final_status["total_requests_processed"] == 1
        assert final_status["total_prompt_tokens"] == 6
        assert final_status["total_completion_tokens"] == 0
        assert final_status["cancelled_requests"] == 1

    asyncio.run(scenario())


def test_cancel_during_prefill_keeps_request_accounted_until_runner_safe_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.inference.engine.bind_generation_streams", lambda: None)

    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            task = asyncio.create_task(
                engine.submit(
                    InferenceRequest(
                        prompt="cancel in flight",
                        max_tokens=4,
                        trace_id="cancel-in-flight",
                    )
                )
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not runner.prefill_started.is_set():
                await asyncio.sleep(0.01)
            assert runner.prefill_started.is_set()

            assert await engine.cancel("cancel-in-flight") is True
            status = engine.status()
            assert status["active_estimated_bytes"] > 0
            assert status["num_running"] == 1
            request_info = status["requests"][0]
            assert request_info["request_id"] == "cancel-in-flight"
            assert request_info["cancel_requested"] is True
            assert "cancel-in-flight" in engine._requests

            with pytest.raises(AsterError) as exc_info:
                await task
            assert exc_info.value.code == "request_cancelled"
        finally:
            await engine.stop()

        final_status = engine.status()
        assert final_status["active_estimated_bytes"] == 0
        assert final_status["requests"] == []
        assert final_status["cancelled_requests"] == 1
        assert "cancel-in-flight" not in engine._requests

    asyncio.run(scenario())


def test_engine_trims_checkpoint_clone_to_cache_token_count() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            await engine.submit(InferenceRequest(prompt="alpha beta gamma", max_tokens=1))
        finally:
            await engine.stop()

        assert runner.clone_cache_token_counts == [2, 3]

    asyncio.run(scenario())


def test_engine_cleans_up_cancelled_requests() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            task = asyncio.create_task(
                engine.submit(
                    InferenceRequest(prompt="cancel me now", max_tokens=4, trace_id="cancel-me"),
                )
            )
            await asyncio.sleep(0.01)
            cancelled = await engine.cancel("cancel-me")
            assert cancelled is True
            try:
                await task
            except AsterError as exc:
                assert exc.code == "request_cancelled"
            else:
                raise AssertionError("cancelled request unexpectedly completed")
        finally:
            await engine.stop()

        status = engine.status()
        assert status["cancelled_requests"] == 1
        assert "cancel-me" not in engine._requests

    asyncio.run(scenario())


def test_cancelled_prefill_checkpoint_is_reused_by_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aster.inference.engine.bind_generation_streams", lambda: None)

    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            task = asyncio.create_task(
                engine.submit(
                    InferenceRequest(
                        prompt="alpha",
                        max_tokens=4,
                        trace_id="cancel-prefill-cache",
                    ),
                )
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not runner.prefill_started.is_set():
                await asyncio.sleep(0.01)
            assert runner.prefill_started.is_set()
            assert await engine.cancel("cancel-prefill-cache") is True
            with pytest.raises(AsterError) as exc_info:
                await task
            assert exc_info.value.code == "request_cancelled"

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if engine.status()["cancelled_prefill_checkpoints"] == 1:
                    break
                await asyncio.sleep(0.01)

            first_prefill_calls = runner.prefill_calls
            result = await engine.submit(InferenceRequest(prompt="alpha", max_tokens=1))
        finally:
            await engine.stop()

        assert result.text == "a"
        assert runner.prefill_calls == first_prefill_calls
        status = engine.status()
        assert status["cancelled_prefill_checkpoints"] == 1
        assert status["prefix_reuse_hits"] == 1

    asyncio.run(scenario())


def test_submit_timeout_cancels_orphaned_non_stream_request() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(api_overrides={"request_timeout_seconds": 0.01})
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            with pytest.raises(AsterError) as exc_info:
                await engine.submit(
                    InferenceRequest(
                        prompt="timeout me during prefill",
                        max_tokens=4,
                        trace_id="timeout-me",
                    )
                )

            assert exc_info.value.code == "request_timeout"
            assert exc_info.value.status_code == 504

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "timeout-me" not in engine._requests:
                    break
                await asyncio.sleep(0.01)
        finally:
            await engine.stop()

        status = engine.status()
        assert status["cancelled_requests"] == 1
        assert status["timed_out_requests"] == 1
        assert status["completed_requests"] == 0
        assert engine.metrics.errors.value == 1
        assert "timeout-me" not in engine._requests

    asyncio.run(scenario())


def test_submit_uses_per_request_timeout() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(api_overrides={"request_timeout_seconds": 10.0})
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            with pytest.raises(AsterError) as exc_info:
                await engine.submit(
                    InferenceRequest(
                        prompt="timeout me with request budget",
                        max_tokens=4,
                        trace_id="request-timeout",
                        timeout_seconds=0.01,
                    )
                )

            assert exc_info.value.code == "request_timeout"
            assert exc_info.value.details == {"timeout_seconds": 0.01}
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and "request-timeout" in engine._requests:
                await asyncio.sleep(0.01)
        finally:
            await engine.stop()

        assert "request-timeout" not in engine._requests
        assert engine.status()["timed_out_requests"] == 1
        assert engine.metrics.errors.value == 1

    asyncio.run(scenario())


def test_engine_preserves_generated_whitespace_in_final_response() -> None:
    async def scenario() -> None:
        engine, _runner = _make_engine()
        future = asyncio.get_running_loop().create_future()
        state = RequestState(
            request_id="preserve-whitespace",
            request=InferenceRequest(prompt="ignored"),
            prompt_tokens=[1, 2],
            response_future=future,
            output_parts=[" leading", " text\n"],
            completion_tokens=2,
        )
        engine._requests[state.request_id] = state
        try:
            await engine._complete_request(state)
            result = await future
        finally:
            await engine.aclose()

        assert result.text == " leading text\n"

    asyncio.run(scenario())


def test_engine_runs_model_and_cache_work_on_single_runner_thread() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            result = await engine.submit(InferenceRequest(prompt="thread ownership", max_tokens=2))
        finally:
            await engine.stop()

        assert result.text == "ab"
        assert runner.thread_ids
        assert len(set(runner.thread_ids)) == 1

    asyncio.run(scenario())


def test_engine_binds_mlx_generation_streams_on_runner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_bind() -> None:
        calls.append(threading.get_ident())

    monkeypatch.setattr("aster.inference.engine.bind_generation_streams", fake_bind)

    async def scenario() -> None:
        engine, runner = _make_engine()
        await engine.start()
        try:
            result = await engine.submit(InferenceRequest(prompt="stream binding", max_tokens=2))
        finally:
            await engine.stop()

        assert result.text == "ab"
        assert calls
        assert runner.thread_ids
        assert set(calls) == set(runner.thread_ids)

    asyncio.run(scenario())


def test_stream_disconnect_cancels_orphaned_request() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine()
        runner.decode_delay_seconds = 0.02
        await engine.start()
        try:
            stream = engine.stream(
                InferenceRequest(
                    prompt="stream disconnect",
                    max_tokens=8,
                    trace_id="stream-disconnect",
                )
            )
            first = await anext(stream)
            assert first.token == "a"
            await stream.aclose()

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "stream-disconnect" not in engine._requests:
                    break
                await asyncio.sleep(0.01)
        finally:
            await engine.stop()

        status = engine.status()
        assert status["cancelled_requests"] == 1
        assert "stream-disconnect" not in engine._requests

    asyncio.run(scenario())


def test_stream_timeout_cancels_orphaned_request() -> None:
    async def scenario() -> None:
        engine, runner = _make_engine(api_overrides={"request_timeout_seconds": 10.0})
        runner.prefill_delay_seconds = 0.05
        await engine.start()
        try:
            stream = engine.stream(
                InferenceRequest(
                    prompt="stream timeout",
                    max_tokens=8,
                    trace_id="stream-timeout",
                    timeout_seconds=0.01,
                )
            )
            with pytest.raises(AsterError) as exc_info:
                await anext(stream)

            assert exc_info.value.code == "request_timeout"
            assert exc_info.value.details == {"timeout_seconds": 0.01}
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and "stream-timeout" in engine._requests:
                await asyncio.sleep(0.01)
            assert "stream-timeout" not in engine._requests
        finally:
            await engine.stop()

        status = engine.status()
        assert status["cancelled_requests"] == 1
        assert status["timed_out_requests"] == 1
        assert engine.metrics.errors.value == 1

    asyncio.run(scenario())


def test_engine_warmup_uses_strict_prefix_prompts(tmp_path: Path) -> None:
    warm_path = tmp_path / "warm-prompts.json"
    warm_path.write_text(
        json.dumps([[{"role": "system", "content": "shared persona"}]]),
    )

    async def scenario() -> None:
        engine, runner = _make_engine(
            {
                "warm_prompts_path": str(warm_path),
                "warm_prompts_max_tokens": 1,
            }
        )
        await engine.start()
        try:
            await engine.warmup()
            first_prefill_calls = runner.prefill_calls
            result = await engine.submit(
                InferenceRequest(prompt="warm strict prefix", max_tokens=1)
            )
        finally:
            await engine.stop()

        assert result.text == "a"
        assert runner.strict_prefix_calls == [[{"role": "system", "content": "shared persona"}]]
        assert runner.prefill_calls == first_prefill_calls
        assert engine.status()["prefix_reuse_hits"] >= 1

    asyncio.run(scenario())
