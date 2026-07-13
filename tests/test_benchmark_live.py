from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scripts.dev.benchmark_live import (
    _build_workload,
    _collect_runtime_metadata,
    _extract_staggered_latency_metrics,
    _engine_timing_memory_gb,
    _max_response_peak_memory_gb,
    _prefix_cache_counter_delta,
    _run_requests,
)
from aster.inference.contracts import InferenceRequest


def test_build_workload_uses_explicit_sampling_temperature() -> None:
    requests = _build_workload("mixed", 2, temperature=0.0)

    assert requests
    assert {request.temperature for request in requests} == {0.0}


def test_reuse_workload_uses_exact_duplicate_prompts() -> None:
    requests = _build_workload("reuse", 2, temperature=0.0)

    assert len({request.prompt for request in requests}) == 1


def test_divergent_reuse_workload_has_a_cacheable_shared_prefix() -> None:
    requests = _build_workload("reuse-divergent", 2, temperature=0.0)
    first_prompt = requests[0].prompt or ""
    second_prompt = requests[1].prompt or ""
    shared_prefix_length = 0
    for first, second in zip(first_prompt, second_prompt, strict=False):
        if first != second:
            break
        shared_prefix_length += 1

    assert shared_prefix_length > 256


def test_collect_runtime_metadata_reports_reproducibility_fields() -> None:
    metadata = _collect_runtime_metadata()

    assert metadata["platform"]
    assert metadata["python_version"]
    assert metadata["system_memory_total_bytes"] > 0
    assert metadata["swap_used_bytes"] >= 0


def test_build_staggered_workload_starts_with_long_prompt() -> None:
    requests = _build_workload("staggered", 4, temperature=0.0)

    assert len(requests) == 4
    assert requests[0].trace_id == "staggered-long"
    assert len(requests[0].prompt or "") > len(requests[1].prompt or "")


def test_build_long_workload_accepts_configured_prompt_length() -> None:
    requests = _build_workload("long", 1, temperature=0.0, long_prompt_words=12)

    assert (requests[0].prompt or "").split().count("section") == 12


def test_extract_staggered_latency_metrics_separates_long_and_short_requests() -> None:
    timelines = [
        {"request_id": "staggered-long", "total_latency_s": 4.0},
        {"request_id": "staggered-short-1", "total_latency_s": 1.0},
        {"request_id": "staggered-short-2", "total_latency_s": 2.0},
    ]

    metrics = _extract_staggered_latency_metrics(timelines)

    assert metrics == {
        "long_request_latency_seconds": 4.0,
        "short_request_p95_latency_seconds": 2.0,
    }


def test_prefix_cache_counter_delta_reads_nested_status_stats() -> None:
    before = {"prefix_cache_stats": {"exact_hits": 1}}
    after = {"prefix_cache_stats": {"exact_hits": 3}}

    assert _prefix_cache_counter_delta(before, after, "exact_hits") == 2


def test_max_response_peak_memory_uses_allocator_metric() -> None:
    responses = [
        SimpleNamespace(peak_memory_gb=6.25),
        SimpleNamespace(peak_memory_gb=7.5),
    ]

    assert _max_response_peak_memory_gb(responses) == 7.5


def test_engine_timing_memory_reads_prefill_metrics() -> None:
    status = {"engine_timing": {"max_prefill_peak_memory_gb": 9.25}}

    assert _engine_timing_memory_gb(status, "max_prefill_peak_memory_gb") == 9.25


def test_run_requests_can_serialize_prefix_reuse_requests() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.completed: list[str] = []

        async def submit(self, request: InferenceRequest) -> object:
            self.completed.append(request.trace_id or "")
            return object()

    async def scenario() -> None:
        engine = FakeEngine()
        requests = [
            InferenceRequest(prompt="shared", trace_id="reuse-0"),
            InferenceRequest(prompt="shared", trace_id="reuse-1"),
        ]

        await _run_requests(engine, requests, sequential=True)

        assert engine.completed == ["reuse-0", "reuse-1"]

    asyncio.run(scenario())
