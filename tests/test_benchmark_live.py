from __future__ import annotations

from scripts.dev.benchmark_live import (
    _build_workload,
    _collect_runtime_metadata,
    _extract_staggered_latency_metrics,
)


def test_build_workload_uses_explicit_sampling_temperature() -> None:
    requests = _build_workload("mixed", 2, temperature=0.0)

    assert requests
    assert {request.temperature for request in requests} == {0.0}


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
