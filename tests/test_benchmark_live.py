from __future__ import annotations

from scripts.dev.benchmark_live import _build_workload, _collect_runtime_metadata


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
