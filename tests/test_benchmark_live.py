from __future__ import annotations

from scripts.dev.benchmark_live import _build_workload


def test_build_workload_uses_explicit_sampling_temperature() -> None:
    requests = _build_workload("mixed", 2, temperature=0.0)

    assert requests
    assert {request.temperature for request in requests} == {0.0}
