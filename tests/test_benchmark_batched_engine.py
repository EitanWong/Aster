from __future__ import annotations

import asyncio

import pytest

from aster.core.config import RuntimeSettings
from aster.inference.contracts import InferenceRequest
from scripts.dev.benchmark_batched_engine import (
    _apply_benchmark_overrides,
    _build_structured_workload,
    _requests_for_workload,
    _submit_requests,
)


def test_structured_workload_attaches_schema_to_every_request() -> None:
    requests = _build_structured_workload(2, temperature=0.0)

    assert len(requests) == 2
    assert all(request.structured_output_schema is not None for request in requests)
    assert {request.temperature for request in requests} == {0.0}


def test_requests_for_workload_delegates_standard_workloads() -> None:
    requests = _requests_for_workload("reuse", 2, temperature=0.0, long_prompt_words=8)

    assert len(requests) == 2
    assert requests[0].prompt == requests[1].prompt


def test_submit_requests_preserves_sequential_order() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.completed: list[str] = []

        async def submit(self, request: InferenceRequest) -> object:
            self.completed.append(request.trace_id or "")
            return object()

    async def scenario() -> None:
        engine = FakeEngine()
        requests = [
            InferenceRequest(prompt="a", trace_id="one"),
            InferenceRequest(prompt="b", trace_id="two"),
        ]

        await _submit_requests(engine, requests, staggered=False, sequential=True)  # type: ignore[arg-type]

        assert engine.completed == ["one", "two"]

    asyncio.run(scenario())


def test_benchmark_overrides_lane_limit_without_mutating_settings() -> None:
    settings = RuntimeSettings()

    overridden = _apply_benchmark_overrides(
        settings,
        concurrency_levels=[2, 4],
        prefix_cache_enabled=False,
        max_lanes=2,
        lane_admission_window_ms=200.0,
    )

    assert settings.engine.batch_generator_max_lanes == 1
    assert overridden.engine.batch_generator_max_lanes == 2
    assert overridden.engine.batch_generator_lane_admission_window_ms == 200.0
    assert overridden.engine.prefix_cache_enabled is False
    assert overridden.engine.max_active_requests >= 4

    with pytest.raises(ValueError):
        _apply_benchmark_overrides(
            settings,
            concurrency_levels=[2],
            prefix_cache_enabled=False,
            max_lanes=2,
        )
