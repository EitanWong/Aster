from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module() -> ModuleType:
    path = Path(__file__).with_name("candidate_benchmark.py")
    spec = importlib.util.spec_from_file_location("iter052_candidate_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Row:
    def __sub__(self, other: Any) -> tuple[_Row, Any]:
        return self, other


class _MX:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, bool]] = []

    def logsumexp(self, row: Any, *, axis: int, keepdims: bool) -> str:
        self.calls.append((row, axis, keepdims))
        return "normalizer"


class _Metrics:
    direct_logit_rows = 0
    normalized_rows = 0


def test_marked_greedy_sampler_receives_raw_logits() -> None:
    module = _load_module()
    sampler = lambda value: value  # noqa: E731
    setattr(sampler, module.SAMPLER_ACCEPTS_LOGITS, True)
    mx = _MX()
    row = _Row()
    metrics = _Metrics()

    result = module._sampler_input(mx, row, sampler, metrics)

    assert result is row
    assert mx.calls == []
    assert metrics.direct_logit_rows == 1
    assert metrics.normalized_rows == 0


def test_unmarked_sampler_preserves_logprob_normalization() -> None:
    module = _load_module()
    mx = _MX()
    row = _Row()
    metrics = _Metrics()

    result = module._sampler_input(mx, row, lambda value: value, metrics)

    assert result == (row, "normalizer")
    assert mx.calls == [(row, -1, True)]
    assert metrics.direct_logit_rows == 0
    assert metrics.normalized_rows == 1


def test_profile_percentile_is_distribution_free() -> None:
    module = _load_module()
    profile_path = Path(__file__).with_name("operator_profile.py")
    spec = importlib.util.spec_from_file_location("iter052_operator_profile", profile_path)
    assert spec is not None and spec.loader is not None
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)

    assert profile._percentile([1.0, 2.0, 4.0, 8.0], 0.5) == 3.0
    assert profile._summary([])["count"] == 0
    assert module.SAMPLER_ACCEPTS_LOGITS.endswith("logits")


def test_aggregate_keeps_cells_and_exactness() -> None:
    aggregate_path = Path(__file__).with_name("aggregate.py")
    spec = importlib.util.spec_from_file_location("iter052_aggregate", aggregate_path)
    assert spec is not None and spec.loader is not None
    aggregate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aggregate)
    payload = {
        "workload": "greedy",
        "batch_size": 2,
        "timings": {
            "baseline_step_seconds": [2.0, 2.0],
            "production_step_seconds": [1.0, 1.0],
        },
        "parity": {"exact_token_text_cache": True},
        "policy_metrics": {"production": {"direct_logit_rows": 4, "normalized_rows": 0}},
        "memory": {"swap_before_bytes": 10, "swap_after_bytes": 10},
    }

    result = aggregate.summarize([payload])

    assert result["record_count"] == 1
    cell = result["cells"]["greedy-b2"]
    assert cell["speed_percent"]["median"] == 100.0
    assert cell["all_exact_token_text_cache"] is True
