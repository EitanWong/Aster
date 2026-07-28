from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).with_name("candidate_benchmark.py")
    spec = importlib.util.spec_from_file_location("iter055_candidate_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_tokens_include_current_input_and_keep_latest_window() -> None:
    module = _load_module()
    item = SimpleNamespace(
        logits_processor_tokens=list(range(30)),
        input_token=30,
    )

    tokens = module._bounded_tokens(item, 20)

    assert tokens == list(range(11, 31))


def test_bounded_tokens_preserve_short_histories() -> None:
    module = _load_module()
    item = SimpleNamespace(logits_processor_tokens=[1, 2], input_token=3)

    assert module._bounded_tokens(item, 20) == [1, 2, 3]


def test_recent_processor_tokens_avoid_full_history_copy() -> None:
    module = _load_module()
    lane = SimpleNamespace(
        prompt_tokens=list(range(100)),
        output_tokens=list(range(100, 111)),
        input_token=110,
    )

    tokens, logical_source_tokens = module._recent_processor_tokens(lane, 20)

    assert tokens == list(range(91, 110))
    assert logical_source_tokens == 111


def test_recent_processor_tokens_exclude_initial_input_before_readding() -> None:
    module = _load_module()
    lane = SimpleNamespace(
        prompt_tokens=list(range(30)),
        output_tokens=[],
        input_token=29,
    )

    tokens, logical_source_tokens = module._recent_processor_tokens(lane, 20)

    assert tokens == list(range(10, 29))
    assert logical_source_tokens == 30


def test_descriptive_statistics_use_interpolated_p95() -> None:
    path = Path(__file__).with_name("descriptive_summary.py")
    spec = importlib.util.spec_from_file_location("iter055_descriptive_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._stats([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary == {
        "count": 5,
        "min": 1.0,
        "median": 3.0,
        "p95": 4.8,
        "max": 5.0,
        "population_stdev": 2.0**0.5,
    }


def test_no_regression_gate_checks_every_order_stratum() -> None:
    path = Path(__file__).with_name("admission.py")
    spec = importlib.util.spec_from_file_location("iter055_admission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cell = {
        "intervals": {
            "balanced": {"confidence_met": True, "lower": -0.2},
            "baseline_first": {"confidence_met": True, "lower": -0.9},
            "production_first": {"confidence_met": True, "lower": -1.01},
        }
    }

    assert module._intervals_clear_floor(cell, -1.0) is False
    cell["intervals"]["production_first"]["lower"] = -1.0
    assert module._intervals_clear_floor(cell, -1.0) is True
