from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/benchmark_shared_prefix_batch_attention.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260801-086-shared-prefix-batch-attention-feasibility/attention-screen-summary.json"
)


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_shared_prefix_batch_attention", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario(
    *,
    batch_size: int,
    p95_ratio: float,
    total_reduction: float,
    full_reduction: float,
    max_abs_difference: float = 0.001,
) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "total_tokens": 10_334,
        "correctness": {
            "max_abs_difference": max_abs_difference,
            "tolerance": 0.003,
        },
        "construction": {
            "materialized_batch_prefix": False,
            "native_merge_invoked": False,
            "total_merge_growth_reduction_percent": total_reduction,
            "full_attention_growth_reduction_percent": full_reduction,
        },
        "timing": {"candidate_over_native_p95_ratio": p95_ratio},
        "lifecycle": {
            "allocated_blocks_after_release": 0,
            "pool_nbytes_after_release": 0,
        },
    }


def test_integer_csv_parser_normalizes_and_rejects_ambiguous_values() -> None:
    tool = load_tool()

    assert tool.parse_integer_csv("8,2,4", minimum=2) == (2, 4, 8)
    with pytest.raises(tool.BenchmarkError, match="unique"):
        tool.parse_integer_csv("2,2", minimum=2)
    with pytest.raises(tool.BenchmarkError, match="at least 2"):
        tool.parse_integer_csv("1,2", minimum=2)
    with pytest.raises(tool.BenchmarkError, match="comma-separated integers"):
        tool.parse_integer_csv("2,eight", minimum=2)


def test_screen_summary_requires_every_predeclared_gate() -> None:
    tool = load_tool()
    passing = [
        _scenario(
            batch_size=batch_size,
            p95_ratio=1.02,
            total_reduction=86.7,
            full_reduction=99.9,
        )
        for batch_size in (2, 4, 8)
    ]

    summary = tool.summarize_screen(passing)

    assert summary["gates"] == {
        "numerical_contract": True,
        "no_batch_prefix_materialization": True,
        "b8_total_merge_growth_reduction_at_least_75_percent": True,
        "b8_full_attention_growth_reduction_at_least_90_percent": True,
        "p95_latency_no_regression_3_percent": True,
        "release_clean": True,
    }
    assert summary["decision"] == "screen-passed"

    failing = list(passing)
    failing[-1] = _scenario(
        batch_size=8,
        p95_ratio=1.031,
        total_reduction=86.7,
        full_reduction=99.9,
    )
    rejected = tool.summarize_screen(failing)

    assert rejected["gates"]["p95_latency_no_regression_3_percent"] is False
    assert rejected["decision"] == "screen-rejected"


def test_screen_summary_rejects_missing_b8_evidence() -> None:
    tool = load_tool()
    scenarios = [
        _scenario(
            batch_size=batch_size,
            p95_ratio=1.0,
            total_reduction=90.0,
            full_reduction=99.0,
        )
        for batch_size in (2, 4)
    ]

    with pytest.raises(tool.BenchmarkError, match="B8"):
        tool.summarize_screen(scenarios)


def test_retained_summary_recomputes_rejection_and_source_hashes() -> None:
    payload = json.loads(ARTIFACT_PATH.read_text())
    primary = payload["primary_screen"]["results"]
    memory = payload["locked_b8_memory"]
    confirmation = payload["confirmation"]

    recomputed_gates = {
        "numerical_contract": max(row["max_abs"] for row in primary) <= 3e-3,
        "no_batch_prefix_materialization": payload["candidate"]["production_routing_changed"]
        is False,
        "b8_total_merge_growth_reduction_at_least_75_percent": memory[
            "estimated_total_merge_growth_reduction_percent"
        ]
        >= 75.0,
        "b8_full_attention_growth_reduction_at_least_90_percent": memory[
            "estimated_full_attention_growth_reduction_percent"
        ]
        >= 90.0,
        "p95_latency_no_regression_3_percent": confirmation["median_of_process_p95_ratio"] <= 1.03,
        "release_clean": confirmation["all_release_clean"],
    }

    assert len(primary) == 9
    assert recomputed_gates == payload["gates"]
    assert payload["decision"]["status"] == "screen-rejected"
    assert payload["decision"]["run_locked_9b_ab"] is False
    for relative_path, expected in payload["source_sha256"].items():
        actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected
