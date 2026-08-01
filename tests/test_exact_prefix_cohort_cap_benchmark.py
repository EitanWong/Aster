from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/benchmark_exact_prefix_cohort_cap.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260801-087-exact-prefix-active-cohort-cap/active-cohort-cap-summary.json"
)


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_exact_prefix_cohort_cap", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    lane: str,
    replicate: int,
    order: str,
    peak_mlx_gb: float,
    replay_tps: float,
    p95_ttft_s: float,
    p95_latency_s: float,
    max_latency_s: float,
    active_cache_equivalents: int,
    contract_passed: bool = True,
) -> dict[str, object]:
    return {
        "lane": lane,
        "replicate": replicate,
        "order": order,
        "plan_sha256": "plan",
        "workload_sha256": "workload",
        "model": "test-model",
        "max_decode_batch": 4,
        "configured_max_active_requests": 16 if lane == "baseline" else 4,
        "observed_peak_submitted_requests": 7,
        "observed_peak_active_cache_equivalents": active_cache_equivalents,
        "peak_mlx_memory_gb": peak_mlx_gb,
        "aggregate_replay_tps": replay_tps,
        "replay_p95_ttft_seconds": p95_ttft_s,
        "replay_p95_latency_seconds": p95_latency_s,
        "replay_max_latency_seconds": max_latency_s,
        "contract_passed": contract_passed,
    }


def _passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in range(1, 6):
        order = "baseline-first" if replicate % 2 else "candidate-first"
        rows.extend(
            (
                _row(
                    lane="baseline",
                    replicate=replicate,
                    order=order,
                    peak_mlx_gb=10.6,
                    replay_tps=100.0,
                    p95_ttft_s=4.0,
                    p95_latency_s=6.0,
                    max_latency_s=6.5,
                    active_cache_equivalents=7,
                ),
                _row(
                    lane="candidate",
                    replicate=replicate,
                    order=order,
                    peak_mlx_gb=9.0,
                    replay_tps=98.0,
                    p95_ttft_s=4.05,
                    p95_latency_s=6.05,
                    max_latency_s=6.55,
                    active_cache_equivalents=4,
                ),
            )
        )
    return rows


def test_matrix_summary_requires_every_predeclared_gate() -> None:
    tool = load_tool()
    passing = tool.summarize_matrix(_passing_rows())

    assert passing["decision"] == "screen-passed"
    assert all(passing["gates"].values())
    assert passing["paired"]["aggregate_replay_tps_ratio_median"] == pytest.approx(0.98)
    assert passing["lanes"]["baseline"]["peak_mlx_memory_gb_median"] == pytest.approx(10.6)
    assert passing["lanes"]["candidate"]["peak_mlx_memory_gb_median"] == pytest.approx(9.0)

    failing = _passing_rows()
    for row in failing:
        if row["lane"] == "candidate":
            row["aggregate_replay_tps"] = 96.0
    rejected = tool.summarize_matrix(failing)

    assert rejected["gates"]["aggregate_replay_tps_no_regression_3_percent"] is False
    assert rejected["decision"] == "screen-rejected"


def test_matrix_summary_rejects_incomplete_or_mismatched_pairs() -> None:
    tool = load_tool()
    rows = _passing_rows()

    with pytest.raises(tool.BenchmarkError, match="five complete pairs"):
        tool.summarize_matrix(rows[:-1])

    rows[1]["plan_sha256"] = "different"
    with pytest.raises(tool.BenchmarkError, match="plan"):
        tool.summarize_matrix(rows)


def test_cancellation_summary_requires_clean_follow_up() -> None:
    tool = load_tool()
    payload = {
        "execution": {"max_active_requests": 4},
        "resources": {
            "swap_delta_bytes": 0,
            "engine_lifecycle_sampling": {
                "final": {
                    "active_requests": 0,
                    "pending_requests": 0,
                    "prefill_requests": 0,
                    "decode_requests": 0,
                    "failed_requests": 0,
                    "cancelled_requests": 1,
                    "completed_requests": 1,
                    "prefix_cache": {"pinned_entries": 0, "pinned_bytes": 0},
                }
            },
        },
        "result": {
            "cancel_accepted": True,
            "events": [
                {"key": "long-primary", "error": {"code": "request_cancelled"}},
                {
                    "key": "cancel-follow-up",
                    "error": None,
                    "response": {"finish_reason": "length", "completion_tokens": 8},
                    "timeline": {"decode_steps": 8},
                },
            ],
        },
    }

    compact = tool.compact_cancellation(payload)

    assert compact["passed"] is True
    payload["result"]["cancel_accepted"] = False
    assert tool.compact_cancellation(payload)["passed"] is False


def test_retained_summary_recomputes_gates_and_source_hashes() -> None:
    tool = load_tool()
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert tool.summarize_matrix(payload["rows"]) == payload["matrix"]
    assert payload["decision"] == {
        "change_production_defaults": False,
        "run_conditional_scheduler_implementation": True,
        "status": "screen-passed",
    }
    assert payload["post_pass"]["candidate_cancellation_cleanup"]["passed"] is True
    assert len(payload["raw_sha256"]) == 11
    assert all(len(value) == 64 for value in payload["raw_sha256"].values())
    for relative_path, expected in payload["source_sha256"].items():
        actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected
