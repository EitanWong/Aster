from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/benchmark_active_cap_frontier.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260801-088-active-cap-workload-frontier"
    / "active-cap-frontier-evidence.json"
)
CAPS = (2, 3, 4, 5, 6, 16)
WORKLOADS = ("exact-long", "short-simultaneous", "mixed")


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_active_cap_frontier", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    workload: str,
    cap: int,
    *,
    peak_mlx_gb: float,
    throughput: float,
    ttft: float,
    latency: float,
    max_latency: float,
) -> dict[str, object]:
    return {
        "workload": workload,
        "cap": cap,
        "plan_sha256": f"plan-{workload}",
        "workload_sha256": "workload-source",
        "model": "test-model",
        "max_decode_batch": 4,
        "peak_mlx_memory_gb": peak_mlx_gb,
        "aggregate_tps": throughput,
        "p95_ttft_seconds": ttft,
        "p95_latency_seconds": latency,
        "max_latency_seconds": max_latency,
        "output_fingerprints": {"request-0": ["tokens", "text", "length"]},
        "contract_passed": True,
    }


def _pilot_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for cap in CAPS:
            if cap == 16:
                values = (10.0, 100.0, 4.0, 6.0, 6.5)
            elif cap == 4:
                values = (8.5, 102.0, 3.9, 5.9, 6.4)
            elif cap < 4:
                values = (7.5, 90.0, 5.0, 7.0, 7.5)
            else:
                values = (9.5, 96.0, 4.2, 6.3, 6.8)
            rows.append(
                _record(
                    workload,
                    cap,
                    peak_mlx_gb=values[0],
                    throughput=values[1],
                    ttft=values[2],
                    latency=values[3],
                    max_latency=values[4],
                )
            )
    return rows


def _workload() -> dict[str, object]:
    interactive = [
        {
            "workload_id": f"short-{index}",
            "max_tokens": 8,
            "scenario": {"family": "interactive"},
            "source": {"dataset": "mt-bench"},
        }
        for index in range(8)
    ]
    qmsum = [
        {
            "workload_id": f"qmsum-{index}",
            "max_tokens": 8,
            "scenario": {"family": "long-context"},
            "source": {"dataset": "qmsum"},
        }
        for index in range(2)
    ]
    return {"kind": "public-cross-engine-workload", "records": [*interactive, *qmsum]}


def _diagnostic(
    cap: int,
    *,
    output_hash: str,
    divergent_token: int,
    divergent_mode: str,
) -> dict[str, object]:
    prefix = [271, 12646, 25, 357, 2526, 2923]
    selected = [*prefix, divergent_token, 999]
    return {
        "kind": "active-cap-greedy-logit-diagnostic",
        "performance_measurement_valid": False,
        "cap": cap,
        "target_request_id": "public-arrival:mixed-short-3",
        "candidate_token_ids": [364, 421, 8574],
        "output_token_ids_sha256": output_hash,
        "text_sha256": f"text-{output_hash}",
        "request_contract_passed": True,
        "cohorts": [
            {
                "mode": divergent_mode if index == 6 else "batch",
                "completion_tokens": index,
                "request_ids": ["public-arrival:mixed-short-3"],
            }
            for index in range(8)
        ],
        "trace": [
            {
                "completion_tokens": index,
                "input_token": 13 if index == 0 else selected[index - 1],
                "selected_token": token,
                "candidate_logits": {
                    "364": 21.0 if token == 364 else 20.875,
                    "421": 21.0 if token == 421 else 20.875,
                    "8574": 20.875,
                },
            }
            for index, token in enumerate(selected)
        ],
    }


def test_mixed_plan_contains_exact_distinct_and_short_followups() -> None:
    tool = load_tool()

    plan = tool.build_frontier_plan(_workload(), workload="mixed")

    assert plan.scenario == "active-cap-mixed"
    assert plan.concurrency == 8
    assert len(plan.entries) == 8
    assert [entry.key for entry in plan.entries] == [
        "long-primary",
        "mixed-exact-0",
        "mixed-exact-1",
        "mixed-distinct-0",
        "mixed-short-0",
        "mixed-short-1",
        "mixed-short-2",
        "mixed-short-3",
    ]
    assert {entry.depends_on for entry in plan.entries[1:]} == {"long-primary"}


def test_pilot_summary_selects_only_cross_workload_eligible_cap() -> None:
    tool = load_tool()

    summary = tool.summarize_pilot(_pilot_rows())

    assert summary["contracts_passed"] is True
    assert summary["global_eligible_caps"] == [4]
    assert summary["confirmation_caps"] == [4, 16]
    assert summary["decision"] == "confirm-global-candidate"


def test_pilot_summary_rejects_incomplete_grid() -> None:
    tool = load_tool()
    rows = _pilot_rows()

    with pytest.raises(tool.BenchmarkError, match="complete 18-row grid"):
        tool.summarize_pilot(rows[:-1])


def test_pilot_summary_retains_output_drift_as_no_go_evidence() -> None:
    tool = load_tool()
    rows = _pilot_rows()

    rows[0]["output_fingerprints"] = {"request-0": ["drift", "text", "length"]}
    summary = tool.summarize_pilot(rows)

    assert summary["cell_contracts_passed"] is True
    assert summary["cross_cap_output_consistent"] is False
    assert summary["contracts_passed"] is False
    assert summary["performance_global_eligible_caps"] == [4]
    assert summary["global_eligible_caps"] == []
    assert summary["confirmation_caps"] == []
    assert summary["diagnostic_caps"] == [2, 16]
    assert summary["decision"] == "reject-output-drift"
    exact = summary["output_consistency"]["exact-long"]
    assert exact == {
        "consistent": False,
        "baseline_cap": 16,
        "divergent_caps": [2],
        "divergent_keys": {"request-0": [2]},
    }


def test_diagnostic_summary_confirms_batch_shape_near_tie() -> None:
    tool = load_tool()
    diagnostics = [
        _diagnostic(2, output_hash="single", divergent_token=364, divergent_mode="single"),
        _diagnostic(3, output_hash="batch", divergent_token=421, divergent_mode="batch"),
        _diagnostic(5, output_hash="single", divergent_token=364, divergent_mode="single"),
        _diagnostic(16, output_hash="batch", divergent_token=421, divergent_mode="batch"),
    ]

    summary = tool.summarize_diagnostics(diagnostics)

    assert summary["contracts_passed"] is True
    assert summary["output_groups"] == [
        {"caps": [3, 16], "output_token_ids_sha256": "batch"},
        {"caps": [2, 5], "output_token_ids_sha256": "single"},
    ]
    assert summary["shared_selected_prefix"] == [271, 12646, 25, 357, 2526, 2923]
    assert summary["first_divergent_completion_index"] == 6
    assert summary["divergent_step"]["2"]["mode"] == "single"
    assert summary["divergent_step"]["3"]["mode"] == "batch"
    assert summary["diagnosis"] == "batch-shape-sensitive-near-tie"


def test_retained_evidence_is_source_bound() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())

    assert artifact["kind"] == "active-cap-workload-frontier-evidence"
    assert artifact["pilot"]["decision"] == "reject-output-drift"
    assert artifact["diagnostics"]["diagnosis"] == "batch-shape-sensitive-near-tie"
    assert len(artifact["raw_sha256"]) == 18
    assert len(artifact["diagnostic_sha256"]) == 4
    for relative_path, expected_sha256 in artifact["source_sha256"].items():
        actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_sha256
