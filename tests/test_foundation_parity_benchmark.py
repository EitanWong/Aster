from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/benchmark_foundation_parity.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260820-090-qwen35-9b-foundation-parity"
    / "foundation-parity-evidence.json"
)
CELLS = ("b1-short", "b1-long", "b4-short", "b4-mixed")
ENGINES = ("aster", "mlx-lm")


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_foundation_parity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workload() -> dict[str, object]:
    interactive = [
        {
            "workload_id": f"mt-bench:{81 + index}:turn-1",
            "max_tokens": 256,
            "scenario": {"family": "interactive"},
            "source": {"dataset": "mt-bench"},
        }
        for index in range(4)
    ]
    qmsum = [
        {
            "workload_id": "longbench:qmsum:one",
            "max_tokens": 64,
            "scenario": {"family": "long-context"},
            "source": {"dataset": "qmsum"},
        }
    ]
    return {"kind": "public-cross-engine-workload", "records": [*interactive, *qmsum]}


def _row(
    tool: ModuleType,
    *,
    cell: str,
    engine: str,
    repetition: int,
    input_sha256: str | None = None,
    terminal_clean: bool = True,
    finish_reason: str = "length",
    throughput: float | None = None,
    prefill_tps: float | None = None,
    decode_tps: float | None = None,
) -> dict[str, object]:
    concurrency = 1 if cell.startswith("b1-") else 4
    order = tool.engine_order_for_pair(cell, repetition)
    engine_position = order.index(engine)
    request_count = concurrency
    engine_scale = 0.90 if engine == "aster" else 1.0
    request_metrics = [
        {
            "key": f"request-{index}",
            "workload_id": f"public:{cell}:{index}",
            "prompt_tokens": 100 + index,
            "completion_tokens": 8,
            "finish_reason": finish_reason,
            "output_token_ids_sha256": f"tokens-{cell}-{index}",
            "text_sha256": f" text-{cell}-{index}",
            "ttft_seconds": 1.0 / engine_scale,
            "end_to_end_seconds": 2.0 / engine_scale,
        }
        for index in range(request_count)
    ]
    return {
        "schema_version": 1,
        "kind": "foundation-parity-cell-result",
        "cell": cell,
        "engine": engine,
        "repetition": repetition,
        "pair_order": list(order),
        "engine_position": engine_position,
        "source": {
            "workload_sha256": "workload-source",
            "source_lock_sha256": "lock-source",
            "model_sha256": "model-source",
            "tokenizer_sha256": "tokenizer-source",
            "common_source_sha256": "common-source",
            "engine_source_sha256": f"{engine}-source",
        },
        "plan_sha256": f"plan-{cell}",
        "input_manifest_sha256": input_sha256 or f"input-{cell}",
        "execution": {
            "max_output_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "warmup_requests": 1,
        },
        "requests": request_metrics,
        "metrics": {
            "prompt_tokens": sum(item["prompt_tokens"] for item in request_metrics),
            "completion_tokens": 8 * request_count,
            "prefill_model_tps": prefill_tps or 100.0 * engine_scale,
            "decode_driver_tps": decode_tps or 50.0 * engine_scale,
            "aggregate_generation_tps": throughput or 40.0 * engine_scale,
            "ttft_p50_seconds": 1.0 / engine_scale,
            "ttft_p95_seconds": 1.0 / engine_scale,
            "end_to_end_p50_seconds": 2.0 / engine_scale,
            "end_to_end_p95_seconds": 2.0 / engine_scale,
            "peak_mlx_memory_gb": 1.0,
            "peak_rss_bytes": 1_000_000,
            "swap_delta_bytes": 0,
        },
        "lifecycle": {
            "terminal_clean": terminal_clean,
            "completed_requests": request_count,
            "failed_requests": 0,
            "cancelled_requests": 0,
        },
        "contract": {"passed": terminal_clean and finish_reason == "length"},
    }


def _matrix_rows(tool: ModuleType) -> list[dict[str, object]]:
    return [
        _row(tool, cell=cell, engine=engine, repetition=repetition)
        for repetition in range(1, 5)
        for cell in CELLS
        for engine in ENGINES
    ]


def test_foundation_plan_freezes_b1_b4_and_mixed_public_cohorts() -> None:
    tool = load_tool()

    plans = {cell: tool.build_foundation_plan(_workload(), cell=cell) for cell in CELLS}

    assert [entry.workload_id for entry in plans["b1-short"].entries] == ["mt-bench:81:turn-1"]
    assert [entry.workload_id for entry in plans["b1-long"].entries] == ["longbench:qmsum:one"]
    assert [entry.workload_id for entry in plans["b4-short"].entries] == [
        f"mt-bench:{question}:turn-1" for question in range(81, 85)
    ]
    assert [entry.workload_id for entry in plans["b4-mixed"].entries] == [
        "longbench:qmsum:one",
        "mt-bench:81:turn-1",
        "mt-bench:82:turn-1",
        "mt-bench:83:turn-1",
    ]
    assert {entry.release for plan in plans.values() for entry in plan.entries} == {"at-start"}
    assert [plans[cell].concurrency for cell in CELLS] == [1, 1, 4, 4]
    assert {entry.max_tokens for plan in plans.values() for entry in plan.entries} == {8}


def test_engine_order_is_balanced_per_cell_across_four_repetitions() -> None:
    tool = load_tool()

    for cell in CELLS:
        orders = [tool.engine_order_for_pair(cell, repetition) for repetition in range(1, 5)]
        assert {order for order in orders} == {
            ("aster", "mlx-lm"),
            ("mlx-lm", "aster"),
        }
        assert sum(order[0] == "aster" for order in orders) == 2
        assert sum(order[0] == "mlx-lm" for order in orders) == 2


def test_summary_rejects_an_incomplete_matrix() -> None:
    tool = load_tool()
    rows = _matrix_rows(tool)

    with pytest.raises(tool.BenchmarkError, match="complete 32-row matrix"):
        tool.summarize_matrix(rows[:-1], repetitions=4)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("input", "input manifest"),
        ("terminal", "contract"),
        ("source", "source"),
        ("order", "order"),
    ],
)
def test_summary_rejects_comparability_drift(mutation: str, message: str) -> None:
    tool = load_tool()
    rows = _matrix_rows(tool)
    target = rows[-1]
    if mutation == "input":
        target["input_manifest_sha256"] = "drifted-input"
    elif mutation == "terminal":
        target["contract"] = {"passed": False}
    elif mutation == "source":
        target["source"] = {**target["source"], "model_sha256": "drifted-model"}
    else:
        target["pair_order"] = ["aster", "mlx-lm"]

    with pytest.raises(tool.BenchmarkError, match=message):
        tool.summarize_matrix(rows, repetitions=4)


def test_summary_selects_only_a_repeated_three_percent_owned_gap() -> None:
    tool = load_tool()
    rows = _matrix_rows(tool)

    summary = tool.summarize_matrix(rows, repetitions=4)

    assert summary["contracts_passed"] is True
    assert summary["source_comparable"] is True
    assert summary["input_comparable"] is True
    assert summary["order_balanced"] is True
    assert summary["cross_engine_terminal_identity"] is True
    assert summary["priority_gap"]["owner"] == "aster-manual-decode-driver"
    assert summary["priority_gap"]["metric"] == "decode_driver_tps"
    assert set(summary["priority_gap"]["qualifying_cells"]) == set(CELLS)
    assert summary["decision"] == "select-decode-driver-profile-for-i091"


def test_sub_three_percent_noise_does_not_select_a_candidate() -> None:
    tool = load_tool()
    rows = _matrix_rows(tool)
    for row in rows:
        engine_scale = 0.98 if row["engine"] == "aster" else 1.0
        row["metrics"] = {
            **row["metrics"],
            "prefill_model_tps": 100.0 * engine_scale,
            "decode_driver_tps": 50.0 * engine_scale,
            "aggregate_generation_tps": 40.0 * engine_scale,
            "ttft_p50_seconds": 1.0 / engine_scale,
            "ttft_p95_seconds": 1.0 / engine_scale,
            "end_to_end_p50_seconds": 2.0 / engine_scale,
            "end_to_end_p95_seconds": 2.0 / engine_scale,
        }

    summary = tool.summarize_matrix(rows, repetitions=4)

    assert summary["priority_gap"] is None
    assert summary["decision"] == "baseline-only-no-reproducible-3-percent-gap"


def test_stable_cross_engine_cohort_drift_is_reported_without_hiding_metrics() -> None:
    tool = load_tool()
    rows = _matrix_rows(tool)
    for row in rows:
        if row["cell"] == "b4-mixed" and row["engine"] == "mlx-lm":
            row["requests"][0]["output_token_ids_sha256"] = "cohort-native-output"

    summary = tool.summarize_matrix(rows, repetitions=4)

    assert summary["cross_engine_output_identity"] is False
    assert summary["cross_engine_output_divergences"] == {"b4-mixed": ["request-0"]}
    assert summary["cross_engine_terminal_identity"] is True
    assert summary["priority_gap"]["metric"] == "decode_driver_tps"


def test_retained_foundation_parity_artifact_recomputes_and_binds_current_source() -> None:
    tool = load_tool()
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert payload["kind"] == "foundation-parity-evidence"
    assert payload["iteration"] == "ITER-20260820-090-qwen35-9b-foundation-parity"
    assert payload["model_fingerprint"] == {
        "model_sha256": "d77667c10dd92f5f94e7a2b3d290e411dd9564d88940a31286648cfa8b138b2a",
        "tokenizer_sha256": "94b66525e309d7ce24691be8194369f880e4f8a5ba82b726782e70fc97e1559e",
    }
    assert tool.summarize_matrix(payload["rows"], repetitions=4) == payload["summary"]

    source_hashes = {engine: tool._source_hashes(engine) for engine in ("aster", "mlx-lm")}
    for row in payload["rows"]:
        expected = source_hashes[row["engine"]]
        assert row["source"]["common_source_sha256"] == expected["common_source_sha256"]
        assert row["source"]["engine_source_sha256"] == expected["engine_source_sha256"]
        assert row["source"]["files"] == expected["files"]
    assert (
        payload["source"]["benchmark_source_sha256"]
        == source_hashes["aster"]["common_source_sha256"]
    )
