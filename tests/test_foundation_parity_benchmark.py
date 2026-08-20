from __future__ import annotations

import importlib.util
import json
import statistics
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
I091_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260820-091-decode-driver-attribution"
    / "decode-tensorized-logprobs-rejection.json"
)
I092_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260821-092-decode-driver-roofline-attribution"
    / "decode-stage-observer-rejection.json"
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


def test_execution_contract_keeps_tensorized_candidate_explicitly_off() -> None:
    tool = load_tool()

    contract = tool._execution_contract()

    assert contract["decode_tensorized_logprobs_enabled"] is False
    assert contract["decode_stage_observer_max_events"] == 0
    assert contract["decode_stage_observer_sample_interval"] == 1


def test_decode_stage_observer_sample_interval_is_recorded_in_contract() -> None:
    tool = load_tool()

    contract = tool._execution_contract()

    assert contract["decode_stage_observer_sample_interval"] == 1


def test_decode_stage_observer_delta_excludes_warmup_events() -> None:
    tool = load_tool()
    before = {
        "configured_max_events": 64,
        "batch_steps": 1,
        "single_steps": 2,
        "dropped_events": 0,
        "seconds": {
            "cache_prepare": 0.1,
            "model_enqueue": 0.2,
            "sampling_enqueue": 0.3,
            "evaluation_window": 0.4,
            "result_delivery": 0.5,
            "eager_completion": 0.0,
            "observed_total": 1.5,
        },
        "events": [{"path": "warmup-0"}, {"path": "warmup-1"}],
    }
    after = {
        "configured_max_events": 64,
        "batch_steps": 4,
        "single_steps": 3,
        "dropped_events": 1,
        "seconds": {
            "cache_prepare": 0.4,
            "model_enqueue": 0.6,
            "sampling_enqueue": 0.8,
            "evaluation_window": 1.0,
            "result_delivery": 1.2,
            "eager_completion": 0.0,
            "observed_total": 4.0,
        },
        "events": [
            {"path": "warmup-0"},
            {"path": "warmup-1"},
            {"path": "timed-0"},
            {"path": "timed-1"},
        ],
    }

    delta = tool._decode_stage_observer_delta(before, after)

    assert delta["batch_steps"] == 3
    assert delta["single_steps"] == 1
    assert delta["dropped_events"] == 1
    assert delta["events"] == [{"path": "timed-0"}, {"path": "timed-1"}]
    assert delta["seconds"] == pytest.approx(
        {
            "cache_prepare": 0.3,
            "model_enqueue": 0.4,
            "sampling_enqueue": 0.5,
            "evaluation_window": 0.6,
            "result_delivery": 0.7,
            "eager_completion": 0.0,
            "observed_total": 2.5,
        }
    )


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


def test_retained_foundation_parity_artifact_recomputes_and_binds_recorded_source() -> None:
    tool = load_tool()
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert payload["kind"] == "foundation-parity-evidence"
    assert payload["iteration"] == "ITER-20260820-090-qwen35-9b-foundation-parity"
    assert payload["model_fingerprint"] == {
        "model_sha256": "d77667c10dd92f5f94e7a2b3d290e411dd9564d88940a31286648cfa8b138b2a",
        "tokenizer_sha256": "94b66525e309d7ce24691be8194369f880e4f8a5ba82b726782e70fc97e1559e",
    }
    assert tool.summarize_matrix(payload["rows"], repetitions=4) == payload["summary"]

    common_hashes = set()
    benchmark_hashes = set()
    for row in payload["rows"]:
        assert len(row["source"]["common_source_sha256"]) == 64
        assert len(row["source"]["engine_source_sha256"]) == 64
        common_hashes.add(row["source"]["common_source_sha256"])
        benchmark_hashes.add(row["source"]["files"]["benchmark_foundation_parity.py"])
    assert len(common_hashes) == 1
    assert len(benchmark_hashes) == 1
    assert payload["source"]["benchmark_source_sha256"] == common_hashes.pop()


def test_i091_rejection_artifact_recomputes_balanced_primary_result() -> None:
    payload = json.loads(I091_ARTIFACT_PATH.read_text())

    assert payload["kind"] == "decode-tensorized-logprobs-rejection-evidence"
    assert payload["iteration"] == "ITER-20260820-091-decode-driver-attribution"
    assert payload["candidate"]["default_enabled"] is False
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["candidate_admitted"] is False
    assert payload["summary"]["decision"] == "reject-below-3-percent-and-resource-regression"

    rows = payload["rows"]
    assert len(rows) == 16
    assert [
        (entry["cell"], entry["repetition"], entry["first"], entry["second"])
        for entry in payload["execution"]["collection_sequence"]
    ] == [
        (
            cell,
            repetition,
            "baseline" if repetition % 2 else "candidate",
            "candidate" if repetition % 2 else "baseline",
        )
        for cell in ("b4-short", "b4-mixed")
        for repetition in range(1, 5)
    ]
    assert all(entry["statuses"] == [0, 0] for entry in payload["execution"]["collection_sequence"])
    assert {
        (row["result"]["cell"], row["result"]["repetition"], row["variant"]) for row in rows
    } == {
        (cell, repetition, variant)
        for cell in ("b4-short", "b4-mixed")
        for repetition in range(1, 5)
        for variant in ("baseline", "candidate")
    }

    expected_primary = {
        "b4-short": (54.10592946695242, 53.248829436424245),
        "b4-mixed": (33.85063472258099, 33.8590963262843),
    }
    for cell, (expected_baseline, expected_candidate) in expected_primary.items():
        by_variant = {
            variant: [
                row["result"]["metrics"]["decode_driver_tps"]
                for row in rows
                if row["result"]["cell"] == cell and row["variant"] == variant
            ]
            for variant in ("baseline", "candidate")
        }
        baseline = statistics.median(by_variant["baseline"])
        candidate = statistics.median(by_variant["candidate"])
        summary = payload["summary"]["cell_summaries"][cell]
        assert baseline == pytest.approx(expected_baseline)
        assert candidate == pytest.approx(expected_candidate)
        assert summary["medians"]["baseline"]["decode_driver_tps"] == pytest.approx(baseline)
        assert summary["medians"]["candidate"]["decode_driver_tps"] == pytest.approx(candidate)
        assert summary["relative_candidate_vs_baseline_ratio"][
            "decode_driver_tps"
        ] == pytest.approx(candidate / baseline - 1)
        assert summary["candidate_order_balance"] == {
            "baseline-first": 2,
            "candidate-first": 2,
        }
        assert summary["reproducible_primary_gain_at_least_3_percent"] is False

    gates = payload["summary"]["correctness_and_resource_gates"]
    assert gates["source_comparable"] is True
    assert gates["exact_output_identity"] is True
    assert gates["terminal_clean"] is True
    assert gates["zero_decode_fallbacks"] is True
    assert gates["candidate_path_exercised"] is True
    assert gates["baseline_path_inactive"] is True
    assert gates["zero_candidate_swap_growth"] is False
    assert gates["candidate_swap_growth_rows"] == [
        {
            "cell": "b4-mixed",
            "repetition": 3,
            "swap_delta_bytes": 317_587_456,
        }
    ]


def test_i092_stage_observer_artifact_recomputes_noop_rejection() -> None:
    payload = json.loads(I092_ARTIFACT_PATH.read_text())

    assert payload["kind"] == "decode-stage-observer-rejection-evidence"
    assert payload["iteration"] == "ITER-20260821-092-decode-driver-roofline-attribution"
    assert payload["candidate"]["default_max_events"] == 0
    assert payload["candidate"]["benchmark_max_events"] == 64
    assert payload["candidate"]["no_forced_evaluation"] is True
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["candidate_admitted"] is False
    assert payload["summary"]["decision"] == (
        "reject-observer-no-op-gate-resource-and-tail-regression"
    )

    rows = payload["rows"]
    assert len(rows) == 16
    assert {
        (row["cell"], row["repetition"], row["state"])
        for row in rows
    } == {
        (cell, repetition, state)
        for cell in ("b4-short", "b4-mixed")
        for repetition in range(1, 5)
        for state in ("observer-off", "observer-on")
    }
    assert all(status == 0 for status in payload["execution"]["statuses"])

    expected_primary = {
        "b4-short": (52.37678479191009, 51.820240558396776, -0.010625780786744254),
        "b4-mixed": (33.0196851026291, 31.43343696755776, -0.0480394688847301),
    }
    for cell, (expected_off, expected_on, expected_delta) in expected_primary.items():
        by_state = {
            state: [
                row["result"]["metrics"]["decode_driver_tps"]
                for row in rows
                if row["cell"] == cell and row["state"] == state
            ]
            for state in ("observer-off", "observer-on")
        }
        off = statistics.median(by_state["observer-off"])
        on = statistics.median(by_state["observer-on"])
        summary = payload["summary"]["cell_summaries"][cell]
        assert off == pytest.approx(expected_off)
        assert on == pytest.approx(expected_on)
        assert summary["medians"]["observer-off"]["decode_driver_tps"] == pytest.approx(off)
        assert summary["medians"]["observer-on"]["decode_driver_tps"] == pytest.approx(on)
        assert summary["relative_observer_on_vs_off_ratio"]["decode_driver_tps"] == pytest.approx(
            expected_delta
        )
        assert summary["sample_count_per_state"] == 4
        assert summary["observer_on_steps"]["dropped_events"] == [0]

    gates = payload["summary"]["correctness_and_resource_gates"]
    assert gates["source_comparable"] is True
    assert gates["input_manifest_comparable"] is True
    assert gates["exact_output_identity_off_vs_on"] is True
    assert gates["finish_identity_off_vs_on"] is True
    assert gates["terminal_clean"] is True
    assert gates["zero_decode_fallbacks"] is True
    assert gates["observer_off_has_zero_events"] is True
    assert gates["observer_on_event_bound"] is True
    assert gates["primary_decode_driver_no_op"] is False
    assert gates["tail_no_op"] is False
    assert gates["resource_no_op"] is False
    assert gates["all_no_op_gates"] is False
