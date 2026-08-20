from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "scripts/dev/benchmark_decode_observer.py"
FOUNDATION_TEST_PATH = PROJECT_ROOT / "tests/test_foundation_parity_benchmark.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260821-093-low-overhead-decode-stage-attribution"
    / "decode-stage-observer-sampled-matrix.json"
)


def load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_decode_observer", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_foundation_tests() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_foundation_parity_benchmark_fixture", FOUNDATION_TEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_order_is_balanced_per_cell() -> None:
    harness = load_harness()

    for cell in harness.CELLS:
        orders = [harness.state_order(cell, repetition) for repetition in range(1, 5)]
        assert orders.count(("observer-off", "observer-on")) == 2
        assert orders.count(("observer-on", "observer-off")) == 2


def _synthetic_rows() -> list[dict[str, object]]:
    harness = load_harness()
    foundation_tests = load_foundation_tests()
    foundation = harness.load_foundation()
    rows: list[dict[str, object]] = []
    for result in foundation_tests._matrix_rows(foundation):
        if result["cell"] not in harness.CELLS:
            continue
        result = json.loads(json.dumps(result))
        result["metrics"]["decode_driver_seconds"] = (
            result["metrics"]["completion_tokens"] / result["metrics"]["decode_driver_tps"]
        )
        result["execution"]["decode_stage_observer_max_events"] = 0
        result["execution"]["decode_stage_observer_sample_interval"] = 8
        result["lifecycle"] = {
            "terminal_clean": True,
            "decode_batch_diagnostics": {"batch_fallbacks": 0},
            "decode_stage_observer": {
                "sample_interval": 8,
                "sampled_steps": 1,
                "batch_steps": 1,
                "single_steps": 0,
                "dropped_events": 0,
                "events": [{"mode": "batch"}],
                "seconds": {"observed_total": 1.0},
            },
        }
        for state in harness.STATES:
            rows.append(
                {
                    "cell": result["cell"],
                    "repetition": result["repetition"],
                    "state": state,
                    "engine": result["engine"],
                    "source_path": f"/tmp/{result['cell']}-{result['engine']}-{state}.json",
                    "source_file_sha256": "source-file",
                    "result": json.loads(json.dumps(result)),
                }
            )
            rows[-1]["result"]["lifecycle"]["decode_stage_observer"]["events"] = (
                [] if state == "observer-off" else [{"mode": "batch"}]
            )
            rows[-1]["result"]["execution"]["decode_stage_observer_max_events"] = (
                0 if state == "observer-off" else 64
            )
    return rows


def test_summary_requires_exact_output_and_all_order_strata_under_one_percent() -> None:
    harness = load_harness()
    foundation = harness.load_foundation()

    payload = harness.summarize(
        foundation,
        _synthetic_rows(),
        [
            {"cell": cell, "repetition": repetition, "state": state, "engine": engine, "status": 0}
            for cell in harness.CELLS
            for repetition in range(1, 5)
            for state in harness.STATES
            for engine in harness.ENGINES
        ],
        repetitions=4,
        expected_sample_interval=8,
    )

    assert payload["measurement_status"] == "valid"
    assert payload["source_comparable"] is True
    assert payload["exact_output_identity_off_vs_on"] is True
    assert payload["terminal_clean"] is True
    assert payload["zero_decode_fallbacks"] is True
    assert payload["observer_contract"] is True
    assert payload["observer_off_empty"] is True
    assert payload["observer_on_bounded"] is True
    assert payload["observer_on_zero_drops"] is True
    assert payload["aster_no_op_gate_all_metrics_and_strata"] is True
    assert payload["control_engine_stable_all_metrics_and_strata"] is True
    assert payload["measurement_confounded_by_control_variance"] is False
    assert payload["no_op_gate_all_metrics_and_strata"] is True


def test_summary_rejects_state_output_drift() -> None:
    harness = load_harness()
    foundation = harness.load_foundation()
    rows = _synthetic_rows()
    target = next(row for row in rows if row["state"] == "observer-on")
    target["result"]["requests"][0]["text_sha256"] = "drift"
    collection = [
        {"cell": cell, "repetition": repetition, "state": state, "engine": engine, "status": 0}
        for cell in harness.CELLS
        for repetition in range(1, 5)
        for state in harness.STATES
        for engine in harness.ENGINES
    ]

    payload = harness.summarize(
        foundation, rows, collection, repetitions=4, expected_sample_interval=8
    )

    assert payload["exact_output_identity_off_vs_on"] is False
    assert payload["candidate_admitted"] is False


def test_retained_i093_artifact_recomputes_paired_matrix() -> None:
    harness = load_harness()
    foundation = harness.load_foundation()
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert payload["kind"] == "decode-stage-observer-sampled-matrix-evidence"
    assert payload["iteration"] == "ITER-20260821-093-low-overhead-decode-stage-attribution"
    assert len(payload["rows"]) == 32
    assert payload["candidate"]["sample_interval"] == 8
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["candidate_admitted"] is False
    assert payload["summary"]["exact_output_identity_off_vs_on"] is True
    assert payload["summary"]["source_comparable"] is True
    assert payload["summary"]["aster_no_op_gate_all_metrics_and_strata"] is False
    assert payload["summary"]["control_engine_stable_all_metrics_and_strata"] is False
    assert payload["summary"]["measurement_confounded_by_control_variance"] is True

    recomputed = harness.summarize(
        foundation,
        payload["rows"],
        payload["execution"]["collection_statuses"],
        repetitions=4,
        expected_sample_interval=8,
    )
    for key, value in recomputed.items():
        assert payload["summary"][key] == value

    for cell, expected_events in (("b4-short", 2), ("b4-mixed", 3)):
        observer_rows = [
            row
            for row in payload["rows"]
            if row["cell"] == cell and row["engine"] == "aster" and row["state"] == "observer-on"
        ]
        assert {
            len(row["result"]["lifecycle"]["decode_stage_observer"]["events"])
            for row in observer_rows
        } == {expected_events}


@pytest.mark.parametrize("repetitions", [2, 4])
def test_state_order_has_equal_strata(repetitions: int) -> None:
    harness = load_harness()

    for cell in harness.CELLS:
        assert (
            sum(
                harness.state_order(cell, repetition)[0] == "observer-off"
                for repetition in range(1, repetitions + 1)
            )
            == repetitions // 2
        )
