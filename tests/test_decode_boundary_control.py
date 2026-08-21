from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/benchmark_decode_boundary_control.py"
OBSERVER_PATH = PROJECT_ROOT / "scripts/dev/benchmark_decode_observer.py"
FOUNDATION_TEST_PATH = PROJECT_ROOT / "tests/test_foundation_parity_benchmark.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260823-095-decode-boundary-control"
    / "decode-boundary-control.json"
)


def load_control() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_decode_boundary_control", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_observer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_decode_observer_for_control", OBSERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_foundation_tests() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_foundation_parity_for_control", FOUNDATION_TEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_control_order_is_balanced_per_cell() -> None:
    control = load_control()

    for cell in control.CELLS:
        orders = [control.control_order(cell, repetition) for repetition in range(1, 5)]
        assert orders.count(("observer-off", "control-off")) == 2
        assert orders.count(("control-off", "observer-off")) == 2


def test_summary_accepts_identical_off_control_rows() -> None:
    control = load_control()
    observer = load_observer()
    foundation = observer.load_foundation()
    foundation_tests = load_foundation_tests()
    base_rows = [
        row
        for row in foundation_tests._matrix_rows(foundation)
        if row["cell"] in control.CELLS
    ]
    for row in base_rows:
        row["metrics"]["decode_driver_seconds"] = 0.16
        row["metrics"]["swap_delta_bytes"] = 0.0
    control_rows = []
    for base in base_rows:
        row = json.loads(json.dumps(base))
        row["state"] = "control-off"
        control_rows.append(
            {
                "cell": row["cell"],
                "engine": row["engine"],
                "repetition": row["repetition"],
                "state": "control-off",
                "state_first": control.control_order(
                    row["cell"], int(row["repetition"])
                )[0],
                "result": row,
            }
        )

    payload = control.summarize_control(
        foundation,
        base_rows,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
    )

    assert payload["measurement_status"] == "valid"
    assert payload["control_contract"] is True
    assert payload["exact_output_identity_off_vs_control"] is True
    assert payload["control_stable_primary_and_strata"] is True
    assert payload["candidate_admitted"] is False


def test_summary_rejects_a_material_control_drift() -> None:
    control = load_control()
    observer = load_observer()
    foundation = observer.load_foundation()
    foundation_tests = load_foundation_tests()
    base_rows = [
        row
        for row in foundation_tests._matrix_rows(foundation)
        if row["cell"] in control.CELLS
    ]
    for row in base_rows:
        row["metrics"]["decode_driver_seconds"] = 0.16
        row["metrics"]["swap_delta_bytes"] = 0.0
    control_rows = []
    for base in base_rows:
        row = json.loads(json.dumps(base))
        row["state"] = "control-off"
        if row["cell"] == "b4-mixed" and row["engine"] == "aster":
            row["metrics"]["decode_driver_tps"] *= 1.02
        control_rows.append(
            {
                "cell": row["cell"],
                "engine": row["engine"],
                "repetition": row["repetition"],
                "state": "control-off",
                "state_first": control.control_order(
                    row["cell"], int(row["repetition"])
                )[0],
                "result": row,
            }
        )

    payload = control.summarize_control(
        foundation,
        base_rows,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
    )

    assert payload["measurement_status"] == "valid"
    assert payload["control_stable_primary_and_strata"] is False
    assert payload["measurement_confounded_by_control_variance"] is True
    assert payload["candidate_admitted"] is False


def test_summary_rejects_control_output_cap_mismatch() -> None:
    control = load_control()
    observer = load_observer()
    foundation = observer.load_foundation()
    foundation_tests = load_foundation_tests()
    base_rows = [
        row
        for row in foundation_tests._matrix_rows(foundation)
        if row["cell"] in control.CELLS
    ]
    for row in base_rows:
        row["metrics"]["decode_driver_seconds"] = 0.16
        row["metrics"]["swap_delta_bytes"] = 0.0
    control_rows = []
    for base in base_rows:
        row = json.loads(json.dumps(base))
        row["state"] = "control-off"
        row["execution"]["max_output_tokens"] = 32
        control_rows.append(
            {
                "cell": row["cell"],
                "engine": row["engine"],
                "repetition": row["repetition"],
                "state": "control-off",
                "state_first": control.control_order(
                    row["cell"], int(row["repetition"])
                )[0],
                "result": row,
            }
        )

    with pytest.raises(ValueError, match="output cap"):
        control.summarize_control(
            foundation,
            base_rows,
            control_rows,
            repetitions=4,
            expected_max_output_tokens=8,
        )


def test_retained_i095_artifact_recomputes_control_matrix() -> None:
    control = load_control()
    payload = json.loads(ARTIFACT_PATH.read_text())
    foundation = control.load_foundation()

    recomputed = control.summarize_control(
        foundation,
        payload["observer_rows"],
        payload["rows"],
        repetitions=4,
        expected_max_output_tokens=32,
    )

    assert payload["kind"] == "decode-boundary-control-evidence"
    assert payload["iteration"] == "ITER-20260823-095-decode-boundary-control"
    assert len(payload["rows"]) == 16
    assert recomputed == payload["summary"]
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["control_contract"] is True
    assert payload["summary"]["prewarm_contract"] is True
    assert payload["summary"]["control_stable_primary_and_strata"] is False
    assert payload["summary"]["measurement_confounded_by_control_variance"] is True
