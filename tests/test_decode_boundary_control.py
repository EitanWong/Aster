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
I096_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260824-096-host-state-trace"
    / "host-state-trace.json"
)


def load_control() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_decode_boundary_control", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_observer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_decode_observer_for_control", OBSERVER_PATH
    )
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
        row for row in foundation_tests._matrix_rows(foundation) if row["cell"] in control.CELLS
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
                "state_first": control.control_order(row["cell"], int(row["repetition"]))[0],
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
        row for row in foundation_tests._matrix_rows(foundation) if row["cell"] in control.CELLS
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
                "state_first": control.control_order(row["cell"], int(row["repetition"]))[0],
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
        row for row in foundation_tests._matrix_rows(foundation) if row["cell"] in control.CELLS
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
                "state_first": control.control_order(row["cell"], int(row["repetition"]))[0],
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


def _valid_telemetry() -> dict[str, object]:
    snapshot = {
        "schema_version": 1,
        "process": {"pid": 42},
        "system": {
            "memory_total_bytes": 100,
            "memory_available_bytes": 80,
            "memory_available_percent": 80.0,
            "swap_used_bytes": 10,
        },
    }
    return {
        "enabled": True,
        "host_state_before": snapshot,
        "host_state_after": snapshot,
        "process": {
            "status": "complete",
            "sample_count": 2,
            "rss_before_bytes": 10,
            "peak_rss_bytes": 12,
            "cpu_percent_avg": 50.0,
            "cpu_percent_max": 60.0,
            "system_cpu_percent_avg": 25.0,
            "system_cpu_percent_max": 30.0,
            "system_available_memory_min_bytes": 75,
            "system_available_memory_min_percent": 75.0,
            "system_swap_used_max_bytes": 10,
            "load_average_one_min_max": 1.0,
        },
        "thermal_power": {
            "schema_version": 1,
            "probes": {
                "powermetrics": {"status": "unavailable"},
                "pmset_thermal": {"status": "unavailable"},
                "memory_pressure": {"status": "available"},
            },
        },
    }


def test_summary_requires_a_complete_telemetry_envelope_when_requested() -> None:
    control = load_control()
    observer = load_observer()
    foundation = observer.load_foundation()
    foundation_tests = load_foundation_tests()
    base_rows = [
        row for row in foundation_tests._matrix_rows(foundation) if row["cell"] in control.CELLS
    ]
    for row in base_rows:
        row["metrics"]["decode_driver_seconds"] = 0.16
        row["metrics"]["swap_delta_bytes"] = 0.0
        row["lifecycle"]["mlx_allocator"] = {
            "before_timed": {
                "active_memory_bytes": 100,
                "cache_memory_bytes": 20,
                "peak_memory_bytes": 0,
            },
            "after_timed": {
                "active_memory_bytes": 110,
                "cache_memory_bytes": 30,
                "peak_memory_bytes": 120,
            },
        }
    baseline_with_telemetry = [
        {
            "cell": row["cell"],
            "engine": row["engine"],
            "repetition": row["repetition"],
            "state": "observer-off",
            "state_first": control.control_order(row["cell"], int(row["repetition"]))[0],
            "telemetry": _valid_telemetry(),
            "result": row,
        }
        for row in base_rows
    ]
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
                "state_first": control.control_order(row["cell"], int(row["repetition"]))[0],
                "telemetry": _valid_telemetry(),
                "result": row,
            }
        )

    payload = control.summarize_control(
        foundation,
        baseline_with_telemetry,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
        require_telemetry=True,
    )

    assert payload["measurement_status"] == "valid"
    assert payload["telemetry_contract"] is True
    assert payload["allocator_contract"] is True
    assert payload["host_state_diagnostics"]["pair_count"] == 16

    baseline_with_telemetry[0]["result"]["contract"]["passed"] = False
    rejected = control.summarize_control(
        foundation,
        baseline_with_telemetry,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
        require_telemetry=True,
    )
    assert rejected["measurement_status"] == "invalid-contract"
    baseline_with_telemetry[0]["result"]["contract"]["passed"] = True

    control_rows[0]["result"]["metrics"]["swap_delta_bytes"] = 1
    rejected = control.summarize_control(
        foundation,
        baseline_with_telemetry,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
        require_telemetry=True,
    )
    assert rejected["measurement_status"] == "invalid-contract"
    control_rows[0]["result"]["metrics"]["swap_delta_bytes"] = 0

    control_rows[0].pop("telemetry")
    rejected = control.summarize_control(
        foundation,
        baseline_with_telemetry,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
        require_telemetry=True,
    )
    assert rejected["measurement_status"] == "invalid-contract"
    assert rejected["telemetry_contract"] is False

    control_rows[0]["telemetry"] = _valid_telemetry()
    control_rows[0]["result"]["lifecycle"].pop("mlx_allocator")
    rejected = control.summarize_control(
        foundation,
        baseline_with_telemetry,
        control_rows,
        repetitions=4,
        expected_max_output_tokens=8,
        require_telemetry=True,
    )
    assert rejected["measurement_status"] == "invalid-contract"
    assert rejected["allocator_contract"] is False


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


def test_retained_i096_artifact_recomputes_host_state_matrix() -> None:
    control = load_control()
    payload = json.loads(I096_ARTIFACT_PATH.read_text())
    foundation = control.load_foundation()

    recomputed = control.summarize_control(
        foundation,
        payload["paired_baseline_rows"],
        payload["paired_control_rows"],
        repetitions=4,
        expected_max_output_tokens=32,
        require_telemetry=True,
        observer_reference_rows=payload["observer_reference_rows"],
    )

    assert payload["kind"] == "decode-boundary-host-state-evidence"
    assert payload["iteration"] == "ITER-20260824-096-host-state-trace"
    assert len(payload["rows"]) == 32
    assert recomputed == payload["summary"]
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["control_contract"] is True
    assert payload["summary"]["telemetry_contract"] is True
    assert payload["summary"]["allocator_contract"] is True
    assert payload["summary"]["host_state_diagnostics"]["pair_count"] == 16
    assert payload["summary"]["control_stable_primary_and_strata"] is False
    assert all(row["result"]["contract"]["passed"] for row in payload["rows"])
    assert all(row["result"]["execution"]["warmup_requests"] > 0 for row in payload["rows"])
    assert all(row["result"]["metrics"]["swap_delta_bytes"] == 0 for row in payload["rows"])
