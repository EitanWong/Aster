from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_record(common: ModuleType) -> dict[str, object]:
    samples = {
        "baseline": [1.0, 2.0, 3.0],
        "candidate": [1.5, 2.5, 3.5],
    }
    return {
        "warmups": 2,
        "iterations": 3,
        "samples_ms": samples,
        "methods": {name: common.summarize(values) for name, values in samples.items()},
    }


@pytest.mark.parametrize(
    ("relative_path", "baseline", "candidate"),
    [
        ("pure-mlx/aggregate.py", "separate", "combined"),
        ("reference-vllm-metal/aggregate.py", "mlx_scatter", "fused_primitive"),
        ("aster-layout/aggregate.py", "mlx_scatter", "fused_primitive"),
    ],
)
def test_point_delta_preserves_process_pairing(
    relative_path: str,
    baseline: str,
    candidate: str,
) -> None:
    module = load_module(relative_path, relative_path.replace("/", "_"))
    baseline_medians = [1.0, 2.0, 3.0, 100.0, 101.0]
    candidate_medians = [100.0, 200.0, 3.0, 100.0, 101.0]
    records = [
        {
            "samples_ms": {
                baseline: [baseline_median] * 3,
                candidate: [candidate_median] * 3,
            }
        }
        for baseline_median, candidate_median in zip(
            baseline_medians,
            candidate_medians,
            strict=True,
        )
    ]

    assert module.paired_point_delta(records, baseline, candidate) == 0.0


def test_measurement_validation_rejects_sample_count_drift() -> None:
    common = load_module("aggregate_common.py", "aggregate_common_sample_count")
    record = valid_record(common)
    record["samples_ms"]["candidate"].pop()  # type: ignore[index]

    with pytest.raises(ValueError, match="sample count"):
        common.validate_measurements(
            record,
            frozenset({"baseline", "candidate"}),
            expected_warmups=2,
            expected_iterations=3,
        )


def test_measurement_validation_rejects_summary_drift() -> None:
    common = load_module("aggregate_common.py", "aggregate_common_summary")
    record = deepcopy(valid_record(common))
    record["methods"]["candidate"]["median_ms"] = 999.0  # type: ignore[index]

    with pytest.raises(ValueError, match="summary"):
        common.validate_measurements(
            record,
            frozenset({"baseline", "candidate"}),
            expected_warmups=2,
            expected_iterations=3,
        )


def test_cell_validation_rejects_duplicate_records() -> None:
    common = load_module("aggregate_common.py", "aggregate_common_cells")

    with pytest.raises(ValueError, match="Duplicate"):
        common.validate_cells([1, 1, 2], frozenset({1, 2}))
