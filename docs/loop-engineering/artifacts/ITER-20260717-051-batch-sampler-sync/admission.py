#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import strict_aggregate

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
RESULTS_DIR = ARTIFACT_DIR / "results"
STRICT_COMPONENTS = {
    "short": ("strict-final-v7-short-r18", 180),
    "long_screen": ("strict-final-v7-long-r18", 72),
    "long_greedy_b2_focus": (
        "strict-final-v7-long-greedy-b2-r18-s1024",
        18,
    ),
    "sustained_mixed_b8": (
        "strict-final-v7-stress-mixed-b8-r18-s1024",
        18,
    ),
}
STRUCTURED_RESULT = "structured-final-v7-stop-validation.json"
NEGATIVE_EVIDENCE = (
    "production-confirmation/aggregate.json",
    "production-long-confirmation/aggregate.json",
    "structured-stop-validation-lane0-failed.json",
    "strict-final-short-n9/execution-manifest.json",
    "strict-fix-screen-v2/execution-manifest.json",
    "strict-eager-row-screen-v3/execution-manifest.json",
    "runner-balanced-screen-v4/execution-manifest.json",
)


def _current_files(hashes: dict[str, str]) -> bool:
    return all(
        (path := PROJECT_ROOT / relative).is_file()
        and strict_aggregate.legacy._sha256(path) == expected
        for relative, expected in hashes.items()
    )


def _cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(cell["passed"]),
        "balanced_interval_lower_percent": float(
            cell["intervals"]["balanced"]["lower"]
        ),
        "baseline_first_interval_lower_percent": float(
            cell["intervals"]["baseline_first"]["lower"]
        ),
        "production_first_interval_lower_percent": float(
            cell["intervals"]["production_first"]["lower"]
        ),
        "failed_gates": sorted(
            name for name, passed in cell["gates"].items() if not passed
        ),
    }


def _load_strict(name: str, expected_records: int) -> tuple[dict[str, Any], dict[str, Any]]:
    root = RESULTS_DIR / name
    manifest_path = root / "execution-manifest.json"
    aggregate_path = root / "strict-aggregate.json"
    manifest, payloads = strict_aggregate.legacy._load(manifest_path)
    archived = json.loads(aggregate_path.read_text())
    recomputed = strict_aggregate.aggregate(manifest_path)
    if recomputed != archived:
        raise ValueError(f"strict aggregate does not recompute: {name}")
    if not payloads:
        raise ValueError(f"strict component has no payloads: {name}")

    component = {
        "result_directory": name,
        "execution_manifest_sha256": strict_aggregate.legacy._sha256(manifest_path),
        "strict_aggregate_sha256": strict_aggregate.legacy._sha256(aggregate_path),
        "records": int(archived["records"]),
        "expected_records": expected_records,
        "unique_pids": int(archived["unique_pids"]),
        "all_cells_passed": bool(archived["all_cells_passed"]),
        "measurement_sources_current": _current_files(payloads[0]["source_sha256"]),
        "model_inputs_current": _current_files(payloads[0]["model_input_sha256"]),
        "cells": {
            key: _cell_summary(cell) for key, cell in archived["cells"].items()
        },
    }
    return archived, component


def _structured_component() -> dict[str, Any]:
    path = RESULTS_DIR / STRUCTURED_RESULT
    payload = json.loads(path.read_text())
    swap_delta = int(payload["memory"]["swap_after_bytes"]) - int(
        payload["memory"]["swap_before_bytes"]
    )
    return {
        "result": STRUCTURED_RESULT,
        "result_sha256": strict_aggregate.legacy._sha256(path),
        "all_schema_valid": bool(payload["all_schema_valid"]),
        "all_stopped_before_limit": bool(payload["all_stopped_before_limit"]),
        "membership_shrank": len(set(payload["membership_sizes"])) > 1,
        "swap_delta_bytes": swap_delta,
        "swap_non_growth": swap_delta <= 0,
        "sources_current": _current_files(payload["source_sha256"]),
        "model_inputs_current": _current_files(payload["model_input_sha256"]),
    }


def build() -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {}
    components: dict[str, dict[str, Any]] = {}
    for alias, (name, expected_records) in STRICT_COMPONENTS.items():
        aggregates[alias], components[alias] = _load_strict(name, expected_records)

    structured = _structured_component()
    long_screen = aggregates["long_screen"]["cells"]
    long_weak = long_screen["greedy-b2"]
    expected_screen_failure = {
        "production_first_median_interval_clears_floor",
        "order_strata_speed_floor_required",
    }
    observed_screen_failure = {
        name for name, passed in long_weak["gates"].items() if not passed
    }
    long_other_cells_pass = all(
        bool(cell["passed"])
        for name, cell in long_screen.items()
        if name != "greedy-b2"
    )
    long_screen_is_resolvable = (
        not bool(long_weak["passed"])
        and bool(long_weak["intervals"]["balanced"]["lower_meets_speed_floor"])
        and observed_screen_failure == expected_screen_failure
    )

    gates = {
        "short_strict_matrix_pass": bool(aggregates["short"]["all_cells_passed"]),
        "long_nonweak_cells_pass": long_other_cells_pass,
        "long_greedy_b2_focus_pass": bool(
            aggregates["long_greedy_b2_focus"]["all_cells_passed"]
        ),
        "long_screen_failure_resolved": (
            long_screen_is_resolvable
            and bool(aggregates["long_greedy_b2_focus"]["all_cells_passed"])
        ),
        "sustained_stress_pass": bool(
            aggregates["sustained_mixed_b8"]["all_cells_passed"]
        ),
        "all_measurement_sources_current": all(
            component["measurement_sources_current"]
            for component in components.values()
        )
        and structured["sources_current"],
        "all_model_inputs_current": all(
            component["model_inputs_current"] for component in components.values()
        )
        and structured["model_inputs_current"],
        "all_expected_record_counts": all(
            component["records"] == component["expected_records"]
            and component["unique_pids"] == component["records"]
            for component in components.values()
        ),
        "structured_stop_aware_valid": all(
            structured[name]
            for name in (
                "all_schema_valid",
                "all_stopped_before_limit",
                "membership_shrank",
                "swap_non_growth",
                "sources_current",
                "model_inputs_current",
            )
        ),
        "negative_evidence_retained": all(
            (RESULTS_DIR / relative).is_file() for relative in NEGATIVE_EVIDENCE
        ),
    }
    return {
        "schema_version": 2,
        "admission_source_sha256": strict_aggregate.legacy._sha256(
            Path(__file__).resolve()
        ),
        "strict_aggregate_source_sha256": strict_aggregate.legacy._sha256(
            Path(strict_aggregate.__file__).resolve()
        ),
        "components": components,
        "structured_stop_aware": structured,
        "gates": gates,
        "admitted": all(gates.values()),
    }


def main() -> None:
    payload = build()
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output = RESULTS_DIR / "final-admission.json"
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
