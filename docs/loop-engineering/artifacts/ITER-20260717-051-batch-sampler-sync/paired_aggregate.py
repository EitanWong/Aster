#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _speed_floor(workload: str) -> float:
    return 0.0 if workload == "structured" else 3.0


def _block_speedups(
    baseline: list[float],
    production: list[float],
    block_size: int,
) -> list[float]:
    if len(baseline) != len(production) or len(baseline) % block_size:
        raise ValueError("paired timings must have equal complete blocks")
    return [
        _percent_change(
            sum(baseline[start : start + block_size]),
            sum(production[start : start + block_size]),
        )
        for start in range(0, len(baseline), block_size)
    ]


def _order_speedup(
    baseline: list[float],
    production: list[float],
    first_policies: list[str],
    first_policy: str,
) -> float:
    indices = [
        index for index, observed in enumerate(first_policies) if observed == first_policy
    ]
    if not indices:
        raise ValueError(f"missing {first_policy}-first timing stratum")
    return _percent_change(
        sum(baseline[index] for index in indices),
        sum(production[index] for index in indices),
    )


def _bootstrap_median_ci(
    values: list[float],
    *,
    seed: int,
    samples: int = 20_000,
) -> tuple[float, float]:
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    last = len(medians) - 1
    return medians[int(last * 0.025)], medians[int(last * 0.975)]


def _exact_process_median_resample_ci(values: list[float]) -> tuple[float, float]:
    medians = sorted(
        statistics.median(sample)
        for sample in itertools.product(values, repeat=len(values))
    )
    last = len(medians) - 1
    return medians[int(last * 0.025)], medians[int(last * 0.975)]


def _load(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    payloads: list[dict[str, Any]] = []
    for record in manifest["records"]:
        output = ARTIFACT_DIR / record["output"]
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in payload["source_sha256"].items()
        ):
            raise ValueError(f"source hash mismatch: {output}")
        payloads.append(payload)
    return manifest, payloads


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest, payloads = _load(manifest_path)
    expected = len(manifest["matrix"]["cells"]) * int(manifest["matrix"]["runs"])
    if len(payloads) != expected:
        raise ValueError(f"expected {expected} payloads, found {len(payloads)}")
    if len({int(payload["pid"]) for payload in payloads}) != expected:
        raise ValueError("paired payload PIDs are not unique")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[(str(payload["workload"]), int(payload["batch_size"]))].append(payload)

    cells: dict[str, Any] = {}
    all_runs_passed = True
    cell_speedups: list[float] = []
    for descriptor in manifest["matrix"]["cells"]:
        key = (str(descriptor["workload"]), int(descriptor["batch_size"]))
        runs: list[dict[str, Any]] = []
        for payload in sorted(grouped[key], key=lambda item: int(item["run_id"])):
            baseline = [float(value) for value in payload["timings"]["baseline_step_seconds"]]
            production = [
                float(value) for value in payload["timings"]["production_step_seconds"]
            ]
            block_size = int(payload["settings"]["block_size"])
            block_speedups = _block_speedups(baseline, production, block_size)
            first_policies = payload["timings"].get("first_policy_by_step")
            if first_policies is None:
                first_policies = [
                    "baseline" if step % 2 == 0 else "production"
                    for step in range(len(baseline))
                ]
            baseline_first = _order_speedup(
                baseline, production, first_policies, "baseline"
            )
            production_first = _order_speedup(
                baseline, production, first_policies, "production"
            )
            first_counts = {
                policy: first_policies.count(policy)
                for policy in ("baseline", "production")
            }
            lower, upper = _bootstrap_median_ci(
                block_speedups,
                seed=20260718 + int(payload["run_id"]),
            )
            end_to_end = _percent_change(sum(baseline), sum(production))
            median = statistics.median(block_speedups)
            positive_fraction = sum(value > 0.0 for value in block_speedups) / len(
                block_speedups
            )
            speed_floor = _speed_floor(key[0])
            positive_floor = 0.5 if key[0] == "structured" else 0.75
            gate = {
                "exact_token_text_cache_parity": bool(
                    payload["parity"]["exact_token_text_cache"]
                ),
                "swap_zero": (
                    int(payload["memory"]["swap_after_bytes"])
                    - int(payload["memory"]["swap_before_bytes"])
                    == 0
                ),
                "end_to_end_speedup_meets_workload_floor": end_to_end >= speed_floor,
                "block_median_speedup_meets_workload_floor": median >= speed_floor,
                "block_resample_lower_meets_core_floor": (
                    key[0] == "structured" or lower >= speed_floor
                ),
                "positive_block_fraction_meets_workload_floor": (
                    positive_fraction >= positive_floor
                ),
                "balanced_ab_ba_order": (
                    first_counts["baseline"] == first_counts["production"]
                ),
            }
            passed = all(gate.values())
            all_runs_passed &= passed
            runs.append(
                {
                    "run_id": int(payload["run_id"]),
                    "end_to_end_speedup_percent": end_to_end,
                    "block_speedup_percent_median": median,
                    "block_resample_stability_95_percent": [lower, upper],
                    "positive_block_fraction": positive_fraction,
                    "baseline_first_speedup_percent": baseline_first,
                    "production_first_speedup_percent": production_first,
                    "order_interaction_percentage_points": abs(
                        baseline_first - production_first
                    ),
                    "first_policy_counts": first_counts,
                    "structured_document_observed_during_fixed_steps": (
                        payload["parity"].get("structured_schema_valid")
                    ),
                    "block_speedups_percent": block_speedups,
                    "gate": gate,
                    "passed": passed,
                }
            )
        median_speedup = statistics.median(
            run["end_to_end_speedup_percent"] for run in runs
        )
        process_speedups = [run["end_to_end_speedup_percent"] for run in runs]
        process_lower, process_upper = _exact_process_median_resample_ci(
            process_speedups
        )
        process_gate = process_lower >= _speed_floor(key[0])
        all_runs_passed &= process_gate
        cell_speedups.append(median_speedup)
        cells[f"{key[0]}-b{key[1]}"] = {
            "end_to_end_speedup_percent_median": median_speedup,
            "independent_process_speedups_percent": process_speedups,
            "independent_process_median_resample_95_percent": [
                process_lower,
                process_upper,
            ],
            "minimum_independent_process_speedup_percent": min(process_speedups),
            "all_independent_processes_meet_workload_floor": all(
                run["end_to_end_speedup_percent"] >= _speed_floor(key[0])
                for run in runs
            ),
            "independent_process_resample_lower_meets_workload_floor": process_gate,
            "all_runs_passed": all(run["passed"] for run in runs) and process_gate,
            "runs": runs,
        }

    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "cells": cells,
        "speedup_percent_median_across_cells": statistics.median(cell_speedups),
        "all_runs_passed": all_runs_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
