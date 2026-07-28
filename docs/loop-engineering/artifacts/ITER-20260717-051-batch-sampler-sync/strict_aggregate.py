#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import paired_aggregate as legacy

DEFAULT_CONFIDENCE = 0.95
DEFAULT_MIN_PROCESSES = 18
DEFAULT_MIN_REPLICATES = 9


def _speed_floor(workload: str) -> float:
    return -1.0 if workload == "structured" else legacy._speed_floor(workload)


def _median_interval_coverage(processes: int, rank: int) -> float:
    tail = sum(math.comb(processes, index) for index in range(rank))
    return max(0.0, 1.0 - (2.0 * tail / (2**processes)))


def distribution_free_median_interval(
    values: list[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float | int | bool]:
    if not values:
        raise ValueError("median interval requires at least one process")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    ordered = sorted(float(value) for value in values)
    selected_rank = 1
    selected_coverage = _median_interval_coverage(len(ordered), selected_rank)
    for rank in range(2, (len(ordered) + 1) // 2 + 1):
        coverage = _median_interval_coverage(len(ordered), rank)
        if coverage < confidence:
            break
        selected_rank = rank
        selected_coverage = coverage

    return {
        "confidence_target": confidence,
        "coverage": selected_coverage,
        "lower": ordered[selected_rank - 1],
        "upper": ordered[-selected_rank],
        "order_statistic_rank": selected_rank,
        "observations": len(ordered),
        "confidence_met": selected_coverage >= confidence,
    }


def _expected_record_order(
    cells: tuple[tuple[str, int], ...],
    runs: int,
) -> tuple[tuple[str, int, int], ...]:
    scheduled: list[tuple[str, int, int]] = []
    for run_id in range(1, runs + 1):
        shift = (run_id - 1) % len(cells)
        rotated = (*cells[shift:], *cells[:shift])
        scheduled.extend(
            (workload, batch_size, run_id)
            for workload, batch_size in rotated
        )
    return tuple(scheduled)


def _timing_series(
    payload: dict[str, Any],
) -> tuple[list[float], list[float], list[str]]:
    baseline = [float(value) for value in payload["timings"]["baseline_step_seconds"]]
    production = [
        float(value) for value in payload["timings"]["production_step_seconds"]
    ]
    first_policies = [str(value) for value in payload["timings"]["first_policy_by_step"]]
    if not baseline or len(baseline) != len(production) or len(baseline) != len(
        first_policies
    ):
        raise ValueError("timing series must be non-empty and equal length")
    phase = int(payload["run_id"]) - 1
    expected_order = [
        "baseline" if (step + phase) % 2 == 0 else "production"
        for step in range(len(first_policies))
    ]
    if first_policies != expected_order:
        raise ValueError("first-policy order must strictly alternate for the run phase")
    return baseline, production, first_policies


def _swap_delta(payload: dict[str, Any]) -> int:
    memory = payload["memory"]
    return int(memory["swap_after_bytes"]) - int(memory["swap_before_bytes"])


def validate_evidence(
    manifest: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> None:
    matrix = manifest["matrix"]
    cells = [
        (str(cell["workload"]), int(cell["batch_size"]))
        for cell in matrix["cells"]
    ]
    if not cells or len(cells) != len(set(cells)):
        raise ValueError("manifest cells must be non-empty and unique")
    runs = int(matrix["runs"])
    if runs < 2 or runs % 2:
        raise ValueError("strict aggregate requires an even process count")
    if int(matrix.get("independent_replicates", 0)) != runs // 2:
        raise ValueError("manifest independent replicate count is inconsistent")
    expected_records = len(cells) * runs
    records = manifest["records"]
    if len(records) != expected_records or len(payloads) != expected_records:
        raise ValueError(
            f"expected {expected_records} manifest records and payloads, "
            f"found {len(records)} and {len(payloads)}"
        )
    for flag in (
        "fresh_processes",
        "within_process_adjacent_pairing",
        "alternating_ab_ba_order",
        "round_robin_cell_execution",
        "alternating_policy_runner_assignment",
        "paired_runner_assignment_replicates",
    ):
        if matrix.get(flag) is not True:
            raise ValueError(f"manifest must assert {flag}")

    if len({str(record["output"]) for record in records}) != expected_records:
        raise ValueError("manifest outputs must be unique")
    if len({int(payload["pid"]) for payload in payloads}) != expected_records:
        raise ValueError("strict aggregate requires one fresh process per payload")

    source_signature = payloads[0]["source_sha256"]
    model_signature = payloads[0]["model_input_sha256"]
    grouped_run_ids: dict[tuple[str, int], list[int]] = defaultdict(list)
    for record, payload in zip(records, payloads, strict=True):
        record_key = (
            str(record["workload"]),
            int(record["batch_size"]),
            int(record["run_id"]),
            int(record["pid"]),
        )
        payload_key = (
            str(payload["workload"]),
            int(payload["batch_size"]),
            int(payload["run_id"]),
            int(payload["pid"]),
        )
        if record_key != payload_key:
            raise ValueError("manifest record metadata does not match payload")
        if payload["source_sha256"] != source_signature:
            raise ValueError("payload source signatures differ")
        if payload["model_input_sha256"] != model_signature:
            raise ValueError("payload model signatures differ")
        settings = payload["settings"]
        expected_settings = {
            "context_words": int(matrix["context_words"]),
            "steps": int(matrix["steps"]),
            "pair_warmup_steps": int(matrix["pair_warmup_steps"]),
            "block_size": int(matrix["block_size"]),
            "seed": int(matrix["base_seed"])
            + ((int(payload["run_id"]) + 1) // 2),
        }
        observed_settings = {
            "context_words": int(payload["context_words"]),
            "steps": int(settings["steps"]),
            "pair_warmup_steps": int(settings["pair_warmup_steps"]),
            "block_size": int(settings["block_size"]),
            "seed": int(settings["seed"]),
        }
        if observed_settings != expected_settings:
            raise ValueError("payload settings do not match manifest matrix")
        expected_assignment = (
            {"baseline": "runner_a", "production": "runner_b"}
            if int(payload["run_id"]) % 2 == 1
            else {"baseline": "runner_b", "production": "runner_a"}
        )
        comparison_design = payload.get("comparison_design", {})
        if (
            comparison_design.get("runner_assignment_alternates_by_run") is not True
            or comparison_design.get("policy_runner_assignment")
            != expected_assignment
            or int(comparison_design.get("assignment_balanced_replicate_id", 0))
            != (int(payload["run_id"]) + 1) // 2
        ):
            raise ValueError("payload policy runner assignment does not match run ID")
        _timing_series(payload)
        grouped_run_ids[payload_key[:2]].append(int(payload["run_id"]))

    expected_run_ids = list(range(1, runs + 1))
    for cell in cells:
        if sorted(grouped_run_ids[cell]) != expected_run_ids:
            raise ValueError(f"cell {cell} run IDs must be exactly {expected_run_ids}")
    record_order = tuple(
        (
            str(record["workload"]),
            int(record["batch_size"]),
            int(record["run_id"]),
        )
        for record in records
    )
    if record_order != _expected_record_order(tuple(cells), runs):
        raise ValueError("manifest records do not follow round-robin execution order")


def _timing_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    baseline, production, first_policies = _timing_series(payload)
    workload = str(payload["workload"])
    block_speedups = legacy._block_speedups(
        baseline,
        production,
        int(payload["settings"]["block_size"]),
    )
    speed_floor = _speed_floor(workload)
    block_median = statistics.median(block_speedups)
    positive_fraction = sum(value > 0.0 for value in block_speedups) / len(
        block_speedups
    )
    block_floor_fraction = sum(
        value >= speed_floor for value in block_speedups
    ) / len(block_speedups)
    first_counts = {
        policy: first_policies.count(policy)
        for policy in ("baseline", "production")
    }
    swap_delta = _swap_delta(payload)
    return {
        "run_id": int(payload["run_id"]),
        "pid": int(payload["pid"]),
        "end_to_end_speedup_percent": legacy._percent_change(
            sum(baseline), sum(production)
        ),
        "baseline_first_speedup_percent": legacy._order_speedup(
            baseline, production, first_policies, "baseline"
        ),
        "production_first_speedup_percent": legacy._order_speedup(
            baseline, production, first_policies, "production"
        ),
        "block_speedup_percent_median": block_median,
        "positive_block_fraction": positive_fraction,
        "block_floor_fraction": block_floor_fraction,
        "within_process_stable": (
            block_median >= speed_floor and block_floor_fraction >= 0.75
        ),
        "balanced_ab_ba_order": first_counts["baseline"] == first_counts["production"],
        "exact_token_text_cache_parity": bool(
            payload["parity"]["exact_token_text_cache"]
        ),
        "swap_delta_bytes": swap_delta,
        "swap_zero": swap_delta == 0,
        "swap_non_growth": swap_delta <= 0,
    }


def _interval_gate(values: list[float], speed_floor: float) -> dict[str, Any]:
    interval = distribution_free_median_interval(values)
    return {
        **interval,
        "speed_floor_percent": speed_floor,
        "lower_meets_speed_floor": (
            bool(interval["confidence_met"])
            and float(interval["lower"]) >= speed_floor
        ),
    }


def _replicate_metrics(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if len(payloads) != 2:
        raise ValueError("assignment-balanced replicate requires two processes")
    ordered = sorted(payloads, key=lambda payload: int(payload["run_id"]))
    run_ids = [int(payload["run_id"]) for payload in ordered]
    if run_ids[0] % 2 != 1 or run_ids[1] != run_ids[0] + 1:
        raise ValueError("assignment-balanced replicate requires odd/even run pairs")
    workload = str(ordered[0]["workload"])
    batch_size = int(ordered[0]["batch_size"])
    if any(
        str(payload["workload"]) != workload
        or int(payload["batch_size"]) != batch_size
        for payload in ordered
    ):
        raise ValueError("assignment-balanced replicate payloads must share one cell")

    series = [_timing_series(payload) for payload in ordered]

    def speedup(first_policy: str | None = None) -> float:
        ratios: list[float] = []
        for baseline, production, first_policies in series:
            indices = range(len(baseline))
            if first_policy is not None:
                indices = [
                    index
                    for index, observed in enumerate(first_policies)
                    if observed == first_policy
                ]
            baseline_total = sum(baseline[index] for index in indices)
            production_total = sum(production[index] for index in indices)
            if baseline_total <= 0.0 or production_total <= 0.0:
                raise ValueError("timing totals must be positive")
            ratios.append(baseline_total / production_total)
        return (math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)) - 1.0) * 100.0

    block_size = int(ordered[0]["settings"]["block_size"])
    steps = len(series[0][0])
    block_speedups = []
    for start in range(0, steps, block_size):
        ratios = []
        for baseline, production, _first_policies in series:
            baseline_total = sum(baseline[start : start + block_size])
            production_total = sum(production[start : start + block_size])
            if baseline_total <= 0.0 or production_total <= 0.0:
                raise ValueError("block timing totals must be positive")
            ratios.append(baseline_total / production_total)
        block_speedups.append(
            (
                math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
                - 1.0
            )
            * 100.0
        )

    speed_floor = _speed_floor(workload)
    block_median = statistics.median(block_speedups)
    block_floor_fraction = sum(
        value >= speed_floor for value in block_speedups
    ) / len(block_speedups)
    assignments = [
        payload["comparison_design"]["policy_runner_assignment"]
        for payload in ordered
    ]
    swap_deltas = [_swap_delta(payload) for payload in ordered]
    return {
        "replicate_id": (run_ids[0] + 1) // 2,
        "run_ids": run_ids,
        "end_to_end_speedup_percent": speedup(),
        "baseline_first_speedup_percent": speedup("baseline"),
        "production_first_speedup_percent": speedup("production"),
        "block_speedup_percent_median": block_median,
        "block_floor_fraction": block_floor_fraction,
        "within_replicate_stable": (
            block_median >= speed_floor and block_floor_fraction >= 0.75
        ),
        "runner_assignment_balanced": assignments
        == [
            {"baseline": "runner_a", "production": "runner_b"},
            {"baseline": "runner_b", "production": "runner_a"},
        ],
        "exact_token_text_cache_parity": all(
            bool(payload["parity"]["exact_token_text_cache"])
            for payload in ordered
        ),
        "swap_delta_bytes": swap_deltas,
        "swap_zero": all(delta == 0 for delta in swap_deltas),
        "swap_non_growth": all(delta <= 0 for delta in swap_deltas),
    }


def aggregate_cell(
    workload: str,
    batch_size: int,
    payloads: list[dict[str, Any]],
    *,
    min_processes: int = DEFAULT_MIN_PROCESSES,
    min_replicates: int = DEFAULT_MIN_REPLICATES,
) -> dict[str, Any]:
    ordered_payloads = sorted(payloads, key=lambda item: int(item["run_id"]))
    runs = [
        _timing_metrics(payload)
        for payload in ordered_payloads
    ]
    replicates = [
        _replicate_metrics(ordered_payloads[index : index + 2])
        for index in range(0, len(ordered_payloads), 2)
        if len(ordered_payloads[index : index + 2]) == 2
    ]
    speed_floor = _speed_floor(workload)
    intervals = {
        "balanced": _interval_gate(
            [float(replicate["end_to_end_speedup_percent"]) for replicate in replicates],
            speed_floor,
        ),
        "baseline_first": _interval_gate(
            [
                float(replicate["baseline_first_speedup_percent"])
                for replicate in replicates
            ],
            speed_floor,
        ),
        "production_first": _interval_gate(
            [
                float(replicate["production_first_speedup_percent"])
                for replicate in replicates
            ],
            speed_floor,
        ),
    }
    stable_replicates = sum(
        bool(replicate["within_replicate_stable"]) for replicate in replicates
    )
    order_strata_required = workload != "structured"
    stability_required = workload != "structured"
    baseline_first_clears = bool(
        intervals["baseline_first"]["lower_meets_speed_floor"]
    )
    production_first_clears = bool(
        intervals["production_first"]["lower_meets_speed_floor"]
    )
    gates = {
        "minimum_independent_processes": len(runs) >= min_processes,
        "minimum_assignment_balanced_replicates": len(replicates) >= min_replicates,
        "all_exact_token_text_cache_parity": all(
            bool(run["exact_token_text_cache_parity"]) for run in runs
        ),
        "all_swap_non_growth": all(bool(run["swap_non_growth"]) for run in runs),
        "all_balanced_ab_ba_order": all(
            bool(run["balanced_ab_ba_order"]) for run in runs
        ),
        "all_runner_assignment_pairs_balanced": all(
            bool(replicate["runner_assignment_balanced"])
            for replicate in replicates
        ),
        "balanced_median_interval_clears_floor": bool(
            intervals["balanced"]["lower_meets_speed_floor"]
        ),
        "baseline_first_median_interval_clears_floor": (
            not order_strata_required or baseline_first_clears
        ),
        "production_first_median_interval_clears_floor": (
            not order_strata_required or production_first_clears
        ),
        "order_strata_speed_floor_required": (
            not order_strata_required
            or (baseline_first_clears and production_first_clears)
        ),
        "within_replicate_stability": (
            not stability_required or stable_replicates >= len(replicates) - 1
        ),
    }
    return {
        "workload": workload,
        "batch_size": batch_size,
        "speed_floor_percent": speed_floor,
        "independent_processes": len(runs),
        "independent_replicates": len(replicates),
        "stable_replicates": stable_replicates,
        "all_swap_zero": all(bool(run["swap_zero"]) for run in runs),
        "order_strata_inference_required": order_strata_required,
        "within_replicate_stability_required": stability_required,
        "intervals": intervals,
        "gates": gates,
        "passed": all(gates.values()),
        "replicates": replicates,
        "runs": runs,
    }


def aggregate(
    manifest_path: Path,
    *,
    min_processes: int = DEFAULT_MIN_PROCESSES,
    min_replicates: int = DEFAULT_MIN_REPLICATES,
) -> dict[str, Any]:
    manifest, payloads = legacy._load(manifest_path)
    validate_evidence(manifest, payloads)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[(str(payload["workload"]), int(payload["batch_size"]))].append(payload)

    cells: dict[str, Any] = {}
    for descriptor in manifest["matrix"]["cells"]:
        key = (str(descriptor["workload"]), int(descriptor["batch_size"]))
        cells[f"{key[0]}-b{key[1]}"] = aggregate_cell(
            key[0],
            key[1],
            grouped[key],
            min_processes=min_processes,
            min_replicates=min_replicates,
        )

    return {
        "schema_version": 2,
        "manifest_sha256": legacy._sha256(manifest_path),
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "minimum_processes_per_cell": min_processes,
        "minimum_replicates_per_cell": min_replicates,
        "cells": cells,
        "all_cells_passed": all(cell["passed"] for cell in cells.values()),
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
