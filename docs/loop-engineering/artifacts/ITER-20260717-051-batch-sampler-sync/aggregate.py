#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percent_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _bootstrap_median_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    medians = sorted(
        statistics.median(sample)
        for sample in itertools.product(values, repeat=len(values))
    )
    last = len(medians) - 1
    return medians[int(last * 0.025)], medians[int(last * 0.975)]


def _speed_floor(workload: str) -> float:
    return 0.0 if workload == "structured" else 3.0


def _load(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    payloads: list[dict[str, Any]] = []
    for record in manifest["records"]:
        output = _resolve(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        observed = payload["source_sha256"]
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in observed.items()
        ):
            raise ValueError(f"source mismatch: {output}")
        payloads.append(payload)
    return manifest, payloads


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest, payloads = _load(manifest_path)
    matrix = manifest["matrix"]
    expected = (
        len(matrix["policies"]) * len(matrix["cells"]) * int(matrix["runs"])
    )
    if len(payloads) != expected:
        raise ValueError(f"expected {expected} payloads, found {len(payloads)}")
    if len({int(payload["pid"]) for payload in payloads}) != expected:
        raise ValueError("screen payload PIDs are not unique")

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_run: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for payload in payloads:
        key = (
            str(payload["workload"]),
            int(payload["batch_size"]),
            str(payload["policy"]),
        )
        grouped[key].append(payload)
        by_run[(*key[:2], int(payload["run_id"]), key[2])] = payload

    cells: dict[str, Any] = {}
    policy_passes: dict[str, list[bool]] = defaultdict(list)
    policy_speedups: dict[str, list[float]] = defaultdict(list)
    for descriptor in matrix["cells"]:
        workload = str(descriptor["workload"])
        batch_size = int(descriptor["batch_size"])
        baseline = grouped[(workload, batch_size, "baseline")]
        baseline_tps = _median(
            [float(payload["decode"]["tokens_per_second"]) for payload in baseline]
        )
        baseline_rss = _median(
            [float(payload["memory"]["rss_peak_bytes"]) for payload in baseline]
        )
        policies: dict[str, Any] = {}
        for policy in matrix["policies"]:
            candidates = grouped[(workload, batch_size, policy)]
            candidate_tps = _median(
                [
                    float(payload["decode"]["tokens_per_second"])
                    for payload in candidates
                ]
            )
            paired_speedups: list[float] = []
            exact_parity = True
            swap_zero = True
            for run_id in range(1, int(matrix["runs"]) + 1):
                reference = by_run[(workload, batch_size, run_id, "baseline")]
                candidate = by_run[(workload, batch_size, run_id, policy)]
                paired_speedups.append(
                    _percent_change(
                        float(candidate["decode"]["tokens_per_second"]),
                        float(reference["decode"]["tokens_per_second"]),
                    )
                )
                exact_parity &= (
                    candidate["decode"]["token_ids"]
                    == reference["decode"]["token_ids"]
                    and candidate["decode"]["text_sha256"]
                    == reference["decode"]["text_sha256"]
                    and candidate["decode"]["cache_digest"]
                    == reference["decode"]["cache_digest"]
                )
                swap_zero &= (
                    int(candidate["memory"]["swap_after_bytes"])
                    - int(candidate["memory"]["swap_before_bytes"])
                    == 0
                )
            candidate_rss = _median(
                [float(payload["memory"]["rss_peak_bytes"]) for payload in candidates]
            )
            speedup = _median(paired_speedups)
            bootstrap_low, bootstrap_high = _bootstrap_median_ci(paired_speedups)
            rss_change = _percent_change(candidate_rss, baseline_rss)
            clear_counts = [
                int(payload["policy_metrics"]["clear_requests"])
                for payload in candidates
            ]
            expected_clears = int(matrix["max_tokens"]) * batch_size // 512
            speed_floor = _speed_floor(workload)
            structured_wins = sum(change >= 0.0 for change in paired_speedups)
            gate = {
                "exact_token_text_cache_parity": exact_parity,
                "swap_zero": swap_zero,
                "clear_count_exact": all(
                    count == expected_clears for count in clear_counts
                ),
                "median_speedup_meets_workload_floor": (
                    policy == "baseline" or speedup >= speed_floor
                ),
                "paired_bootstrap_lower_meets_core_floor": (
                    policy == "baseline"
                    or workload == "structured"
                    or bootstrap_low >= speed_floor
                ),
                "structured_nonnegative_in_at_least_4_of_5_pairs": (
                    policy == "baseline"
                    or workload != "structured"
                    or structured_wins >= 4
                ),
                "rss_peak_regression_at_most_2_percent": rss_change <= 2.0,
            }
            passed = all(gate.values())
            if policy != "baseline":
                policy_passes[policy].append(passed)
                policy_speedups[policy].append(speedup)
            sample_sync_ms = _median(
                [
                    float(payload["policy_metrics"]["sample_sync_seconds"]["median"])
                    * 1000.0
                    for payload in candidates
                ]
            )
            grouped_eval_ms = _median(
                [
                    float(
                        payload["policy_metrics"]["sample_group_eval_seconds"][
                            "median"
                        ]
                    )
                    * 1000.0
                    for payload in candidates
                ]
            )
            policies[policy] = {
                "decode_tps_median": candidate_tps,
                "speedup_percent_median": speedup,
                "paired_speedups_percent": paired_speedups,
                "paired_bootstrap_95_percent": [bootstrap_low, bootstrap_high],
                "rss_peak_regression_percent": rss_change,
                "sample_sync_ms_median": sample_sync_ms,
                "grouped_eval_ms_median": grouped_eval_ms,
                "gate": gate,
                "passed": passed,
            }
        cells[f"{workload}-b{batch_size}"] = {
            "baseline_decode_tps_median": baseline_tps,
            "policies": policies,
        }

    ranking = sorted(
        (
            {
                "policy": policy,
                "all_cells_passed": all(policy_passes[policy]),
                "speedup_percent_median_across_cells": _median(
                    policy_speedups[policy]
                ),
            }
            for policy in policy_passes
        ),
        key=lambda item: float(item["speedup_percent_median_across_cells"]),
        reverse=True,
    )
    winner = next(
        (item["policy"] for item in ranking if item["all_cells_passed"]),
        None,
    )
    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "cells": cells,
        "ranking": ranking,
        "screen_winner": winner,
        "confirmation_required": winner is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/screen/execution-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/screen/aggregate.json",
    )
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
