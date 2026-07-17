#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(values: list[float], *, seed: int, samples: int = 50_000) -> list[float]:
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    ]
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _load(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    matrix = manifest["matrix"]
    expected = len(matrix["policies"]) * len(matrix["cells"]) * int(matrix["runs"])
    records = manifest["records"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    if len({int(record["pid"]) for record in records}) != expected:
        raise ValueError("records do not have unique PIDs")

    payloads: list[dict[str, Any]] = []
    occupied: set[tuple[str, int, int, str, int]] = set()
    for record in records:
        output = _resolve(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        if payload["environment"]["git_commit"] != manifest["environment"]["git_commit"]:
            raise ValueError(f"commit mismatch: {output}")
        observed = payload["source_sha256"] | payload["model_input_sha256"]
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in observed.items()
        ):
            raise ValueError(f"source hash mismatch: {output}")
        cell = (
            str(payload["cache_kind"]),
            int(payload["batch_size"]),
            int(payload["context_words"]),
            str(payload["policy"]),
            int(payload["run_id"]),
        )
        if cell in occupied:
            raise ValueError(f"duplicate cell: {cell}")
        occupied.add(cell)
        payloads.append(payload)
    return manifest, payloads


def _change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _speedup(candidate_seconds: float, baseline_seconds: float) -> float:
    return (baseline_seconds / candidate_seconds - 1.0) * 100.0


def _paired_metric(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    candidate_path: tuple[str, ...],
    baseline_path: tuple[str, ...],
    *,
    speedup: bool = False,
) -> list[float]:
    def value(payload: dict[str, Any], path: tuple[str, ...]) -> float:
        current: Any = payload
        for key in path:
            current = current[key]
        return float(current)

    return [
        (
            _speedup(value(candidate, candidate_path), value(baseline, baseline_path))
            if speedup
            else _change(value(candidate, candidate_path), value(baseline, baseline_path))
        )
        for baseline, candidate in pairs
    ]


def _metric_summary(values: list[float], seed: int) -> dict[str, Any]:
    return {
        "values": values,
        "median": statistics.median(values),
        "bootstrap95": _bootstrap(values, seed=seed),
    }


def aggregate(
    manifest_path: Path,
    *,
    candidate_policy: str = "periodic-512",
) -> dict[str, Any]:
    manifest, payloads = _load(manifest_path)
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[
            (
                str(payload["cache_kind"]),
                int(payload["batch_size"]),
                int(payload["context_words"]),
                str(payload["policy"]),
            )
        ].append(payload)

    cells: dict[str, Any] = {}
    all_pass = True
    for cell_index, descriptor in enumerate(manifest["matrix"]["cells"]):
        cache_kind = str(descriptor["cache_kind"])
        batch_size = int(descriptor["batch_size"])
        context_words = int(descriptor["context_words"])
        baseline = sorted(
            grouped[(cache_kind, batch_size, context_words, "baseline")],
            key=lambda payload: int(payload["run_id"]),
        )
        candidate = sorted(
            grouped[(cache_kind, batch_size, context_words, candidate_policy)],
            key=lambda payload: int(payload["run_id"]),
        )
        if [item["run_id"] for item in baseline] != [item["run_id"] for item in candidate]:
            raise ValueError(f"unpaired cell: {descriptor}")
        pairs = list(zip(baseline, candidate, strict=True))
        parity = all(
            base["decode"]["token_ids"] == cand["decode"]["token_ids"]
            and base["decode"]["text_sha256"] == cand["decode"]["text_sha256"]
            and base["decode"]["cache_digest"] == cand["decode"]["cache_digest"]
            for base, cand in pairs
        )
        seed = 20260717 + cell_index * 100
        speed = _metric_summary(
            _paired_metric(
                pairs,
                ("decode", "elapsed_seconds"),
                ("decode", "elapsed_seconds"),
                speedup=True,
            ),
            seed,
        )
        rss = _metric_summary(
            _paired_metric(
                pairs,
                ("memory", "rss_peak_bytes"),
                ("memory", "rss_peak_bytes"),
            ),
            seed + 1,
        )
        active = _metric_summary(
            _paired_metric(
                pairs,
                ("memory", "mlx_after_decode", "active_bytes"),
                ("memory", "mlx_after_decode", "active_bytes"),
            ),
            seed + 2,
        )
        peak = _metric_summary(
            _paired_metric(
                pairs,
                ("memory", "mlx_after_decode", "peak_bytes"),
                ("memory", "mlx_after_decode", "peak_bytes"),
            ),
            seed + 3,
        )
        step_p95 = _metric_summary(
            _paired_metric(
                pairs,
                ("decode", "step_seconds", "p95"),
                ("decode", "step_seconds", "p95"),
            ),
            seed + 4,
        )
        policy_counts = all(
            int(base["policy_metrics"]["cache_eval_executed"])
            == int(manifest["matrix"]["max_tokens"])
            and int(base["policy_metrics"]["clear_executed"])
            == int(manifest["matrix"]["max_tokens"])
            and int(cand["policy_metrics"]["cache_eval_skipped"])
            == int(manifest["matrix"]["max_tokens"])
            and int(cand["policy_metrics"]["clear_executed"])
            == (
                int(manifest["matrix"]["max_tokens"]) * batch_size // 512
                if candidate_policy == "periodic-token-512"
                else int(manifest["matrix"]["max_tokens"]) // 512
            )
            for base, cand in pairs
        )
        swap_zero = all(
            int(payload["memory"]["swap_after_bytes"])
            - int(payload["memory"]["swap_before_bytes"])
            == 0
            for pair in pairs
            for payload in pair
        )
        gate = {
            "parity": parity,
            "policy_counts": policy_counts,
            "swap_zero": swap_zero,
            "speed_median_at_least_3_percent": speed["median"] >= 3.0,
            "speed_bootstrap_lower_at_least_3_percent": speed["bootstrap95"][0] >= 3.0,
            "rss_bootstrap_upper_at_most_1_percent": rss["bootstrap95"][1] <= 1.0,
            "active_bootstrap_upper_at_most_1_percent": active["bootstrap95"][1] <= 1.0,
            "peak_bootstrap_upper_at_most_1_percent": peak["bootstrap95"][1] <= 1.0,
            "step_p95_median_regression_at_most_1_percent": step_p95["median"] <= 1.0,
        }
        passed = all(gate.values())
        all_pass &= passed
        key = f"{cache_kind}-b{batch_size}-{context_words}w"
        cells[key] = {
            "pairs": len(pairs),
            "prompt_tokens": [item["prefill"]["prompt_tokens"] for item in baseline],
            "decode_speedup_percent": speed,
            "rss_peak_regression_percent": rss,
            "mlx_active_regression_percent": active,
            "mlx_peak_regression_percent": peak,
            "step_p95_regression_percent": step_p95,
            "gate": gate,
            "passed": passed,
        }

    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "cells": cells,
        "admission": {
            "all_confirmation_cells_passed": all_pass,
            "long_stress_required": True,
            "integration_approved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/confirmation/execution-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/confirmation/aggregate.json",
    )
    parser.add_argument("--candidate-policy", default="periodic-512")
    args = parser.parse_args()
    payload = aggregate(
        args.manifest.resolve(),
        candidate_policy=args.candidate_policy,
    )
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
