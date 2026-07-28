#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = ARTIFACT_DIR.parent / "ITER-20260724-057-structured-python-tokens"
RESULTS = (
    ("short_structured_b4", "strict-short-b4-r18", "structured-b4"),
    ("long_structured_b2", "strict-long-b2-r18", "structured-b2"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("statistics require at least one value")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "population_stdev": statistics.pstdev(values),
    }


def _payloads(root: Path, result_name: str) -> list[dict[str, Any]]:
    result_dir = root / "results" / result_name
    manifest = _load(result_dir / "execution-manifest.json")
    return [_load(root / record["output"]) for record in manifest["records"]]


def _summarize(result_name: str, cell_name: str) -> dict[str, Any]:
    result_dir = ARTIFACT_DIR / "results" / result_name
    aggregate = _load(result_dir / "aggregate.json")
    payloads = _payloads(ARTIFACT_DIR, result_name)
    reference_payloads = _payloads(REFERENCE_DIR, result_name)
    baseline_steps = [
        float(value) * 1000.0
        for payload in payloads
        for value in payload["timings"]["baseline_step_seconds"]
    ]
    production_steps = [
        float(value) * 1000.0
        for payload in payloads
        for value in payload["timings"]["production_step_seconds"]
    ]
    baseline_tps = [
        float(payload["timings"]["baseline_tokens_per_second"])
        for payload in payloads
    ]
    production_tps = [
        float(payload["timings"]["production_tokens_per_second"])
        for payload in payloads
    ]
    process_speedups = [
        (production / baseline - 1.0) * 100.0
        for baseline, production in zip(baseline_tps, production_tps, strict=True)
    ]
    current_peaks = [
        float(payload["memory"]["mlx_peak_bytes"]) for payload in payloads
    ]
    reference_peaks = [
        float(payload["memory"]["mlx_peak_bytes"])
        for payload in reference_payloads
    ]
    cell = aggregate["cells"][cell_name]
    settings = payloads[0]["settings"]
    return {
        "batch_size": int(payloads[0]["batch_size"]),
        "context_words": int(payloads[0]["context_words"]),
        "actual_prompt_tokens": int(settings["actual_prompt_tokens"]),
        "measured_steps_per_process": int(settings["steps"]),
        "paired_warmup_steps": int(settings["pair_warmup_steps"]),
        "processes": len(payloads),
        "assignment_balanced_replicates": int(cell["independent_replicates"]),
        "decode_step_latency_ms": {
            "forced_miss_baseline": _stats(baseline_steps),
            "production": _stats(production_steps),
            "median_change_percent": (
                statistics.median(production_steps)
                / statistics.median(baseline_steps)
                - 1.0
            )
            * 100.0,
        },
        "process_throughput_tokens_per_second": {
            "forced_miss_baseline": _stats(baseline_tps),
            "production": _stats(production_tps),
            "median_change_percent": (
                statistics.median(production_tps)
                / statistics.median(baseline_tps)
                - 1.0
            )
            * 100.0,
        },
        "process_speedup_percent": _stats(process_speedups),
        "balanced_median_interval_percent": cell["intervals"]["balanced"],
        "baseline_first_interval_percent": cell["intervals"]["baseline_first"],
        "production_first_interval_percent": cell["intervals"]["production_first"],
        "dual_runner_mlx_peak_bytes": _stats(current_peaks),
        "dual_runner_reference_mlx_peak_median_bytes": statistics.median(
            reference_peaks
        ),
        "dual_runner_mlx_peak_median_delta_bytes": (
            statistics.median(current_peaks) - statistics.median(reference_peaks)
        ),
        "rss_after_bytes": _stats(
            [float(payload["memory"]["rss_after_bytes"]) for payload in payloads]
        ),
        "all_exact_token_text_cache_parity": all(
            payload["parity"]["exact_token_text_cache"] is True
            for payload in payloads
        ),
        "all_swap_non_growth": all(
            payload["memory"]["swap_after_bytes"]
            <= payload["memory"]["swap_before_bytes"]
            for payload in payloads
        ),
        "admission_passed": bool(cell["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "summary.json")
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "metric_scope": "paired structured decode mask reuse",
        "cache_capacity_per_processor": 1,
        "not_measured": [
            "TTFT",
            "prefill throughput",
            "prefix cache hit rate",
            "power",
        ],
        "cells": {
            label: _summarize(result_name, cell_name)
            for label, result_name, cell_name in RESULTS
        },
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
