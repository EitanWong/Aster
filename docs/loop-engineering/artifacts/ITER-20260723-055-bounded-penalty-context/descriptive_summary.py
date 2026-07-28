#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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


def _load_payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in manifest["records"]:
        path = ARTIFACT_DIR / record["output"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"payload hash mismatch: {path}")
        payloads.append(json.loads(path.read_text()))
    return payloads


def summarize(manifest_path: Path, aggregate_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    payloads = _load_payloads(manifest)
    cells: dict[str, Any] = {}
    for cell_name, admission in aggregate["cells"].items():
        workload = str(admission["workload"])
        batch_size = int(admission["batch_size"])
        selected = [
            payload
            for payload in payloads
            if payload["workload"] == workload
            and int(payload["batch_size"]) == batch_size
        ]
        baseline_steps = [
            float(value)
            for payload in selected
            for value in payload["timings"]["baseline_step_seconds"]
        ]
        production_steps = [
            float(value)
            for payload in selected
            for value in payload["timings"]["production_step_seconds"]
        ]
        production_metrics = [
            payload["policy_metrics"]["production"] for payload in selected
        ]
        cells[cell_name] = {
            "settings": {
                "context_words": int(manifest["matrix"]["context_words"]),
                "actual_prompt_tokens": sorted(
                    {
                        int(payload["settings"]["actual_prompt_tokens"])
                        for payload in selected
                    }
                ),
                "steps_per_process": int(manifest["matrix"]["steps"]),
                "batch_size": batch_size,
            },
            "run_speedup_percent": _stats(
                [
                    float(run["end_to_end_speedup_percent"])
                    for run in admission["runs"]
                ]
            ),
            "replicate_speedup_percent": _stats(
                [
                    float(replicate["end_to_end_speedup_percent"])
                    for replicate in admission["replicates"]
                ]
            ),
            "decode_tokens_per_second": {
                "baseline": _stats(
                    [
                        float(payload["timings"]["baseline_tokens_per_second"])
                        for payload in selected
                    ]
                ),
                "production": _stats(
                    [
                        float(payload["timings"]["production_tokens_per_second"])
                        for payload in selected
                    ]
                ),
            },
            "decode_step_seconds": {
                "baseline": _stats(baseline_steps),
                "production": _stats(production_steps),
            },
            "processor_context": {
                "bounded_rows": sum(
                    int(metrics["bounded_rows"]) for metrics in production_metrics
                ),
                "logical_source_tokens": sum(
                    int(metrics["source_tokens_total"])
                    for metrics in production_metrics
                ),
                "device_tokens": sum(
                    int(metrics["device_tokens_total"])
                    for metrics in production_metrics
                ),
                "max_logical_source_tokens": max(
                    int(metrics["max_source_tokens"])
                    for metrics in production_metrics
                ),
                "max_device_tokens_per_row": 20,
            },
            "memory": {
                "mlx_peak_bytes": _stats(
                    [float(payload["memory"]["mlx_peak_bytes"]) for payload in selected]
                ),
                "swap_delta_bytes": _stats(
                    [
                        float(payload["memory"]["swap_after_bytes"])
                        - float(payload["memory"]["swap_before_bytes"])
                        for payload in selected
                    ]
                ),
            },
            "correctness": {
                "exact_processes": sum(
                    payload["parity"]["exact_token_text_cache"] is True
                    for payload in selected
                ),
                "processes": len(selected),
            },
            "strict_admission": {
                "passed": bool(admission["passed"]),
                "intervals": admission["intervals"],
                "gates": admission["gates"],
            },
        }
    source = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "inputs_sha256": {
            "manifest": _sha256(manifest_path),
            "strict_aggregate": _sha256(aggregate_path),
            "descriptive_summary": _sha256(source),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.manifest.resolve(), args.strict_aggregate.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
