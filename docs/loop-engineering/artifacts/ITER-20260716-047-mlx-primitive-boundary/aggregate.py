from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_METHODS = frozenset(
    {"public_fast", "direct_fast", "guarded_fast", "primitive"}
)
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aster_paged_ops.cpp",
    "bench.py",
    "aggregate.py",
)
EXPECTED_ASTER_IDENTITY = {
    "commit": "22865cd0e290acdfe02e0b845eb680eef7fc0a76",
    "source": "aster/inference/metal_paged_attention.py",
    "source_dirty": False,
    "source_sha256": "b7b4bea2ead78057d4d4759d99fc1de62f674a6ba3dd603b0d7233bc8bbd8796",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def local_source_hashes() -> dict[str, str]:
    artifact_dir = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
        for name in SOURCE_FILENAMES
    }


def bootstrap_delta(
    records: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    *,
    resamples: int,
    generator: random.Random,
) -> tuple[float, float]:
    deltas: list[float] = []
    for _ in range(resamples):
        selected = generator.choices(records, k=len(records))
        baseline_medians = []
        candidate_medians = []
        for record in selected:
            baseline_samples = record["samples_ms"][baseline]
            candidate_samples = record["samples_ms"][candidate]
            block_size = max(2, math.isqrt(len(baseline_samples)))
            indices: list[int] = []
            while len(indices) < len(baseline_samples):
                start = generator.randrange(len(baseline_samples))
                indices.extend(
                    (start + offset) % len(baseline_samples) for offset in range(block_size)
                )
            selected_indices = indices[: len(baseline_samples)]
            baseline_medians.append(
                statistics.median(baseline_samples[index] for index in selected_indices)
            )
            candidate_medians.append(
                statistics.median(candidate_samples[index] for index in selected_indices)
            )
        baseline_value = statistics.median(baseline_medians)
        candidate_value = statistics.median(candidate_medians)
        deltas.append(100.0 * (candidate_value / baseline_value - 1.0))
    return percentile(deltas, 0.025), percentile(deltas, 0.975)


def main() -> None:
    args = parse_args()
    if args.resamples < 1:
        raise ValueError("resamples must be positive")
    if len(args.files) < 5:
        raise ValueError("At least five independent process files are required")
    resolved_files = [path.resolve() for path in args.files]
    if len(set(resolved_files)) != len(resolved_files):
        raise ValueError("Duplicate input files are not allowed")
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    run_ids: set[int] = set()
    process_ids: set[int] = set()
    expected_environment: str | None = None
    expected_hashes = local_source_hashes()
    expected_baseline: dict[str, Any] | None = None
    expected_cells: set[tuple[int, int]] | None = None
    expected_configs: dict[tuple[int, int], tuple[int, int]] | None = None
    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = payload["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id: {run_id}")
        run_ids.add(run_id)
        process_id = payload["environment"]["pid"]
        if process_id in process_ids:
            raise ValueError(f"Duplicate process id: {process_id}")
        process_ids.add(process_id)
        if payload["source_hashes"] != expected_hashes:
            raise ValueError("Result source hashes do not match the archived sources")
        baseline = payload["aster_baseline"]
        identity = {key: baseline[key] for key in EXPECTED_ASTER_IDENTITY}
        if identity != EXPECTED_ASTER_IDENTITY:
            raise ValueError("Result does not use the archived Aster baseline")
        environment = {
            key: value for key, value in payload["environment"].items() if key != "pid"
        }
        environment_key = json.dumps(environment, sort_keys=True)
        if expected_environment is None:
            expected_environment = environment_key
            expected_baseline = baseline
        else:
            if environment_key != expected_environment:
                raise ValueError("Benchmark environments do not match")
            if baseline != expected_baseline:
                raise ValueError("Aster baseline identities do not match")
        records = payload["results"]
        if not records:
            raise ValueError("Benchmark result set must not be empty")
        cells = {(record["tokens"], record["batch"]) for record in records}
        if len(cells) != len(records):
            raise ValueError("Duplicate benchmark cells are not allowed")
        configs = {
            (record["tokens"], record["batch"]): (
                record["warmups"],
                record["iterations"],
            )
            for record in records
        }
        if expected_cells is None:
            expected_cells = cells
            expected_configs = configs
        elif cells != expected_cells:
            raise ValueError("Benchmark cell sets do not match")
        elif configs != expected_configs:
            raise ValueError("Benchmark warmup or iteration configurations do not match")
        for record in records:
            if set(record["samples_ms"]) != EXPECTED_METHODS:
                raise ValueError("Benchmark method sets do not match")
            sample_lengths = {len(record["samples_ms"][name]) for name in record["samples_ms"]}
            if sample_lengths != {record["iterations"]}:
                raise ValueError("Method sample counts do not match the iteration count")
            if record["parity_max_abs"] != 0.0:
                raise ValueError("Benchmark parity must be exact across compared paths")
            if any(
                not math.isfinite(sample) or sample <= 0.0
                for samples in record["samples_ms"].values()
                for sample in samples
            ):
                raise ValueError("Benchmark samples must be finite and positive")
            grouped[(record["tokens"], record["batch"])].append(
                {**record, "_run_id": run_id}
            )

    generator = random.Random(0xA57E047)
    cells = []
    for (tokens, batch), records in sorted(grouped.items()):
        cells.append(
            {
                "tokens": tokens,
                "batch": batch,
                "primitive_vs_guarded_ci95_pct": bootstrap_delta(
                    records,
                    "guarded_fast",
                    "primitive",
                    resamples=args.resamples,
                    generator=generator,
                ),
                "primitive_vs_public_ci95_pct": bootstrap_delta(
                    records,
                    "public_fast",
                    "primitive",
                    resamples=args.resamples,
                    generator=generator,
                ),
                "primitive_vs_direct_ci95_pct": bootstrap_delta(
                    records,
                    "direct_fast",
                    "primitive",
                    resamples=args.resamples,
                    generator=generator,
                ),
                "guarded_vs_direct_ci95_pct": bootstrap_delta(
                    records,
                    "direct_fast",
                    "guarded_fast",
                    resamples=args.resamples,
                    generator=generator,
                ),
            }
        )
    output = {
        "interval_note": (
            "Exploratory paired moving-block bootstrap intervals nested inside "
            "independent process resampling; not calibrated significance tests."
        ),
        "process_count": len(args.files),
        "process_ids": sorted(process_ids),
        "run_ids": sorted(run_ids),
        "aster_baseline": expected_baseline,
        "source_hashes": expected_hashes,
        "cells": cells,
    }
    payload = json.dumps(output, allow_nan=False, indent=2)
    if args.output is not None:
        compact_payload = json.dumps(output, allow_nan=False, separators=(",", ":"))
        args.output.write_text(compact_payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
