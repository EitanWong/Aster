from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_ROOT))

from aggregate_common import (  # noqa: E402
    bootstrap_delta,
    paired_point_delta,
    sha256,
    validate_cells,
    validate_measurements,
    validate_recorded_delta,
    verify_manifest,
)

METHODS = frozenset({"mlx_scatter", "fused_primitive"})
EXPECTED_CELLS = frozenset((tokens, batch) for tokens in (1, 16, 64) for batch in (1, 2, 4, 8))
EXPECTED_WARMUPS = 30
EXPECTED_ITERATIONS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_resources(payload: dict[str, Any]) -> int:
    resources = payload.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "before",
        "after",
        "power",
    }:
        raise ValueError("Resource evidence is incomplete")
    snapshots = []
    for phase in ("before", "after"):
        snapshot = resources[phase]
        swap = snapshot.get("swap")
        thermal = snapshot.get("thermal")
        if not isinstance(swap, dict) or swap.get("returncode") != 0:
            raise ValueError("Swap evidence is unavailable")
        if not isinstance(swap.get("used_bytes"), int) or swap["used_bytes"] < 0:
            raise ValueError("Swap evidence is invalid")
        if not isinstance(thermal, dict) or thermal.get("returncode") != 0:
            raise ValueError("Thermal evidence is unavailable")
        if thermal.get("warning") is not False:
            raise ValueError("Thermal warning gate failed")
        snapshots.append(snapshot)
    growth = snapshots[1]["swap"]["used_bytes"] - snapshots[0]["swap"]["used_bytes"]
    if growth > 0:
        raise ValueError("Swap growth gate failed")
    power = resources["power"]
    if not isinstance(power, dict) or set(power) != {"measured", "reason"}:
        raise ValueError("Power evidence is invalid")
    return growth


def main() -> None:
    args = parse_args()
    if len(args.files) != 5:
        raise ValueError("Exactly five independent process files are required")
    if args.resamples < 1:
        raise ValueError("resamples must be positive")
    resolved = [path.resolve() for path in args.files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate files are not allowed")

    manifest, manifest_hash = verify_manifest(ARTIFACT_ROOT)
    relative_source = Path(__file__).resolve().relative_to(ARTIFACT_ROOT).as_posix()
    if manifest["artifact_sources"].get(relative_source) != sha256(Path(__file__)):
        raise ValueError("Aggregator is missing from the verified manifest")
    reference_hashes = manifest["reference"]["source_hashes"]
    expected_source_hashes = {
        "Aster/paged_kv_adapter.py": manifest["repository_sources"][
            "aster/inference/paged_kv_adapter.py"
        ],
        "vllm-metal/paged_ops.cpp": reference_hashes["paged_ops.cpp"],
        "vllm-metal/reshape_and_cache.metal": reference_hashes["reshape_and_cache.metal"],
        "CMakeLists.txt": manifest["artifact_sources"]["aster-layout/CMakeLists.txt"],
        "kv_scatter.cpp": manifest["artifact_sources"]["aster-layout/kv_scatter.cpp"],
        "bench.py": manifest["artifact_sources"]["aster-layout/bench.py"],
    }
    toolchain = manifest["toolchain"]

    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    run_ids: set[int] = set()
    process_ids: set[int] = set()
    environment: str | None = None
    swap_growth: list[int] = []
    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = payload["run_id"]
        process_id = payload["environment"]["pid"]
        if run_id in run_ids or process_id in process_ids:
            raise ValueError("Run IDs and process IDs must be unique")
        run_ids.add(run_id)
        process_ids.add(process_id)
        if payload["manifest_sha256"] != manifest_hash:
            raise ValueError("Benchmark manifest hash does not match")
        if payload["source_hashes"] != expected_source_hashes:
            raise ValueError("Benchmark source hashes do not match manifest")
        if payload["correctness"]["max_abs"] != 0.0:
            raise ValueError("Correctness parity must be exact")
        if payload["environment"]["python"] != toolchain["python"]:
            raise ValueError("Python version does not match manifest")
        if payload["environment"]["mlx"] != toolchain["mlx"]:
            raise ValueError("MLX version does not match manifest")
        if payload["environment"]["nanobind"] != toolchain["root_nanobind"]:
            raise ValueError("nanobind version does not match manifest")
        payload_environment = {
            key: value for key, value in payload["environment"].items() if key != "pid"
        }
        environment_key = json.dumps(payload_environment, sort_keys=True)
        if environment is None:
            environment = environment_key
        elif environment_key != environment:
            raise ValueError("Benchmark environments must match")
        swap_growth.append(validate_resources(payload))

        cell_keys = [(record["tokens"], record["batch"]) for record in payload["results"]]
        validate_cells(cell_keys, EXPECTED_CELLS)
        for record in payload["results"]:
            validate_measurements(
                record,
                METHODS,
                expected_warmups=EXPECTED_WARMUPS,
                expected_iterations=EXPECTED_ITERATIONS,
            )
            validate_recorded_delta(record, "fused_vs_mlx_pct", "mlx_scatter", "fused_primitive")
            if record.get("post_benchmark_max_abs") != 0.0:
                raise ValueError("Post-benchmark parity must be exact")
            grouped[(record["tokens"], record["batch"])].append(record)

    if any(len(records) != 5 for records in grouped.values()):
        raise ValueError("Every benchmark cell requires five process records")

    generator = random.Random(0xA57E048)
    output = {
        "note": "Exploratory paired moving-block/process bootstrap intervals.",
        "run_ids": sorted(run_ids),
        "process_ids": sorted(process_ids),
        "manifest_sha256": manifest_hash,
        "source_hashes": expected_source_hashes,
        "aggregator_sha256": sha256(Path(__file__)),
        "max_swap_growth_bytes": max(swap_growth),
        "thermal_warning": False,
        "cells": [
            {
                "tokens": tokens,
                "batch": batch,
                "fused_vs_mlx_pct": paired_point_delta(records, "mlx_scatter", "fused_primitive"),
                "fused_vs_mlx_ci95_pct": bootstrap_delta(
                    records,
                    "mlx_scatter",
                    "fused_primitive",
                    resamples=args.resamples,
                    generator=generator,
                ),
            }
            for (tokens, batch), records in sorted(grouped.items())
        ],
    }
    encoded = json.dumps(output, allow_nan=False, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(output, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
