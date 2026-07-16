#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from run_matrix import preserve_executable_path
from turboquant_aggregate import aggregate_turboquant

ARTIFACT_DIR = Path(__file__).resolve().parent
BENCHMARK = ARTIFACT_DIR / "turboquant_bench.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", nargs="+", type=int, default=(2048, 8192, 32768, 65536))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--seed", type=int, default=49_117)
    args = parser.parse_args()

    python = preserve_executable_path(args.python, cwd=Path.cwd())
    results = args.results.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"benchmark Python is missing: {python}")
    if args.runs < 1 or args.iterations < 1 or args.warmups < 1:
        raise ValueError("runs, iterations, and warmups must be positive")
    if len(set(args.tokens)) != len(args.tokens):
        raise ValueError("token counts must be unique")
    results.mkdir(parents=True, exist_ok=True)

    output_paths = [results / f"run-{run_id}.json" for run_id in range(1, args.runs + 1)]
    manifest_path = results / "execution-manifest.json"
    aggregate_path = results / "aggregate.json"
    occupied = [path for path in (*output_paths, manifest_path, aggregate_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"matrix outputs already exist: {occupied[:3]}")

    base_manifest: dict[str, object] = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": str(python),
        "runs": args.runs,
        "tokens": args.tokens,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "seed": args.seed,
        "benchmark_sha256": _sha256(BENCHMARK),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "cells": [],
    }
    completed: list[dict[str, object]] = []
    _write_json(manifest_path, base_manifest)

    for run_id, output in enumerate(output_paths, start=1):
        print(f"[{run_id}/{args.runs}] TurboQuant fresh process", flush=True)
        command = [
            str(python),
            str(BENCHMARK),
            "--run-id",
            str(run_id),
            "--tokens",
            *(str(tokens) for tokens in args.tokens),
            "--iterations",
            str(args.iterations),
            "--warmups",
            str(args.warmups),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
        ]
        started = time.perf_counter()
        process = subprocess.run(command, check=False)
        completed = [
            *completed,
            {
                "run_id": run_id,
                "output": output.name,
                "elapsed_seconds": time.perf_counter() - started,
                "exit_code": process.returncode,
            },
        ]
        _write_json(
            manifest_path,
            {
                **base_manifest,
                "cells": completed,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    records = [json.loads(path.read_text()) for path in output_paths]
    aggregate = aggregate_turboquant(
        records,
        expected_runs=args.runs,
        expected_tokens=args.tokens,
        expected_iterations=args.iterations,
    )
    _write_json(aggregate_path, aggregate)
    _write_json(
        manifest_path,
        {
            **base_manifest,
            "cells": completed,
            "completed_utc": datetime.now(UTC).isoformat(),
            "aggregate": aggregate_path.name,
            "aggregate_sha256": _sha256(aggregate_path),
        },
    )


if __name__ == "__main__":
    main()
