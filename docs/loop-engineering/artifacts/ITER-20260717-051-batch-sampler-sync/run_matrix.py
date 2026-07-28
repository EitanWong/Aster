#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BENCHMARK = ARTIFACT_DIR / "sampling_benchmark.py"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
POLICIES = ("baseline", "grouped-eager", "grouped-lazy", "grouped-async")
CELLS = (
    ("greedy", 2),
    ("greedy", 4),
    ("mixed", 2),
    ("mixed", 4),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _policy_order(run_id: int, cell_index: int) -> tuple[str, ...]:
    shift = (run_id - 1 + cell_index) % len(POLICIES)
    return (*POLICIES[shift:], *POLICIES[:shift])


def _source_hashes(config: Path) -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        PROJECT_ROOT / "aster/inference/paged_attention_bridge.py",
        PROJECT_ROOT / "aster/inference/metal_paged_attention.py",
        config,
        BENCHMARK,
        Path(__file__).resolve(),
        ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync/benchmark.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
        if path.is_file()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("*.json")) and not args.resume:
        raise FileExistsError(f"{args.output_dir} already contains JSON files")
    expected_hashes = _source_hashes(args.config)
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for run_id in range(1, args.runs + 1):
        for cell_index, (workload, batch_size) in enumerate(CELLS):
            for order_index, policy in enumerate(_policy_order(run_id, cell_index)):
                output = args.output_dir / (
                    f"{workload}-b{batch_size}-run-{run_id}-{policy}.json"
                )
                command = [
                    str(PYTHON),
                    str(BENCHMARK),
                    "--config",
                    str(args.config),
                    "--model",
                    str(args.model),
                    "--policy",
                    policy,
                    "--workload",
                    workload,
                    "--batch-size",
                    str(batch_size),
                    "--context-words",
                    str(args.context_words),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--warmup-tokens",
                    str(args.warmup_tokens),
                    "--seed",
                    str(args.seed + run_id),
                    "--run-id",
                    str(run_id),
                    "--output",
                    str(output),
                ]
                if output.is_file() and args.resume:
                    payload = json.loads(output.read_text())
                    records.append(
                        {
                            "output": str(output.relative_to(ARTIFACT_DIR)),
                            "sha256": _sha256(output),
                            "pid": int(payload["pid"]),
                            "workload": workload,
                            "batch_size": batch_size,
                            "run_id": run_id,
                            "policy": policy,
                            "order_index": order_index,
                            "resumed": True,
                            "elapsed_seconds": None,
                            "command": command,
                        }
                    )
                    continue

                started = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONHASHSEED": "0"},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                elapsed = time.perf_counter() - started
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark failed ({completed.returncode}): {' '.join(command)}\n"
                        f"{completed.stderr}"
                    )
                payload = json.loads(output.read_text())
                observed = payload["source_sha256"]
                if any(expected_hashes.get(path) != digest for path, digest in observed.items()):
                    raise RuntimeError(f"source hash mismatch: {output}")
                records.append(
                    {
                        "output": str(output.relative_to(ARTIFACT_DIR)),
                        "sha256": _sha256(output),
                        "pid": int(payload["pid"]),
                        "workload": workload,
                        "batch_size": batch_size,
                        "run_id": run_id,
                        "policy": policy,
                        "order_index": order_index,
                        "resumed": False,
                        "elapsed_seconds": elapsed,
                        "command": command,
                    }
                )
                print(
                    f"run={run_id} workload={workload} batch={batch_size} "
                    f"policy={policy} tps={payload['decode']['tokens_per_second']:.3f}",
                    flush=True,
                )

    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "matrix": {
            "policies": list(POLICIES),
            "cells": [
                {"workload": workload, "batch_size": batch_size}
                for workload, batch_size in CELLS
            ],
            "runs": args.runs,
            "context_words": args.context_words,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "base_seed": args.seed,
            "fresh_processes": True,
            "rotated_policy_order": True,
        },
        "source_sha256": expected_hashes,
        "records": records,
        "wall_seconds": time.perf_counter() - started_all,
    }
    path = args.output_dir / "execution-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR / "results/screen",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.runs < 1 or args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("run and token counts must be positive")
    manifest = run(args)
    print(
        json.dumps(
            {
                "records": len(manifest["records"]),
                "wall_seconds": manifest["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
