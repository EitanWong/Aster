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
BENCHMARK = ARTIFACT_DIR / "paired_benchmark.py"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
SHORT_CELLS = (
    ("greedy", 2),
    ("greedy", 4),
    ("greedy", 8),
    ("mixed", 2),
    ("mixed", 4),
    ("mixed", 8),
    ("penalties", 2),
    ("penalties", 4),
    ("structured", 2),
    ("structured", 4),
)
LONG_CELLS = (
    ("greedy", 2),
    ("greedy", 4),
    ("mixed", 2),
    ("mixed", 4),
)
WORKLOADS = frozenset(workload for workload, _batch_size in SHORT_CELLS)
MODEL_INPUT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _run_order(run_count: int, cell_index: int) -> tuple[int, ...]:
    if run_count < 1:
        raise ValueError("run_count must be positive")
    runs = tuple(range(1, run_count + 1))
    shift = cell_index % len(runs)
    return (*runs[shift:], *runs[:shift])


def _replicate_id(run_id: int) -> int:
    if run_id < 1:
        raise ValueError("run_id must be positive")
    return (run_id + 1) // 2


def _execution_order(
    cells: tuple[tuple[str, int], ...],
    run_count: int,
) -> tuple[tuple[str, int, int], ...]:
    if not cells:
        raise ValueError("at least one cell is required")
    scheduled: list[tuple[str, int, int]] = []
    for run_id in range(1, run_count + 1):
        shift = (run_id - 1) % len(cells)
        rotated = (*cells[shift:], *cells[:shift])
        scheduled.extend((workload, batch_size, run_id) for workload, batch_size in rotated)
    return tuple(scheduled)


def _parse_cell(value: str) -> tuple[str, int]:
    try:
        workload, raw_batch_size = value.rsplit(":", 1)
        batch_size = int(raw_batch_size)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("cell must use WORKLOAD:BATCH_SIZE") from exc
    if workload not in WORKLOADS or batch_size not in (2, 4, 8):
        raise argparse.ArgumentTypeError(f"unsupported paired cell: {value}")
    return workload, batch_size


def _source_hashes(config: Path) -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        config,
        BENCHMARK,
        ARTIFACT_DIR / "production_benchmark.py",
        ARTIFACT_DIR / "sampling_benchmark.py",
        ARTIFACT_DIR / "strict_aggregate.py",
        ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync/benchmark.py",
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in paths}


def _payload_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    analysis_paths = {
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        str((ARTIFACT_DIR / "strict_aggregate.py").relative_to(PROJECT_ROOT)),
    }
    return {
        path: digest
        for path, digest in source_hashes.items()
        if path not in analysis_paths
    }


def _model_input_hashes(model: Path) -> dict[str, str]:
    paths = [
        path
        for name in MODEL_INPUT_NAMES
        if (path := model / name).is_file()
    ]
    if not paths:
        raise FileNotFoundError(f"no benchmark model inputs found: {model}")
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
    }


def _validate_payload(
    payload: dict[str, Any],
    *,
    expected_source_hashes: dict[str, str],
    expected_model_hashes: dict[str, str],
    workload: str,
    batch_size: int,
    run_id: int,
    args: argparse.Namespace,
) -> None:
    observed_sources = payload["source_sha256"]
    if set(observed_sources) != set(expected_source_hashes):
        raise RuntimeError("paired payload source key set mismatch")
    if observed_sources != expected_source_hashes:
        raise RuntimeError("paired payload source hash mismatch")
    if payload["model_input_sha256"] != expected_model_hashes:
        raise RuntimeError("paired payload model input hash mismatch")
    identity = (
        str(payload["workload"]),
        int(payload["batch_size"]),
        int(payload["run_id"]),
    )
    if identity != (workload, batch_size, run_id):
        raise RuntimeError("paired payload identity mismatch")
    settings = payload["settings"]
    observed_settings = (
        int(payload["context_words"]),
        int(settings["steps"]),
        int(settings["pair_warmup_steps"]),
        int(settings["block_size"]),
        int(settings["seed"]),
    )
    expected_settings = (
        int(args.context_words),
        int(args.steps),
        int(args.pair_warmup_steps),
        int(args.block_size),
        int(args.seed) + _replicate_id(run_id),
    )
    if observed_settings != expected_settings:
        raise RuntimeError("paired payload settings mismatch")
    expected_assignment = (
        {"baseline": "runner_a", "production": "runner_b"}
        if run_id % 2 == 1
        else {"baseline": "runner_b", "production": "runner_a"}
    )
    comparison_design = payload.get("comparison_design", {})
    if (
        comparison_design.get("runner_assignment_alternates_by_run") is not True
        or comparison_design.get("policy_runner_assignment") != expected_assignment
        or int(comparison_design.get("assignment_balanced_replicate_id", 0))
        != _replicate_id(run_id)
    ):
        raise RuntimeError("paired payload policy runner assignment mismatch")
    baseline = payload["timings"]["baseline_step_seconds"]
    production = payload["timings"]["production_step_seconds"]
    first_policies = payload["timings"]["first_policy_by_step"]
    expected_first = [
        "baseline" if (step + run_id - 1) % 2 == 0 else "production"
        for step in range(args.steps)
    ]
    if (
        len(baseline) != args.steps
        or len(production) != args.steps
        or first_policies != expected_first
    ):
        raise RuntimeError("paired payload timing/order mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cells = (
        tuple(args.cells)
        if args.cells
        else (SHORT_CELLS if args.profile == "short" else LONG_CELLS)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("*.json")) and not args.resume:
        raise FileExistsError(f"{args.output_dir} already contains JSON files")
    source_hashes = _source_hashes(args.config)
    expected_payload_sources = _payload_source_hashes(source_hashes)
    expected_model_hashes = _model_input_hashes(args.model)
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for workload, batch_size, run_id in _execution_order(cells, args.runs):
        output = args.output_dir / f"{workload}-b{batch_size}-run-{run_id}.json"
        command = [
            str(PYTHON),
            str(BENCHMARK),
            "--config",
            str(args.config),
            "--model",
            str(args.model),
            "--workload",
            workload,
            "--batch-size",
            str(batch_size),
            "--context-words",
            str(args.context_words),
            "--steps",
            str(args.steps),
            "--pair-warmup-steps",
            str(args.pair_warmup_steps),
            "--block-size",
            str(args.block_size),
            "--seed",
            str(args.seed + _replicate_id(run_id)),
            "--run-id",
            str(run_id),
            "--output",
            str(output),
        ]
        if output.is_file() and args.resume:
            payload = json.loads(output.read_text())
        else:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"paired benchmark failed: {' '.join(command)}\n{completed.stderr}"
                )
            payload = json.loads(output.read_text())
        _validate_payload(
            payload,
            expected_source_hashes=expected_payload_sources,
            expected_model_hashes=expected_model_hashes,
            workload=workload,
            batch_size=batch_size,
            run_id=run_id,
            args=args,
        )
        records.append(
            {
                "output": str(output.relative_to(ARTIFACT_DIR)),
                "sha256": _sha256(output),
                "pid": int(payload["pid"]),
                "workload": workload,
                "batch_size": batch_size,
                "run_id": run_id,
                "command": command,
            }
        )
        baseline = float(payload["timings"]["baseline_tokens_per_second"])
        current = float(payload["timings"]["production_tokens_per_second"])
        speedup = (current / baseline - 1.0) * 100.0
        print(
            f"workload={workload} batch={batch_size} run={run_id} "
            f"speedup={speedup:.3f}% exact={payload['parity']['exact_token_text_cache']}",
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
            "profile": args.profile,
            "cells": [
                {"workload": workload, "batch_size": batch_size} for workload, batch_size in cells
            ],
            "runs": args.runs,
            "independent_replicates": args.runs // 2,
            "context_words": args.context_words,
            "steps": args.steps,
            "pair_warmup_steps": args.pair_warmup_steps,
            "block_size": args.block_size,
            "base_seed": args.seed,
            "fresh_processes": True,
            "within_process_adjacent_pairing": True,
            "alternating_ab_ba_order": True,
            "round_robin_cell_execution": True,
            "alternating_policy_runner_assignment": True,
            "paired_runner_assignment_replicates": True,
        },
        "source_sha256": source_hashes,
        "records": records,
        "wall_seconds": time.perf_counter() - started_all,
    }
    output = args.output_dir / "execution-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), required=True)
    parser.add_argument(
        "--cell",
        action="append",
        dest="cells",
        type=_parse_cell,
        help="limit the matrix to WORKLOAD:BATCH_SIZE; may be repeated",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runs", type=int, default=18)
    parser.add_argument("--context-words", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--pair-warmup-steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    short = args.profile == "short"
    args.context_words = args.context_words or (128 if short else 2048)
    args.steps = args.steps or (256 if short else 128)
    default_output = "paired-confirmation" if short else "paired-long-confirmation"
    args.output_dir = (args.output_dir or ARTIFACT_DIR / "results" / default_output).resolve()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    if args.runs < 2 or args.runs > 30 or args.runs % 2:
        raise ValueError("runs must be an even number between 2 and 30")
    if args.cells and len(set(args.cells)) != len(args.cells):
        raise ValueError("focused cells must be unique")
    if args.steps < 1 or args.steps % args.block_size:
        raise ValueError("steps must be positive and divisible by block size")

    manifest = run(args)
    print(
        json.dumps({"records": len(manifest["records"]), "wall_seconds": manifest["wall_seconds"]})
    )


if __name__ == "__main__":
    main()
