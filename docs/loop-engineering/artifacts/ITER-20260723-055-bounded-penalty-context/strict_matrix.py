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
BENCHMARK = ARTIFACT_DIR / "candidate_benchmark.py"
AGGREGATE = ARTIFACT_DIR / "strict_aggregate.py"
PYTHON = PROJECT_ROOT / ".venv/bin/python"
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


def _replicate_id(run_id: int) -> int:
    if run_id < 1:
        raise ValueError("run_id must be positive")
    return (run_id + 1) // 2


def _source_hashes(config: Path) -> dict[str, str]:
    base_dir = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
    paths = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/engine.py",
        PROJECT_ROOT / "aster/inference/request_state.py",
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        config,
        base_dir / "paired_benchmark.py",
        base_dir / "production_benchmark.py",
        base_dir / "sampling_benchmark.py",
        ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync/benchmark.py",
        BENCHMARK,
        AGGREGATE,
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in paths}


def _payload_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    analysis_paths = {
        str(AGGREGATE.relative_to(PROJECT_ROOT)),
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
    }
    return {
        path: digest
        for path, digest in source_hashes.items()
        if path not in analysis_paths
    }


def _model_input_hashes(model: Path) -> dict[str, str]:
    paths = [path for name in MODEL_INPUT_NAMES if (path := model / name).is_file()]
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
    run_id: int,
    args: argparse.Namespace,
) -> None:
    if payload["source_sha256"] != expected_source_hashes:
        raise RuntimeError("paired payload source hashes differ from the matrix")
    if payload["model_input_sha256"] != expected_model_hashes:
        raise RuntimeError("paired payload model input hashes differ from the matrix")
    if (
        str(payload["workload"]),
        int(payload["batch_size"]),
        int(payload["run_id"]),
    ) != ("penalties", args.batch_size, run_id):
        raise RuntimeError("paired payload identity mismatch")
    settings = payload["settings"]
    if (
        int(payload["context_words"]),
        int(settings["steps"]),
        int(settings["pair_warmup_steps"]),
        int(settings["model_warmup_tokens"]),
        int(settings["prefill_step"]),
        int(settings["block_size"]),
        int(settings["seed"]),
    ) != (
        args.context_words,
        args.steps,
        args.pair_warmup_steps,
        args.model_warmup_tokens,
        args.prefill_step,
        args.block_size,
        args.seed + _replicate_id(run_id),
    ):
        raise RuntimeError("paired payload settings mismatch")
    assignment = (
        {"baseline": "runner_a", "production": "runner_b"}
        if run_id % 2 == 1
        else {"baseline": "runner_b", "production": "runner_a"}
    )
    design = payload.get("comparison_design", {})
    if (
        design.get("runner_assignment_alternates_by_run") is not True
        or design.get("policy_runner_assignment") != assignment
        or int(design.get("assignment_balanced_replicate_id", 0))
        != _replicate_id(run_id)
    ):
        raise RuntimeError("paired payload runner assignment mismatch")
    expected_first = [
        "baseline" if (step + run_id - 1) % 2 == 0 else "production"
        for step in range(args.steps)
    ]
    timings = payload["timings"]
    if (
        len(timings["baseline_step_seconds"]) != args.steps
        or len(timings["production_step_seconds"]) != args.steps
        or timings["first_policy_by_step"] != expected_first
    ):
        raise RuntimeError("paired payload timing order mismatch")
    if payload["parity"]["exact_token_text_cache"] is not True:
        raise RuntimeError("paired payload lost exact parity")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("*.json")) and not args.resume:
        raise FileExistsError(f"{args.output_dir} already contains JSON files")
    source_hashes = _source_hashes(args.config)
    payload_source_hashes = _payload_source_hashes(source_hashes)
    model_hashes = _model_input_hashes(args.model)
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for run_id in range(1, args.runs + 1):
        output = args.output_dir / f"penalties-b{args.batch_size}-run-{run_id}.json"
        command = [
            str(PYTHON),
            str(BENCHMARK),
            "--config",
            str(args.config),
            "--model",
            str(args.model),
            "--workload",
            "penalties",
            "--batch-size",
            str(args.batch_size),
            "--context-words",
            str(args.context_words),
            "--steps",
            str(args.steps),
            "--pair-warmup-steps",
            str(args.pair_warmup_steps),
            "--model-warmup-tokens",
            str(args.model_warmup_tokens),
            "--prefill-step",
            str(args.prefill_step),
            "--block-size",
            str(args.block_size),
            "--seed",
            str(args.seed + _replicate_id(run_id)),
            "--run-id",
            str(run_id),
            "--output",
            str(output),
        ]
        resumed = output.is_file() and args.resume
        started = time.perf_counter()
        if resumed:
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
            expected_source_hashes=payload_source_hashes,
            expected_model_hashes=model_hashes,
            run_id=run_id,
            args=args,
        )
        records.append(
            {
                "output": str(output.relative_to(ARTIFACT_DIR)),
                "sha256": _sha256(output),
                "pid": int(payload["pid"]),
                "workload": "penalties",
                "batch_size": args.batch_size,
                "run_id": run_id,
                "resumed": resumed,
                "elapsed_seconds": time.perf_counter() - started,
                "command": command,
            }
        )
        baseline = float(payload["timings"]["baseline_tokens_per_second"])
        production = float(payload["timings"]["production_tokens_per_second"])
        print(
            f"run={run_id} speedup={(production / baseline - 1.0) * 100.0:.3f}% "
            f"exact={payload['parity']['exact_token_text_cache']}",
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
            "profile": "ultra-long-penalty",
            "cells": [{"workload": "penalties", "batch_size": args.batch_size}],
            "runs": args.runs,
            "independent_replicates": args.runs // 2,
            "context_words": args.context_words,
            "steps": args.steps,
            "pair_warmup_steps": args.pair_warmup_steps,
            "model_warmup_tokens": args.model_warmup_tokens,
            "prefill_step": args.prefill_step,
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
        default=ARTIFACT_DIR / "results/strict-ultra-long-r18",
    )
    parser.add_argument("--runs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--context-words", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--pair-warmup-steps", type=int, default=8)
    parser.add_argument("--model-warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.runs < 2 or args.runs > 30 or args.runs % 2:
        raise ValueError("runs must be an even number between 2 and 30")
    if min(args.steps, args.pair_warmup_steps, args.model_warmup_tokens) < 1:
        raise ValueError("step counts must be positive")
    if args.block_size < 1 or args.steps % args.block_size:
        raise ValueError("steps must be divisible by block size")

    manifest = run(args)
    print(
        json.dumps(
            {"records": len(manifest["records"]), "wall_seconds": manifest["wall_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
