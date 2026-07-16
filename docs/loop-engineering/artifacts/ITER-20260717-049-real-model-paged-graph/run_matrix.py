#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
BENCHMARK = ARTIFACT_DIR / "benchmark.py"


@dataclass(frozen=True, slots=True)
class Cell:
    mode: str
    context_words: int
    run_id: int
    variant: str


def build_cells(
    *,
    modes: tuple[str, ...],
    contexts: tuple[int, ...],
    runs: tuple[int, ...],
    seed: int,
) -> tuple[Cell, ...]:
    if not modes or any(mode not in {"control", "profile"} for mode in modes):
        raise ValueError("invalid matrix modes")
    if not contexts or any(context < 1 for context in contexts):
        raise ValueError("invalid matrix contexts")
    if not runs or len(set(runs)) != len(runs) or any(run < 0 for run in runs):
        raise ValueError("invalid matrix run ids")

    rng = random.Random(seed)
    blocks = [(mode, context, run_id) for mode in modes for context in contexts for run_id in runs]
    rng.shuffle(blocks)
    cells: list[Cell] = []
    for mode, context, run_id in blocks:
        variants = ["native", "direct"]
        rng.shuffle(variants)
        cells.extend(
            Cell(mode=mode, context_words=context, run_id=run_id, variant=variant)
            for variant in variants
        )
    return tuple(cells)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_runtime_hashes(model: Path) -> dict[str, str]:
    runtime_suffixes = {".json", ".jinja", ".model", ".safetensors", ".txt"}
    return {
        path.relative_to(model).as_posix(): _sha256(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
        and path.suffix in runtime_suffixes
        and not any(part.startswith(".") for part in path.relative_to(model).parts)
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def preserve_executable_path(path: Path, *, cwd: Path) -> Path:
    return path if path.is_absolute() else cwd / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("control", "profile"), default=("control", "profile"))
    parser.add_argument("--contexts", nargs="+", type=int, default=(2048, 8192))
    parser.add_argument("--runs", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=49_017)
    args = parser.parse_args()

    python = preserve_executable_path(args.python, cwd=Path.cwd())
    config = args.config.resolve()
    model = args.model.resolve()
    results = args.results.resolve()
    if not python.is_file() or not config.is_file() or not model.is_dir():
        raise FileNotFoundError("python, config, or model path is missing")
    if args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("token counts must be positive")
    results.mkdir(parents=True, exist_ok=True)

    cells = build_cells(
        modes=tuple(args.modes),
        contexts=tuple(args.contexts),
        runs=tuple(args.runs),
        seed=args.seed,
    )
    manifest_path = results / "execution-manifest.json"
    expected_outputs = [
        results / f"{cell.mode}-{cell.context_words}w-run-{cell.run_id}-{cell.variant}.json"
        for cell in cells
    ]
    occupied = [path for path in expected_outputs if path.exists()]
    if occupied or manifest_path.exists():
        raise FileExistsError(f"matrix outputs already exist: {occupied[:3] or [manifest_path]}")

    base_manifest: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": str(python),
        "config": str(config),
        "config_sha256": _sha256(config),
        "model": str(model),
        "model_file_sha256": _model_runtime_hashes(model),
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "benchmark_sha256": _sha256(BENCHMARK),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "cells": [],
    }
    completed: list[dict[str, object]] = []
    _write_manifest(manifest_path, base_manifest)

    for index, (cell, output) in enumerate(
        zip(cells, expected_outputs, strict=True), start=1
    ):
        print(
            f"[{index}/{len(cells)}] {cell.mode} {cell.context_words}w "
            f"run={cell.run_id} {cell.variant}",
            flush=True,
        )
        command = [
            str(python),
            str(BENCHMARK),
            "--config",
            str(config),
            "--model",
            str(model),
            "--variant",
            cell.variant,
            "--mode",
            cell.mode,
            "--context-words",
            str(cell.context_words),
            "--max-tokens",
            str(args.max_tokens),
            "--warmup-tokens",
            str(args.warmup_tokens),
            "--run-id",
            str(cell.run_id),
            "--output",
            str(output),
        ]
        started = time.perf_counter()
        process = subprocess.run(command, stdout=subprocess.DEVNULL, check=False)
        output_sha256 = _sha256(output) if process.returncode == 0 and output.is_file() else None
        completed = [
            *completed,
            {
                **asdict(cell),
                "output": output.name,
                "elapsed_seconds": time.perf_counter() - started,
                "exit_code": process.returncode,
                "output_sha256": output_sha256,
            },
        ]
        _write_manifest(
            manifest_path,
            {
                **base_manifest,
                "cells": completed,
                "completed_utc": datetime.now(UTC).isoformat(),
            },
        )
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    main()
