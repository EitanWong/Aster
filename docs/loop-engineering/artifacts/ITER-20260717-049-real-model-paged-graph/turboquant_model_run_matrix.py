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

from run_matrix import preserve_executable_path
from turboquant_model_aggregate import aggregate_model_records

ARTIFACT_DIR = Path(__file__).resolve().parent
BENCHMARK = ARTIFACT_DIR / "turboquant_model_bench.py"


@dataclass(frozen=True, slots=True)
class ModelCell:
    context_tokens: int
    run_id: int
    variant: str


def model_cell_offset(*, base: int, stride: int, run_id: int) -> int:
    if base < 0 or stride < 0 or run_id < 1:
        raise ValueError("base/stride must be non-negative and run id positive")
    return base + (run_id - 1) * stride


def build_model_cells(
    *, contexts: tuple[int, ...], runs: tuple[int, ...], seed: int
) -> tuple[ModelCell, ...]:
    if not contexts or any(context < 1 for context in contexts):
        raise ValueError("contexts must be positive")
    if not runs or len(set(runs)) != len(runs) or any(run < 1 for run in runs):
        raise ValueError("run ids must be positive and unique")
    rng = random.Random(seed)
    blocks = [(context, run_id) for context in contexts for run_id in runs]
    rng.shuffle(blocks)
    cells: list[ModelCell] = []
    for context, run_id in blocks:
        variants = ["fp16", "turboquant"]
        rng.shuffle(variants)
        cells.extend(
            ModelCell(context_tokens=context, run_id=run_id, variant=variant)
            for variant in variants
        )
    return tuple(cells)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=(2048, 8192))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--teacher-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=1024)
    parser.add_argument("--offset-stride", type=int, default=16_384)
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=49_217)
    parser.add_argument("--fallback-reference-tests-passed", action="store_true")
    args = parser.parse_args()

    python = preserve_executable_path(args.python, cwd=Path.cwd())
    model = args.model.resolve()
    corpus = args.corpus.resolve()
    results = args.results.resolve()
    if not python.is_file() or not model.is_dir() or not corpus.is_file():
        raise FileNotFoundError("benchmark Python, model, or corpus is missing")
    if min(
        args.runs,
        args.teacher_tokens,
        args.generation_tokens,
        args.prefill_step,
    ) < 1:
        raise ValueError("runs and token counts must be positive")
    if len(set(args.contexts)) != len(args.contexts):
        raise ValueError("contexts must be unique")
    if args.offset < 0 or args.offset_stride < 0:
        raise ValueError("offset and offset stride must be non-negative")
    results.mkdir(parents=True, exist_ok=True)

    cells = build_model_cells(
        contexts=tuple(args.contexts),
        runs=tuple(range(1, args.runs + 1)),
        seed=args.seed,
    )
    outputs = [
        results
        / f"{cell.context_tokens}t-run-{cell.run_id}-{cell.variant}.json"
        for cell in cells
    ]
    manifest_path = results / "execution-manifest.json"
    aggregate_path = results / "aggregate.json"
    occupied = [path for path in (*outputs, manifest_path, aggregate_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"model matrix outputs already exist: {occupied[:3]}")

    base_manifest: dict[str, object] = {
        "schema_version": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "python": str(python),
        "model": str(model),
        "corpus": str(corpus),
        "corpus_sha256": _sha256(corpus),
        "contexts": args.contexts,
        "runs": args.runs,
        "teacher_tokens": args.teacher_tokens,
        "generation_tokens": args.generation_tokens,
        "prefill_step": args.prefill_step,
        "offset": args.offset,
        "offset_stride": args.offset_stride,
        "bits": args.bits,
        "seed": args.seed,
        "fallback_reference_tests_passed": args.fallback_reference_tests_passed,
        "benchmark_sha256": _sha256(BENCHMARK),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "cells": [],
    }
    completed: list[dict[str, object]] = []
    _write_json(manifest_path, base_manifest)

    for index, (cell, output) in enumerate(zip(cells, outputs, strict=True), start=1):
        print(
            f"[{index}/{len(cells)}] {cell.context_tokens}t "
            f"run={cell.run_id} {cell.variant}",
            flush=True,
        )
        command = [
            str(python),
            str(BENCHMARK),
            "--model",
            str(model),
            "--corpus",
            str(corpus),
            "--variant",
            cell.variant,
            "--context-tokens",
            str(cell.context_tokens),
            "--teacher-tokens",
            str(args.teacher_tokens),
            "--generation-tokens",
            str(args.generation_tokens),
            "--prefill-step",
            str(args.prefill_step),
            "--offset",
            str(
                model_cell_offset(
                    base=args.offset,
                    stride=args.offset_stride,
                    run_id=cell.run_id,
                )
            ),
            "--bits",
            str(args.bits),
            "--seed",
            str(args.seed),
            "--run-id",
            str(cell.run_id),
            "--output",
            str(output),
        ]
        started = time.perf_counter()
        process = subprocess.run(command, check=False)
        completed = [
            *completed,
            {
                **asdict(cell),
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

    records = [json.loads(path.read_text()) for path in outputs]
    aggregate = aggregate_model_records(
        records,
        expected_runs=args.runs,
        expected_contexts=args.contexts,
        expected_teacher_tokens=args.teacher_tokens,
        expected_generation_tokens=args.generation_tokens,
        fallback_reference_tests_passed=args.fallback_reference_tests_passed,
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
