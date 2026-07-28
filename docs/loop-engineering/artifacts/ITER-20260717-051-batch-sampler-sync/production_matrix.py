#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_matrix as matrix

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BENCHMARK = ARTIFACT_DIR / "production_benchmark.py"
POLICIES = ("baseline", "production")
SHORT_CELLS = (
    ("greedy", 2),
    ("greedy", 4),
    ("mixed", 2),
    ("mixed", 4),
    ("penalties", 2),
    ("penalties", 4),
    ("structured", 2),
    ("structured", 4),
    ("greedy", 8),
    ("mixed", 8),
)
LONG_CELLS = (
    ("greedy", 2),
    ("greedy", 4),
    ("mixed", 2),
    ("mixed", 4),
)


def _policy_order(run_id: int, cell_index: int) -> tuple[str, ...]:
    shift = (run_id - 1 + cell_index) % len(POLICIES)
    return (*POLICIES[shift:], *POLICIES[:shift])


def run(args: argparse.Namespace) -> dict[str, Any]:
    cells = SHORT_CELLS if args.profile == "short" else LONG_CELLS
    original = {
        "benchmark": matrix.BENCHMARK,
        "policies": matrix.POLICIES,
        "cells": matrix.CELLS,
        "policy_order": matrix._policy_order,
        "source_hashes": matrix._source_hashes,
    }

    def source_hashes(config: Path) -> dict[str, str]:
        hashes = original["source_hashes"](config)
        for path in (
            ARTIFACT_DIR / "sampling_benchmark.py",
            Path(__file__).resolve(),
        ):
            hashes[str(path.relative_to(PROJECT_ROOT))] = matrix._sha256(path)
        return hashes

    matrix.BENCHMARK = BENCHMARK
    matrix.POLICIES = POLICIES
    matrix.CELLS = cells
    matrix._policy_order = _policy_order
    matrix._source_hashes = source_hashes
    try:
        manifest = matrix.run(args)
    finally:
        matrix.BENCHMARK = original["benchmark"]
        matrix.POLICIES = original["policies"]
        matrix.CELLS = original["cells"]
        matrix._policy_order = original["policy_order"]
        matrix._source_hashes = original["source_hashes"]

    manifest["matrix"]["profile"] = args.profile
    output = args.output_dir / "execution-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--context-words", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    short = args.profile == "short"
    args.runs = args.runs or (5 if short else 3)
    args.context_words = args.context_words or (128 if short else 2048)
    default_output = "production-confirmation" if short else "production-long-confirmation"
    args.output_dir = (args.output_dir or ARTIFACT_DIR / "results" / default_output).resolve()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    if args.runs < 1 or args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("run and token counts must be positive")

    manifest = run(args)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "records": len(manifest["records"]),
                "wall_seconds": manifest["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
