#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_matrix as matrix

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
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

    if args.profile == "short":
        cells = SHORT_CELLS
        default_runs = 5
        default_context_words = 128
        default_output = ARTIFACT_DIR / "results/confirmation"
    else:
        cells = LONG_CELLS
        default_runs = 3
        default_context_words = 2048
        default_output = ARTIFACT_DIR / "results/long-confirmation"

    args.runs = args.runs or default_runs
    args.context_words = args.context_words or default_context_words
    args.output_dir = (args.output_dir or default_output).resolve()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    if args.runs < 1 or args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("run and token counts must be positive")

    matrix.POLICIES = ("baseline", "grouped-async")
    matrix.CELLS = cells
    manifest = matrix.run(args)
    source = Path(__file__).resolve()
    manifest["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = matrix._sha256(
        source
    )
    manifest["matrix"]["profile"] = args.profile
    output = args.output_dir / "execution-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
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
