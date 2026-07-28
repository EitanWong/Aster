#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
)
for path in (PROJECT_ROOT, BASE_ARTIFACT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sampling_benchmark as sampling  # noqa: E402

from aster.inference.model_runner import ModelRunner  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_seconds": statistics.median(values) if values else 0.0,
        "p95_seconds": _percentile(values, 0.95),
        "mean_seconds": statistics.mean(values) if values else 0.0,
    }


def _measure(mx: Any, rows: list[Any], repeats: int) -> dict[str, Any]:
    normalized: list[float] = []
    raw: list[float] = []
    normalizer_only: list[float] = []

    # Warm both graph shapes before collecting timings.
    for _ in range(2):
        mx.eval([mx.argmax(row, axis=-1) for row in rows])
        mx.eval(
            [
                mx.argmax(
                    row - mx.logsumexp(row, axis=-1, keepdims=True),
                    axis=-1,
                )
                for row in rows
            ]
        )

    for index in range(repeats):
        if index % 2 == 0:
            started = time.perf_counter()
            mx.eval(
                [
                    mx.argmax(
                        row - mx.logsumexp(row, axis=-1, keepdims=True),
                        axis=-1,
                    )
                    for row in rows
                ]
            )
            normalized.append(time.perf_counter() - started)

            started = time.perf_counter()
            mx.eval([mx.argmax(row, axis=-1) for row in rows])
            raw.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            mx.eval([mx.argmax(row, axis=-1) for row in rows])
            raw.append(time.perf_counter() - started)

            started = time.perf_counter()
            mx.eval(
                [
                    mx.argmax(
                        row - mx.logsumexp(row, axis=-1, keepdims=True),
                        axis=-1,
                    )
                    for row in rows
                ]
            )
            normalized.append(time.perf_counter() - started)

        started = time.perf_counter()
        mx.eval([mx.logsumexp(row, axis=-1, keepdims=True) for row in rows])
        normalizer_only.append(time.perf_counter() - started)

    return {
        "normalized_argmax": _summary(normalized),
        "raw_argmax": _summary(raw),
        "logsumexp_only": _summary(normalizer_only),
        "median_graph_delta_seconds": statistics.median(normalized)
        - statistics.median(raw),
        "median_graph_delta_fraction_of_normalized": (
            (statistics.median(normalized) - statistics.median(raw))
            / statistics.median(normalized)
            if normalized and statistics.median(normalized)
            else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = sampling.base._settings(
        args.config,
        args.model,
        cache_kind="native",
        batch_size=8,
    )
    runner = ModelRunner(settings)
    runner.warmup()
    mx = runner._mx
    if mx is None:
        raise RuntimeError("MLX failed to load")
    sampling._ACTIVE_ARGS = args
    try:
        sampling.base._warmup(
            runner,
            tokens=args.model_warmup_tokens,
            prefill_step=args.prefill_step,
        )
        lanes = [
            sampling._prepare_lane(
                runner,
                request_id=f"iter052-profile-lane-{index}",
                prompt=sampling.base._prompt(args.context_words),
                max_tokens=args.max_tokens,
                prefill_step=args.prefill_step,
            )[0]
            for index in range(8)
        ]
        results: dict[str, Any] = {}
        for batch_size in (1, 2, 4, 8):
            selected = lanes[:batch_size]
            merged_cache, _ = runner._get_decode_batch_cache(
                [sampling.base._work_item(lane, args.max_tokens) for lane in selected]
            )
            input_tokens = mx.array(
                [[lane.input_token] for lane in selected], dtype=mx.uint32
            )
            logits = runner._model(input_tokens, cache=merged_cache)[:, -1, :]
            mx.eval(logits)
            rows = [logits[index : index + 1] for index in range(batch_size)]
            measured = _measure(mx, rows, args.repeats)
            measured.update(
                {
                    "batch_size": batch_size,
                    "vocab_size": int(logits.shape[-1]),
                    "dtype": str(logits.dtype),
                }
            )
            results[str(batch_size)] = measured
            mx.clear_cache()
        return {
            "schema_version": 1,
            "pid": os.getpid(),
            "settings": {
                "context_words": args.context_words,
                "repeats": args.repeats,
                "model_warmup_tokens": args.model_warmup_tokens,
                "prefill_step": args.prefill_step,
                "actual_prompt_tokens": len(lanes[0].prompt_tokens),
            },
            "results": results,
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sampling.base._sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    PROJECT_ROOT / "aster/inference/model_runner.py",
                    BASE_ARTIFACT_DIR / "sampling_benchmark.py",
                    args.config,
                )
            },
            "model_input_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sampling.base._sha256(path)
                for path in sampling.base._model_inputs(args.model)
            },
        }
    finally:
        sampling._ACTIVE_ARGS = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--model-warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.workload = "greedy"
    if min(args.repeats, args.model_warmup_tokens) < 1:
        raise ValueError("repeats and warmup tokens must be positive")

    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
