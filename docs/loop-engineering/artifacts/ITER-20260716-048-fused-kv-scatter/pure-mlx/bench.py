from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("separate", "combined", "combined_prestacked")


@dataclass(frozen=True)
class Case:
    batch: int
    tokens: int
    heads: int = 8
    dimension: int = 128
    block_size: int = 64
    capacity: int = 8


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> tuple[dict[str, Any], str]:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_source = Path(__file__).resolve().relative_to(ARTIFACT_ROOT).as_posix()
    if manifest["artifact_sources"].get(relative_source) != sha256(Path(__file__)):
        raise RuntimeError("Benchmark source does not match manifest")
    toolchain = manifest["toolchain"]
    if platform.python_version() != toolchain["python"]:
        raise RuntimeError("Python version does not match manifest")
    if mx.__version__ != toolchain["mlx"]:
        raise RuntimeError("MLX version does not match manifest")
    return manifest, sha256(manifest_path)


def make_inputs(case: Case) -> tuple[Any, Any]:
    shape = (case.batch, case.heads, case.tokens, case.dimension)
    keys = (mx.random.normal(shape) * 0.125).astype(mx.float16)
    values = (mx.random.normal(shape) * 0.125).astype(mx.float16)
    mx.eval(keys, values)
    return keys, values


def separate_write(
    key_pool: Any,
    value_pool: Any,
    block_id: int,
    block_offset: int,
    keys: Any,
    values: Any,
) -> None:
    end = block_offset + int(keys.shape[2])
    key_pool[block_id, ..., block_offset:end, :] = keys
    value_pool[block_id, ..., block_offset:end, :] = values


def combined_write(
    pool: Any,
    block_id: int,
    block_offset: int,
    keys: Any,
    values: Any,
) -> None:
    end = block_offset + int(keys.shape[2])
    pool[:, block_id, ..., block_offset:end, :] = mx.stack((keys, values), axis=0)


def combined_prestacked_write(
    pool: Any,
    block_id: int,
    block_offset: int,
    updates: Any,
) -> None:
    end = block_offset + int(updates.shape[3])
    pool[:, block_id, ..., block_offset:end, :] = updates


def empty_separate(case: Case) -> tuple[Any, Any]:
    shape = (
        case.capacity,
        case.batch,
        case.heads,
        case.block_size,
        case.dimension,
    )
    return mx.zeros(shape, dtype=mx.float16), mx.zeros(shape, dtype=mx.float16)


def empty_combined(case: Case) -> Any:
    key_pool, _ = empty_separate(case)
    return mx.zeros((2, *key_pool.shape), dtype=mx.float16)


def verify_case(case: Case) -> float:
    keys, values = make_inputs(case)
    key_pool, value_pool = empty_separate(case)
    combined_pool = empty_combined(case)
    block_id = case.capacity - 1
    block_offset = case.block_size - case.tokens
    separate_write(key_pool, value_pool, block_id, block_offset, keys, values)
    combined_write(combined_pool, block_id, block_offset, keys, values)
    mx.eval(key_pool, value_pool, combined_pool)
    key_error = float(mx.max(mx.abs(key_pool - combined_pool[0])).item())
    value_error = float(mx.max(mx.abs(value_pool - combined_pool[1])).item())
    return max(key_error, value_error)


def benchmark_case(
    case: Case,
    *,
    warmups: int,
    iterations: int,
    generator: random.Random,
) -> dict[str, Any]:
    keys, values = make_inputs(case)
    stacked = mx.stack((keys, values), axis=0)
    mx.eval(stacked)
    key_pool, value_pool = empty_separate(case)
    combined_pool = empty_combined(case)
    prestacked_pool = empty_combined(case)
    mx.eval(key_pool, value_pool, combined_pool, prestacked_pool)

    def call_separate(block_id: int) -> None:
        separate_write(key_pool, value_pool, block_id, 0, keys, values)
        mx.eval(key_pool, value_pool)

    def call_combined(block_id: int) -> None:
        combined_write(combined_pool, block_id, 0, keys, values)
        mx.eval(combined_pool)

    def call_prestacked(block_id: int) -> None:
        combined_prestacked_write(prestacked_pool, block_id, 0, stacked)
        mx.eval(prestacked_pool)

    calls: dict[str, Callable[[int], None]] = {
        "separate": call_separate,
        "combined": call_combined,
        "combined_prestacked": call_prestacked,
    }
    for index in range(warmups):
        order = list(METHODS)
        generator.shuffle(order)
        for name in order:
            calls[name](index % case.capacity)

    samples: dict[str, list[float]] = {name: [] for name in METHODS}
    mx.reset_peak_memory()
    for index in range(iterations):
        order = list(METHODS)
        generator.shuffle(order)
        block_id = index % case.capacity
        for name in order:
            started = time.perf_counter_ns()
            calls[name](block_id)
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)

    mx.eval(key_pool, value_pool, combined_pool, prestacked_pool)
    post_benchmark_error = max(
        float(mx.max(mx.abs(key_pool - combined_pool[0])).item()),
        float(mx.max(mx.abs(value_pool - combined_pool[1])).item()),
        float(mx.max(mx.abs(key_pool - prestacked_pool[0])).item()),
        float(mx.max(mx.abs(value_pool - prestacked_pool[1])).item()),
    )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "batch": case.batch,
        "tokens": case.tokens,
        "heads": case.heads,
        "dimension": case.dimension,
        "capacity": case.capacity,
        "warmups": warmups,
        "iterations": iterations,
        "methods": {name: summary(values) for name, values in samples.items()},
        "samples_ms": samples,
        "combined_vs_separate_pct": 100.0 * (medians["combined"] / medians["separate"] - 1.0),
        "prestacked_vs_separate_pct": 100.0
        * (medians["combined_prestacked"] / medians["separate"] - 1.0),
        "post_benchmark_max_abs": post_benchmark_error,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")
    _, manifest_hash = verify_manifest()
    mx.random.seed(0xA57E048 + args.run_id)
    cases = [Case(batch=batch, tokens=tokens) for tokens in (1, 16, 64) for batch in (1, 2)]
    correctness_cases = cases + [Case(batch=1, tokens=1, heads=1, dimension=64)]
    max_error = max(verify_case(case) for case in correctness_cases)
    if max_error != 0.0:
        raise AssertionError(f"Combined pool parity failed: {max_error}")
    results = []
    if not args.smoke_only:
        generator = random.Random(0xA57E048 + args.run_id)
        ordered_cases = list(cases)
        generator.shuffle(ordered_cases)
        results = [
            benchmark_case(
                case,
                warmups=args.warmups,
                iterations=args.iterations,
                generator=generator,
            )
            for case in ordered_cases
        ]
    payload = {
        "run_id": args.run_id,
        "environment": {
            "machine": platform.machine(),
            "mlx": mx.__version__,
            "device": mx.device_info(),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "manifest_sha256": manifest_hash,
        "source_sha256": sha256(Path(__file__)),
        "correctness_max_abs": max_error,
        "results": results,
    }
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
