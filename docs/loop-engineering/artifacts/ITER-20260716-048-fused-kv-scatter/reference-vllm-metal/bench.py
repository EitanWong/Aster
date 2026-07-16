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
import nanobind

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(os.environ["VLLM_METAL_REFERENCE_ROOT"]).resolve()
METHODS = ("mlx_scatter", "fused_primitive")


@dataclass(frozen=True)
class Case:
    tokens: int
    heads: int = 8
    dimension: int = 128
    block_size: int = 64
    blocks: int = 16


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
    }


def reference_scatter(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slots: Any,
) -> tuple[Any, Any]:
    heads, dimension = int(key.shape[1]), int(key.shape[2])
    key_shape = key_cache.shape
    value_shape = value_cache.shape
    flat_key = key_cache.reshape(-1, heads, dimension)
    flat_key[slots] = key
    flat_value = value_cache.reshape(-1, heads, dimension)
    flat_value[slots] = value
    return flat_key.reshape(key_shape), flat_value.reshape(value_shape)


def make_inputs(case: Case, dtype: Any) -> tuple[Any, Any]:
    shape = (case.tokens, case.heads, case.dimension)
    key = (mx.random.normal(shape) * 0.125).astype(dtype)
    value = (mx.random.normal(shape) * 0.125).astype(dtype)
    mx.eval(key, value)
    return key, value


def make_caches(case: Case, dtype: Any) -> tuple[Any, Any]:
    shape = (case.blocks, case.block_size, case.heads, case.dimension)
    key_cache = mx.zeros(shape, dtype=dtype)
    value_cache = mx.zeros(shape, dtype=dtype)
    mx.eval(key_cache, value_cache)
    return key_cache, value_cache


def make_slot_maps(case: Case) -> list[Any]:
    capacity = case.blocks * case.block_size
    maps = []
    for shift in range(8):
        slots = [((index * 67) + shift * 131) % capacity for index in range(case.tokens)]
        slot_map = mx.array(slots, dtype=mx.int64)
        mx.eval(slot_map)
        maps.append(slot_map)
    return maps


def verify(ops: Any) -> dict[str, Any]:
    cases = []
    max_error = 0.0
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for heads, dimension in ((2, 64), (4, 128)):
            case = Case(tokens=5, heads=heads, dimension=dimension, blocks=4, block_size=16)
            key, value = make_inputs(case, dtype)
            slots = mx.array([3, 20, 7, 40, 1], dtype=mx.int64)
            ref_key, ref_value = reference_scatter(key, value, *make_caches(case, dtype), slots)
            out_key, out_value = ops.reshape_and_cache(key, value, *make_caches(case, dtype), slots)
            mx.eval(ref_key, ref_value, out_key, out_value)
            error = max(
                float(mx.max(mx.abs(ref_key - out_key)).item()),
                float(mx.max(mx.abs(ref_value - out_value)).item()),
            )
            max_error = max(max_error, error)
            cases.append(
                {
                    "dtype": str(dtype),
                    "heads": heads,
                    "dimension": dimension,
                    "max_abs": error,
                }
            )

    padding_case = Case(tokens=3, heads=4, dimension=128, blocks=4, block_size=16)
    key, value = make_inputs(padding_case, mx.float16)
    slots = mx.array([5, -1, 9], dtype=mx.int64)
    out_key, out_value = ops.reshape_and_cache(
        key, value, *make_caches(padding_case, mx.float16), slots
    )
    ref_key, ref_value = reference_scatter(
        key[[0, 2]],
        value[[0, 2]],
        *make_caches(padding_case, mx.float16),
        mx.array([5, 9], dtype=mx.int64),
    )
    mx.eval(ref_key, ref_value, out_key, out_value)
    padding_error = max(
        float(mx.max(mx.abs(ref_key - out_key)).item()),
        float(mx.max(mx.abs(ref_value - out_value)).item()),
    )
    max_error = max(max_error, padding_error)
    if max_error != 0.0:
        raise AssertionError(f"Reference fused scatter parity failed: {max_error}")
    return {"cases": cases, "padding_max_abs": padding_error, "max_abs": max_error}


def benchmark_case(
    ops: Any,
    case: Case,
    *,
    warmups: int,
    iterations: int,
    generator: random.Random,
) -> dict[str, Any]:
    key, value = make_inputs(case, mx.float16)
    slot_maps = make_slot_maps(case)
    reference_state = list(make_caches(case, mx.float16))
    fused_state = list(make_caches(case, mx.float16))

    def call_reference(slot_map: Any) -> None:
        reference_state[:] = reference_scatter(
            key, value, reference_state[0], reference_state[1], slot_map
        )
        mx.eval(*reference_state)

    def call_fused(slot_map: Any) -> None:
        fused_state[:] = ops.reshape_and_cache(key, value, fused_state[0], fused_state[1], slot_map)
        mx.eval(*fused_state)

    calls: dict[str, Callable[[Any], None]] = {
        "mlx_scatter": call_reference,
        "fused_primitive": call_fused,
    }
    for index in range(warmups):
        order = list(METHODS)
        generator.shuffle(order)
        for name in order:
            calls[name](slot_maps[index % len(slot_maps)])

    samples: dict[str, list[float]] = {name: [] for name in METHODS}
    mx.reset_peak_memory()
    for index in range(iterations):
        order = list(METHODS)
        generator.shuffle(order)
        slot_map = slot_maps[index % len(slot_maps)]
        for name in order:
            started = time.perf_counter_ns()
            calls[name](slot_map)
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)

    mx.eval(*reference_state, *fused_state)
    post_benchmark_error = max(
        float(mx.max(mx.abs(reference_state[0] - fused_state[0])).item()),
        float(mx.max(mx.abs(reference_state[1] - fused_state[1])).item()),
    )
    reference_median = statistics.median(samples["mlx_scatter"])
    fused_median = statistics.median(samples["fused_primitive"])
    return {
        "tokens": case.tokens,
        "heads": case.heads,
        "dimension": case.dimension,
        "block_size": case.block_size,
        "blocks": case.blocks,
        "warmups": warmups,
        "iterations": iterations,
        "methods": {name: summarize(values) for name, values in samples.items()},
        "samples_ms": samples,
        "fused_vs_mlx_pct": 100.0 * (fused_median / reference_median - 1.0),
        "post_benchmark_max_abs": post_benchmark_error,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> tuple[dict[str, Any], str, dict[str, str]]:
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
    if nanobind.__version__ != toolchain["reference_nanobind"]:
        raise RuntimeError("nanobind version does not match manifest")

    sources = {
        "paged_ops.cpp": sha256(REFERENCE_ROOT / "metal/paged_ops.cpp"),
        "reshape_and_cache.metal": sha256(
            REFERENCE_ROOT / "metal/kernels_v2/reshape_and_cache.metal"
        ),
        "upstream_test": sha256(REFERENCE_ROOT.parent / "tests/test_reshape_and_cache.py"),
        "benchmark": sha256(Path(__file__)),
    }
    expected = {
        **manifest["reference"]["source_hashes"],
        "benchmark": manifest["artifact_sources"][relative_source],
    }
    if sources != expected:
        raise RuntimeError("Reference source does not match manifest")
    return manifest, sha256(manifest_path), sources


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
    manifest, manifest_hash, sources = verify_manifest()
    from vllm_metal.metal import get_ops

    mx.random.seed(0xA57E048 + args.run_id)
    ops = get_ops()
    correctness = verify(ops)
    cases = [Case(tokens=tokens) for tokens in (1, 2, 4, 8, 16, 64, 128)]
    results = []
    if not args.smoke_only:
        generator = random.Random(0xA57E048 + args.run_id)
        generator.shuffle(cases)
        results = [
            benchmark_case(
                ops,
                case,
                warmups=args.warmups,
                iterations=args.iterations,
                generator=generator,
            )
            for case in cases
        ]
    payload = {
        "run_id": args.run_id,
        "environment": {
            "device": mx.device_info(),
            "machine": platform.machine(),
            "mlx": mx.__version__,
            "nanobind": nanobind.__version__,
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "manifest_sha256": manifest_hash,
        "reference_commit": manifest["reference"]["commit"],
        "source_hashes": sources,
        "correctness": correctness,
        "results": results,
    }
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
