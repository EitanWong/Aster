#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
METHODS = ("mlx_fp16", "aster_paged_fp16", "tq_fused", "tq_dequant")


def build_method_orders(
    methods: Sequence[str], *, iterations: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Build deterministic Latin-square cycles to balance dispatch position."""
    names = tuple(methods)
    if not names or len(set(names)) != len(names):
        raise ValueError("methods must be non-empty and unique")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    rng = random.Random(seed)
    orders: list[tuple[str, ...]] = []
    while len(orders) < iterations:
        cycle = list(names)
        rng.shuffle(cycle)
        for offset in range(len(cycle)):
            rotated = tuple(cycle[offset:] + cycle[:offset])
            orders.append(rotated)
            if len(orders) == iterations:
                break
    return tuple(orders)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _command(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_commit(path: Path) -> str:
    return _command("git", "rev-parse", "HEAD", cwd=path)


def _swap_used_bytes() -> int | None:
    output = _command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _thermal_state() -> str:
    return _command("pmset", "-g", "therm")


def _summarize_ms(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("cannot summarize empty samples")
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(ordered),
    }


def _max_abs(mx: Any, left: Any, right: Any) -> float:
    value = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    return float(value.item())


def _mse(mx: Any, left: Any, right: Any) -> float:
    delta = left.astype(mx.float32) - right.astype(mx.float32)
    return float(mx.mean(delta * delta).item())


def _cosine(mx: Any, left: Any, right: Any) -> float:
    left32 = left.astype(mx.float32)
    right32 = right.astype(mx.float32)
    numerator = mx.sum(left32 * right32)
    denominator = mx.sqrt(mx.sum(left32 * left32) * mx.sum(right32 * right32))
    return float((numerator / mx.maximum(denominator, mx.array(1e-20))).item())


def _all_finite(mx: Any, value: Any) -> bool:
    return bool(mx.all(mx.isfinite(value)).item())


def _mlx_memory(mx: Any) -> dict[str, int]:
    values: dict[str, int] = {}
    for key, function_name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        function = getattr(mx, function_name, None)
        values[key] = int(function()) if callable(function) else 0
    return values


def _evaluate(mx: Any, operation: Callable[[], Any]) -> Any:
    output = operation()
    mx.eval(output)
    return output


def _time_operation_ms(mx: Any, operation: Callable[[], Any]) -> float:
    started = time.perf_counter_ns()
    output = operation()
    mx.eval(output)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    del output
    return elapsed


def _source_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _optional_source(path: Path) -> dict[str, str] | None:
    return _source_record(path) if path.is_file() else None


def _benchmark_context(
    *,
    mx: Any,
    KVCache: type,
    TurboQuantKVCache: type,
    paged_block_attention: Callable[..., Any],
    tokens: int,
    iterations: int,
    warmups: int,
    seed: int,
) -> dict[str, object]:
    batch = 1
    query_heads = 8
    kv_heads = 2
    query_tokens = 1
    head_dim = 256
    block_size = 64
    if tokens < block_size or tokens % block_size != 0:
        raise ValueError("token counts must be positive multiples of 64")

    mx.random.seed(seed)
    query = mx.random.normal(
        (batch, query_heads, query_tokens, head_dim)
    ).astype(mx.float16)
    keys = mx.random.normal((batch, kv_heads, tokens, head_dim)).astype(mx.float16)
    values = mx.random.normal((batch, kv_heads, tokens, head_dim)).astype(mx.float16)
    mx.eval(query, keys, values)

    fp_cache = KVCache()
    fp_keys, fp_values = fp_cache.update_and_fetch(keys, values)
    mx.eval(fp_keys, fp_values)
    tq_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0, seed=seed)
    tq_keys, tq_values = tq_cache.state

    physical_blocks = tokens // block_size
    key_pool = keys.reshape(
        batch, kv_heads, physical_blocks, block_size, head_dim
    ).transpose(2, 0, 1, 3, 4)
    value_pool = values.reshape(
        batch, kv_heads, physical_blocks, block_size, head_dim
    ).transpose(2, 0, 1, 3, 4)
    block_indices = mx.arange(physical_blocks, dtype=mx.uint32)
    scale = head_dim**-0.5
    mx.eval(key_pool, value_pool, block_indices)

    def mlx_fp16() -> Any:
        return mx.fast.scaled_dot_product_attention(
            query, fp_keys, fp_values, scale=scale
        )

    def aster_paged_fp16() -> Any:
        return paged_block_attention(
            query,
            key_pool,
            value_pool,
            block_indices,
            query_offset=tokens - query_tokens,
            total_kv_tokens=tokens,
            scale=scale,
        )

    def tq_fused() -> Any:
        return tq_cache.decode_attention(
            query,
            tq_keys,
            tq_values,
            scale=scale,
            mask=None,
        )

    def tq_dequant() -> Any:
        dequantized_keys, dequantized_values = tq_cache.dequantize(
            tq_keys, tq_values
        )
        return mx.fast.scaled_dot_product_attention(
            query.astype(mx.float32),
            dequantized_keys,
            dequantized_values,
            scale=scale,
        ).astype(query.dtype)

    operations: dict[str, Callable[[], Any]] = {
        "mlx_fp16": mlx_fp16,
        "aster_paged_fp16": aster_paged_fp16,
        "tq_fused": tq_fused,
        "tq_dequant": tq_dequant,
    }

    warmup_orders = build_method_orders(METHODS, iterations=warmups, seed=seed + 1)
    for order in warmup_orders:
        for name in order:
            _time_operation_ms(mx, operations[name])

    outputs = {name: _evaluate(mx, operation) for name, operation in operations.items()}
    dequantized_keys, dequantized_values = tq_cache.dequantize(tq_keys, tq_values)
    mx.eval(dequantized_keys, dequantized_values)
    correctness = {
        "all_finite": {name: _all_finite(mx, output) for name, output in outputs.items()},
        "aster_vs_mlx_max_abs": _max_abs(
            mx, outputs["aster_paged_fp16"], outputs["mlx_fp16"]
        ),
        "tq_fused_vs_dequant_max_abs": _max_abs(
            mx, outputs["tq_fused"], outputs["tq_dequant"]
        ),
        "tq_fused_vs_mlx_max_abs": _max_abs(
            mx, outputs["tq_fused"], outputs["mlx_fp16"]
        ),
        "tq_fused_vs_mlx_mse": _mse(
            mx, outputs["tq_fused"], outputs["mlx_fp16"]
        ),
        "key_dequant_mse": _mse(mx, dequantized_keys, fp_keys),
        "key_dequant_cosine": _cosine(mx, dequantized_keys, fp_keys),
        "value_dequant_mse": _mse(mx, dequantized_values, fp_values),
        "value_dequant_cosine": _cosine(mx, dequantized_values, fp_values),
    }
    del dequantized_keys, dequantized_values, outputs

    raw_samples: dict[str, list[float]] = {name: [] for name in METHODS}
    orders = build_method_orders(METHODS, iterations=iterations, seed=seed + 2)
    swap_before = _swap_used_bytes()
    thermal_before = _thermal_state()
    for order in orders:
        for name in order:
            raw_samples[name].append(_time_operation_ms(mx, operations[name]))
    thermal_after = _thermal_state()
    swap_after = _swap_used_bytes()

    method_memory: dict[str, dict[str, int]] = {}
    for name in build_method_orders(METHODS, iterations=1, seed=seed + 3)[0]:
        mx.synchronize()
        gc.collect()
        mx.clear_cache()
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        before = _mlx_memory(mx)
        _time_operation_ms(mx, operations[name])
        after = _mlx_memory(mx)
        method_memory[name] = {
            "active_before_bytes": before["active_bytes"],
            "active_after_bytes": after["active_bytes"],
            "peak_bytes": after["peak_bytes"],
            "incremental_peak_bytes": max(0, after["peak_bytes"] - before["active_bytes"]),
        }

    fp_bytes = int(fp_cache.nbytes)
    tq_bytes = int(tq_cache.nbytes)
    return {
        "tokens": tokens,
        "shape": {
            "batch": batch,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "query_tokens": query_tokens,
            "head_dim": head_dim,
            "block_size": block_size,
        },
        "scale": scale,
        "iterations": iterations,
        "warmups": warmups,
        "latency_ms": {
            name: {**_summarize_ms(samples), "samples": samples}
            for name, samples in raw_samples.items()
        },
        "correctness": correctness,
        "storage": {
            "fp16_bytes": fp_bytes,
            "turboquant_bytes": tq_bytes,
            "compression_ratio": fp_bytes / tq_bytes,
        },
        "method_memory": method_memory,
        "swap_used_before_bytes": swap_before,
        "swap_used_after_bytes": swap_after,
        "swap_delta_bytes": (
            None
            if swap_before is None or swap_after is None
            else swap_after - swap_before
        ),
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    from aster.inference.metal_paged_attention import paged_block_attention

    if not mx.metal.is_available():
        raise RuntimeError("Metal is required for this benchmark")
    runtime_source = Path(inspect.getsourcefile(TurboQuantKVCache) or "")
    source_candidates = {
        "benchmark": Path(__file__),
        "aster_paged_attention": PROJECT_ROOT
        / "aster/inference/metal_paged_attention.py",
        "mlx_vlm_turboquant_runtime": runtime_source,
        "omlx_turboquant_patch": PROJECT_ROOT
        / "examples/omlx/omlx/patches/turboquant_attention.py",
        "vllm_metal_turboquant": PROJECT_ROOT
        / "examples/vllm-metal/vllm_metal/metal/kernels_v2/turboquant.metal",
        "gemma4metal_turboquant": PROJECT_ROOT
        / "examples/gemma4metal/lib/turboquant.metal",
    }
    sources = {
        name: record
        for name, path in source_candidates.items()
        if (record := _optional_source(path)) is not None
    }

    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    started = time.perf_counter()
    contexts = [
        _benchmark_context(
            mx=mx,
            KVCache=KVCache,
            TurboQuantKVCache=TurboQuantKVCache,
            paged_block_attention=paged_block_attention,
            tokens=tokens,
            iterations=args.iterations,
            warmups=args.warmups,
            seed=args.seed + args.run_id * 10_000 + tokens,
        )
        for tokens in args.tokens
    ]
    return {
        "schema_version": 1,
        "benchmark": "turboquant_decode_attention",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "pid": os.getpid(),
        "seed": args.seed,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "mlx_vlm": _version("mlx-vlm"),
            "cpu": _command("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_bytes": _command("sysctl", "-n", "hw.memsize"),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "provenance": {
            "aster_commit": _git_commit(PROJECT_ROOT),
            "omlx_commit": _git_commit(PROJECT_ROOT / "examples/omlx"),
            "vllm_metal_commit": _git_commit(PROJECT_ROOT / "examples/vllm-metal"),
            "gemma4metal_commit": _git_commit(PROJECT_ROOT / "examples/gemma4metal"),
            "sources": sources,
        },
        "methods": list(METHODS),
        "total_elapsed_seconds": time.perf_counter() - started,
        "contexts": contexts,
        "process_memory": _mlx_memory(mx),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, default=(2048, 8192, 32768, 65536))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--seed", type=int, default=49_117)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_id < 1 or args.iterations < 1 or args.warmups < 1:
        raise ValueError("run id, iterations, and warmups must be positive")
    if len(set(args.tokens)) != len(args.tokens):
        raise ValueError("token counts must be unique")
    _write_json(args.output.resolve(), run(args))


if __name__ == "__main__":
    main()
