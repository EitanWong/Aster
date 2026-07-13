"""Benchmark the experimental block-indexed Metal attention boundary."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from typing import Any

import mlx.core as mx

from aster.inference.metal_paged_attention import paged_block_attention


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(max(math.ceil(percentile / 100 * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _measure(
    kind: str,
    query: Any,
    key_pool: Any,
    value_pool: Any,
    block_indices: Any,
    dense_keys: Any,
    dense_values: Any,
    total_tokens: int,
) -> tuple[float, float]:
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()
    start = time.perf_counter()
    if kind == "native":
        output = mx.fast.scaled_dot_product_attention(
            query,
            dense_keys,
            dense_values,
            scale=1.0 / math.sqrt(query.shape[-1]),
        )
    else:
        output = paged_block_attention(
            query,
            key_pool,
            value_pool,
            block_indices,
            query_offset=total_tokens - 1,
            total_kv_tokens=total_tokens,
            scale=1.0 / math.sqrt(query.shape[-1]),
        )
    mx.eval(output)
    elapsed = time.perf_counter() - start
    peak_memory = (
        float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else 0.0
    )
    return elapsed, peak_memory


def _summary(records: list[dict[str, float | int | str]], kind: str) -> dict[str, float]:
    elapsed = [float(record["elapsed_s"]) for record in records if record["kind"] == kind]
    peaks = [float(record["peak_memory_gb"]) for record in records if record["kind"] == kind]
    return {
        "median_elapsed_s": _percentile(elapsed, 50),
        "p95_elapsed_s": _percentile(elapsed, 95),
        "min_elapsed_s": min(elapsed),
        "max_elapsed_s": max(elapsed),
        "median_peak_memory_gb": _percentile(peaks, 50),
    }


def benchmark(total_tokens: int, measurements: int) -> dict[str, Any]:
    block_size = 64
    num_blocks = (total_tokens + block_size - 1) // block_size
    mx.random.seed(20260714 + total_tokens)
    query = mx.random.normal((1, 8, 1, 256)).astype(mx.float16)
    key_pool = mx.random.normal((num_blocks, 1, 2, block_size, 256)).astype(mx.float16)
    value_pool = mx.random.normal((num_blocks, 1, 2, block_size, 256)).astype(mx.float16)
    block_indices = mx.array(list(reversed(range(num_blocks))), dtype=mx.uint32)
    dense_keys = mx.concatenate(
        [key_pool[index] for index in reversed(range(num_blocks))], axis=2
    )[..., :total_tokens, :]
    dense_values = mx.concatenate(
        [value_pool[index] for index in reversed(range(num_blocks))], axis=2
    )[..., :total_tokens, :]
    mx.eval(query, key_pool, value_pool, block_indices, dense_keys, dense_values)

    scale = 1.0 / math.sqrt(query.shape[-1])
    native_output = mx.fast.scaled_dot_product_attention(
        query, dense_keys, dense_values, scale=scale
    )
    paged_output = paged_block_attention(
        query,
        key_pool,
        value_pool,
        block_indices,
        query_offset=total_tokens - 1,
        total_kv_tokens=total_tokens,
        scale=scale,
    )
    mx.eval(native_output, paged_output)

    for kind in ("native", "paged"):
        _measure(
            kind,
            query,
            key_pool,
            value_pool,
            block_indices,
            dense_keys,
            dense_values,
            total_tokens,
        )

    order = [
        (trial, kind)
        for trial in range(1, measurements + 1)
        for kind in ("native", "paged")
    ]
    random.Random(20260714 + total_tokens).shuffle(order)
    records: list[dict[str, float | int | str]] = []
    for trial, kind in order:
        elapsed, peak_memory = _measure(
            kind,
            query,
            key_pool,
            value_pool,
            block_indices,
            dense_keys,
            dense_values,
            total_tokens,
        )
        records.append(
            {
                "trial": trial,
                "kind": kind,
                "elapsed_s": elapsed,
                "peak_memory_gb": peak_memory,
            }
        )

    native = _summary(records, "native")
    paged = _summary(records, "paged")
    return {
        "shape": {"queries": list(query.shape), "pool": list(key_pool.shape)},
        "block_size": block_size,
        "total_tokens": total_tokens,
        "max_abs_difference": float(mx.max(mx.abs(native_output - paged_output))),
        "warmups_per_kind": 1,
        "measurements_per_kind": measurements,
        "native": native,
        "tiled_paged_kernel": paged,
        "paged_over_native_median_ratio": (
            paged["median_elapsed_s"] / native["median_elapsed_s"]
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--measurements", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "kernel": "lane0-softmax-broadcast-correct-dispatch",
                "results": {
                    str(total_tokens): benchmark(total_tokens, args.measurements)
                    for total_tokens in args.tokens
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
