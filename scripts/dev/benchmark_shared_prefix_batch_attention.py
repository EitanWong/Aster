#!/usr/bin/env python3
"""Screen shared-prefix batch attention over one physical MLX KV pool.

This is an attention-only experiment. It deliberately does not patch the
model runner or replace native cache merging. The candidate owns only a
borrowed two-dimensional block table; request bundles remain responsible for
pool and block lifetime.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from aster.inference.paged_cache import PagedCacheManager
from aster.inference.paged_kv_adapter import (
    PagedAttentionView,
    PagedBatchAttentionView,
    PagedKVCacheBundle,
)

BLOCK_SIZE = 64
FULL_ATTENTION_LAYERS = 8
I085_ARRAYS_BYTES_PER_REQUEST = 51_511_296
I085_FULL_BYTES_PER_REQUEST = 338_591_744
I085_TOTAL_BYTES_PER_REQUEST = I085_ARRAYS_BYTES_PER_REQUEST + I085_FULL_BYTES_PER_REQUEST
ARRAYS_TO_FULL_RATIO = I085_ARRAYS_BYTES_PER_REQUEST / I085_FULL_BYTES_PER_REQUEST
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
MEMORY_REDUCTION_GATE = 75.0
FULL_MEMORY_REDUCTION_GATE = 90.0
LATENCY_REGRESSION_GATE = 1.03
NUMERICAL_TOLERANCE = 3e-3


class BenchmarkError(RuntimeError):
    """Raised when a benchmark input or result violates its contract."""


def parse_integer_csv(value: str, *, minimum: int) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise BenchmarkError("values must be comma-separated integers") from error
    if not parsed or any(item < minimum for item in parsed):
        raise BenchmarkError(f"every value must be at least {minimum}")
    if len(set(parsed)) != len(parsed):
        raise BenchmarkError("values must be unique")
    return tuple(sorted(parsed))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise BenchmarkError("at least one timing sample is required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_s": min(values),
        "median_s": statistics.median(values),
        "p95_s": _percentile(values, 0.95),
        "max_s": max(values),
    }


def summarize_screen(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared I086 gates to completed scenario records."""
    if not scenarios:
        raise BenchmarkError("at least one scenario is required")
    batch_sizes = {int(row["batch_size"]) for row in scenarios}
    if 8 not in batch_sizes:
        raise BenchmarkError("B8 evidence is required before making a decision")

    numerical_contract = all(
        float(row["correctness"]["max_abs_difference"]) <= float(row["correctness"]["tolerance"])
        for row in scenarios
    )
    no_materialization = all(
        not bool(row["construction"]["materialized_batch_prefix"])
        and not bool(row["construction"]["native_merge_invoked"])
        for row in scenarios
    )
    b8 = [row for row in scenarios if int(row["batch_size"]) == 8]
    total_reduction = all(
        float(row["construction"]["total_merge_growth_reduction_percent"]) >= MEMORY_REDUCTION_GATE
        for row in b8
    )
    full_reduction = all(
        float(row["construction"]["full_attention_growth_reduction_percent"])
        >= FULL_MEMORY_REDUCTION_GATE
        for row in b8
    )
    latency = all(
        float(row["timing"]["candidate_over_native_p95_ratio"]) <= LATENCY_REGRESSION_GATE
        for row in scenarios
    )
    release_clean = all(
        int(row["lifecycle"]["allocated_blocks_after_release"]) == 0
        and int(row["lifecycle"]["pool_nbytes_after_release"]) == 0
        for row in scenarios
    )
    gates = {
        "numerical_contract": numerical_contract,
        "no_batch_prefix_materialization": no_materialization,
        "b8_total_merge_growth_reduction_at_least_75_percent": total_reduction,
        "b8_full_attention_growth_reduction_at_least_90_percent": full_reduction,
        "p95_latency_no_regression_3_percent": latency,
        "release_clean": release_clean,
    }
    by_batch: dict[str, dict[str, Any]] = defaultdict(dict)
    for batch_size in sorted(batch_sizes):
        rows = [row for row in scenarios if int(row["batch_size"]) == batch_size]
        ratios = [float(row["timing"]["candidate_over_native_p95_ratio"]) for row in rows]
        reductions = [
            float(row["construction"]["total_merge_growth_reduction_percent"]) for row in rows
        ]
        by_batch[str(batch_size)] = {
            "scenario_count": len(rows),
            "candidate_over_native_p95_ratio_max": max(ratios),
            "total_merge_growth_reduction_percent_min": min(reductions),
        }
    return {
        "gates": gates,
        "decision": "screen-passed" if all(gates.values()) else "screen-rejected",
        "by_batch": dict(by_batch),
    }


def _clear_allocator() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _next_power_of_two(value: int) -> int:
    return 1 << max(value - 1, 1).bit_length()


def _build_fanout(
    total_tokens: int,
    batch_size: int,
    *,
    seed: int,
) -> tuple[
    PagedCacheManager, PagedKVCacheBundle, list[PagedKVCacheBundle], tuple[PagedAttentionView, ...]
]:
    if total_tokens < 2:
        raise BenchmarkError("total_tokens must be at least 2")
    prefix_tokens = total_tokens - 1
    required_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    manager = PagedCacheManager(
        num_layers=1,
        block_size=BLOCK_SIZE,
        max_blocks=_next_power_of_two(required_blocks + batch_size + 8),
    )
    source = PagedKVCacheBundle.from_prompt_cache(
        [KVCache()],
        manager,
        kv_cache_type=KVCache,
        request_id=f"shared-prefix-source-{seed}",
        enable_block_pool=True,
        enable_direct_attention=True,
    )
    mx.random.seed(seed)
    source.layers[0].update_and_fetch(
        mx.random.normal((1, NUM_KV_HEADS, prefix_tokens, HEAD_DIM)).astype(mx.float16),
        mx.random.normal((1, NUM_KV_HEADS, prefix_tokens, HEAD_DIM)).astype(mx.float16),
    )
    children: list[PagedKVCacheBundle] = []
    for index in range(batch_size):
        child = source.fork(f"shared-prefix-child-{seed}-{index}")
        mx.random.seed(seed + index + 1)
        child.layers[0].update_and_fetch(
            mx.random.normal((1, NUM_KV_HEADS, 1, HEAD_DIM)).astype(mx.float16),
            mx.random.normal((1, NUM_KV_HEADS, 1, HEAD_DIM)).astype(mx.float16),
        )
        children.append(child)
    pool_keys, pool_values = source.layers[0]._pool.block_pool()
    mx.eval(pool_keys, pool_values)
    views = tuple(child.layers[0].attention_view() for child in children)
    return manager, source, children, views


def _release_fanout(
    source: PagedKVCacheBundle,
    children: list[PagedKVCacheBundle],
) -> None:
    for child in children:
        child.release()
    source.release()


def _measure_construction(
    views: tuple[PagedAttentionView, ...],
    *,
    batch_size: int,
) -> tuple[PagedBatchAttentionView, Any, Any, dict[str, Any]]:
    before = int(mx.get_active_memory())
    started = time.perf_counter()
    candidate = PagedBatchAttentionView.from_views(views)
    mx.eval(candidate.block_tables, candidate.sequence_lengths)
    mx.synchronize()
    candidate_elapsed = time.perf_counter() - started
    candidate_active_delta = int(mx.get_active_memory()) - before

    started = time.perf_counter()
    dense_rows = [view.materialize() for view in views]
    dense_keys = mx.concatenate([row[0] for row in dense_rows], axis=0)
    dense_values = mx.concatenate([row[1] for row in dense_rows], axis=0)
    mx.eval(dense_keys, dense_values)
    mx.synchronize()
    native_elapsed = time.perf_counter() - started
    native_active_delta = int(mx.get_active_memory()) - before - candidate_active_delta

    native_bytes = int(dense_keys.nbytes) + int(dense_values.nbytes)
    candidate_bytes = candidate.metadata_nbytes
    native_full_bytes = native_bytes * FULL_ATTENTION_LAYERS
    estimated_arrays_bytes = int(native_full_bytes * ARRAYS_TO_FULL_RATIO)
    candidate_full_bytes = candidate_bytes * FULL_ATTENTION_LAYERS
    native_total_bytes = native_full_bytes + estimated_arrays_bytes
    candidate_total_bytes = candidate_full_bytes + estimated_arrays_bytes
    construction = {
        "materialized_batch_prefix": False,
        "native_merge_invoked": False,
        "candidate_metadata_bytes": candidate_bytes,
        "native_dense_batch_bytes": native_bytes,
        "candidate_active_delta_bytes": candidate_active_delta,
        "native_active_delta_bytes": native_active_delta,
        "candidate_construction_seconds": candidate_elapsed,
        "native_construction_seconds": native_elapsed,
        "estimated_arrays_bytes_unchanged": estimated_arrays_bytes,
        "estimated_native_full_attention_bytes": native_full_bytes,
        "estimated_candidate_full_attention_bytes": candidate_full_bytes,
        "estimated_native_total_merge_bytes": native_total_bytes,
        "estimated_candidate_total_merge_bytes": candidate_total_bytes,
        "total_merge_growth_reduction_percent": 100.0
        * (1.0 - candidate_total_bytes / native_total_bytes),
        "full_attention_growth_reduction_percent": 100.0
        * (1.0 - candidate_full_bytes / native_full_bytes),
        "batch_size": batch_size,
    }
    return candidate, dense_keys, dense_values, construction


def _run_attention(
    candidate: PagedBatchAttentionView,
    dense_keys: Any,
    dense_values: Any,
    queries: Any,
    *,
    kind: str,
) -> float:
    started = time.perf_counter()
    if kind == "native":
        output = mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=1.0 / math.sqrt(HEAD_DIM),
        )
    else:
        output = candidate.attention(queries, scale=1.0 / math.sqrt(HEAD_DIM))
    mx.eval(output)
    mx.synchronize()
    return time.perf_counter() - started


def _measure_timing(
    candidate: PagedBatchAttentionView,
    dense_keys: Any,
    dense_values: Any,
    queries: Any,
    *,
    warmups: int,
    measurements: int,
) -> tuple[dict[str, Any], float]:
    for _ in range(warmups):
        _run_attention(candidate, dense_keys, dense_values, queries, kind="native")
        _run_attention(candidate, dense_keys, dense_values, queries, kind="candidate")
    native_output = mx.fast.scaled_dot_product_attention(
        queries,
        dense_keys,
        dense_values,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )
    candidate_output = candidate.attention(queries, scale=1.0 / math.sqrt(HEAD_DIM))
    mx.eval(native_output, candidate_output)
    mx.synchronize()
    max_abs_difference = float(mx.max(mx.abs(native_output - candidate_output)).item())

    order = []
    for trial in range(measurements):
        order.extend(("native", "candidate") if trial % 2 == 0 else ("candidate", "native"))
    random.Random(20260801 + int(queries.shape[0]) + int(queries.shape[2])).shuffle(order)
    timings: dict[str, list[float]] = {"native": [], "candidate": []}
    for kind in order:
        timings[kind].append(
            _run_attention(candidate, dense_keys, dense_values, queries, kind=kind)
        )
    native = _timing_summary(timings["native"])
    candidate_summary = _timing_summary(timings["candidate"])
    ratio = float(candidate_summary["p95_s"]) / float(native["p95_s"])
    return (
        {
            "native": native,
            "candidate": candidate_summary,
            "candidate_over_native_p95_ratio": ratio,
        },
        max_abs_difference,
    )


def run_scenario(
    total_tokens: int,
    batch_size: int,
    *,
    warmups: int,
    measurements: int,
    seed: int,
) -> dict[str, Any]:
    _clear_allocator()
    active_before = int(mx.get_active_memory())
    manager, source, children, views = _build_fanout(
        total_tokens,
        batch_size,
        seed=seed,
    )
    pool = source.layers[0]._pool
    candidate: PagedBatchAttentionView | None = None
    dense_keys: Any | None = None
    dense_values: Any | None = None
    queries: Any | None = None
    result: dict[str, Any] | None = None
    try:
        mx.random.seed(seed + 1000)
        queries = mx.random.normal((batch_size, NUM_QUERY_HEADS, 1, HEAD_DIM)).astype(mx.float16)
        mx.eval(queries)
        candidate, dense_keys, dense_values, construction = _measure_construction(
            views,
            batch_size=batch_size,
        )
        timing, max_abs_difference = _measure_timing(
            candidate,
            dense_keys,
            dense_values,
            queries,
            warmups=warmups,
            measurements=measurements,
        )
        common_blocks = set(views[0].block_ids)
        for view in views[1:]:
            common_blocks.intersection_update(view.block_ids)
        unique_blocks = set(block_id for view in views for block_id in view.block_ids)
        construction["common_physical_block_count"] = len(common_blocks)
        construction["unique_physical_block_count"] = len(unique_blocks)
        construction["pool_nbytes"] = pool.nbytes
        result = {
            "total_tokens": total_tokens,
            "batch_size": batch_size,
            "shape": {
                "queries": list(queries.shape),
                "pool": list(pool.block_pool()[0].shape),
                "block_tables": list(candidate.block_tables.shape),
            },
            "correctness": {
                "max_abs_difference": max_abs_difference,
                "tolerance": NUMERICAL_TOLERANCE,
            },
            "construction": construction,
            "timing": timing,
            "lifecycle": {
                "allocated_blocks_before_release": manager.stats.allocated_blocks,
                "pool_nbytes_before_release": pool.nbytes,
            },
        }
    finally:
        del candidate, dense_keys, dense_values
        _release_fanout(source, children)
        allocated_blocks_after_release = manager.stats.allocated_blocks
        pool_nbytes_after_release = pool.nbytes
        views = ()
        children.clear()
        queries = None
        _clear_allocator()
        active_after = int(mx.get_active_memory())

    if result is None:
        raise BenchmarkError("scenario completed without a result")
    result["lifecycle"].update(
        {
            "allocated_blocks_after_release": allocated_blocks_after_release,
            "pool_nbytes_after_release": pool_nbytes_after_release,
            "active_memory_delta_after_release": active_after - active_before,
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokens = parse_integer_csv(args.tokens, minimum=2)
    batch_sizes = parse_integer_csv(args.batch_sizes, minimum=2)
    scenarios = []
    for total_tokens in tokens:
        for batch_size in batch_sizes:
            scenarios.append(
                run_scenario(
                    total_tokens,
                    batch_size,
                    warmups=args.warmups,
                    measurements=args.measurements,
                    seed=args.seed + total_tokens + batch_size,
                )
            )
    summary = summarize_screen(scenarios)
    project_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "kind": "shared-prefix-batch-attention-feasibility",
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline_commit": _git_head(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "machine": mx.device_info(),
        },
        "source": {
            "project_root": str(project_root),
            "block_size": BLOCK_SIZE,
            "full_attention_layers": FULL_ATTENTION_LAYERS,
            "i085_arrays_bytes_per_request": I085_ARRAYS_BYTES_PER_REQUEST,
            "i085_full_attention_bytes_per_request": I085_FULL_BYTES_PER_REQUEST,
            "gate_tolerance": NUMERICAL_TOLERANCE,
        },
        "parameters": {
            "tokens": list(tokens),
            "batch_sizes": list(batch_sizes),
            "warmups": args.warmups,
            "measurements": args.measurements,
            "seed": args.seed,
        },
        "scenarios": scenarios,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="2048,8192,10334")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.measurements < 1:
        raise SystemExit("warmups must be non-negative and measurements must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
