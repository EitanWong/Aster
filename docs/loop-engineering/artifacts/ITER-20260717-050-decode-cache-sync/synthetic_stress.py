#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import ArraysCache, KVCache

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.inference.paged_cache import PagedCacheManager  # noqa: E402
from aster.inference.paged_kv_adapter import PagedKVCacheLayer  # noqa: E402

POLICIES = ("baseline", "periodic-512", "skip-eval-no-clear")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mlx_memory() -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "rss_bytes": int(psutil.Process().memory_info().rss),
    }


def _apply_policy(policy: str, step: int, targets: Any) -> None:
    if policy == "baseline":
        mx.eval(targets)
        mx.clear_cache()
    elif policy == "periodic-512" and (step + 1) % 512 == 0:
        mx.clear_cache()


def _array_digest(arrays: list[Any]) -> str:
    mx.eval(arrays)
    digest = hashlib.sha256()
    for index, array in enumerate(arrays):
        digest.update(f"{index}:{tuple(array.shape)}:{array.dtype};".encode())
        digest.update(np.asarray(array.view(mx.uint8)).tobytes(order="C"))
    return digest.hexdigest()


def _sample_digest_update(digest: Any, value: int) -> None:
    digest.update(int(value).to_bytes(4, byteorder="little", signed=True))


def _run_native(policy: str, steps: int, sample_interval: int) -> dict[str, Any]:
    cache = KVCache()
    cache.step = 64
    sampled = hashlib.sha256()
    curve: list[dict[str, int]] = []
    mx.reset_peak_memory()
    started = time.perf_counter()
    for step in range(steps):
        phase = (step % 257) - 128
        keys = mx.full((1, 2, 1, 8), phase / 129.0, dtype=mx.float32)
        values = mx.full((1, 2, 1, 8), ((step * 17) % 251) / 251.0, dtype=mx.float32)
        all_keys, all_values = cache.update_and_fetch(keys, values)
        score = mx.sum(all_keys[..., -1, :] * 0.75 + all_values[..., -1, :] * 0.25)
        token = int(mx.argmax(mx.stack([score, -score])).item())
        _sample_digest_update(sampled, token)
        _apply_policy(policy, step, cache.state)
        if step == 0 or (step + 1) % sample_interval == 0:
            curve.append({"step": step + 1, **_mlx_memory()})
    elapsed = time.perf_counter() - started
    arrays = list(cache.state)
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "sample_digest": sampled.hexdigest(),
        "state_digest": _array_digest(arrays),
        "final_offset": cache.offset,
        "curve": curve,
        "memory": _mlx_memory(),
    }


def _run_recurrent(policy: str, steps: int, sample_interval: int) -> dict[str, Any]:
    cache = ArraysCache(2)
    cache[0] = mx.zeros((1, 3, 8), dtype=mx.float32)
    cache[1] = mx.zeros((1, 4, 8, 8), dtype=mx.float32)
    weights = mx.arange(64, dtype=mx.float32).reshape(1, 1, 8, 8) / 64.0
    sampled = hashlib.sha256()
    curve: list[dict[str, int]] = []
    mx.reset_peak_memory()
    started = time.perf_counter()
    for step in range(steps):
        signal = mx.full((1, 1, 8), ((step * 13) % 127 - 63) / 64.0, dtype=mx.float32)
        old_conv = cache[0]
        old_state = cache[1]
        output = mx.sum(old_conv) + mx.sum(old_state * weights)
        cache[0] = mx.contiguous(mx.concatenate([old_conv[:, 1:, :], signal], axis=1))
        delta = signal.reshape(1, 1, 1, 8) * weights
        cache[1] = old_state * 0.999 + delta
        token = int(mx.argmax(mx.stack([output, -output])).item())
        _sample_digest_update(sampled, token)
        _apply_policy(policy, step, cache.state)
        if step == 0 or (step + 1) % sample_interval == 0:
            curve.append({"step": step + 1, **_mlx_memory()})
    elapsed = time.perf_counter() - started
    arrays = [array for array in cache.state if array is not None]
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "sample_digest": sampled.hexdigest(),
        "state_digest": _array_digest(arrays),
        "curve": curve,
        "memory": _mlx_memory(),
    }


def _run_paged(policy: str, steps: int, sample_interval: int) -> dict[str, Any]:
    block_size = 64
    manager = PagedCacheManager(
        num_layers=1,
        block_size=block_size,
        max_blocks=math.ceil(steps / block_size) + 2,
    )
    layer = PagedKVCacheLayer(
        manager,
        layer_index=0,
        enable_block_pool=True,
        enable_direct_attention=True,
    )
    sampled = hashlib.sha256()
    curve: list[dict[str, int]] = []
    mx.reset_peak_memory()
    started = time.perf_counter()
    for step in range(steps):
        keys = mx.full((1, 2, 1, 8), ((step * 7) % 113 - 56) / 57.0, dtype=mx.float32)
        values = mx.full((1, 2, 1, 8), ((step * 11) % 109) / 109.0, dtype=mx.float32)
        layer.update_and_fetch(keys, values)
        pool_keys, pool_values, _ = layer.attention_view().block_pool()
        block_id = layer.block_table.block_ids[step // block_size]
        block_offset = step % block_size
        score = mx.sum(
            pool_keys[block_id, ..., block_offset, :] * 0.5
            + pool_values[block_id, ..., block_offset, :] * 0.5
        )
        token = int(mx.argmax(mx.stack([score, -score])).item())
        _sample_digest_update(sampled, token)
        _apply_policy(policy, step, (pool_keys, pool_values))
        if step == 0 or (step + 1) % sample_interval == 0:
            curve.append({"step": step + 1, **_mlx_memory()})
    elapsed = time.perf_counter() - started
    arrays = list(layer.state)
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "sample_digest": sampled.hexdigest(),
        "state_digest": _array_digest(arrays),
        "final_offset": layer.offset,
        "allocated_blocks": len(layer.block_table.block_ids),
        "curve": curve,
        "memory": _mlx_memory(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases: tuple[tuple[str, Callable[[str, int, int], dict[str, Any]]], ...] = (
        ("native_kv_waw", _run_native),
        ("recurrent_sibling_raw", _run_recurrent),
        ("paged_pool_waw", _run_paged),
    )
    results: dict[str, Any] = {}
    swap_before = int(psutil.swap_memory().used)
    for case_name, function in cases:
        case_results: dict[str, Any] = {}
        for policy in POLICIES:
            mx.clear_cache()
            case_results[policy] = function(policy, args.steps, args.sample_interval)
        baseline = case_results["baseline"]
        parity = {
            policy: (
                result["sample_digest"] == baseline["sample_digest"]
                and result["state_digest"] == baseline["state_digest"]
            )
            for policy, result in case_results.items()
        }
        results[case_name] = {"policies": case_results, "parity": parity}
    swap_after = int(psutil.swap_memory().used)
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "steps": args.steps,
        "sample_interval": args.sample_interval,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                PROJECT_ROOT / "aster/inference/paged_cache.py",
                PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
                Path(__file__).resolve(),
            )
        },
        "results": results,
        "all_parity": all(
            parity
            for case in results.values()
            for parity in case["parity"].values()
        ),
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--sample-interval", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/synthetic-stress.json",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.sample_interval < 1:
        raise ValueError("steps and sample interval must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
