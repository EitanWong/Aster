#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
for path in (PROJECT_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paired_benchmark as paired  # noqa: E402

from aster.inference.constrained.json_schema_processor import (  # noqa: E402
    JSONSchemaLogitsProcessor,
    ThinkingAwareJsonLogitsProcessor,
)
from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

sampling = paired.sampling
production = paired.production
base = paired.base


class MaskCacheMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.mask_cache_hits = 0
        self.mask_cache_misses = 0
        self.mask_cache_collisions = 0
        self.mask_cache_evictions = 0
        self.mask_cache_peak_entries = 0
        self.mask_cache_compare_seconds = 0.0
        self.instrumented_processors: set[int] = set()
        self.processor_restorers: list[Any] = []

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "mask_cache_hits": self.mask_cache_hits,
                "mask_cache_misses": self.mask_cache_misses,
                "mask_cache_collisions": self.mask_cache_collisions,
                "mask_cache_evictions": self.mask_cache_evictions,
                "mask_cache_peak_entries": self.mask_cache_peak_entries,
                "mask_cache_compare_seconds": self.mask_cache_compare_seconds,
            }
        )
        return payload


def _mask_key(allowed: list[int], logits: Any) -> tuple[object, ...]:
    shape = tuple(int(dimension) for dimension in logits.shape)
    if not allowed:
        return (shape, 0)
    middle = len(allowed) // 2
    return (
        shape,
        len(allowed),
        int(allowed[0]),
        int(allowed[middle]),
        int(allowed[-1]),
    )


def _install_mask_cache(
    processor: Any,
    metrics: MaskCacheMetrics,
    *,
    capacity: int,
) -> None:
    target = processor
    if isinstance(target, ThinkingAwareJsonLogitsProcessor):
        target = target._inner
    if not isinstance(target, JSONSchemaLogitsProcessor):
        return
    if id(target) in metrics.instrumented_processors:
        return
    metrics.instrumented_processors.add(id(target))
    original_mask = target._mask
    had_instance_method = "_mask" in target.__dict__
    instance_method = target.__dict__.get("_mask")
    cache: OrderedDict[tuple[object, ...], tuple[list[int], Any]] = OrderedDict()

    def cached_mask(allowed: list[int], logits: Any) -> Any:
        key = _mask_key(allowed, logits)
        entry = cache.get(key)
        if entry is not None:
            started = time.perf_counter()
            matches = entry[0] == allowed
            metrics.mask_cache_compare_seconds += time.perf_counter() - started
            if matches:
                metrics.mask_cache_hits += 1
                cache.move_to_end(key)
                return entry[1]
            metrics.mask_cache_collisions += 1

        metrics.mask_cache_misses += 1
        mask = original_mask(allowed, logits)
        cache[key] = (allowed, mask)
        cache.move_to_end(key)
        if len(cache) > capacity:
            cache.popitem(last=False)
            metrics.mask_cache_evictions += 1
        metrics.mask_cache_peak_entries = max(
            metrics.mask_cache_peak_entries,
            len(cache),
        )
        return mask

    target._mask = cached_mask

    def restore_processor() -> None:
        if had_instance_method:
            target._mask = instance_method
        else:
            target.__dict__.pop("_mask", None)

    metrics.processor_restorers.append(restore_processor)


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[MaskCacheMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("mask-cache benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = MaskCacheMetrics(mlx, policy)
    original_apply = runner._apply_logits_processors

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        if policy == "production":
            for processor in item.logits_processors:
                _install_mask_cache(
                    processor,
                    metrics,
                    capacity=args.cache_capacity,
                )
        return original_apply(logits, item=item)

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]

    def restore() -> None:
        for restore_processor in metrics.processor_restorers:
            restore_processor()
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = production._install_policy
    production._install_policy = _install_policy
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    payload["comparison"] = {
        "baseline": "current JSON mask construction on every row",
        "production": f"request-local exact mask LRU with capacity {args.cache_capacity}",
        "cache_key": "shape, allowed length, and first/middle/last IDs; full list equality verifies hits",
        "candidate_scope": (
            "allowed-token generation, key/EOS filters, processor order, samplers, "
            "RNG, cache extraction, and host materialization are unchanged"
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--workload", choices=("structured",), default="structured")
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--pair-warmup-steps", type=int, default=32)
    parser.add_argument("--model-warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cache-capacity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if min(
        args.steps,
        args.pair_warmup_steps,
        args.model_warmup_tokens,
        args.cache_capacity,
    ) < 1:
        raise ValueError("step counts and cache capacity must be positive")
    if args.block_size < 1 or args.steps % args.block_size:
        raise ValueError("steps must be divisible by block size")

    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
