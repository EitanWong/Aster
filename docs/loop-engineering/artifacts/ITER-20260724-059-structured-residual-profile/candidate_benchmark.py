#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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


class CandidateMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.eos_cache_hits = 0
        self.eos_cache_misses = 0
        self.eos_cache_collisions = 0
        self.eos_compare_seconds = 0.0
        self.eos_membership_seconds = 0.0
        self.instrumented_processors: set[int] = set()
        self.processor_restorers: list[Any] = []

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "eos_cache_hits": self.eos_cache_hits,
                "eos_cache_misses": self.eos_cache_misses,
                "eos_cache_collisions": self.eos_cache_collisions,
                "eos_compare_seconds": self.eos_compare_seconds,
                "eos_membership_seconds": self.eos_membership_seconds,
            }
        )
        return payload


def _target_processor(processor: Any) -> JSONSchemaLogitsProcessor | None:
    target = processor
    if isinstance(target, ThinkingAwareJsonLogitsProcessor):
        target = target._inner
    return target if isinstance(target, JSONSchemaLogitsProcessor) else None


def _same_allowed(
    cached: list[int] | None,
    allowed: list[int],
    metrics: CandidateMetrics,
) -> bool:
    if cached is None or len(cached) != len(allowed):
        return False
    if allowed:
        middle = len(allowed) // 2
        if (
            cached[0] != allowed[0]
            or cached[middle] != allowed[middle]
            or cached[-1] != allowed[-1]
        ):
            return False
    started = time.perf_counter()
    matches = cached == allowed
    metrics.eos_compare_seconds += time.perf_counter() - started
    if not matches:
        metrics.eos_cache_collisions += 1
    return matches


def _install_eos_cache(processor: Any, metrics: CandidateMetrics) -> None:
    target = _target_processor(processor)
    if target is None or id(target) in metrics.instrumented_processors:
        return
    metrics.instrumented_processors.add(id(target))
    original_allowed = target._allowed_tokens
    original_mask = target._mask
    had_cached_eos = hasattr(target, "_iter059_mask_cache_contains_eos")
    cached_eos = getattr(target, "_iter059_mask_cache_contains_eos", None)
    target._iter059_mask_cache_contains_eos = None
    target._iter059_pending_allowed = None
    target._iter059_pending_contains_eos = False

    def allowed_tokens(suffix: list[int]) -> list[int]:
        allowed_result = target._enforcer.get_allowed_tokens(suffix)
        allowed = getattr(allowed_result, "allowed_tokens", allowed_result)
        if allowed is None:
            return []
        if isinstance(allowed, list) and (not allowed or isinstance(allowed[0], int)):
            allowed_values = allowed
        else:
            allowed_values = [int(token_id) for token_id in allowed]

        context = target._json_context(suffix)
        if context in {"key_start", "in_key"}:
            allowed_values = target._filter_at_key_context(
                context,
                suffix,
                allowed_values,
            )

        contains_eos = False
        if target._eos_token_ids:
            cached_contains_eos = target._iter059_mask_cache_contains_eos
            if cached_contains_eos is not None and _same_allowed(
                target._mask_cache_allowed,
                allowed_values,
                metrics,
            ):
                metrics.eos_cache_hits += 1
                contains_eos = cached_contains_eos
            else:
                metrics.eos_cache_misses += 1
                started = time.perf_counter()
                contains_eos = any(
                    token_id in allowed_values
                    for token_id in target._eos_token_ids
                )
                metrics.eos_membership_seconds += time.perf_counter() - started

        if contains_eos and not target._is_complete_json(suffix):
            allowed_values = [
                token_id
                for token_id in allowed_values
                if token_id not in target._eos_token_ids
            ]
            contains_eos = False

        target._iter059_pending_allowed = allowed_values
        target._iter059_pending_contains_eos = contains_eos
        return allowed_values

    def mask(allowed: list[int], logits: Any) -> Any:
        value = original_mask(allowed, logits)
        if allowed is target._iter059_pending_allowed:
            target._iter059_mask_cache_contains_eos = (
                target._iter059_pending_contains_eos
            )
        else:
            target._iter059_mask_cache_contains_eos = any(
                token_id in allowed for token_id in target._eos_token_ids
            )
        return value

    target._allowed_tokens = allowed_tokens
    target._mask = mask

    def restore_processor() -> None:
        target._allowed_tokens = original_allowed
        target._mask = original_mask
        if had_cached_eos:
            target._iter059_mask_cache_contains_eos = cached_eos
        else:
            target.__dict__.pop("_iter059_mask_cache_contains_eos", None)
        target.__dict__.pop("_iter059_pending_allowed", None)
        target.__dict__.pop("_iter059_pending_contains_eos", None)

    metrics.processor_restorers.append(restore_processor)


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[CandidateMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("EOS membership benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = CandidateMetrics(mlx, policy)
    original_apply = runner._apply_logits_processors

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        if policy == "production":
            for processor in item.logits_processors:
                _install_eos_cache(processor, metrics)
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
        "baseline": "current production JSONSchemaLogitsProcessor",
        "production": "exact EOS-membership reuse from the prior mask snapshot",
        "cache_key": (
            "allowed length and first/middle/last IDs, followed by full list equality"
        ),
        "candidate_scope": (
            "LMFE state advancement, allowed-token contents, key filtering, masks, "
            "processor order, samplers, RNG, and cache extraction are unchanged"
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
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if min(args.steps, args.pair_warmup_steps, args.model_warmup_tokens) < 1:
        raise ValueError("step counts must be positive")
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
