#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
ITER055_DIR = ARTIFACT_DIR.parent / "ITER-20260723-055-bounded-penalty-context"
for path in (PROJECT_ROOT, ITER055_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import candidate_benchmark as iter055  # noqa: E402

from aster.inference.constrained.json_schema_processor import (  # noqa: E402
    JSONSchemaLogitsProcessor,
    ThinkingAwareJsonLogitsProcessor,
)
from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

paired = iter055.paired
sampling = iter055.sampling
production = iter055.production
base = iter055.base


class CandidateMetrics(iter055.ProcessorMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.borrowed_allowed_lists = 0
        self.converted_allowed_lists = 0
        self.mask_cache_hits = 0
        self.mask_cache_misses = 0
        self.python_token_calls = 0
        self.instrumented_processors: set[int] = set()

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "borrowed_allowed_lists": self.borrowed_allowed_lists,
                "converted_allowed_lists": self.converted_allowed_lists,
                "mask_cache_hits": self.mask_cache_hits,
                "mask_cache_misses": self.mask_cache_misses,
                "python_token_calls": self.python_token_calls,
            }
        )
        return payload


def _processor_tokens(item: DecodeWorkItem) -> list[int]:
    context_size = item.logits_processor_context_size
    if context_size is None:
        return [*item.logits_processor_tokens, item.input_token]
    preceding = max(context_size - 1, 0)
    return [
        *(item.logits_processor_tokens[-preceding:] if preceding else ()),
        item.input_token,
    ]


def _install_processor_fastpath(processor: Any, metrics: CandidateMetrics) -> bool:
    target = processor
    if isinstance(target, ThinkingAwareJsonLogitsProcessor):
        target = target._inner
    if not isinstance(target, JSONSchemaLogitsProcessor):
        return False
    if id(target) in metrics.instrumented_processors:
        return True
    metrics.instrumented_processors.add(id(target))

    original_mask = target._mask
    last_allowed: list[int] | None = None
    last_shape: tuple[int, ...] | None = None
    last_mask: Any | None = None

    def allowed_tokens(suffix: list[int]) -> list[int]:
        allowed_result = target._enforcer.get_allowed_tokens(suffix)
        allowed = getattr(allowed_result, "allowed_tokens", allowed_result)
        if allowed is None:
            return []
        if isinstance(allowed, list) and (
            not allowed or isinstance(allowed[0], int)
        ):
            result = allowed
            metrics.borrowed_allowed_lists += 1
        else:
            result = [int(token_id) for token_id in allowed]
            metrics.converted_allowed_lists += 1

        context = target._json_context(suffix)
        if context in {"key_start", "in_key"}:
            result = target._filter_at_key_context(context, suffix, result)
        eos_ids = target._eos_token_ids
        if eos_ids and any(token_id in result for token_id in eos_ids):
            if not target._is_complete_json(suffix):
                result = [token_id for token_id in result if token_id not in eos_ids]
        return result

    def mask(allowed: list[int], logits: Any) -> Any:
        nonlocal last_allowed, last_shape, last_mask
        shape = tuple(int(size) for size in logits.shape)
        if allowed is last_allowed and shape == last_shape:
            metrics.mask_cache_hits += 1
            return last_mask
        metrics.mask_cache_misses += 1
        last_allowed = allowed
        last_shape = shape
        last_mask = original_mask(allowed, logits)
        return last_mask

    target._allowed_tokens = allowed_tokens
    target._mask = mask
    return True


def _apply_candidate(
    runner: ModelRunner,
    logits: Any,
    *,
    item: DecodeWorkItem,
    metrics: CandidateMetrics,
) -> Any:
    if not item.logits_processors:
        return logits
    mx = runner._mx
    assert mx is not None
    processor_tokens = _processor_tokens(item)
    device_tokens: Any | None = None
    for processor in item.logits_processors:
        if _install_processor_fastpath(processor, metrics):
            metrics.python_token_calls += 1
            tokens: Any = processor_tokens
        else:
            if device_tokens is None:
                device_tokens = mx.array(processor_tokens, dtype=mx.uint32)
            tokens = device_tokens
        logits = processor(tokens, logits)
    return logits


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[CandidateMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("optimized processor benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = CandidateMetrics(mlx, policy)
    original_apply = runner._apply_logits_processors
    original_bounded = getattr(runner, "_iter055_bounded_work_items", None)
    original_metrics = getattr(runner, "_iter055_metrics", None)
    runner._iter055_bounded_work_items = True
    runner._iter055_metrics = metrics

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        if policy == "production":
            return _apply_candidate(runner, logits, item=item, metrics=metrics)
        return original_apply(logits, item=item)

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]
        if original_bounded is None:
            del runner._iter055_bounded_work_items
        else:
            runner._iter055_bounded_work_items = original_bounded
        if original_metrics is None:
            del runner._iter055_metrics
        else:
            runner._iter055_metrics = original_metrics

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = production._install_policy
    original_prepare = sampling._prepare_lane
    original_advance = paired._advance
    paired._ITER055_ORIGINAL_ADVANCE = original_advance
    production._install_policy = _install_policy
    sampling._prepare_lane = iter055._prepare_lane
    paired._advance = iter055._advance
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install
        sampling._prepare_lane = original_prepare
        paired._advance = original_advance
        del paired._ITER055_ORIGINAL_ADVANCE

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(source)
    payload["source_sha256"]["aster/inference/constrained/json_schema_processor.py"] = (
        base._sha256(PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py")
    )
    payload["comparison"] = {
        "baseline": "current structured processor token/list/mask path",
        "production": (
            "Python tokens, borrowed TokenEnforcer lists, EOS-driven membership, "
            "and one-entry identity mask cache"
        ),
        "candidate_scope": (
            "parser calls, filters, allowed token values, masks, eager row order, "
            "samplers, cache extraction, and host materialization are unchanged"
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
