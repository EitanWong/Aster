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
_PYTHON_TOKEN_PROCESSORS = (
    JSONSchemaLogitsProcessor,
    ThinkingAwareJsonLogitsProcessor,
)


class CandidateMetrics(iter055.ProcessorMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.python_token_calls = 0
        self.device_token_arrays = 0
        self.processor_seconds: list[float] = []
        self.allowed_seconds: list[float] = []
        self.mask_seconds: list[float] = []
        self.mask_cardinalities: list[int] = []
        self.mask_keys: set[tuple[int, ...]] = set()
        self.instrumented_processors: set[int] = set()

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "python_token_calls": self.python_token_calls,
                "device_token_arrays": self.device_token_arrays,
                "processor_seconds": {
                    **base._summary(self.processor_seconds),
                    "total": sum(self.processor_seconds),
                },
                "allowed_seconds": {
                    **base._summary(self.allowed_seconds),
                    "total": sum(self.allowed_seconds),
                },
                "mask_seconds": {
                    **base._summary(self.mask_seconds),
                    "total": sum(self.mask_seconds),
                },
                "mask_cardinality": base._summary(self.mask_cardinalities),
                "mask_unique_keys": len(self.mask_keys),
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


def _apply_python_tokens(
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
        if isinstance(processor, _PYTHON_TOKEN_PROCESSORS):
            _instrument_processor(processor, metrics)
            metrics.python_token_calls += 1
            tokens: Any = processor_tokens
        else:
            if device_tokens is None:
                metrics.device_token_arrays += 1
                device_tokens = mx.array(processor_tokens, dtype=mx.uint32)
            tokens = device_tokens
        logits = processor(tokens, logits)
    return logits


def _instrument_processor(processor: Any, metrics: CandidateMetrics) -> None:
    target = processor
    if isinstance(target, ThinkingAwareJsonLogitsProcessor):
        target = target._inner
    if not isinstance(target, JSONSchemaLogitsProcessor):
        return
    if id(target) in metrics.instrumented_processors:
        return
    metrics.instrumented_processors.add(id(target))
    original_allowed = target._allowed_tokens
    original_mask = target._mask

    def allowed_tokens(suffix: list[int]) -> list[int]:
        started = time.perf_counter()
        try:
            return original_allowed(suffix)
        finally:
            metrics.allowed_seconds.append(time.perf_counter() - started)

    def mask(allowed: list[int], logits: Any) -> Any:
        key = tuple(allowed)
        metrics.mask_cardinalities.append(len(key))
        metrics.mask_keys.add(key)
        started = time.perf_counter()
        try:
            return original_mask(allowed, logits)
        finally:
            metrics.mask_seconds.append(time.perf_counter() - started)

    target._allowed_tokens = allowed_tokens
    target._mask = mask


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[CandidateMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("structured Python-token benchmark arguments are not active")
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
        started = time.perf_counter()
        try:
            if policy == "production":
                return _apply_python_tokens(runner, logits, item=item, metrics=metrics)
            return original_apply(logits, item=item)
        finally:
            metrics.processor_seconds.append(time.perf_counter() - started)

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
        "baseline": "full token history converted to MLX then back to Python",
        "production": "Python token history passed directly to Aster JSON processors",
        "candidate_scope": (
            "model logits, eager row order, parser ownership, masks, sampler calls, "
            "cache extraction, and host materialization are unchanged"
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
