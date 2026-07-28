#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
ITER055_DIR = ARTIFACT_DIR.parent / "ITER-20260723-055-bounded-penalty-context"
for path in (PROJECT_ROOT, ITER055_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import candidate_benchmark as iter055  # noqa: E402

from aster.inference.model_runner import (  # noqa: E402
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)

paired = iter055.paired
sampling = iter055.sampling
production = iter055.production
base = iter055.base
_PENALTY_SPEC = "_iter056_penalty_spec"


@dataclass(frozen=True, slots=True)
class PenaltySpec:
    repetition: float
    presence: float
    frequency: float


class CandidateMetrics(iter055.ProcessorMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.vectorized_batches = 0
        self.vectorized_rows = 0

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "vectorized_batches": self.vectorized_batches,
                "vectorized_rows": self.vectorized_rows,
            }
        )
        return payload


def _prepare_lane(
    runner: ModelRunner,
    *,
    request_id: str,
    prompt: str,
    max_tokens: int,
    prefill_step: int,
) -> tuple[Any, float]:
    lane, elapsed = iter055._prepare_lane(
        runner,
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
        prefill_step=prefill_step,
    )
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("batched penalty benchmark arguments are not active")
    workload, lane_index = sampling._lane_workload(request_id, args.workload)
    request = sampling._request_for_lane(
        workload=workload,
        lane_index=lane_index,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    setattr(
        lane.sampler,
        _PENALTY_SPEC,
        PenaltySpec(
            repetition=request.repetition_penalty,
            presence=request.presence_penalty,
            frequency=request.frequency_penalty,
        ),
    )
    return lane, elapsed


def _penalty_inputs(
    items: list[DecodeWorkItem],
) -> tuple[list[list[int]], list[PenaltySpec]] | None:
    token_rows: list[list[int]] = []
    specs: list[PenaltySpec] = []
    width: int | None = None
    for item in items:
        spec = getattr(item.sampler, _PENALTY_SPEC, None)
        context_size = item.logits_processor_context_size
        if not isinstance(spec, PenaltySpec) or not isinstance(context_size, int):
            return None
        if len(item.logits_processors) != 3:
            return None
        tokens = iter055._bounded_tokens(item, context_size)
        if not tokens or (width is not None and len(tokens) != width):
            return None
        width = len(tokens)
        token_rows.append(tokens)
        specs.append(spec)
    return token_rows, specs


def _apply_batched_penalties(
    mx: Any,
    logits: Any,
    *,
    token_rows: list[list[int]],
    specs: list[PenaltySpec],
) -> Any:
    tokens = mx.array(token_rows, dtype=mx.uint32)
    rows = mx.arange(len(token_rows))[:, None]

    repetition = mx.array([spec.repetition for spec in specs])[:, None]
    selected = logits[rows, tokens]
    selected = mx.where(selected < 0, selected * repetition, selected / repetition)
    logits[rows, tokens] = selected

    presence = mx.array([spec.presence for spec in specs])[:, None]
    logits[rows, tokens] = logits[rows, tokens] - presence

    frequency = mx.array([spec.frequency for spec in specs])[:, None]
    return logits.at[rows, tokens].subtract(frequency)


def _decode_vectorized(
    runner: ModelRunner,
    items: list[DecodeWorkItem],
    *,
    token_rows: list[list[int]],
    specs: list[PenaltySpec],
) -> list[DecodeResult]:
    runner._ensure_loaded()
    mx = runner._mx
    model = runner._model
    assert mx is not None and model is not None

    merged_cache, batch_cache_state = runner._get_decode_batch_cache(items)
    input_tokens = mx.array([[item.input_token] for item in items], dtype=mx.uint32)
    logits = model(input_tokens, cache=merged_cache)[:, -1, :]
    logits = _apply_batched_penalties(
        mx,
        logits,
        token_rows=token_rows,
        specs=specs,
    )

    sampled_tokens: list[Any] = []
    for index, item in enumerate(items):
        row = logits[index : index + 1]
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        sampled_tokens.append(item.sampler(logprobs))

    lazy_samples, trusted_array_type = runner._mlx_sample_arrays(mx, sampled_tokens)
    if not lazy_samples:
        evaluation_targets: Any = logits
    elif trusted_array_type and len(lazy_samples) == len(sampled_tokens):
        evaluation_targets = lazy_samples
    else:
        evaluation_targets = [logits, *lazy_samples]
    mx.async_eval(evaluation_targets)
    peak_memory_gb = runner.current_peak_memory_gb()
    prompt_caches = [
        (
            runner._decode_cache_ref(batch_cache_state, index)
            if batch_cache_state is not None
            else runner._extract_prompt_cache(merged_cache, index)
        )
        for index in range(len(items))
    ]
    mx.eval(evaluation_targets)

    return [
        runner._decode_result(
            item=item,
            token=runner._materialize_sampled_token(sampled),
            prompt_cache=prompt_cache,
            peak_memory_gb=peak_memory_gb,
        )
        for item, sampled, prompt_cache in zip(
            items,
            sampled_tokens,
            prompt_caches,
            strict=True,
        )
    ]


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[CandidateMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("batched penalty benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = CandidateMetrics(mlx, policy)
    original_batch = runner._decode_batch
    original_apply = runner._apply_logits_processors
    original_bounded = getattr(runner, "_iter055_bounded_work_items", None)
    original_metrics = getattr(runner, "_iter055_metrics", None)
    runner._iter055_bounded_work_items = True
    runner._iter055_metrics = metrics

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        context_size = item.logits_processor_context_size
        if isinstance(context_size, int) and item.logits_processors:
            metrics.bounded_rows += 1
            metrics.device_tokens_total += len(iter055._bounded_tokens(item, context_size))
        return original_apply(logits, item=item)

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        inputs = _penalty_inputs(items) if policy == "production" else None
        if inputs is None:
            return original_batch(items)
        token_rows, specs = inputs
        metrics.vectorized_batches += 1
        metrics.vectorized_rows += len(items)
        metrics.bounded_rows += len(items)
        metrics.device_tokens_total += sum(len(tokens) for tokens in token_rows)
        return _decode_vectorized(
            runner,
            items,
            token_rows=token_rows,
            specs=specs,
        )

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]
        runner._decode_batch = original_batch  # type: ignore[method-assign]
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
    sampling._prepare_lane = _prepare_lane
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
    payload["comparison"] = {
        "baseline": "current bounded per-row MLX-LM penalty processors",
        "production": "batched gather/scatter penalties over homogeneous active rows",
        "candidate_scope": (
            "20-token context, model path, per-row sampler order, RNG state, "
            "normalization, cache extraction, and host materialization are unchanged"
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
    parser.add_argument("--workload", choices=("penalties",), default="penalties")
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
