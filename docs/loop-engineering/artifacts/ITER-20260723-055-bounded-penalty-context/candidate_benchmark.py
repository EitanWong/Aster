#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
)
for path in (PROJECT_ROOT, BASE_ARTIFACT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paired_benchmark as paired  # noqa: E402

from aster.inference.model_runner import (  # noqa: E402
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)

sampling = paired.sampling
production = paired.production
base = paired.base
PENALTY_CONTEXT_SIZE = "_aster_penalty_context_size"

_BASE_PREPARE_LANE = sampling._prepare_lane


class ProcessorMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.bounded_rows = 0
        self.source_tokens_total = 0
        self.device_tokens_total = 0
        self.max_source_tokens = 0

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "bounded_rows": self.bounded_rows,
                "source_tokens_total": self.source_tokens_total,
                "device_tokens_total": self.device_tokens_total,
                "max_source_tokens": self.max_source_tokens,
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
    lane, elapsed = _BASE_PREPARE_LANE(
        runner,
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
        prefill_step=prefill_step,
    )
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("candidate benchmark arguments are not active")
    workload, lane_index = sampling._lane_workload(request_id, args.workload)
    request = sampling._request_for_lane(
        workload=workload,
        lane_index=lane_index,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    active_penalty = (
        request.repetition_penalty != 1.0
        or request.presence_penalty != 0.0
        or request.frequency_penalty != 0.0
    )
    context_size = (
        20
        if active_penalty and request.structured_output_schema is None
        else None
    )
    setattr(lane.sampler, PENALTY_CONTEXT_SIZE, context_size)
    return lane, elapsed


def _bounded_tokens(item: DecodeWorkItem, context_size: int) -> list[int]:
    return (item.logits_processor_tokens + [item.input_token])[-context_size:]


def _recent_processor_tokens(lane: Any, context_size: int) -> tuple[list[int], int]:
    prompt_end = len(lane.prompt_tokens)
    output_end = len(lane.output_tokens)
    if output_end and lane.output_tokens[-1] == lane.input_token:
        output_end -= 1
    elif not output_end and prompt_end and lane.prompt_tokens[-1] == lane.input_token:
        prompt_end -= 1
    logical_source_tokens = prompt_end + output_end + 1
    preceding = max(context_size - 1, 0)
    output_start = max(output_end - preceding, 0)
    recent_output = lane.output_tokens[output_start:output_end]
    remaining = preceding - len(recent_output)
    prompt_start = max(prompt_end - remaining, 0)
    recent_prompt = lane.prompt_tokens[prompt_start:prompt_end] if remaining else []
    return [*recent_prompt, *recent_output], logical_source_tokens


def _bounded_work_item(lane: Any, max_tokens: int) -> tuple[DecodeWorkItem, int]:
    context_size = getattr(lane.sampler, PENALTY_CONTEXT_SIZE, None)
    if not isinstance(context_size, int):
        return base._work_item(lane, max_tokens), 0
    processor_tokens, logical_source_tokens = _recent_processor_tokens(
        lane,
        context_size,
    )
    return (
        DecodeWorkItem(
            prompt_cache=lane.prompt_cache,
            input_token=lane.input_token,
            sampler=lane.sampler,
            detokenizer=lane.detokenizer,
            stop_token_ids=frozenset(),
            logits_processors=lane.logits_processors,
            logits_processor_tokens=processor_tokens,
            completion_tokens=len(lane.output_tokens),
            max_tokens=max_tokens,
            request_id=lane.request_id,
            logits_processor_context_size=context_size,
        ),
        logical_source_tokens,
    )


def _advance(
    runner: ModelRunner,
    lanes: list[Any],
    *,
    max_tokens: int,
) -> None:
    if not getattr(runner, "_iter055_bounded_work_items", False):
        return paired._ITER055_ORIGINAL_ADVANCE(  # type: ignore[attr-defined]
            runner,
            lanes,
            max_tokens=max_tokens,
        )
    built = [_bounded_work_item(lane, max_tokens) for lane in lanes]
    metrics = runner._iter055_metrics
    metrics.source_tokens_total += sum(source_tokens for _, source_tokens in built)
    metrics.max_source_tokens = max(
        metrics.max_source_tokens,
        *(source_tokens for _, source_tokens in built),
    )
    results = runner.decode_batch_step([item for item, _ in built])
    if len(results) != len(lanes):
        raise RuntimeError("paired decode result count mismatch")
    for lane, result in zip(lanes, results, strict=True):
        if not isinstance(result, DecodeResult) or result.token_id is None:
            raise RuntimeError(f"paired decode failed: {result!r}")
        lane.prompt_cache = result.prompt_cache
        lane.input_token = result.token_id
        lane.output_tokens.append(result.token_id)
        lane.text_segments.append(result.text)


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[ProcessorMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("candidate benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = ProcessorMetrics(mlx, policy)
    original_apply = runner._apply_logits_processors
    original_bounded = getattr(runner, "_iter055_bounded_work_items", None)
    original_metrics = getattr(runner, "_iter055_metrics", None)
    runner._iter055_bounded_work_items = policy == "production"
    runner._iter055_metrics = metrics

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        context_size = item.logits_processor_context_size
        if policy != "production" or not isinstance(context_size, int):
            return original_apply(logits, item=item)
        if not item.logits_processors:
            return logits
        bounded = _bounded_tokens(item, context_size)
        metrics.bounded_rows += 1
        metrics.device_tokens_total += len(bounded)
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
    sampling._prepare_lane = _prepare_lane
    paired._advance = _advance
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install
        sampling._prepare_lane = original_prepare
        paired._advance = original_advance
        del paired._ITER055_ORIGINAL_ADVANCE

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    for relative in (
        "aster/inference/engine.py",
        "aster/inference/request_state.py",
    ):
        path = PROJECT_ROOT / relative
        payload["source_sha256"][relative] = base._sha256(path)
    payload["comparison"] = {
        "baseline": "current full-history logits processor token array",
        "production": (
            "20-token work item and MLX input for built-in penalty-only processors"
        ),
        "candidate_scope": (
            "processor graph, order, and parameters are unchanged; only the "
            "documented default context window is applied before mx.array"
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
    parser.add_argument("--seed", type=int, default=20260723)
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
