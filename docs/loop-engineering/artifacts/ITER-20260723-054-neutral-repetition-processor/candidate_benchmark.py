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

from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

sampling = paired.sampling
production = paired.production
base = paired.base
NEUTRAL_REPETITION_PROCESSOR = "_aster_neutral_repetition_processor"
WORKLOADS = ("greedy", "penalties", "mixed")

_BASE_PREPARE_LANE = sampling._prepare_lane


class ProcessorMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.neutral_processors_skipped = 0
        self.processor_rows_preserved = 0

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "neutral_processors_skipped": self.neutral_processors_skipped,
                "processor_rows_preserved": self.processor_rows_preserved,
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
    setattr(
        lane.sampler,
        NEUTRAL_REPETITION_PROCESSOR,
        request.repetition_penalty == 1.0,
    )
    return lane, elapsed


def _candidate_processors(item: DecodeWorkItem) -> tuple[tuple[Any, ...], bool]:
    processors = item.logits_processors
    if (
        processors
        and getattr(item.sampler, NEUTRAL_REPETITION_PROCESSOR, False)
    ):
        return processors[1:], True
    return processors, False


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

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        if policy != "production":
            return original_apply(logits, item=item)
        processors, skipped = _candidate_processors(item)
        if not skipped:
            metrics.processor_rows_preserved += 1
            return original_apply(logits, item=item)
        metrics.neutral_processors_skipped += 1
        if not processors:
            return logits
        mx = runner._mx
        assert mx is not None
        tokens = mx.array(
            item.logits_processor_tokens + [item.input_token],
            dtype=mx.uint32,
        )
        for processor in processors:
            logits = processor(tokens, logits)
        return logits

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = production._install_policy
    original_prepare = sampling._prepare_lane
    production._install_policy = _install_policy
    sampling._prepare_lane = _prepare_lane
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install
        sampling._prepare_lane = original_prepare

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    payload["comparison"] = {
        "baseline": "current ModelRunner._decode_batch processor construction",
        "production": (
            "current decode with repetition_penalty=1.0 processor omitted"
        ),
        "candidate_scope": (
            "benchmark skips only the first MLX-LM processor for requests "
            "whose repetition penalty is exactly neutral"
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
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
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
