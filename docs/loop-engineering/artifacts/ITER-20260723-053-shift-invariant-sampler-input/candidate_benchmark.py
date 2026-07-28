#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
PREVIOUS_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260723-052-greedy-logsumexp-elision"
)


def _load_previous() -> ModuleType:
    path = PREVIOUS_ARTIFACT_DIR / "candidate_benchmark.py"
    spec = importlib.util.spec_from_file_location("iter052_candidate_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load previous candidate harness: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = _load_previous()
sampling = previous.sampling
production = previous.production
base = previous.base
SAMPLER_ACCEPTS_LOGITS = previous.SAMPLER_ACCEPTS_LOGITS


def sampler_accepts_raw_logits(*, temperature: float, top_p: float) -> bool:
    """Return whether MLX-LM's built-in sampler is shift-invariant."""
    if temperature == 0.0:
        return True
    # apply_top_p compares cumulative exp(logprobs) to an absolute threshold;
    # all other installed sampler operations consume logits up to a constant.
    return not (0.0 < top_p < 1.0)


def _prepare_lane(
    runner: Any,
    *,
    request_id: str,
    prompt: str,
    max_tokens: int,
    prefill_step: int,
) -> tuple[Any, float]:
    lane, elapsed = previous._BASE_PREPARE_LANE(
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
        SAMPLER_ACCEPTS_LOGITS,
        sampler_accepts_raw_logits(
            temperature=request.temperature,
            top_p=request.top_p,
        ),
    )
    return lane, elapsed


def run(args: argparse.Namespace) -> dict[str, object]:
    original_prepare = previous._prepare_lane
    previous._prepare_lane = _prepare_lane
    try:
        payload = previous.run(args)
    finally:
        previous._prepare_lane = original_prepare

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    payload["comparison"] = {
        "baseline": "current ModelRunner._decode_batch",
        "production": (
            "current decode graph with logsumexp elided for shift-invariant "
            "built-in samplers"
        ),
        "normalization_contract": (
            "only built-in samplers with active top_p (0 < top_p < 1) "
            "receive normalized logprobs"
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
    parser.add_argument("--workload", choices=("greedy", "penalties", "mixed"), required=True)
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
