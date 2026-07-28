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

from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.model_runner import (  # noqa: E402
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)

paired = iter055.paired
sampling = iter055.sampling
production = iter055.production
base = iter055.base
_BASE_REQUEST_FOR_LANE = sampling._request_for_lane


class CandidateMetrics(iter055.ProcessorMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.batched_normalization_batches = 0
        self.batched_normalization_rows = 0

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "batched_normalization_batches": self.batched_normalization_batches,
                "batched_normalization_rows": self.batched_normalization_rows,
            }
        )
        return payload


def _request_for_lane(
    *,
    workload: str,
    lane_index: int,
    prompt: str,
    max_tokens: int,
) -> InferenceRequest:
    if workload != "top-p":
        return _BASE_REQUEST_FOR_LANE(
            workload=workload,
            lane_index=lane_index,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    return InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        trace_id=f"iter056-{workload}-lane-{lane_index}",
    )


def _decode_batched_normalization(
    runner: ModelRunner,
    items: list[DecodeWorkItem],
) -> list[DecodeResult]:
    runner._ensure_loaded()
    mx = runner._mx
    model = runner._model
    assert mx is not None and model is not None

    merged_cache, batch_cache_state = runner._get_decode_batch_cache(items)
    input_tokens = mx.array([[item.input_token] for item in items], dtype=mx.uint32)
    logits = model(input_tokens, cache=merged_cache)[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sampled_tokens = [
        item.sampler(logprobs[index : index + 1])
        for index, item in enumerate(items)
    ]

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
        raise RuntimeError("batched normalization benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = CandidateMetrics(mlx, policy)
    original_batch = runner._decode_batch
    original_bounded = getattr(runner, "_iter055_bounded_work_items", None)
    original_metrics = getattr(runner, "_iter055_metrics", None)
    runner._iter055_bounded_work_items = True
    runner._iter055_metrics = metrics

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        if policy != "production" or any(item.logits_processors for item in items):
            return original_batch(items)
        metrics.batched_normalization_batches += 1
        metrics.batched_normalization_rows += len(items)
        return _decode_batched_normalization(runner, items)

    runner._mx = metrics
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
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
    original_request = sampling._request_for_lane
    original_advance = paired._advance
    paired._ITER055_ORIGINAL_ADVANCE = original_advance
    production._install_policy = _install_policy
    sampling._prepare_lane = iter055._prepare_lane
    sampling._request_for_lane = _request_for_lane
    paired._advance = iter055._advance
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install
        sampling._prepare_lane = original_prepare
        sampling._request_for_lane = original_request
        paired._advance = original_advance
        del paired._ITER055_ORIGINAL_ADVANCE

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(source)
    payload["comparison"] = {
        "baseline": "one full-vocabulary logsumexp graph per processor-free row",
        "production": "one full-vocabulary logsumexp graph for the processor-free batch",
        "candidate_scope": (
            "per-row sampler calls, RNG order, logits values, cache extraction, "
            "and host materialization are unchanged"
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
    parser.add_argument("--workload", choices=("greedy", "top-p"), required=True)
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
