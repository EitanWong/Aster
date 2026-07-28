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
    _BatchPostSampleError,
)

sampling = paired.sampling
production = paired.production
base = paired.base
SAMPLER_ACCEPTS_LOGITS = "_aster_accepts_unnormalized_logits"
WORKLOADS = ("greedy", "penalties", "mixed")

_BASE_PREPARE_LANE = sampling._prepare_lane


class CandidateMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.direct_logit_rows = 0
        self.normalized_rows = 0

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "direct_logit_rows": self.direct_logit_rows,
                "normalized_rows": self.normalized_rows,
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
        SAMPLER_ACCEPTS_LOGITS,
        request.temperature == 0.0,
    )
    return lane, elapsed


def _sampler_input(
    mx: Any,
    row: Any,
    sampler: Any,
    metrics: CandidateMetrics,
) -> Any:
    if getattr(sampler, SAMPLER_ACCEPTS_LOGITS, False):
        metrics.direct_logit_rows += 1
        return row
    metrics.normalized_rows += 1
    return row - mx.logsumexp(row, axis=-1, keepdims=True)


def _decode_candidate(
    runner: ModelRunner,
    items: list[DecodeWorkItem],
    metrics: CandidateMetrics,
    original_batch: Any,
) -> list[DecodeResult]:
    runner._ensure_loaded()
    mx = runner._mx
    model = runner._model
    assert mx is not None and model is not None

    if runner._uses_eager_row_sampling(items):
        return original_batch(items)

    merged_cache, batch_cache_state = runner._get_decode_batch_cache(items)
    input_tokens = mx.array([[item.input_token] for item in items], dtype=mx.uint32)
    logits = model(input_tokens, cache=merged_cache)[:, -1, :]

    sampled_tokens: list[Any] = []
    for index, item in enumerate(items):
        row = runner._apply_logits_processors(
            logits[index : index + 1],
            item=item,
        )
        sampler_input = _sampler_input(mx, row, item.sampler, metrics)
        sampled_tokens.append(item.sampler(sampler_input))

    lazy_samples, trusted_array_type = runner._mlx_sample_arrays(mx, sampled_tokens)
    if not lazy_samples:
        evaluation_targets: Any = logits
    elif trusted_array_type and len(lazy_samples) == len(sampled_tokens):
        evaluation_targets = lazy_samples
    else:
        evaluation_targets = [logits, *lazy_samples]
    try:
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
    except MemoryError:
        raise
    except Exception as exc:
        raise _BatchPostSampleError(exc) from exc

    try:
        return [
            runner._decode_result(
                item=item,
                token=runner._materialize_sampled_token(sampled),
                prompt_cache=prompt_cache,
                peak_memory_gb=peak_memory_gb,
            )
            for item, sampled, prompt_cache in zip(
                items, sampled_tokens, prompt_caches, strict=True
            )
        ]
    except MemoryError:
        raise
    except Exception as exc:
        raise _BatchPostSampleError(exc) from exc


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[CandidateMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("candidate benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = CandidateMetrics(mlx, policy)
    original_batch = runner._decode_batch

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        with metrics.decode_scope():
            if policy == "baseline":
                return original_batch(items)
            return _decode_candidate(runner, items, metrics, original_batch)

    runner._mx = metrics
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._decode_batch = original_batch  # type: ignore[method-assign]

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
        "baseline": "current ModelRunner._decode_batch",
        "production": "current decode graph with temperature-zero logsumexp elided",
        "normalization_contract": (
            "only samplers created for request.temperature == 0 receive raw logits"
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
