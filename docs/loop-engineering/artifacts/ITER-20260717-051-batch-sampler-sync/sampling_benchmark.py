#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync"
)
for path in (PROJECT_ROOT, BASE_ARTIFACT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark as base  # noqa: E402

from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.model_runner import (  # noqa: E402
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)

POLICIES = ("baseline", "grouped-eager", "grouped-lazy", "grouped-async")
WORKLOADS = ("greedy", "mixed", "penalties", "structured")
_ACTIVE_ARGS: argparse.Namespace | None = None


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["answer", "score"],
        "additionalProperties": False,
    }


def _request_for_lane(
    *,
    workload: str,
    lane_index: int,
    prompt: str,
    max_tokens: int,
) -> InferenceRequest:
    common = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "trace_id": f"iter051-{workload}-lane-{lane_index}",
    }
    if workload == "greedy":
        return InferenceRequest(**common, temperature=0.0, top_p=1.0)
    if workload == "mixed":
        variants = (
            {"temperature": 0.0, "top_p": 1.0},
            {"temperature": 0.7, "top_p": 0.9},
            {"temperature": 0.9, "top_p": 1.0, "top_k": 40, "min_p": 0.05},
            {
                "temperature": 0.4,
                "top_p": 0.95,
                "repetition_penalty": 1.08,
            },
        )
        return InferenceRequest(**common, **variants[lane_index % len(variants)])
    if workload == "penalties":
        return InferenceRequest(
            **common,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05 + 0.01 * (lane_index % 4),
            presence_penalty=0.1 + 0.05 * (lane_index % 3),
            frequency_penalty=0.05 + 0.05 * (lane_index % 2),
        )
    if workload == "structured":
        return InferenceRequest(
            **common,
            temperature=0.0,
            top_p=1.0,
            structured_output_schema=_schema(),
        )
    raise ValueError(f"unsupported workload: {workload}")


def _lane_workload(request_id: str, workload: str) -> tuple[str, int]:
    if request_id == "iter050-warmup":
        return "greedy", 0
    return workload, int(request_id.rsplit("-", 1)[-1])


def _prepare_lane(
    runner: ModelRunner,
    *,
    request_id: str,
    prompt: str,
    max_tokens: int,
    prefill_step: int,
) -> tuple[base.Lane, float]:
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("sampling benchmark arguments are not active")
    workload, lane_index = _lane_workload(request_id, args.workload)
    request = _request_for_lane(
        workload=workload,
        lane_index=lane_index,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    prepared = runner.encode_request(request)
    target = len(prepared.prompt_tokens) - 1
    prompt_cache = None
    cache_token_count = 0
    started = time.perf_counter()
    while cache_token_count < target:
        result = runner.prefill_to(
            prompt_tokens=prepared.prompt_tokens,
            prompt_cache=prompt_cache,
            cache_token_count=cache_token_count,
            target_cache_token_count=min(cache_token_count + prefill_step, target),
        )
        prompt_cache = result.prompt_cache
        cache_token_count = result.cache_token_count
    decode = runner.initialize_decode(
        prompt_tokens=prepared.prompt_tokens,
        cache_token_count=cache_token_count,
        prompt_cache=prompt_cache,
        request=request,
    )
    return (
        base.Lane(
            request_id=request_id,
            prompt_tokens=prepared.prompt_tokens,
            prompt_cache=decode.prompt_cache,
            input_token=decode.next_input_token,
            sampler=decode.sampler,
            detokenizer=decode.detokenizer,
            logits_processors=decode.logits_processors,
        ),
        time.perf_counter() - started,
    )


class SamplingMetrics:
    def __init__(self, mlx: Any, policy: str) -> None:
        self._mlx = mlx
        self.policy = policy
        self._decode_depth = 0
        self.eval_seconds: list[float] = []
        self.async_eval_seconds: list[float] = []
        self.sample_sync_seconds: list[float] = []
        self.sample_enqueue_seconds: list[float] = []
        self.sample_group_eval_seconds: list[float] = []
        self.materialize_seconds: list[float] = []
        self.clear_requests = 0
        self.clear_failures = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mlx, name)

    @contextlib.contextmanager
    def decode_scope(self) -> Iterator[None]:
        self._decode_depth += 1
        try:
            yield
        finally:
            self._decode_depth -= 1

    def eval(self, *values: Any) -> None:
        started = time.perf_counter()
        try:
            self._mlx.eval(*values)
        finally:
            if self._decode_depth:
                self.eval_seconds.append(time.perf_counter() - started)

    def async_eval(self, *values: Any) -> None:
        started = time.perf_counter()
        try:
            self._mlx.async_eval(*values)
        finally:
            if self._decode_depth:
                self.async_eval_seconds.append(time.perf_counter() - started)

    def clear_cache(self) -> None:
        self.clear_requests += 1
        try:
            self._mlx.clear_cache()
        except Exception:
            self.clear_failures += 1
            raise

    def metrics(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "explicit_eval_seconds": base._summary(self.eval_seconds),
            "async_eval_submit_seconds": base._summary(self.async_eval_seconds),
            "sample_sync_seconds": base._summary(self.sample_sync_seconds),
            "sample_enqueue_seconds": base._summary(self.sample_enqueue_seconds),
            "sample_group_eval_seconds": base._summary(
                self.sample_group_eval_seconds
            ),
            "materialize_seconds": base._summary(self.materialize_seconds),
            "clear_requests": self.clear_requests,
            "clear_failures": self.clear_failures,
        }


def _materialize(sampled: Any) -> int:
    if hasattr(sampled, "item"):
        return int(sampled.item())
    if hasattr(sampled, "tolist"):
        values = sampled.tolist()
        if isinstance(values, list):
            return int(values[0])
        return int(values)
    if isinstance(sampled, (list, tuple)):
        return int(sampled[0])
    return int(sampled)


def _decode_grouped(
    runner: ModelRunner,
    items: list[DecodeWorkItem],
    metrics: SamplingMetrics,
) -> list[DecodeResult]:
    runner._ensure_loaded()
    mx = runner._mx
    model = runner._model
    assert mx is not None and model is not None

    merged_cache, batch_cache_state = runner._get_decode_batch_cache(items)
    input_tokens = mx.array([[item.input_token] for item in items], dtype=mx.uint32)
    logits = model(input_tokens, cache=merged_cache)[:, -1, :]
    if metrics.policy == "grouped-eager":
        mx.eval(logits)

    sampled_values: list[Any] = []
    for index, item in enumerate(items):
        row = runner._apply_logits_processors(
            logits[index : index + 1],
            item=item,
        )
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        started = time.perf_counter()
        sampled_values.append(item.sampler(logprobs))
        metrics.sample_enqueue_seconds.append(time.perf_counter() - started)

    if metrics.policy == "grouped-async":
        mx.async_eval(sampled_values)

    peak_memory_gb = runner.current_peak_memory_gb()
    prompt_caches = [
        (
            runner._decode_cache_ref(batch_cache_state, index)
            if batch_cache_state is not None
            else runner._extract_prompt_cache(merged_cache, index)
        )
        for index in range(len(items))
    ]
    started = time.perf_counter()
    mx.eval(sampled_values)
    metrics.sample_group_eval_seconds.append(time.perf_counter() - started)

    results: list[DecodeResult] = []
    for item, sampled, prompt_cache in zip(
        items, sampled_values, prompt_caches, strict=True
    ):
        started = time.perf_counter()
        token = _materialize(sampled)
        metrics.materialize_seconds.append(time.perf_counter() - started)
        results.append(
            runner._decode_result(
                item=item,
                token=token,
                prompt_cache=prompt_cache,
                peak_memory_gb=peak_memory_gb,
            )
        )
    return results


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[SamplingMetrics, Any]:
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("sampling benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = SamplingMetrics(mlx, policy)
    original_batch = runner._decode_batch
    original_sample = runner._sample_token

    def timed_sample(logprobs: Any, sampler: Any) -> int:
        started = time.perf_counter()
        try:
            return original_sample(logprobs, sampler)
        finally:
            metrics.sample_sync_seconds.append(time.perf_counter() - started)

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        with metrics.decode_scope():
            if policy == "baseline":
                return original_batch(items)
            return _decode_grouped(runner, items, metrics)

    runner._mx = metrics
    runner._sample_token = timed_sample  # type: ignore[method-assign]
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._sample_token = original_sample  # type: ignore[method-assign]
        runner._decode_batch = original_batch  # type: ignore[method-assign]

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    global _ACTIVE_ARGS
    original_prepare = base._prepare_lane
    original_install = base._install_policy
    _ACTIVE_ARGS = args
    base._prepare_lane = _prepare_lane
    base._install_policy = _install_policy
    try:
        payload = base.run(args)
    finally:
        base._prepare_lane = original_prepare
        base._install_policy = original_install
        _ACTIVE_ARGS = None

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    payload["workload"] = args.workload
    payload["settings"].update(
        {
            "sampling_workload": args.workload,
            "sampling_seed": args.seed,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--cache-kind", choices=("native",), default="native")
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--memory-sample-interval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("token counts must be positive")

    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
