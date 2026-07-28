#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sampling_benchmark as sampling

from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
POLICIES = ("baseline", "production")


def _decode_baseline(
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
    try:
        mx.eval(logits)
    except Exception:
        pass

    results: list[DecodeResult] = []
    peak_memory_gb = runner.current_peak_memory_gb()
    for index, item in enumerate(items):
        row = runner._apply_logits_processors(
            logits[index : index + 1],
            item=item,
        )
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        token = runner._sample_token(logprobs, item.sampler)
        prompt_cache = (
            runner._decode_cache_ref(batch_cache_state, index)
            if batch_cache_state is not None
            else runner._extract_prompt_cache(merged_cache, index)
        )
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
) -> tuple[sampling.SamplingMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("production benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = sampling.SamplingMetrics(mlx, policy)
    original_batch = runner._decode_batch

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        with metrics.decode_scope():
            if policy == "baseline":
                return _decode_baseline(runner, items)
            return original_batch(items)

    runner._mx = metrics
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._decode_batch = original_batch  # type: ignore[method-assign]

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = sampling._install_policy
    sampling._install_policy = _install_policy
    try:
        payload = sampling.run(args)
    finally:
        sampling._install_policy = original_install

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = sampling.base._sha256(
        source
    )
    payload["comparison"] = {
        "baseline": "iteration-050 eager-logits plus per-row sample sync",
        "production": "current ModelRunner._decode_batch",
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--workload", choices=sampling.WORKLOADS, required=True)
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
