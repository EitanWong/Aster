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

import candidate_benchmark as candidate  # noqa: E402

from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

base = candidate.base
sampling = candidate.sampling
_BASE_INSTALL_POLICY = candidate._install_policy


class ProfileMetrics(candidate.ProcessorMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.batch_seconds: list[float] = []
        self.processor_seconds: list[float] = []
        self.decode_result_seconds: list[float] = []

    @staticmethod
    def _timing(values: list[float]) -> dict[str, object]:
        return {
            **base._summary(values),
            "total": sum(values),
        }

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "batch_seconds": self._timing(self.batch_seconds),
                "processor_seconds": self._timing(self.processor_seconds),
                "decode_result_seconds": self._timing(self.decode_result_seconds),
                "materialize_seconds": self._timing(self.materialize_seconds),
                "explicit_eval_seconds": self._timing(self.eval_seconds),
                "async_eval_submit_seconds": self._timing(self.async_eval_seconds),
            }
        )
        return payload


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[ProfileMetrics, Any]:
    metrics, restore_base = _BASE_INSTALL_POLICY(runner, policy)
    if not isinstance(metrics, ProfileMetrics):
        raise TypeError(f"unexpected metrics type: {type(metrics)!r}")

    original_batch = runner._decode_batch
    original_apply = runner._apply_logits_processors
    original_materialize = runner._materialize_sampled_token
    original_decode_result = runner._decode_result

    def decode_batch(items: list[DecodeWorkItem]) -> list[Any]:
        started = time.perf_counter()
        try:
            with metrics.decode_scope():
                return original_batch(items)
        finally:
            metrics.batch_seconds.append(time.perf_counter() - started)

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        started = time.perf_counter()
        try:
            return original_apply(logits, item=item)
        finally:
            metrics.processor_seconds.append(time.perf_counter() - started)

    def materialize_sampled_token(sampled: Any) -> int:
        started = time.perf_counter()
        try:
            return original_materialize(sampled)
        finally:
            metrics.materialize_seconds.append(time.perf_counter() - started)

    def decode_result(**kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_decode_result(**kwargs)
        finally:
            metrics.decode_result_seconds.append(time.perf_counter() - started)

    runner._decode_batch = decode_batch  # type: ignore[method-assign]
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]
    runner._materialize_sampled_token = materialize_sampled_token  # type: ignore[method-assign]
    runner._decode_result = decode_result  # type: ignore[method-assign]

    def restore() -> None:
        runner._decode_batch = original_batch  # type: ignore[method-assign]
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]
        runner._materialize_sampled_token = original_materialize  # type: ignore[method-assign]
        runner._decode_result = original_decode_result  # type: ignore[method-assign]
        restore_base()

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_metrics = candidate.ProcessorMetrics
    original_install = candidate._install_policy
    candidate.ProcessorMetrics = ProfileMetrics
    candidate._install_policy = _install_policy
    try:
        payload = candidate.run(args)
    finally:
        candidate._install_policy = original_install
        candidate.ProcessorMetrics = original_metrics

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(source)
    payload["profile"] = {
        "scope": "current grouped asynchronous batch decode",
        "host_post_eval": "sample token materialization plus DecodeResult construction",
        "timing_clock": "time.perf_counter",
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
