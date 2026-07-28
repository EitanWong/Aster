#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
for path in (PROJECT_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paired_benchmark as paired  # noqa: E402
from lmformatenforcer import TokenEnforcer  # noqa: E402

import aster.inference.constrained.json_schema_processor as json_processor_module  # noqa: E402
from aster.inference.constrained.json_schema_processor import (  # noqa: E402
    JSONSchemaLogitsProcessor,
    ThinkingAwareJsonLogitsProcessor,
)
from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

sampling = paired.sampling
production = paired.production
base = paired.base
_ACTIVE_METRICS: ProfileMetrics | None = None


class ProfileMetrics(sampling.SamplingMetrics):
    def __init__(
        self,
        mlx: Any,
        policy: str,
        *,
        measured_calls: int,
    ) -> None:
        super().__init__(mlx, policy)
        self.measured_calls = measured_calls
        self.apply_seconds: list[float] = []
        self.processor_seconds: list[float] = []
        self.token_list_seconds: list[float] = []
        self.suffix_seconds: list[float] = []
        self.allowed_seconds: list[float] = []
        self.enforcer_seconds: list[float] = []
        self.json_context_seconds: list[float] = []
        self.decode_suffix_seconds: list[float] = []
        self.mask_seconds: list[float] = []
        self.mask_hit_flags: list[bool] = []
        self.history_lengths: list[int] = []
        self.suffix_lengths: list[int] = []
        self.allowed_cardinalities: list[int] = []
        self.instrumented_processors: set[int] = set()
        self.processor_restorers: list[Any] = []

    def _tail(self, values: list[Any]) -> list[Any]:
        return values[-self.measured_calls :]

    def _timing(self, values: list[float]) -> dict[str, object]:
        measured = self._tail(values)
        return {
            "all_calls": {
                **base._summary(values),
                "count": len(values),
                "total": sum(values),
            },
            "measured_tail": {
                **base._summary(measured),
                "count": len(measured),
                "total": sum(measured),
            },
        }

    def _sizes(self, values: list[int]) -> dict[str, object]:
        measured = self._tail(values)
        return {
            "all_calls": {**base._summary(values), "count": len(values)},
            "measured_tail": {
                **base._summary(measured),
                "count": len(measured),
            },
        }

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        measured_hits = self._tail(self.mask_hit_flags)
        payload.update(
            {
                "profile_measured_calls": self.measured_calls,
                "profile_seconds": {
                    "apply": self._timing(self.apply_seconds),
                    "processor": self._timing(self.processor_seconds),
                    "token_list": self._timing(self.token_list_seconds),
                    "generated_suffix": self._timing(self.suffix_seconds),
                    "allowed_tokens": self._timing(self.allowed_seconds),
                    "enforcer": self._timing(self.enforcer_seconds),
                    "json_context": self._timing(self.json_context_seconds),
                    "decode_suffix": self._timing(self.decode_suffix_seconds),
                    "mask": self._timing(self.mask_seconds),
                },
                "profile_sizes": {
                    "history": self._sizes(self.history_lengths),
                    "suffix": self._sizes(self.suffix_lengths),
                    "allowed_cardinality": self._sizes(self.allowed_cardinalities),
                },
                "mask_cache": {
                    "all_calls": len(self.mask_hit_flags),
                    "all_hits": sum(self.mask_hit_flags),
                    "measured_calls": len(measured_hits),
                    "measured_hits": sum(measured_hits),
                    "measured_misses": len(measured_hits) - sum(measured_hits),
                    "measured_hit_rate": (
                        sum(measured_hits) / len(measured_hits)
                        if measured_hits
                        else 0.0
                    ),
                },
            }
        )
        return payload


def _target_processor(processor: Any) -> JSONSchemaLogitsProcessor | None:
    target = processor
    if isinstance(target, ThinkingAwareJsonLogitsProcessor):
        target = target._inner
    return target if isinstance(target, JSONSchemaLogitsProcessor) else None


def _timed_method(
    values: list[float],
    original: Any,
) -> Any:
    def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            values.append(time.perf_counter() - started)

    return timed


def _instrument_processor(processor: Any, metrics: ProfileMetrics) -> None:
    target = _target_processor(processor)
    if target is None or id(target) in metrics.instrumented_processors:
        return
    metrics.instrumented_processors.add(id(target))

    original_generated_suffix = target._generated_suffix
    original_allowed = target._allowed_tokens
    original_json_context = target._json_context
    original_decode_suffix = target._decode_suffix
    original_mask = target._mask
    original_enforcer = target._enforcer.get_allowed_tokens

    def generated_suffix(token_ids: list[int]) -> list[int]:
        started = time.perf_counter()
        try:
            suffix = original_generated_suffix(token_ids)
        except Exception:
            metrics.suffix_seconds.append(time.perf_counter() - started)
            raise
        metrics.suffix_seconds.append(time.perf_counter() - started)
        metrics.suffix_lengths.append(len(suffix))
        return suffix

    def allowed_tokens(suffix: list[int]) -> list[int]:
        started = time.perf_counter()
        try:
            allowed = original_allowed(suffix)
        except Exception:
            metrics.allowed_seconds.append(time.perf_counter() - started)
            raise
        metrics.allowed_seconds.append(time.perf_counter() - started)
        metrics.allowed_cardinalities.append(len(allowed))
        return allowed

    def mask(allowed: list[int], logits: Any) -> Any:
        previous_value = target._mask_cache_value
        started = time.perf_counter()
        try:
            value = original_mask(allowed, logits)
        except Exception:
            metrics.mask_seconds.append(time.perf_counter() - started)
            raise
        metrics.mask_seconds.append(time.perf_counter() - started)
        metrics.mask_hit_flags.append(previous_value is not None and value is previous_value)
        return value

    target._generated_suffix = generated_suffix
    target._allowed_tokens = allowed_tokens
    target._json_context = _timed_method(
        metrics.json_context_seconds,
        original_json_context,
    )
    target._decode_suffix = _timed_method(
        metrics.decode_suffix_seconds,
        original_decode_suffix,
    )
    target._mask = mask
    target._enforcer.get_allowed_tokens = _timed_method(
        metrics.enforcer_seconds,
        original_enforcer,
    )

    def restore_processor() -> None:
        target._generated_suffix = original_generated_suffix
        target._allowed_tokens = original_allowed
        target._json_context = original_json_context
        target._decode_suffix = original_decode_suffix
        target._mask = original_mask
        target._enforcer.get_allowed_tokens = original_enforcer

    metrics.processor_restorers.append(restore_processor)


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[ProfileMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("structured profile arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = ProfileMetrics(
        mlx,
        policy,
        measured_calls=args.steps * args.batch_size,
    )
    original_apply = runner._apply_logits_processors

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        global _ACTIVE_METRICS
        for processor in item.logits_processors:
            _instrument_processor(processor, metrics)
        previous_metrics = _ACTIVE_METRICS
        _ACTIVE_METRICS = metrics
        started = time.perf_counter()
        try:
            return original_apply(logits, item=item)
        finally:
            metrics.apply_seconds.append(time.perf_counter() - started)
            _ACTIVE_METRICS = previous_metrics

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]

    def restore() -> None:
        for restore_processor in metrics.processor_restorers:
            restore_processor()
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]

    return metrics, restore


def _add_profile_shares(payload: dict[str, object]) -> None:
    timings = payload["timings"]
    policy_metrics = payload["policy_metrics"]
    assert isinstance(timings, dict) and isinstance(policy_metrics, dict)
    for policy in ("baseline", "production"):
        metrics = policy_metrics[policy]
        assert isinstance(metrics, dict)
        profile_seconds = metrics["profile_seconds"]
        assert isinstance(profile_seconds, dict)
        decode_values = timings[f"{policy}_step_seconds"]
        assert isinstance(decode_values, list)
        decode_total = sum(float(value) for value in decode_values)
        measured_totals = {
            name: float(value["measured_tail"]["total"])
            for name, value in profile_seconds.items()
        }
        processor_total = measured_totals["processor"]
        top_level_total = sum(
            measured_totals[name]
            for name in (
                "token_list",
                "generated_suffix",
                "allowed_tokens",
                "mask",
            )
        )
        metrics["profile_measured_decode_seconds"] = decode_total
        metrics["profile_share_of_decode_percent"] = {
            name: (total / decode_total * 100.0 if decode_total else 0.0)
            for name, total in measured_totals.items()
        }
        metrics["profile_exclusive_seconds"] = {
            "runner_input_setup": measured_totals["apply"] - processor_total,
            "processor_control_and_addition": processor_total - top_level_total,
            "decode_outside_apply": decode_total - measured_totals["apply"],
        }


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = production._install_policy
    original_token_list = json_processor_module._token_list
    original_call = JSONSchemaLogitsProcessor.__call__

    def token_list(tokens: Any) -> list[int]:
        started = time.perf_counter()
        values = original_token_list(tokens)
        metrics = _ACTIVE_METRICS
        if metrics is not None:
            metrics.token_list_seconds.append(time.perf_counter() - started)
            metrics.history_lengths.append(len(values))
        return values

    def processor_call(self: JSONSchemaLogitsProcessor, tokens: Any, logits: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_call(self, tokens, logits)
        finally:
            metrics = _ACTIVE_METRICS
            if metrics is not None:
                metrics.processor_seconds.append(time.perf_counter() - started)

    production._install_policy = _install_policy
    json_processor_module._token_list = token_list
    JSONSchemaLogitsProcessor.__call__ = processor_call
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install
        json_processor_module._token_list = original_token_list
        JSONSchemaLogitsProcessor.__call__ = original_call

    _add_profile_shares(payload)
    source = Path(__file__).resolve()
    lmfe_source = Path(inspect.getsourcefile(TokenEnforcer) or "")
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(
        source
    )
    payload["dependency_sha256"] = {
        str(lmfe_source): base._sha256(lmfe_source),
    }
    payload["comparison"] = {
        "baseline": "current production path on one physical runner",
        "production": "identical current production path on the other physical runner",
        "purpose": "runner-balanced residual profiling, not a candidate A/B claim",
        "timing_clock": "time.perf_counter",
        "measured_window": "last steps * batch_size row calls; paired warmups excluded",
        "nested_timings": (
            "enforcer is nested in allowed_tokens; decode_suffix and json_context are "
            "nested in allowed_tokens and must not be summed as exclusive stages"
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
