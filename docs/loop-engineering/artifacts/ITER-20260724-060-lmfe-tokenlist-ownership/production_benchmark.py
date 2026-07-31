#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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

from aster.inference.constrained.json_schema_processor import (  # noqa: E402
    JSONSchemaLogitsProcessor,
)
from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402

sampling = paired.sampling
production = paired.production
base = paired.base


class ProductionMetrics(sampling.SamplingMetrics):
    def __init__(self, mlx: Any, policy: str) -> None:
        super().__init__(mlx, policy)
        self.instrumented_processors: set[int] = set()
        self.max_prefix_state_count = 0

    def record(self, processor: JSONSchemaLogitsProcessor) -> None:
        self.max_prefix_state_count = max(
            self.max_prefix_state_count,
            len(processor._enforcer.prefix_states),
        )

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload["max_prefix_state_count"] = self.max_prefix_state_count
        return payload


def _target_processor(processor: Any) -> JSONSchemaLogitsProcessor | None:
    target = processor
    visited: set[int] = set()
    while target is not None and id(target) not in visited:
        if isinstance(target, JSONSchemaLogitsProcessor):
            return target
        visited.add(id(target))
        target = getattr(target, "_inner", None)
    return None


def _install_policy(
    runner: ModelRunner,
    policy: str,
) -> tuple[ProductionMetrics, Any]:
    args = sampling._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("LMFE ownership benchmark arguments are not active")
    mlx = runner._mx
    if mlx is None:
        raise RuntimeError("MLX is not loaded")
    mlx.random.seed(args.seed)
    metrics = ProductionMetrics(mlx, policy)
    original_apply = runner._apply_logits_processors

    def apply_logits_processors(logits: Any, *, item: DecodeWorkItem) -> Any:
        targets = [
            target
            for processor in item.logits_processors
            if (target := _target_processor(processor)) is not None
        ]
        for target in targets:
            if id(target) in metrics.instrumented_processors:
                continue
            metrics.instrumented_processors.add(id(target))
            if policy == "baseline":
                target._enforcer = TokenEnforcer(
                    target._tokenizer_data,
                    target._enforcer.root_parser,
                )
                target._enforcer.get_allowed_tokens(())
                target._enforcer_last_suffix = ()
                target._reuse_freetext_token_lists = False
                target._bounded_prefix_states = False
        value = original_apply(logits, item=item)
        for target in targets:
            metrics.record(target)
        return value

    runner._mx = metrics
    runner._apply_logits_processors = apply_logits_processors  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = mlx
        runner._apply_logits_processors = original_apply  # type: ignore[method-assign]

    return metrics, restore


def run(args: argparse.Namespace) -> dict[str, object]:
    original_install = production._install_policy
    production._install_policy = _install_policy
    try:
        payload = paired.run(args)
    finally:
        production._install_policy = original_install

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(PROJECT_ROOT))] = base._sha256(source)
    payload["comparison"] = {
        "baseline": "native LMFE TokenEnforcer retaining request prefix TokenLists",
        "production": "Aster reusable per-request LMFE freetext TokenList backing",
        "candidate_scope": (
            "parser transitions, allowed-token contents, key/EOS filters, masks, processor "
            "order, samplers, RNG, cache extraction, and runner assignment are unchanged"
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
    parser.add_argument("--seed", type=int, default=20260728)
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
