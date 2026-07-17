#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import benchmark as base


class ProductionPolicyMetrics:
    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def metrics(self) -> dict[str, object]:
        diagnostics = self.runner.decode_diagnostics()
        return {
            "policy": "production",
            "cache_eval_requests": 0,
            "cache_eval_executed": 0,
            "cache_eval_skipped": 0,
            "cache_eval_seconds": base._summary([]),
            "clear_requests": diagnostics["cache_clear_attempts"],
            "clear_executed": diagnostics["cache_clears"],
            "clear_skipped": 0,
            "clear_seconds": base._summary([]),
            "clear_failures": diagnostics["cache_clear_failures"],
            "clear_token_budget": diagnostics["cache_clear_token_budget"],
        }


def _install_production_policy(runner: Any, _policy: str):
    return ProductionPolicyMetrics(runner), lambda: None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=base.PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=base.PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--cache-kind", choices=("native", "direct"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-words", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--memory-sample-interval", type=int, default=32)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.policy = "production"

    original_install = base._install_policy
    base._install_policy = _install_production_policy
    try:
        payload = base.run(args)
    finally:
        base._install_policy = original_install

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(base.PROJECT_ROOT))] = base._sha256(source)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
