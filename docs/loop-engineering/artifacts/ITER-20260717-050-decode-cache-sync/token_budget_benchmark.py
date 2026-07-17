#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import benchmark as base

TOKEN_BUDGET_POLICY = "periodic-token-512"
_BATCH_SIZE = 1


class TokenBudgetPolicyMX(base.DecodePolicyMX):
    def clear_cache(self) -> None:
        if self.policy != TOKEN_BUDGET_POLICY:
            super().clear_cache()
            return
        if self._active_depth <= 0:
            self._base.clear_cache()
            return

        self.clear_requests += 1
        interval = max(512 // _BATCH_SIZE, 1)
        execute = self.clear_requests % interval == 0
        started = time.perf_counter()
        try:
            if execute:
                self.clear_executed += 1
                self._base.clear_cache()
            else:
                self.clear_skipped += 1
        finally:
            self.clear_seconds.append(time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=base.PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=base.PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--policy", choices=("baseline", TOKEN_BUDGET_POLICY), required=True)
    parser.add_argument("--cache-kind", choices=("native", "direct"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-words", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--memory-sample-interval", type=int, default=32)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    global _BATCH_SIZE
    _BATCH_SIZE = args.batch_size
    original_policy = base.DecodePolicyMX
    base.DecodePolicyMX = TokenBudgetPolicyMX
    try:
        payload = base.run(args)
    finally:
        base.DecodePolicyMX = original_policy

    source = Path(__file__).resolve()
    payload["source_sha256"][str(source.relative_to(base.PROJECT_ROOT))] = base._sha256(source)
    payload["settings"]["decode_cache_clear_token_budget"] = 512
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
