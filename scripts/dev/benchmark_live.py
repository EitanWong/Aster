from __future__ import annotations

import argparse
import asyncio
import json

from autotune.benchmark import BenchmarkSuite
from core.config import load_settings
from scheduler.policy_engine import PolicyEngine
from telemetry.metrics import MetricsRegistry


async def run(
    config_path: str,
    *,
    mode: str,
    timeout_seconds: float,
    shortlist_size: int,
    progress: bool,
) -> None:
    settings = load_settings(config_path)
    metrics = MetricsRegistry(settings.telemetry.metrics_namespace)
    policy = PolicyEngine(settings)
    results = await BenchmarkSuite(settings, metrics, policy).run(
        mode=mode,
        candidate_timeout_seconds=timeout_seconds,
        shortlist_size=shortlist_size,
        progress=progress,
    )
    output = [
        {
            "mode": r.mode,
            "speculative_enabled": r.speculative_enabled,
            "draft_tokens": r.draft_tokens,
            "batch_window_ms": r.batch_window_ms,
            "max_batch_size": r.max_batch_size,
            "stream_flush_ms": r.stream_flush_ms,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "prompt_tps": r.prompt_tps,
            "generation_tps": r.generation_tps,
            "elapsed_seconds": r.elapsed_seconds,
            "second_elapsed_seconds": r.second_elapsed_seconds,
            "second_cache_hit": r.second_cache_hit,
            "prefill_cache_hit": r.prefill_cache_hit,
            "generation_cache_reuse": r.generation_cache_reuse,
            "speculative_path_mode": r.speculative_path_mode,
            "reuse_gain_ratio": r.reuse_gain_ratio,
            "latency_score": r.latency_score,
            "total_score": r.total_score,
        }
        for r in results[:10]
    ]
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Aster MLX benchmark candidates")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        run(
            args.config,
            mode=args.mode,
            timeout_seconds=args.timeout_seconds,
            shortlist_size=args.shortlist_size,
            progress=not args.no_progress,
        )
    )


if __name__ == "__main__":
    main()
