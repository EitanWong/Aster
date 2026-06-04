#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.engine import InferenceEngine  # noqa: E402
from aster.telemetry.metrics import MetricsRegistry  # noqa: E402


@dataclass(slots=True)
class BenchmarkRecord:
    runtime_kernel: str
    workload: str
    request_count: int
    concurrency: int
    elapsed_seconds: float
    average_latency_seconds: float
    p95_latency_seconds: float
    throughput_completion_tps: float
    average_generation_tps: float
    completion_tokens_per_decode_step: float
    total_completion_tokens: int
    prefix_reuse_hits: int
    prefix_tokens_reused: int
    decode_steps: int
    completed_requests: int
    failed_requests: int
    cancelled_requests: int


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1), 0)
    return ordered[index]


def _build_workload(name: str, concurrency: int) -> list[InferenceRequest]:
    repeated_prefix = (
        "System: You are a local Apple Silicon assistant. "
        "Keep answers precise. Reuse prior context when possible. "
    )
    if name == "single":
        return [
            InferenceRequest(
                prompt="Explain how unified memory changes local LLM inference on Apple Silicon.",
                max_tokens=128,
                trace_id="single-0",
            )
        ]
    if name == "reuse":
        return [
            InferenceRequest(
                prompt=repeated_prefix + f"User turn {index}: summarize the same operating constraints.",
                max_tokens=96,
                trace_id=f"reuse-{index}",
            )
            for index in range(max(concurrency, 2))
        ]
    if name == "mixed":
        prompts = [
            "Summarize scheduler fairness tradeoffs.",
            repeated_prefix + "User asks for a brief plan.",
            " ".join(["context"] * 1024) + " Summarize the document.",
            repeated_prefix + "User asks for the same brief plan again.",
        ]
        return [
            InferenceRequest(
                prompt=prompts[index % len(prompts)],
                max_tokens=96 if index % 2 == 0 else 48,
                trace_id=f"mixed-{index}",
            )
            for index in range(max(concurrency, 4))
        ]
    if name == "long":
        long_prompt = repeated_prefix + " ".join(["section"] * 4096)
        return [
            InferenceRequest(
                prompt=long_prompt,
                max_tokens=128,
                trace_id=f"long-{index}",
            )
            for index in range(max(concurrency, 1))
        ]
    raise ValueError(f"Unknown workload: {name}")


def _parse_concurrency_levels(raw: str | None, fallback: int) -> list[int]:
    if not raw:
        return [max(fallback, 1)]
    levels: list[int] = []
    for part in raw.split(","):
        value = int(part.strip())
        if value < 1:
            raise ValueError("Concurrency levels must be >= 1")
        levels.append(value)
    return levels


async def _run_requests(engine: InferenceEngine, requests: list[InferenceRequest]) -> tuple[list[float], list[object]]:
    latencies: list[float] = []

    async def run_one(request: InferenceRequest) -> object:
        started = time.perf_counter()
        try:
            return await engine.submit(request)
        finally:
            latencies.append(time.perf_counter() - started)

    results = await asyncio.gather(*(run_one(request) for request in requests), return_exceptions=True)
    return latencies, results


async def benchmark_workload(
    engine: InferenceEngine,
    *,
    workload: str,
    concurrency: int,
) -> BenchmarkRecord:
    requests = _build_workload(workload, concurrency)
    before = engine.status()
    started = time.perf_counter()
    latencies, results = await _run_requests(engine, requests)
    elapsed = time.perf_counter() - started
    after = engine.status()

    responses = [result for result in results if not isinstance(result, Exception)]
    total_completion_tokens = sum(getattr(response, "completion_tokens", 0) for response in responses)
    average_generation_tps = (
        sum(getattr(response, "generation_tps", 0.0) for response in responses) / len(responses)
        if responses
        else 0.0
    )

    return BenchmarkRecord(
        runtime_kernel=str(after.get("runtime_kernel", "unknown")),
        workload=workload,
        request_count=len(requests),
        concurrency=concurrency,
        elapsed_seconds=elapsed,
        average_latency_seconds=(sum(latencies) / len(latencies)) if latencies else 0.0,
        p95_latency_seconds=_percentile(latencies, 0.95),
        throughput_completion_tps=total_completion_tokens / elapsed if elapsed > 0 else 0.0,
        average_generation_tps=average_generation_tps,
        completion_tokens_per_decode_step=(
            total_completion_tokens
            / max(int(after["decode_steps"]) - int(before["decode_steps"]), 1)
        ),
        total_completion_tokens=total_completion_tokens,
        prefix_reuse_hits=int(after["prefix_reuse_hits"]) - int(before["prefix_reuse_hits"]),
        prefix_tokens_reused=int(after["prefix_tokens_reused"]) - int(before["prefix_tokens_reused"]),
        decode_steps=int(after["decode_steps"]) - int(before["decode_steps"]),
        completed_requests=int(after["completed_requests"]) - int(before["completed_requests"]),
        failed_requests=int(after["failed_requests"]) - int(before["failed_requests"]),
        cancelled_requests=int(after["cancelled_requests"]) - int(before["cancelled_requests"]),
    )


async def run(
    config_path: str,
    *,
    workloads: list[str],
    concurrency_levels: list[int],
    runtime_kernel: str | None,
) -> list[BenchmarkRecord]:
    settings = load_settings(config_path)
    if runtime_kernel is not None:
        settings = settings.model_copy(
            update={
                "engine": settings.engine.model_copy(
                    update={"runtime_kernel": runtime_kernel}
                )
            }
        )
    metrics = MetricsRegistry(settings.telemetry.metrics_namespace)
    engine = InferenceEngine(settings, metrics)
    await engine.start()
    await engine.warmup()
    try:
        records: list[BenchmarkRecord] = []
        for concurrency in concurrency_levels:
            for workload in workloads:
                records.append(
                    await benchmark_workload(
                        engine,
                        workload=workload,
                        concurrency=concurrency,
                    )
                )
        return records
    finally:
        await engine.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Aster engine benchmarks")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--workload",
        choices=["single", "reuse", "mixed", "long", "all"],
        default="all",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--concurrency-levels",
        default=None,
        help="Comma-separated levels, for example 1,2,4,8. Overrides --concurrency.",
    )
    parser.add_argument(
        "--runtime-kernel",
        choices=["configured", "manual", "batch_generator"],
        default="configured",
        help=(
            "Override engine.runtime_kernel for benchmark comparison. "
            "batch_generator currently verifies the adapter boundary and fails fast."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    workloads = ["single", "reuse", "mixed", "long"] if args.workload == "all" else [args.workload]
    runtime_kernel = None if args.runtime_kernel == "configured" else args.runtime_kernel
    concurrency_levels = _parse_concurrency_levels(
        args.concurrency_levels,
        max(args.concurrency, 1),
    )
    records = asyncio.run(
        run(
            args.config,
            workloads=workloads,
            concurrency_levels=concurrency_levels,
            runtime_kernel=runtime_kernel,
        )
    )
    payload = [asdict(record) for record in records]
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
