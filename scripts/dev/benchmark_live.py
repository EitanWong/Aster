#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import psutil
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
    temperature: float
    platform: str
    python_version: str
    mlx_lm_version: str
    system_memory_total_bytes: int
    process_rss_peak_bytes: int
    swap_used_bytes_before: int
    swap_used_bytes_after: int
    workload: str
    request_count: int
    concurrency: int
    total_prompt_tokens: int
    elapsed_seconds: float
    average_latency_seconds: float
    p95_latency_seconds: float
    long_request_latency_seconds: float | None
    short_request_p95_latency_seconds: float | None
    throughput_completion_tps: float
    average_generation_tps: float
    completion_tokens_per_decode_step: float
    total_completion_tokens: int
    prefix_reuse_hits: int
    prefix_tokens_reused: int
    prefix_cache_stores: int
    prefix_cache_exact_hits: int
    prefix_cache_lcp_hits: int
    prefix_unsafe_lcp_skips: int
    decode_steps: int
    completed_requests: int
    failed_requests: int
    cancelled_requests: int
    admission_rejections: int


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1), 0)
    return ordered[index]


def _extract_staggered_latency_metrics(
    timelines: list[dict[str, object]],
) -> dict[str, float | None]:
    long_latency: float | None = None
    short_latencies: list[float] = []
    for timeline in timelines:
        request_id = timeline.get("request_id")
        latency = timeline.get("total_latency_s")
        if not isinstance(request_id, str) or not isinstance(latency, (int, float)):
            continue
        if request_id == "staggered-long":
            long_latency = float(latency)
        elif request_id.startswith("staggered-short-"):
            short_latencies.append(float(latency))
    return {
        "long_request_latency_seconds": long_latency,
        "short_request_p95_latency_seconds": (
            _percentile(short_latencies, 0.95) if short_latencies else None
        ),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _prefix_cache_counter_delta(
    before: dict[str, object],
    after: dict[str, object],
    name: str,
) -> int:
    def read(status: dict[str, object]) -> int:
        stats = status.get("prefix_cache_stats")
        if not isinstance(stats, dict):
            return 0
        value = stats.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    return read(after) - read(before)


def _collect_runtime_metadata() -> dict[str, int | str]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "mlx_lm_version": _package_version("mlx-lm"),
        "system_memory_total_bytes": int(psutil.virtual_memory().total),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "swap_used_bytes": int(psutil.swap_memory().used),
    }


async def _sample_process_rss(stop_event: asyncio.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop_event.is_set():
        try:
            samples.append(int(process.memory_info().rss))
        except psutil.Error:
            return
        await asyncio.sleep(0.05)


def _build_workload(
    name: str,
    concurrency: int,
    *,
    temperature: float = 0.0,
    long_prompt_words: int = 4096,
) -> list[InferenceRequest]:
    if long_prompt_words < 1:
        raise ValueError("long_prompt_words must be >= 1")
    repeated_prefix = (
        "System: You are a local Apple Silicon assistant. "
        "Keep answers precise. Reuse prior context when possible. "
        + " ".join(
            "Preserve this stable operating policy across every agent turn."
            for _ in range(16)
        )
        + " "
    )
    if name == "single":
        return [
            InferenceRequest(
                prompt="Explain how unified memory changes local LLM inference on Apple Silicon.",
                max_tokens=128,
                temperature=temperature,
                trace_id="single-0",
            )
        ]
    if name in {"reuse", "reuse-divergent"}:
        return [
            InferenceRequest(
                prompt=(
                    repeated_prefix + "Summarize the same operating constraints."
                    if name == "reuse"
                    else repeated_prefix
                    + f"User turn {index}: summarize the same operating constraints."
                ),
                max_tokens=96,
                temperature=temperature,
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
                temperature=temperature,
                trace_id=f"mixed-{index}",
            )
            for index in range(max(concurrency, 4))
        ]
    if name == "staggered":
        long_prompt = repeated_prefix + " ".join(["section"] * long_prompt_words)
        request_count = max(concurrency, 4)
        return [
            InferenceRequest(
                prompt=long_prompt,
                max_tokens=128,
                temperature=temperature,
                trace_id="staggered-long",
            ),
            *[
                InferenceRequest(
                    prompt="Summarize scheduler fairness tradeoffs.",
                    max_tokens=48,
                    temperature=temperature,
                    trace_id=f"staggered-short-{index}",
                )
                for index in range(1, request_count)
            ],
        ]
    if name == "long":
        long_prompt = repeated_prefix + " ".join(["section"] * long_prompt_words)
        return [
            InferenceRequest(
                prompt=long_prompt,
                max_tokens=128,
                temperature=temperature,
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


async def _run_requests(
    engine: InferenceEngine,
    requests: list[InferenceRequest],
    *,
    staggered: bool = False,
    sequential: bool = False,
) -> tuple[list[float], list[object]]:
    latencies: list[float] = []

    async def run_one(request: InferenceRequest) -> object:
        started = time.perf_counter()
        try:
            return await engine.submit(request)
        finally:
            latencies.append(time.perf_counter() - started)

    if sequential:
        results = [await run_one(request) for request in requests]
        return latencies, results

    if not staggered:
        results = await asyncio.gather(
            *(run_one(request) for request in requests),
            return_exceptions=True,
        )
        return latencies, results

    tasks = [asyncio.create_task(run_one(requests[0]))]
    deadline = time.monotonic() + 5.0
    while not tasks[0].done() and time.monotonic() < deadline:
        status = engine.status()
        request_phases = {item.get("phase") for item in status.get("requests", [])}
        if status.get("prefill_requests", 0) > 0 or "prefilling" in request_phases:
            break
        await asyncio.sleep(0.01)
    for request in requests[1:]:
        await asyncio.sleep(0.05)
        tasks.append(asyncio.create_task(run_one(request)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return latencies, results


async def benchmark_workload(
    engine: InferenceEngine,
    *,
    workload: str,
    concurrency: int,
    temperature: float,
    long_prompt_words: int,
) -> BenchmarkRecord:
    requests = _build_workload(
        workload,
        concurrency,
        temperature=temperature,
        long_prompt_words=long_prompt_words,
    )
    runtime_metadata = _collect_runtime_metadata()
    rss_samples = [int(runtime_metadata["process_rss_bytes"])]
    rss_stop_event = asyncio.Event()
    rss_task = asyncio.create_task(_sample_process_rss(rss_stop_event, rss_samples))
    before = engine.status()
    try:
        started = time.perf_counter()
        latencies, results = await _run_requests(
            engine,
            requests,
            staggered=workload == "staggered",
            sequential=workload in {"reuse", "reuse-divergent"},
        )
        elapsed = time.perf_counter() - started
        after = engine.status()
    finally:
        rss_stop_event.set()
        await rss_task
    after_metadata = _collect_runtime_metadata()
    timelines = [
        timeline
        for timeline in after.get("recent_request_timelines", [])
        if isinstance(timeline, dict)
    ]
    staggered_latency_metrics = (
        _extract_staggered_latency_metrics(timelines)
        if workload == "staggered"
        else {
            "long_request_latency_seconds": None,
            "short_request_p95_latency_seconds": None,
        }
    )

    responses = [result for result in results if not isinstance(result, Exception)]
    total_completion_tokens = sum(getattr(response, "completion_tokens", 0) for response in responses)
    average_generation_tps = (
        sum(getattr(response, "generation_tps", 0.0) for response in responses) / len(responses)
        if responses
        else 0.0
    )

    return BenchmarkRecord(
        runtime_kernel=str(after.get("runtime_kernel", "unknown")),
        temperature=temperature,
        platform=str(runtime_metadata["platform"]),
        python_version=str(runtime_metadata["python_version"]),
        mlx_lm_version=str(runtime_metadata["mlx_lm_version"]),
        system_memory_total_bytes=int(runtime_metadata["system_memory_total_bytes"]),
        process_rss_peak_bytes=max(rss_samples),
        swap_used_bytes_before=int(runtime_metadata["swap_used_bytes"]),
        swap_used_bytes_after=int(after_metadata["swap_used_bytes"]),
        workload=workload,
        request_count=len(requests),
        concurrency=concurrency,
        total_prompt_tokens=int(after["total_prompt_tokens"]) - int(before["total_prompt_tokens"]),
        elapsed_seconds=elapsed,
        average_latency_seconds=(sum(latencies) / len(latencies)) if latencies else 0.0,
        p95_latency_seconds=_percentile(latencies, 0.95),
        long_request_latency_seconds=staggered_latency_metrics["long_request_latency_seconds"],
        short_request_p95_latency_seconds=staggered_latency_metrics[
            "short_request_p95_latency_seconds"
        ],
        throughput_completion_tps=total_completion_tokens / elapsed if elapsed > 0 else 0.0,
        average_generation_tps=average_generation_tps,
        completion_tokens_per_decode_step=(
            total_completion_tokens
            / max(int(after["decode_steps"]) - int(before["decode_steps"]), 1)
        ),
        total_completion_tokens=total_completion_tokens,
        prefix_reuse_hits=int(after["prefix_reuse_hits"]) - int(before["prefix_reuse_hits"]),
        prefix_tokens_reused=int(after["prefix_tokens_reused"]) - int(before["prefix_tokens_reused"]),
        prefix_cache_stores=_prefix_cache_counter_delta(before, after, "stores"),
        prefix_cache_exact_hits=_prefix_cache_counter_delta(before, after, "exact_hits"),
        prefix_cache_lcp_hits=_prefix_cache_counter_delta(before, after, "lcp_hits"),
        prefix_unsafe_lcp_skips=_prefix_cache_counter_delta(before, after, "unsafe_lcp_skips"),
        decode_steps=int(after["decode_steps"]) - int(before["decode_steps"]),
        completed_requests=int(after["completed_requests"]) - int(before["completed_requests"]),
        failed_requests=int(after["failed_requests"]) - int(before["failed_requests"]),
        cancelled_requests=int(after["cancelled_requests"]) - int(before["cancelled_requests"]),
        admission_rejections=int(after["admission_rejections"])
        - int(before["admission_rejections"]),
    )


async def run(
    config_path: str,
    *,
    workloads: list[str],
    concurrency_levels: list[int],
    runtime_kernel: str | None,
    temperature: float,
    long_prompt_words: int = 4096,
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
                        temperature=temperature,
                        long_prompt_words=long_prompt_words,
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
        choices=["single", "reuse", "reuse-divergent", "mixed", "staggered", "long", "all"],
        default="all",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--long-prompt-words",
        type=int,
        default=4096,
        help="Number of repeated words in long and staggered prompts. Defaults to 4096.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for every request. Defaults to greedy 0.0 for reproducibility.",
    )
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

    workloads = (
        ["single", "reuse", "reuse-divergent", "mixed", "staggered", "long"]
        if args.workload == "all"
        else [args.workload]
    )
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
            temperature=args.temperature,
            long_prompt_words=args.long_prompt_words,
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
