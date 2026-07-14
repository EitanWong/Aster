#!/usr/bin/env python3
"""Benchmark the experimental mlx-lm BatchGenerator engine strategy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import RuntimeSettings, load_settings  # noqa: E402
from aster.inference.batched_engine import BatchedEngine  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.telemetry.metrics import MetricsRegistry  # noqa: E402
from scripts.dev.benchmark_live import (  # noqa: E402
    _build_workload,
    _collect_runtime_metadata,
    _parse_concurrency_levels,
    _percentile,
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _apply_benchmark_overrides(
    settings: RuntimeSettings,
    *,
    concurrency_levels: list[int],
    prefix_cache_enabled: bool,
    max_lanes: int,
) -> RuntimeSettings:
    return settings.model_copy(
        update={
            "engine": settings.engine.model_copy(
                update={
                    "engine_type": "batched",
                    "max_active_requests": max(
                        settings.engine.max_active_requests,
                        max(concurrency_levels),
                    ),
                    "prefix_cache_enabled": prefix_cache_enabled,
                    "batch_generator_max_lanes": max_lanes,
                }
            )
        }
    )


def _build_structured_workload(concurrency: int, *, temperature: float) -> list[InferenceRequest]:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["ok"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    return [
        InferenceRequest(
            prompt="Return exactly the JSON object with answer ok.",
            max_tokens=32,
            temperature=temperature,
            structured_output_schema=schema,
            trace_id=f"structured-{index}",
        )
        for index in range(max(concurrency, 1))
    ]


def _requests_for_workload(
    workload: str,
    concurrency: int,
    *,
    temperature: float,
    long_prompt_words: int,
) -> list[InferenceRequest]:
    if workload == "structured":
        return _build_structured_workload(concurrency, temperature=temperature)
    return _build_workload(
        workload,
        concurrency,
        temperature=temperature,
        long_prompt_words=long_prompt_words,
    )


async def _submit_requests(
    engine: BatchedEngine,
    requests: list[InferenceRequest],
    *,
    staggered: bool,
    sequential: bool,
) -> tuple[list[float], list[object]]:
    latencies: list[float] = []

    async def run_one(request: InferenceRequest) -> object:
        started = time.perf_counter()
        try:
            return await engine.submit(request)
        finally:
            latencies.append(time.perf_counter() - started)

    if sequential:
        return latencies, [await run_one(request) for request in requests]

    if not staggered:
        return latencies, await asyncio.gather(
            *(run_one(request) for request in requests),
            return_exceptions=True,
        )

    tasks = [asyncio.create_task(run_one(requests[0]))]
    deadline = time.monotonic() + 10.0
    while not tasks[0].done() and time.monotonic() < deadline:
        if int(engine.status().get("running_requests", 0)) > 0:
            break
        await asyncio.sleep(0.01)
    for request in requests[1:]:
        await asyncio.sleep(0.05)
        tasks.append(asyncio.create_task(run_one(request)))
    return latencies, await asyncio.gather(*tasks, return_exceptions=True)


def _response_hash(result: object) -> str | None:
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def _record_round(
    *,
    engine: BatchedEngine,
    workload: str,
    concurrency: int,
    round_index: int,
    requests: list[InferenceRequest],
    latencies: list[float],
    results: list[object],
    elapsed: float,
    metadata_before: dict[str, int | str],
    metadata_after: dict[str, int | str],
    rss_peak: int,
) -> dict[str, Any]:
    responses = [result for result in results if not isinstance(result, Exception)]
    errors = [
        {
            "type": type(result).__name__,
            "code": getattr(result, "code", None),
            "message": str(result),
        }
        for result in results
        if isinstance(result, Exception)
    ]
    total_completion_tokens = sum(
        int(getattr(response, "completion_tokens", 0)) for response in responses
    )
    prefix_stats = engine.prefix_store.stats_snapshot()
    structured_valid: list[bool] | None = None
    if workload == "structured":
        structured_valid = []
        for response in responses:
            try:
                payload = json.loads(str(getattr(response, "text", "")))
                structured_valid.append(
                    isinstance(payload, dict) and isinstance(payload.get("answer"), str)
                )
            except (TypeError, ValueError):
                structured_valid.append(False)
    return {
        "workload": workload,
        "concurrency": concurrency,
        "round": round_index,
        "request_count": len(requests),
        "prompt_tokens": sum(int(getattr(response, "prompt_tokens", 0)) for response in responses),
        "elapsed_s": elapsed,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "completion_tokens": total_completion_tokens,
        "completion_tps": total_completion_tokens / elapsed if elapsed > 0 else 0.0,
        "average_generation_tps": (
            sum(float(getattr(response, "generation_tps", 0.0)) for response in responses)
            / len(responses)
            if responses
            else 0.0
        ),
        "completed": len(responses),
        "errors": errors,
        "cache_hits": sum(bool(getattr(response, "prefill_cache_hit", False)) for response in responses),
        "finish_reasons": [getattr(response, "finish_reason", None) for response in responses],
        "response_hashes": [_response_hash(response) for response in responses],
        "structured_valid": structured_valid,
        "peak_mlx_gb": (
            float(engine._mx.get_peak_memory()) / 1e9 if engine._mx is not None else 0.0
        ),
        "rss_peak_bytes": rss_peak,
        "rss_delta_bytes": int(metadata_after["process_rss_bytes"])
        - int(metadata_before["process_rss_bytes"]),
        "swap_delta_bytes": int(metadata_after["swap_used_bytes"])
        - int(metadata_before["swap_used_bytes"]),
        "prefix_stats": prefix_stats,
    }


async def _sample_rss(stop: asyncio.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.is_set():
        samples.append(int(process.memory_info().rss))
        await asyncio.sleep(0.05)


async def _run_cancellation_probe(
    engine: BatchedEngine,
    *,
    concurrency: int,
) -> dict[str, Any]:
    prompt = "Cancellation and cache ownership probe. " * 120
    requests = [
        InferenceRequest(
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
            trace_id=f"cancel-probe-{index}",
            timeout_seconds=120,
        )
        for index in range(max(concurrency, 2))
    ]
    tasks = [asyncio.create_task(engine.submit(request)) for request in requests]
    await asyncio.sleep(0.05)
    cancelled = await engine.cancel("cancel-probe-1")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    follow_up = await engine.submit(
        InferenceRequest(
            prompt="Follow-up after cancellation.",
            max_tokens=8,
            temperature=0.0,
            trace_id="cancel-probe-follow-up",
            timeout_seconds=120,
        )
    )
    return {
        "request_count": len(requests),
        "cancel_accepted": cancelled,
        "completed": sum(not isinstance(result, Exception) for result in results),
        "cancelled_errors": sum(
            getattr(result, "code", None) == "request_cancelled" for result in results
        ),
        "other_errors": [
            {"type": type(result).__name__, "code": getattr(result, "code", None)}
            for result in results
            if isinstance(result, Exception)
            and getattr(result, "code", None) != "request_cancelled"
        ],
        "follow_up_completion_tokens": follow_up.completion_tokens,
        "running_after": int(engine.status().get("running_requests", 0)),
        "pinned_entries_after": int(engine.prefix_store.stats_snapshot()["pinned_entries"]),
    }


async def run_benchmark(
    config_path: str,
    *,
    workloads: list[str],
    concurrency_levels: list[int],
    rounds: int,
    temperature: float,
    long_prompt_words: int,
    prefix_cache_enabled: bool,
    max_lanes: int,
) -> dict[str, Any]:
    settings = load_settings(config_path)
    settings = _apply_benchmark_overrides(
        settings,
        concurrency_levels=concurrency_levels,
        prefix_cache_enabled=prefix_cache_enabled,
        max_lanes=max_lanes,
    )
    engine = BatchedEngine(settings, MetricsRegistry(settings.telemetry.metrics_namespace))
    await engine.start()
    await engine.warmup()
    records: list[dict[str, Any]] = []
    cancellation: dict[str, Any] | None = None
    try:
        for workload in workloads:
            for concurrency in concurrency_levels:
                for round_index in range(rounds):
                    requests = _requests_for_workload(
                        workload,
                        concurrency,
                        temperature=temperature,
                        long_prompt_words=long_prompt_words,
                    )
                    before = _collect_runtime_metadata()
                    rss_samples = [int(before["process_rss_bytes"])]
                    stop = asyncio.Event()
                    rss_task = asyncio.create_task(_sample_rss(stop, rss_samples))
                    started = time.perf_counter()
                    try:
                        latencies, results = await _submit_requests(
                            engine,
                            requests,
                            staggered=workload == "staggered",
                            sequential=workload in {"reuse", "reuse-divergent"},
                        )
                    finally:
                        elapsed = time.perf_counter() - started
                        stop.set()
                        await rss_task
                    after = _collect_runtime_metadata()
                    records.append(
                        _record_round(
                            engine=engine,
                            workload=workload,
                            concurrency=concurrency,
                            round_index=round_index,
                            requests=requests,
                            latencies=latencies,
                            results=results,
                            elapsed=elapsed,
                            metadata_before=before,
                            metadata_after=after,
                            rss_peak=max(rss_samples),
                        )
                    )
        cancellation = await _run_cancellation_probe(
            engine,
            concurrency=max(concurrency_levels),
        )
    finally:
        await engine.aclose()

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_lm": _package_version("mlx-lm"),
        "config": str(config_path),
        "engine": "batched",
        "prefix_cache_enabled": prefix_cache_enabled,
        "batch_generator_max_lanes": max_lanes,
        "workloads": workloads,
        "concurrency_levels": concurrency_levels,
        "rounds": rounds,
        "records": records,
        "cancellation": cancellation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--workloads",
        default="reuse,mixed,reuse-divergent,staggered,structured,long",
        help="Comma-separated workload names.",
    )
    parser.add_argument("--concurrency-levels", default="2,4,8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--long-prompt-words", type=int, default=1024)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="on")
    parser.add_argument("--max-lanes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    concurrency_levels = _parse_concurrency_levels(args.concurrency_levels, 2)
    payload = asyncio.run(
        run_benchmark(
            args.config,
            workloads=workloads,
            concurrency_levels=concurrency_levels,
            rounds=max(args.rounds, 1),
            temperature=args.temperature,
            long_prompt_words=args.long_prompt_words,
            prefix_cache_enabled=args.prefix_cache == "on",
            max_lanes=args.max_lanes,
        )
    )
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
