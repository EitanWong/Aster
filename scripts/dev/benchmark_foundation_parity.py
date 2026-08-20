#!/usr/bin/env python3
"""Compare Aster and native MLX-LM on fixed public B1/B4 workloads."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.engine import InferenceEngine  # noqa: E402
from aster.telemetry.metrics import MetricsRegistry  # noqa: E402
from scripts.dev import public_arrival_load as arrival  # noqa: E402
from scripts.dev import public_benchmark as public  # noqa: E402
from scripts.dev import public_engine_matrix as public_matrix  # noqa: E402

ITERATION = "ITER-20260820-090-qwen35-9b-foundation-parity"
CELLS = ("b1-short", "b1-long", "b4-short", "b4-mixed")
ENGINES = ("aster", "mlx-lm")
MAX_OUTPUT_TOKENS = 8
WARMUP_OUTPUT_TOKENS = 2
DEFAULT_REPETITIONS = 4
DIRECT_PREFILL_STEP_SIZE = 2048
DEFAULT_COOLDOWN_SECONDS = 2.0
MATERIAL_GAP = 0.03
DEFAULT_WORKLOAD = PROJECT_ROOT / "run/loop-engineering/public-benchmarks/cross-engine-core.json"
DEFAULT_RUN_ROOT = PROJECT_ROOT / f"run/loop-engineering/{ITERATION}"


class BenchmarkError(RuntimeError):
    """Raised when the frozen foundation-parity contract is violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise BenchmarkError("percentile requires at least one value")
    rank = max(1, min(len(values), math.ceil(percentile * len(values))))
    return sorted(values)[rank - 1]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _records_for_plan(
    workload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = arrival._records(workload)
    interactive = [
        record for record in records if arrival._scenario_family(record) == "interactive"
    ]
    qmsum = [record for record in records if arrival._source_dataset(record) == "qmsum"]
    if len(interactive) < 4:
        raise BenchmarkError("foundation parity requires four interactive records")
    if not qmsum:
        raise BenchmarkError("foundation parity requires one QMSUM record")
    return interactive, qmsum


def build_foundation_plan(workload: dict[str, Any], *, cell: str) -> arrival.ArrivalPlan:
    """Build one fixed public cohort without resolving or copying prompt text."""

    if cell not in CELLS:
        raise BenchmarkError(f"unknown foundation-parity cell: {cell}")
    interactive, qmsum = _records_for_plan(workload)
    selected: list[tuple[str, dict[str, Any]]]
    if cell == "b1-short":
        selected = [("short-0", interactive[0])]
    elif cell == "b1-long":
        selected = [("long-0", qmsum[0])]
    elif cell == "b4-short":
        selected = [(f"short-{index}", record) for index, record in enumerate(interactive[:4])]
    else:
        selected = [("long-0", qmsum[0])]
        selected.extend((f"short-{index}", record) for index, record in enumerate(interactive[:3]))
    entries = tuple(
        arrival._entry(
            key=key,
            record=record,
            cap=MAX_OUTPUT_TOKENS,
            release="at-start",
        )
        for key, record in selected
    )
    return arrival.ArrivalPlan(
        scenario=f"foundation-parity:{cell}",
        concurrency=len(entries),
        entries=entries,
    )


def engine_order_for_pair(cell: str, repetition: int) -> tuple[str, str]:
    if cell not in CELLS:
        raise BenchmarkError(f"unknown foundation-parity cell: {cell}")
    if repetition < 1:
        raise BenchmarkError("repetition must be positive")
    aster_first = (repetition + CELLS.index(cell)) % 2 == 0
    return ("aster", "mlx-lm") if aster_first else ("mlx-lm", "aster")


def _plan_payload(plan: arrival.ArrivalPlan) -> dict[str, Any]:
    return {
        "scenario": plan.scenario,
        "concurrency": plan.concurrency,
        "entries": [
            {
                "key": entry.key,
                "workload_id": entry.workload_id,
                "max_tokens": entry.max_tokens,
                "release": entry.release,
                "depends_on": entry.depends_on,
                "delay_seconds": entry.delay_seconds,
                "prompt_suffix": entry.prompt_suffix,
            }
            for entry in plan.entries
        ],
    }


def _source_hashes(engine: str) -> dict[str, str]:
    common_paths = {
        "benchmark_foundation_parity.py": Path(__file__).resolve(),
        "public_arrival_load.py": Path(arrival.__file__).resolve(),
        "public_benchmark.py": Path(public.__file__).resolve(),
    }
    if engine == "aster":
        import aster.inference.engine as engine_module
        import aster.inference.model_runner as runner_module
        import aster.inference.runtime_kernel as kernel_module

        engine_paths = {
            "aster/inference/engine.py": Path(engine_module.__file__).resolve(),
            "aster/inference/model_runner.py": Path(runner_module.__file__).resolve(),
            "aster/inference/runtime_kernel.py": Path(kernel_module.__file__).resolve(),
        }
    elif engine == "mlx-lm":
        from mlx_lm.generate import BatchGenerator

        source_path = inspect.getsourcefile(BatchGenerator)
        if source_path is None:
            raise BenchmarkError("cannot locate MLX-LM BatchGenerator source")
        engine_paths = {"mlx_lm/generate.py": Path(source_path).resolve()}
    else:
        raise BenchmarkError(f"unknown engine: {engine}")
    common = {name: _sha256_file(path) for name, path in common_paths.items()}
    specific = {name: _sha256_file(path) for name, path in engine_paths.items()}
    return {
        "common_source_sha256": _canonical_sha256(common),
        "engine_source_sha256": _canonical_sha256(specific),
        "files": {**common, **specific},
    }


def _input_entry(
    *,
    key: str,
    workload_id: str,
    max_tokens: int,
    token_ids: list[int],
) -> dict[str, Any]:
    if len(token_ids) < 2:
        raise BenchmarkError(f"public input {workload_id} encoded to fewer than two tokens")
    return {
        "key": key,
        "workload_id": workload_id,
        "max_tokens": max_tokens,
        "prompt_tokens": len(token_ids),
        "input_token_ids_sha256": _token_ids_sha256(token_ids),
        "token_ids": token_ids,
    }


def _encode_with_tokenizer(tokenizer: Any, prompt: str) -> list[int]:
    bos_token = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos_token is None or not prompt.startswith(bos_token or "")
    return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def _resolve_records(
    workload: dict[str, Any],
    plan: arrival.ArrivalPlan,
) -> dict[str, dict[str, Any]]:
    records = {str(record["workload_id"]): record for record in arrival._records(workload)}
    missing = [entry.workload_id for entry in plan.entries if entry.workload_id not in records]
    if missing:
        raise BenchmarkError(f"plan references missing public records: {missing}")
    return records


async def _aster_input_manifest(
    engine: InferenceEngine,
    plan: arrival.ArrivalPlan,
    workload: dict[str, Any],
    resolver: public.PublicWorkloadResolver,
) -> list[dict[str, Any]]:
    records = _resolve_records(workload, plan)
    manifest: list[dict[str, Any]] = []
    for entry in plan.entries:
        prompt = resolver.resolve(records[entry.workload_id])
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=entry.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            enable_thinking=False,
        )
        prepared = await engine._runner_call(engine.runtime_kernel.encode_request, request)
        manifest.append(
            _input_entry(
                key=entry.key,
                workload_id=entry.workload_id,
                max_tokens=entry.max_tokens,
                token_ids=list(prepared.prompt_tokens),
            )
        )
    return manifest


def _direct_input_manifest(
    tokenizer: Any,
    plan: arrival.ArrivalPlan,
    workload: dict[str, Any],
    resolver: public.PublicWorkloadResolver,
) -> list[dict[str, Any]]:
    records = _resolve_records(workload, plan)
    return [
        _input_entry(
            key=entry.key,
            workload_id=entry.workload_id,
            max_tokens=entry.max_tokens,
            token_ids=_encode_with_tokenizer(
                tokenizer,
                resolver.resolve(records[entry.workload_id]),
            ),
        )
        for entry in plan.entries
    ]


def _metric_summary(
    requests: list[dict[str, Any]],
    *,
    prefill_model_tokens: int,
    prefill_model_seconds: float,
    decode_driver_tokens: int,
    decode_driver_seconds: float,
    peak_mlx_memory_gb: float,
    rss_before_bytes: int,
    rss_after_bytes: int,
    peak_rss_bytes: int,
    swap_before_bytes: int,
    swap_after_bytes: int,
    swap_delta_bytes: int,
) -> dict[str, Any]:
    ttft = [float(request["ttft_seconds"]) for request in requests]
    end_to_end = [float(request["end_to_end_seconds"]) for request in requests]
    submitted = [float(request.get("submitted_after_seconds", 0.0)) for request in requests]
    completed = [
        submitted_at + latency for submitted_at, latency in zip(submitted, end_to_end, strict=True)
    ]
    service_window = max(completed) - min(submitted)
    if service_window <= 0:
        raise BenchmarkError("service window must be positive")
    completion_tokens = sum(int(request["completion_tokens"]) for request in requests)
    return {
        "prompt_tokens": sum(int(request["prompt_tokens"]) for request in requests),
        "completion_tokens": completion_tokens,
        "prefill_model_tokens": prefill_model_tokens,
        "prefill_model_seconds": prefill_model_seconds,
        "prefill_model_tps": (
            prefill_model_tokens / prefill_model_seconds if prefill_model_seconds > 0 else 0.0
        ),
        "decode_driver_tokens": decode_driver_tokens,
        "decode_driver_seconds": decode_driver_seconds,
        "decode_driver_tps": (
            decode_driver_tokens / decode_driver_seconds if decode_driver_seconds > 0 else 0.0
        ),
        "service_window_seconds": service_window,
        "aggregate_generation_tps": completion_tokens / service_window,
        "ttft_p50_seconds": _nearest_rank(ttft, 0.50),
        "ttft_p95_seconds": _nearest_rank(ttft, 0.95),
        "end_to_end_p50_seconds": _nearest_rank(end_to_end, 0.50),
        "end_to_end_p95_seconds": _nearest_rank(end_to_end, 0.95),
        "peak_mlx_memory_gb": peak_mlx_memory_gb,
        "rss_before_bytes": rss_before_bytes,
        "rss_after_bytes": rss_after_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "swap_before_bytes": swap_before_bytes,
        "swap_after_bytes": swap_after_bytes,
        "swap_delta_bytes": swap_delta_bytes,
    }


def _timing_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | int]:
    fields = (
        "prefill_model_seconds",
        "prefill_model_tokens",
        "decode_runner_seconds",
        "decode_runner_batches",
        "decode_runner_items",
        "decode_runner_tokens",
    )
    return {field: after[field] - before[field] for field in fields}


def _decode_stage_observer_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Keep only observer events from the timed window, excluding warmup work."""
    before_events = list(before.get("events", []))
    after_events = list(after.get("events", []))
    before_seconds = dict(before.get("seconds", {}))
    after_seconds = dict(after.get("seconds", {}))
    seconds = {
        stage: float(after_seconds.get(stage, 0.0)) - float(before_seconds.get(stage, 0.0))
        for stage in (
            "cache_prepare",
            "model_enqueue",
            "sampling_enqueue",
            "evaluation_window",
            "result_delivery",
            "eager_completion",
            "observed_total",
        )
    }
    return {
        "configured_max_events": int(after.get("configured_max_events", 0)),
        "sample_interval": int(after.get("sample_interval", 1)),
        "batch_steps": int(after.get("batch_steps", 0)) - int(before.get("batch_steps", 0)),
        "single_steps": int(after.get("single_steps", 0)) - int(before.get("single_steps", 0)),
        "sampled_steps": int(after.get("sampled_steps", 0))
        - int(before.get("sampled_steps", 0)),
        "dropped_events": int(after.get("dropped_events", 0))
        - int(before.get("dropped_events", 0)),
        "seconds": seconds,
        "events": after_events[len(before_events) :],
    }


async def _warm_aster(
    engine: InferenceEngine,
    workload: dict[str, Any],
    resolver: public.PublicWorkloadResolver,
) -> None:
    interactive, _ = _records_for_plan(workload)
    await engine.submit(
        InferenceRequest(
            prompt=resolver.resolve(interactive[0]),
            max_tokens=WARMUP_OUTPUT_TOKENS,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            trace_id="foundation-parity:warmup",
            timeout_seconds=120.0,
            enable_thinking=False,
        )
    )


def _execution_contract() -> dict[str, Any]:
    return {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "warmup_requests": 1,
        "process_isolation": "fresh-process-per-engine-cell-repetition",
        "prefix_cache": "off",
        "input_mode": "pinned-public-source-resolved-token-ids",
        "decode_tensorized_logprobs_enabled": False,
        "decode_stage_observer_max_events": 0,
        "decode_stage_observer_sample_interval": 1,
        "prefill_model_boundary": "model-prefill-call-time",
        "decode_driver_boundary": "batch-decode-driver-including-cache-sampler-and-result-work",
    }


def _base_source(
    *,
    engine: str,
    workload_path: Path,
    lock_path: Path,
    model_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    source_hashes = _source_hashes(engine)
    return {
        "workload_sha256": _sha256_file(workload_path),
        "source_lock_sha256": _sha256_file(lock_path),
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        **source_hashes,
    }


def _cell_envelope(
    *,
    engine: str,
    cell: str,
    repetition: int,
    pair_order: tuple[str, str],
    source: dict[str, Any],
    plan: arrival.ArrivalPlan,
    input_manifest: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    metrics: dict[str, Any],
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    expected_count = 1 if cell.startswith("b1-") else 4
    request_contract = (
        len(requests) == expected_count
        and all(int(request["completion_tokens"]) == MAX_OUTPUT_TOKENS for request in requests)
        and all(request["finish_reason"] == "length" for request in requests)
        and all(float(request["ttft_seconds"]) > 0 for request in requests)
        and all(float(request["end_to_end_seconds"]) > 0 for request in requests)
    )
    metric_contract = (
        all(
            float(metrics[field]) > 0
            for field in (
                "prefill_model_tps",
                "decode_driver_tps",
                "aggregate_generation_tps",
                "ttft_p50_seconds",
                "ttft_p95_seconds",
                "end_to_end_p50_seconds",
                "end_to_end_p95_seconds",
                "peak_mlx_memory_gb",
                "peak_rss_bytes",
            )
        )
        and int(metrics["swap_delta_bytes"]) >= 0
    )
    lifecycle_contract = (
        bool(lifecycle.get("terminal_clean"))
        and int(lifecycle.get("completed_requests", -1)) == expected_count
    )
    return {
        "schema_version": 1,
        "kind": "foundation-parity-cell-result",
        "cell": cell,
        "engine": engine,
        "repetition": repetition,
        "pair_order": list(pair_order),
        "engine_position": pair_order.index(engine),
        "source": source,
        "plan_sha256": _canonical_sha256(_plan_payload(plan)),
        "input_manifest_sha256": _canonical_sha256(
            [
                {key: value for key, value in item.items() if key != "token_ids"}
                for item in input_manifest
            ]
        ),
        "execution": _execution_contract(),
        "requests": requests,
        "metrics": metrics,
        "lifecycle": lifecycle,
        "contract": {
            "request": request_contract,
            "metrics": metric_contract,
            "lifecycle": lifecycle_contract,
            "passed": request_contract and metric_contract and lifecycle_contract,
        },
    }


async def _run_aster_cell(
    *,
    cell: str,
    repetition: int,
    pair_order: tuple[str, str],
    config_path: Path,
    workload_path: Path,
    lock_path: Path,
    data_root: Path,
    model_sha256: str,
    tokenizer_sha256: str,
    timeout_seconds: float,
    memory_sample_interval: float,
) -> dict[str, Any]:
    import mlx.core as mx

    workload = arrival._load_workload(workload_path)
    if workload.get("lock_sha256") != _sha256_file(lock_path):
        raise BenchmarkError("workload source lock differs from the active source lock")
    lock = public.load_lock(lock_path)
    resolver = public.PublicWorkloadResolver(lock, data_root)
    plan = build_foundation_plan(workload, cell=cell)
    base_settings = load_settings(str(config_path))
    settings = arrival._apply_baseline_settings(
        base_settings,
        concurrency=plan.concurrency,
        prefix_cache_enabled=False,
        decode_active_prefill_token_budget=(
            base_settings.engine.decode_active_prefill_token_budget
        ),
        snapshot_budget_bytes=None,
        snapshot_max_entries=None,
        max_active_requests=plan.concurrency,
    )
    engine = InferenceEngine(settings, MetricsRegistry(settings.telemetry.metrics_namespace))
    lifecycle_summary = arrival._new_engine_lifecycle_sample_summary(
        interval_seconds=memory_sample_interval
    )
    sampler = public_matrix.MemorySampler(memory_sample_interval)
    lifecycle_stop = asyncio.Event()
    lifecycle_task: asyncio.Task[None] | None = None
    try:
        await engine.start()
        await engine.warmup()
        await _warm_aster(engine, workload, resolver)
        input_manifest = await _aster_input_manifest(engine, plan, workload, resolver)
        await engine._runner_call(engine.runtime_kernel.reset_decode_stage_observer_window)
        timing_before = dict(engine.status()["engine_timing"])
        observer_before = dict(
            engine.status()["decode_batch_diagnostics"].get("decode_stage_observer", {})
        )
        mx.reset_peak_memory()
        sampler.start()
        lifecycle_task = asyncio.create_task(
            arrival._sample_engine_lifecycle(engine, lifecycle_stop, lifecycle_summary)
        )
        result = await arrival.execute_arrival_plan(
            engine,
            plan,
            workload,
            resolve_prompt=resolver.resolve,
            timeout_seconds=timeout_seconds,
        )
        lifecycle_stop.set()
        await lifecycle_task
        lifecycle_task = None
        memory = sampler.finish()
        arrival._attach_timelines(result)
        status = result["engine_status"]
        timing = _timing_delta(timing_before, status["engine_timing"])
        input_by_key = {item["key"]: item for item in input_manifest}
        requests: list[dict[str, Any]] = []
        for event in result["events"]:
            response = event.get("response")
            timeline = event.get("timeline")
            if event.get("error") is not None or not isinstance(response, dict):
                raise BenchmarkError(f"Aster request failed: {event!r}")
            if not isinstance(timeline, dict):
                raise BenchmarkError("Aster request has no terminal timeline")
            input_entry = input_by_key[str(event["key"])]
            requests.append(
                {
                    "key": str(event["key"]),
                    "workload_id": str(event["workload_id"]),
                    "prompt_tokens": int(input_entry["prompt_tokens"]),
                    "completion_tokens": int(response["completion_tokens"]),
                    "finish_reason": str(response["finish_reason"]),
                    "output_token_ids_sha256": str(timeline["output_token_ids_sha256"]),
                    "text_sha256": str(response["text_sha256"]),
                    "submitted_after_seconds": float(event["submitted_after_seconds"]),
                    "ttft_seconds": float(timeline["ttft_s"]),
                    "end_to_end_seconds": float(timeline["total_latency_s"]),
                }
            )
        metrics = _metric_summary(
            requests,
            prefill_model_tokens=int(timing["prefill_model_tokens"]),
            prefill_model_seconds=float(timing["prefill_model_seconds"]),
            decode_driver_tokens=int(timing["decode_runner_tokens"]),
            decode_driver_seconds=float(timing["decode_runner_seconds"]),
            peak_mlx_memory_gb=float(mx.get_peak_memory()) / 1e9,
            rss_before_bytes=int(memory["rss_before_bytes"]),
            rss_after_bytes=int(memory["rss_after_bytes"]),
            peak_rss_bytes=int(memory["peak_rss_bytes"]),
            swap_before_bytes=int(memory["swap_before_bytes"]),
            swap_after_bytes=int(memory["swap_after_bytes"]),
            swap_delta_bytes=int(memory["swap_delta_bytes"]),
        )
        final_prefix = status["prefix_cache_stats"]
        lifecycle = {
            "terminal_clean": (
                not status["requests"]
                and int(status["pending_requests"]) == 0
                and int(status["prefill_requests"]) == 0
                and int(status["decode_requests"]) == 0
                and int(status["failed_requests"]) == 0
                and int(status["cancelled_requests"]) == 0
                and int(final_prefix["pinned_entries"]) == 0
            ),
            "completed_requests": int(status["completed_requests"]) - 1,
            "failed_requests": int(status["failed_requests"]),
            "cancelled_requests": int(status["cancelled_requests"]),
            "maxima": lifecycle_summary["maxima"],
            "sample_count": lifecycle_summary["sample_count"],
            "decode_runner_batches": int(timing["decode_runner_batches"]),
            "decode_runner_items": int(timing["decode_runner_items"]),
            "decode_batch_diagnostics": dict(status.get("decode_batch_diagnostics", {})),
            "decode_stage_observer": _decode_stage_observer_delta(
                observer_before,
                dict(status["decode_batch_diagnostics"].get("decode_stage_observer", {})),
            ),
        }
        envelope = _cell_envelope(
            engine="aster",
            cell=cell,
            repetition=repetition,
            pair_order=pair_order,
            source=_base_source(
                engine="aster",
                workload_path=workload_path,
                lock_path=lock_path,
                model_sha256=model_sha256,
                tokenizer_sha256=tokenizer_sha256,
            ),
            plan=plan,
            input_manifest=input_manifest,
            requests=requests,
            metrics=metrics,
            lifecycle=lifecycle,
        )
        envelope["execution"]["decode_tensorized_logprobs_enabled"] = bool(
            settings.engine.decode_tensorized_logprobs_enabled
        )
        envelope["execution"]["decode_stage_observer_max_events"] = int(
            settings.engine.decode_stage_observer_max_events
        )
        envelope["execution"]["decode_stage_observer_sample_interval"] = int(
            settings.engine.decode_stage_observer_sample_interval
        )
        return envelope
    finally:
        lifecycle_stop.set()
        if lifecycle_task is not None:
            await lifecycle_task
        await engine.aclose()


def _new_batch_generator(model: Any, tokenizer: Any, *, batch_size: int) -> Any:
    from mlx_lm.generate import BatchGenerator

    stop_tokens = [[int(token)] for token in tokenizer.eos_token_ids]
    return BatchGenerator(
        model,
        stop_tokens=stop_tokens,
        completion_batch_size=batch_size,
        prefill_batch_size=batch_size,
        prefill_step_size=DIRECT_PREFILL_STEP_SIZE,
    )


def _drain_direct_warmup(model: Any, tokenizer: Any, token_ids: list[int]) -> None:
    generator = _new_batch_generator(model, tokenizer, batch_size=1)
    try:
        generator.insert([token_ids], max_tokens=[WARMUP_OUTPUT_TOKENS])
        while generator.next_generated():
            pass
    finally:
        generator.close()


def _direct_lifecycle(generator: Any) -> dict[str, int]:
    pending = len(generator._unprocessed_sequences)
    prefill = len(generator._prompt_batch)
    decode = len(generator._generation_batch)
    return {
        "pending_requests": pending,
        "prefill_requests": prefill,
        "decode_requests": decode,
        "active_requests": pending + prefill + decode,
    }


def _run_direct_cell(
    *,
    cell: str,
    repetition: int,
    pair_order: tuple[str, str],
    config_path: Path,
    workload_path: Path,
    lock_path: Path,
    data_root: Path,
    model_sha256: str,
    tokenizer_sha256: str,
    timeout_seconds: float,
    memory_sample_interval: float,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchStats, GenerationBatch, PromptProcessingBatch

    workload = arrival._load_workload(workload_path)
    if workload.get("lock_sha256") != _sha256_file(lock_path):
        raise BenchmarkError("workload source lock differs from the active source lock")
    lock = public.load_lock(lock_path)
    resolver = public.PublicWorkloadResolver(lock, data_root)
    plan = build_foundation_plan(workload, cell=cell)
    settings = load_settings(str(config_path))
    model, tokenizer = load(str(settings.model.path))
    input_manifest = _direct_input_manifest(tokenizer, plan, workload, resolver)
    interactive, _ = _records_for_plan(workload)
    warmup_tokens = _encode_with_tokenizer(tokenizer, resolver.resolve(interactive[0]))
    _drain_direct_warmup(model, tokenizer, warmup_tokens)
    mx.reset_peak_memory()

    generator = _new_batch_generator(model, tokenizer, batch_size=plan.concurrency)
    stats = BatchStats()
    sampler = public_matrix.MemorySampler(memory_sample_interval)
    maxima = {
        "active_requests": 0,
        "pending_requests": 0,
        "prefill_requests": 0,
        "decode_requests": 0,
    }
    samples = 0

    def sample_lifecycle() -> None:
        nonlocal samples
        current = _direct_lifecycle(generator)
        for field, value in current.items():
            maxima[field] = max(maxima[field], value)
        samples += 1

    started = time.perf_counter()
    sampler.start()
    decode_driver_seconds = 0.0
    original_generate = PromptProcessingBatch.generate
    original_next = GenerationBatch.next

    def observed_generate(batch: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal decode_driver_seconds
        observed_started = time.perf_counter()
        try:
            return original_generate(batch, *args, **kwargs)
        finally:
            decode_driver_seconds += time.perf_counter() - observed_started

    def observed_next(batch: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal decode_driver_seconds
        observed_started = time.perf_counter()
        try:
            return original_next(batch, *args, **kwargs)
        finally:
            decode_driver_seconds += time.perf_counter() - observed_started

    PromptProcessingBatch.generate = observed_generate
    GenerationBatch.next = observed_next
    try:
        uids = generator.insert(
            [list(item["token_ids"]) for item in input_manifest],
            max_tokens=[MAX_OUTPUT_TOKENS] * plan.concurrency,
        )
        entry_by_uid = dict(zip(uids, plan.entries, strict=True))
        input_by_uid = dict(zip(uids, input_manifest, strict=True))
        output_tokens = {uid: [] for uid in uids}
        first_token_at: dict[int, float] = {}
        completed_at: dict[int, float] = {}
        finish_reasons: dict[int, str] = {}
        sample_lifecycle()
        with generator.stats(stats):
            while len(completed_at) < len(uids):
                if time.perf_counter() - started > timeout_seconds:
                    raise BenchmarkError(f"direct MLX-LM {cell} exceeded timeout")
                _, generation_responses = generator.next()
                now = time.perf_counter() - started
                for response in generation_responses:
                    uid = int(response.uid)
                    first_token_at.setdefault(uid, now)
                    output_tokens[uid].append(int(response.token))
                    if response.finish_reason is not None:
                        completed_at[uid] = now
                        finish_reasons[uid] = str(response.finish_reason)
                sample_lifecycle()
        memory = sampler.finish()
        requests = []
        for uid in uids:
            entry = entry_by_uid[uid]
            tokens = output_tokens[uid]
            input_entry = input_by_uid[uid]
            requests.append(
                {
                    "key": entry.key,
                    "workload_id": entry.workload_id,
                    "prompt_tokens": int(input_entry["prompt_tokens"]),
                    "completion_tokens": len(tokens),
                    "finish_reason": finish_reasons[uid],
                    "output_token_ids_sha256": _token_ids_sha256(tokens),
                    "text_sha256": _sha256_bytes(tokenizer.decode(tokens).encode("utf-8")),
                    "submitted_after_seconds": 0.0,
                    "ttft_seconds": first_token_at[uid],
                    "end_to_end_seconds": completed_at[uid],
                }
            )
        metrics = _metric_summary(
            requests,
            prefill_model_tokens=int(stats.prompt_tokens),
            prefill_model_seconds=float(stats.prompt_time),
            decode_driver_tokens=int(stats.generation_tokens),
            decode_driver_seconds=decode_driver_seconds,
            peak_mlx_memory_gb=float(mx.get_peak_memory()) / 1e9,
            rss_before_bytes=int(memory["rss_before_bytes"]),
            rss_after_bytes=int(memory["rss_after_bytes"]),
            peak_rss_bytes=int(memory["peak_rss_bytes"]),
            swap_before_bytes=int(memory["swap_before_bytes"]),
            swap_after_bytes=int(memory["swap_after_bytes"]),
            swap_delta_bytes=int(memory["swap_delta_bytes"]),
        )
        final = _direct_lifecycle(generator)
        lifecycle = {
            "terminal_clean": all(value == 0 for value in final.values()),
            "completed_requests": len(completed_at),
            "failed_requests": 0,
            "cancelled_requests": 0,
            "maxima": maxima,
            "sample_count": samples,
            "batch_generator_decode_steps": int(generator._steps_counter),
            "official_generation_seconds": float(stats.generation_time),
        }
        return _cell_envelope(
            engine="mlx-lm",
            cell=cell,
            repetition=repetition,
            pair_order=pair_order,
            source=_base_source(
                engine="mlx-lm",
                workload_path=workload_path,
                lock_path=lock_path,
                model_sha256=model_sha256,
                tokenizer_sha256=tokenizer_sha256,
            ),
            plan=plan,
            input_manifest=input_manifest,
            requests=requests,
            metrics=metrics,
            lifecycle=lifecycle,
        )
    finally:
        PromptProcessingBatch.generate = original_generate
        GenerationBatch.next = original_next
        generator.close()


def _request_terminal_fingerprint(row: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            request["key"],
            request["workload_id"],
            request["prompt_tokens"],
            request["completion_tokens"],
            request["finish_reason"],
        ]
        for request in row["requests"]
    ]


def _request_output_fingerprint(row: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            request["key"],
            request["output_token_ids_sha256"],
            request["text_sha256"],
            request["finish_reason"],
            request["completion_tokens"],
        ]
        for request in row["requests"]
    ]


def _median(values: list[float]) -> float:
    if not values:
        raise BenchmarkError("median requires at least one value")
    return float(statistics.median(values))


def _paired_deficits(aster: dict[str, Any], mlx_lm: dict[str, Any]) -> dict[str, float]:
    aster_metrics = aster["metrics"]
    mlx_metrics = mlx_lm["metrics"]
    for field in (
        "prefill_model_tps",
        "decode_driver_tps",
        "aggregate_generation_tps",
        "ttft_p95_seconds",
        "end_to_end_p95_seconds",
        "peak_mlx_memory_gb",
        "peak_rss_bytes",
    ):
        if float(aster_metrics[field]) <= 0 or float(mlx_metrics[field]) <= 0:
            raise BenchmarkError(f"matrix metric must be positive: {field}")
    return {
        "prefill_model_tps": (
            float(mlx_metrics["prefill_model_tps"]) / float(aster_metrics["prefill_model_tps"])
            - 1.0
        ),
        "decode_driver_tps": (
            float(mlx_metrics["decode_driver_tps"]) / float(aster_metrics["decode_driver_tps"])
            - 1.0
        ),
        "aggregate_generation_tps": (
            float(mlx_metrics["aggregate_generation_tps"])
            / float(aster_metrics["aggregate_generation_tps"])
            - 1.0
        ),
        "ttft_p95_seconds": (
            float(aster_metrics["ttft_p95_seconds"]) / float(mlx_metrics["ttft_p95_seconds"]) - 1.0
        ),
        "end_to_end_p95_seconds": (
            float(aster_metrics["end_to_end_p95_seconds"])
            / float(mlx_metrics["end_to_end_p95_seconds"])
            - 1.0
        ),
        "peak_mlx_memory_gb": (
            float(aster_metrics["peak_mlx_memory_gb"]) / float(mlx_metrics["peak_mlx_memory_gb"])
            - 1.0
        ),
        "peak_rss_bytes": (
            float(aster_metrics["peak_rss_bytes"]) / float(mlx_metrics["peak_rss_bytes"]) - 1.0
        ),
    }


def summarize_matrix(rows: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    if repetitions < 2 or repetitions % 2:
        raise BenchmarkError("matrix repetitions must be a positive even count of at least two")
    expected = {
        (cell, engine, repetition)
        for repetition in range(1, repetitions + 1)
        for cell in CELLS
        for engine in ENGINES
    }
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["cell"]), str(row["engine"]), int(row["repetition"]))
        if key in indexed:
            raise BenchmarkError(f"matrix repeats row {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise BenchmarkError(f"foundation parity requires a complete {len(expected)}-row matrix")
    if not all(bool(row.get("contract", {}).get("passed")) for row in rows):
        raise BenchmarkError("matrix contains a failed request/metric/lifecycle contract")

    source_fields = (
        "workload_sha256",
        "source_lock_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "common_source_sha256",
    )
    for field in source_fields:
        if len({str(row["source"].get(field)) for row in rows}) != 1:
            raise BenchmarkError(f"matrix source differs: {field}")
    for engine in ENGINES:
        if (
            len(
                {
                    str(row["source"].get("engine_source_sha256"))
                    for row in rows
                    if row["engine"] == engine
                }
            )
            != 1
        ):
            raise BenchmarkError(f"matrix engine source differs: {engine}")

    pair_deficits: list[dict[str, Any]] = []
    cross_engine_output_identity = True
    cross_engine_output_divergences: dict[str, list[str]] = {}
    for cell in CELLS:
        cell_rows = [row for row in rows if row["cell"] == cell]
        if len({str(row["plan_sha256"]) for row in cell_rows}) != 1:
            raise BenchmarkError(f"{cell} plan differs across matrix rows")
        if len({str(row["input_manifest_sha256"]) for row in cell_rows}) != 1:
            raise BenchmarkError(f"{cell} input manifest differs across matrix rows")
        for engine in ENGINES:
            engine_rows = [row for row in cell_rows if row["engine"] == engine]
            outputs = {_canonical_sha256(_request_output_fingerprint(row)) for row in engine_rows}
            if len(outputs) != 1:
                raise BenchmarkError(f"{cell} {engine} output fingerprint is unstable")
        for repetition in range(1, repetitions + 1):
            expected_order = engine_order_for_pair(cell, repetition)
            pair = {engine: indexed[(cell, engine, repetition)] for engine in ENGINES}
            for engine in ENGINES:
                row = pair[engine]
                if row.get("pair_order") != list(expected_order):
                    raise BenchmarkError(f"{cell} repetition {repetition} order differs")
                if int(row.get("engine_position", -1)) != expected_order.index(engine):
                    raise BenchmarkError(f"{cell} repetition {repetition} order position differs")
            if _request_terminal_fingerprint(pair["aster"]) != _request_terminal_fingerprint(
                pair["mlx-lm"]
            ):
                raise BenchmarkError(f"{cell} repetition {repetition} terminal identity differs")
            outputs_match = _request_output_fingerprint(
                pair["aster"]
            ) == _request_output_fingerprint(pair["mlx-lm"])
            cross_engine_output_identity = cross_engine_output_identity and outputs_match
            if not outputs_match:
                aster_outputs = {
                    str(request["key"]): _canonical_sha256(
                        [
                            request["output_token_ids_sha256"],
                            request["text_sha256"],
                            request["finish_reason"],
                            request["completion_tokens"],
                        ]
                    )
                    for request in pair["aster"]["requests"]
                }
                mlx_lm_outputs = {
                    str(request["key"]): _canonical_sha256(
                        [
                            request["output_token_ids_sha256"],
                            request["text_sha256"],
                            request["finish_reason"],
                            request["completion_tokens"],
                        ]
                    )
                    for request in pair["mlx-lm"]["requests"]
                }
                divergent_keys = sorted(
                    key
                    for key in set(aster_outputs) | set(mlx_lm_outputs)
                    if aster_outputs.get(key) != mlx_lm_outputs.get(key)
                )
                existing = cross_engine_output_divergences.setdefault(cell, [])
                existing.extend(key for key in divergent_keys if key not in existing)
            pair_deficits.append(
                {
                    "cell": cell,
                    "repetition": repetition,
                    "first_engine": expected_order[0],
                    "output_identity": outputs_match,
                    "deficits": _paired_deficits(pair["aster"], pair["mlx-lm"]),
                }
            )

    metrics = tuple(pair_deficits[0]["deficits"])
    cell_summaries: dict[str, Any] = {}
    qualifying: dict[str, list[str]] = {metric: [] for metric in metrics}
    for cell in CELLS:
        pairs = [pair for pair in pair_deficits if pair["cell"] == cell]
        metric_summary: dict[str, Any] = {}
        for metric in metrics:
            strata = {
                engine: _median(
                    [
                        float(pair["deficits"][metric])
                        for pair in pairs
                        if pair["first_engine"] == engine
                    ]
                )
                for engine in ENGINES
            }
            overall = _median([float(pair["deficits"][metric]) for pair in pairs])
            reproducible = all(value >= MATERIAL_GAP for value in strata.values())
            if reproducible:
                qualifying[metric].append(cell)
            metric_summary[metric] = {
                "median_aster_deficit_ratio": overall,
                "order_strata": strata,
                "reproducible_at_least_3_percent": reproducible,
            }
        cell_summaries[cell] = metric_summary

    candidate_specs = (
        (
            "decode_driver_tps",
            "aster-manual-decode-driver",
            2,
            "select-decode-driver-profile-for-i091",
        ),
        (
            "aggregate_generation_tps",
            "aster-concurrent-scheduler",
            2,
            "select-concurrent-scheduler-profile-for-i091",
        ),
        (
            "prefill_model_tps",
            "aster-prefill-driver",
            2,
            "select-prefill-driver-profile-for-i091",
        ),
        (
            "ttft_p95_seconds",
            "aster-request-lifecycle",
            2,
            "select-ttft-lifecycle-profile-for-i091",
        ),
        (
            "end_to_end_p95_seconds",
            "aster-request-lifecycle",
            2,
            "select-end-to-end-lifecycle-profile-for-i091",
        ),
        (
            "peak_mlx_memory_gb",
            "aster-cache-ownership",
            3,
            "select-memory-ownership-profile-for-i091",
        ),
    )
    priority_gap: dict[str, Any] | None = None
    decision = "baseline-only-no-reproducible-3-percent-gap"
    for metric, owner, minimum_cells, candidate_decision in candidate_specs:
        cells = qualifying[metric]
        if len(cells) < minimum_cells:
            continue
        priority_gap = {
            "metric": metric,
            "owner": owner,
            "qualifying_cells": cells,
            "minimum_material_gap_ratio": MATERIAL_GAP,
            "median_aster_deficit_ratio": _median(
                [
                    float(cell_summaries[cell][metric]["median_aster_deficit_ratio"])
                    for cell in cells
                ]
            ),
        }
        decision = candidate_decision
        break

    return {
        "schema_version": 1,
        "kind": "foundation-parity-matrix-summary",
        "row_count": len(rows),
        "repetitions": repetitions,
        "contracts_passed": True,
        "source_comparable": True,
        "input_comparable": True,
        "order_balanced": True,
        "cross_engine_terminal_identity": True,
        "cross_engine_output_identity": cross_engine_output_identity,
        "cross_engine_output_divergences": cross_engine_output_divergences,
        "cell_summaries": cell_summaries,
        "pair_deficits": pair_deficits,
        "qualifying_cells_by_metric": qualifying,
        "priority_gap": priority_gap,
        "decision": decision,
    }


def _machine_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": int(psutil.virtual_memory().total),
        "mlx": _package_version("mlx"),
        "mlx_lm": _package_version("mlx-lm"),
    }


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _cell_command(
    args: argparse.Namespace,
    *,
    cell: str,
    engine: str,
    repetition: int,
    pair_order: tuple[str, str],
    output: Path,
    fingerprint: dict[str, str],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-cell",
        "--cell",
        cell,
        "--engine",
        engine,
        "--repetition",
        str(repetition),
        "--pair-order",
        ",".join(pair_order),
        "--config",
        str(args.config),
        "--workload",
        str(args.workload),
        "--lock",
        str(args.lock),
        "--data-root",
        str(args.data_root),
        "--model-sha256",
        fingerprint["model_sha256"],
        "--tokenizer-sha256",
        fingerprint["tokenizer_sha256"],
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--memory-sample-interval",
        str(args.memory_sample_interval),
        "--output",
        str(output),
    ]


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions < 2 or args.repetitions % 2:
        raise BenchmarkError("matrix repetitions must be a positive even count of at least two")
    settings = load_settings(str(args.config))
    model_path = Path(settings.model.path).resolve()
    fingerprint_before = public_matrix.model_fingerprint(model_path)
    source_before = _source_hashes("aster")["common_source_sha256"]
    args.run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        offset = (repetition - 1) % len(CELLS)
        cell_order = (*CELLS[offset:], *CELLS[:offset])
        for cell in cell_order:
            pair_order = engine_order_for_pair(cell, repetition)
            schedule.append(
                {
                    "repetition": repetition,
                    "cell": cell,
                    "engine_order": list(pair_order),
                }
            )
            for engine in pair_order:
                raw_path = args.run_root / f"r{repetition}-{cell}-{engine}.json"
                completed = subprocess.run(
                    _cell_command(
                        args,
                        cell=cell,
                        engine=engine,
                        repetition=repetition,
                        pair_order=pair_order,
                        output=raw_path,
                        fingerprint=fingerprint_before,
                    ),
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=args.process_timeout_seconds,
                )
                if completed.returncode != 0:
                    tail = completed.stderr[-4000:]
                    raise BenchmarkError(
                        f"cell process failed for r{repetition}/{cell}/{engine}: {tail}"
                    )
                try:
                    row = json.loads(raw_path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise BenchmarkError(f"cannot read cell output {raw_path}: {error}") from error
                rows.append(row)
                time.sleep(args.cooldown_seconds)
    fingerprint_after = public_matrix.model_fingerprint(model_path)
    if fingerprint_after != fingerprint_before:
        raise BenchmarkError("model/tokenizer fingerprint changed during matrix execution")
    source_after = _source_hashes("aster")["common_source_sha256"]
    if source_after != source_before:
        raise BenchmarkError("benchmark common source changed during matrix execution")
    summary = summarize_matrix(rows, repetitions=args.repetitions)
    return {
        "schema_version": 1,
        "kind": "foundation-parity-evidence",
        "iteration": ITERATION,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_commit": _git_head(),
        "machine": _machine_metadata(),
        "model_path": str(model_path),
        "model_fingerprint": fingerprint_before,
        "source": {
            "workload_path": str(args.workload.resolve()),
            "workload_sha256": _sha256_file(args.workload),
            "source_lock_path": str(args.lock.resolve()),
            "source_lock_sha256": _sha256_file(args.lock),
            "benchmark_source_sha256": source_before,
        },
        "execution": {
            **_execution_contract(),
            "decode_stage_observer_max_events": int(
                settings.engine.decode_stage_observer_max_events
            ),
            "decode_stage_observer_sample_interval": int(
                settings.engine.decode_stage_observer_sample_interval
            ),
            "repetitions": args.repetitions,
            "direct_mlx_lm_prefill_step_size": DIRECT_PREFILL_STEP_SIZE,
            "memory_sample_interval_seconds": args.memory_sample_interval,
            "inter_process_cooldown_seconds": args.cooldown_seconds,
            "timeout_seconds": args.timeout_seconds,
            "schedule": schedule,
        },
        "rows": rows,
        "summary": summary,
    }


def _parse_pair_order(value: str) -> tuple[str, str]:
    parts = tuple(value.split(","))
    if len(parts) != 2 or set(parts) != set(ENGINES):
        raise BenchmarkError("pair order must contain aster and mlx-lm exactly once")
    return parts  # type: ignore[return-value]


def _write_payload(payload: dict[str, Any], path: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    else:
        print(rendered, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--pair-order")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--lock", type=Path, default=public.DEFAULT_LOCK_PATH)
    parser.add_argument("--data-root", type=Path, default=public.DEFAULT_DATA_ROOT)
    parser.add_argument("--model-sha256")
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02)
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.workload = args.workload.resolve()
    args.lock = args.lock.resolve()
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    if args.memory_sample_interval <= 0:
        raise BenchmarkError("memory sample interval must be positive")
    if args.cooldown_seconds < 0:
        raise BenchmarkError("cooldown seconds must be non-negative")
    if args.timeout_seconds <= 0 or args.process_timeout_seconds <= 0:
        raise BenchmarkError("timeouts must be positive")

    if args.run_cell:
        if (
            args.cell is None
            or args.engine is None
            or args.repetition is None
            or args.pair_order is None
            or args.model_sha256 is None
            or args.tokenizer_sha256 is None
        ):
            raise BenchmarkError("cell mode requires cell, engine, repetition, order, and hashes")
        pair_order = _parse_pair_order(args.pair_order)
        kwargs = {
            "cell": args.cell,
            "repetition": args.repetition,
            "pair_order": pair_order,
            "config_path": args.config,
            "workload_path": args.workload,
            "lock_path": args.lock,
            "data_root": args.data_root,
            "model_sha256": args.model_sha256,
            "tokenizer_sha256": args.tokenizer_sha256,
            "timeout_seconds": args.timeout_seconds,
            "memory_sample_interval": args.memory_sample_interval,
        }
        payload = (
            asyncio.run(_run_aster_cell(**kwargs))
            if args.engine == "aster"
            else _run_direct_cell(**kwargs)
        )
    else:
        payload = run_matrix(args)
    _write_payload(payload, args.output)


if __name__ == "__main__":
    main()
