#!/usr/bin/env python3
"""Build deterministic public-source arrival/load plans for Aster baselines."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import RuntimeSettings, load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.engine import InferenceEngine  # noqa: E402
from aster.telemetry.metrics import MetricsRegistry  # noqa: E402
from scripts.dev import public_benchmark as public  # noqa: E402

Release = Literal["at-start", "after-prefill", "after-completion", "after-cancellation"]
SCENARIOS = (
    "idle-lifecycle",
    "simultaneous",
    "staggered-long-prefill",
    "shared-prefix",
    "sustained-exact-prefix",
    "strict-prefix-append",
    "distinct-prefix",
    "capacity-replay",
    "capacity-replay-depth",
    "capacity-replay-six",
    "cancel-during-prefill",
)
LIFECYCLE_SCENARIOS = frozenset(("sustained-exact-prefix", "strict-prefix-append"))
STRICT_PREFIX_APPEND_TEXT = "\n\nProvide one additional concise supporting detail."


class ArrivalLoadError(ValueError):
    """Raised when a public arrival/load plan cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class ArrivalEntry:
    key: str
    workload_id: str
    max_tokens: int
    release: Release
    depends_on: str | None = None
    delay_seconds: float = 0.0
    prompt_suffix: str | None = None


@dataclass(frozen=True, slots=True)
class ArrivalPlan:
    scenario: str
    concurrency: int
    entries: tuple[ArrivalEntry, ...]
    cancel_target_key: str | None = None


def _records(workload: dict[str, Any]) -> list[dict[str, Any]]:
    if workload.get("kind") != "public-cross-engine-workload":
        raise ArrivalLoadError("arrival plan requires a public-cross-engine-workload")
    records = workload.get("records")
    if not isinstance(records, list):
        raise ArrivalLoadError("public workload records must be a list")
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ArrivalLoadError("public workload has an invalid record")
        workload_id = record.get("workload_id")
        if not isinstance(workload_id, str) or not workload_id:
            raise ArrivalLoadError("public workload record has no workload_id")
        validated.append(record)
    return validated


def _scenario_family(record: dict[str, Any]) -> str | None:
    scenario = record.get("scenario")
    return scenario.get("family") if isinstance(scenario, dict) else None


def _source_dataset(record: dict[str, Any]) -> str | None:
    source = record.get("source")
    return source.get("dataset") if isinstance(source, dict) else None


def _effective_max_tokens(record: dict[str, Any], cap: int) -> int:
    declared = record.get("max_tokens")
    if not isinstance(declared, int) or declared < 1:
        raise ArrivalLoadError(f"public record {record['workload_id']} has invalid max_tokens")
    return min(declared, cap)


def _entry(
    *,
    key: str,
    record: dict[str, Any],
    cap: int,
    release: Release,
    depends_on: str | None = None,
    delay_seconds: float = 0.0,
    prompt_suffix: str | None = None,
) -> ArrivalEntry:
    return ArrivalEntry(
        key=key,
        workload_id=str(record["workload_id"]),
        max_tokens=_effective_max_tokens(record, cap),
        release=release,
        depends_on=depends_on,
        delay_seconds=delay_seconds,
        prompt_suffix=prompt_suffix,
    )


def build_arrival_plan(
    workload: dict[str, Any],
    *,
    scenario: str,
    concurrency: int,
    max_output_tokens: int,
    stagger_delay_seconds: float,
    qmsum_start_index: int = 0,
    exact_replay_count: int | None = None,
) -> ArrivalPlan:
    """Select public records without materializing their prompt text."""

    if scenario not in SCENARIOS:
        raise ArrivalLoadError(f"unknown arrival scenario: {scenario}")
    if concurrency < 1:
        raise ArrivalLoadError("concurrency must be at least one")
    if max_output_tokens < 1:
        raise ArrivalLoadError("max_output_tokens must be at least one")
    if stagger_delay_seconds < 0:
        raise ArrivalLoadError("stagger_delay_seconds must be non-negative")
    if qmsum_start_index < 0:
        raise ArrivalLoadError("qmsum_start_index must be non-negative")
    if scenario == "sustained-exact-prefix" and concurrency != 1:
        raise ArrivalLoadError("sustained-exact-prefix concurrency must be one")
    if scenario == "strict-prefix-append" and concurrency != 1:
        raise ArrivalLoadError("strict-prefix-append concurrency must be one")
    if exact_replay_count is not None and not 1 <= exact_replay_count <= 64:
        raise ArrivalLoadError("exact_replay_count must be between 1 and 64")
    if exact_replay_count is not None and scenario != "sustained-exact-prefix":
        raise ArrivalLoadError(
            "exact_replay_count is only supported for sustained-exact-prefix"
        )
    if qmsum_start_index and scenario != "capacity-replay-six":
        raise ArrivalLoadError(
            "qmsum_start_index is only supported for capacity-replay-six"
        )

    records = _records(workload)
    if scenario == "idle-lifecycle":
        return ArrivalPlan(scenario=scenario, concurrency=concurrency, entries=())

    interactive = [record for record in records if _scenario_family(record) == "interactive"]
    qmsum = [record for record in records if _source_dataset(record) == "qmsum"]
    if scenario == "simultaneous":
        if len(interactive) < concurrency:
            raise ArrivalLoadError("public workload has too few interactive records for concurrency")
        entries = tuple(
            _entry(
                key=f"simultaneous-{index}",
                record=record,
                cap=max_output_tokens,
                release="at-start",
            )
            for index, record in enumerate(interactive[:concurrency])
        )
        return ArrivalPlan(scenario=scenario, concurrency=concurrency, entries=entries)

    if not qmsum:
        raise ArrivalLoadError("public workload has no QMSUM long-prefill record")
    capacity_six_records = qmsum[qmsum_start_index : qmsum_start_index + 6]
    if scenario == "capacity-replay-six" and len(capacity_six_records) < 6:
        raise ArrivalLoadError("public workload has too few distinct QMSUM records")
    long_entry = _entry(
        key="long-primary",
        record=capacity_six_records[0] if scenario == "capacity-replay-six" else qmsum[0],
        cap=max_output_tokens,
        release="at-start",
    )

    if scenario == "sustained-exact-prefix":
        entries = [long_entry]
        dependency_key = long_entry.key
        for index in range(exact_replay_count or 8):
            entry = _entry(
                key=f"sustained-exact-{index + 1}",
                record=qmsum[0],
                cap=max_output_tokens,
                release="after-completion",
                depends_on=dependency_key,
            )
            entries.append(entry)
            dependency_key = entry.key
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=tuple(entries),
        )

    if scenario == "strict-prefix-append":
        strict_entry = _entry(
            key="strict-prefix-append",
            record=qmsum[0],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=long_entry.key,
            prompt_suffix=STRICT_PREFIX_APPEND_TEXT,
        )
        repeat_entry = _entry(
            key="strict-prefix-repeat",
            record=qmsum[0],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=strict_entry.key,
            prompt_suffix=STRICT_PREFIX_APPEND_TEXT,
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=(long_entry, strict_entry, repeat_entry),
        )

    if scenario == "staggered-long-prefill":
        short_count = max(concurrency - 1, 1)
        if len(interactive) < short_count:
            raise ArrivalLoadError("public workload has too few interactive records for staggered traffic")
        short_entries = tuple(
            _entry(
                key=f"staggered-short-{index}",
                record=record,
                cap=max_output_tokens,
                release="after-prefill",
                depends_on=long_entry.key,
                delay_seconds=stagger_delay_seconds * (index + 1),
            )
            for index, record in enumerate(interactive[:short_count])
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=(long_entry, *short_entries),
        )

    if scenario == "shared-prefix":
        reuse_count = max(concurrency - 1, 1)
        entries = [long_entry]
        for index in range(reuse_count):
            entries.append(
                _entry(
                    key=f"shared-prefix-{index}",
                    record=qmsum[0],
                    cap=max_output_tokens,
                    release="after-completion",
                    depends_on=long_entry.key,
                )
            )
        return ArrivalPlan(scenario=scenario, concurrency=concurrency, entries=tuple(entries))

    if scenario == "distinct-prefix":
        if len(qmsum) < 2:
            raise ArrivalLoadError("public workload has too few distinct QMSUM records")
        distinct_entry = _entry(
            key="distinct-prefix-0",
            record=qmsum[1],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=long_entry.key,
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=(long_entry, distinct_entry),
        )

    if scenario == "capacity-replay":
        if len(qmsum) < 3:
            raise ArrivalLoadError("public workload has too few distinct QMSUM records")
        second_entry = _entry(
            key="capacity-distinct-1",
            record=qmsum[1],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=long_entry.key,
        )
        third_entry = _entry(
            key="capacity-distinct-2",
            record=qmsum[2],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=second_entry.key,
        )
        replay_entry = _entry(
            key="capacity-replay-0",
            record=qmsum[0],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=third_entry.key,
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=(long_entry, second_entry, third_entry, replay_entry),
        )

    if scenario == "capacity-replay-depth":
        if len(qmsum) < 4:
            raise ArrivalLoadError("public workload has too few distinct QMSUM records")
        second_entry = _entry(
            key="capacity-depth-distinct-1",
            record=qmsum[1],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=long_entry.key,
        )
        third_entry = _entry(
            key="capacity-depth-distinct-2",
            record=qmsum[2],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=second_entry.key,
        )
        fourth_entry = _entry(
            key="capacity-depth-distinct-3",
            record=qmsum[3],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=third_entry.key,
        )
        replay_entry = _entry(
            key="capacity-depth-replay-0",
            record=qmsum[0],
            cap=max_output_tokens,
            release="after-completion",
            depends_on=fourth_entry.key,
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=(long_entry, second_entry, third_entry, fourth_entry, replay_entry),
        )

    if scenario == "capacity-replay-six":
        entries = [long_entry]
        dependency_key = long_entry.key
        for index, record in enumerate(capacity_six_records[1:], start=1):
            entry = _entry(
                key=f"capacity-six-distinct-{index}",
                record=record,
                cap=max_output_tokens,
                release="after-completion",
                depends_on=dependency_key,
            )
            entries.append(entry)
            dependency_key = entry.key
        entries.append(
            _entry(
                key="capacity-six-replay-0",
                record=capacity_six_records[0],
                cap=max_output_tokens,
                release="after-completion",
                depends_on=dependency_key,
            )
        )
        return ArrivalPlan(
            scenario=scenario,
            concurrency=concurrency,
            entries=tuple(entries),
        )

    if not interactive:
        raise ArrivalLoadError("public workload has no interactive follow-up record")
    follow_up = _entry(
        key="cancel-follow-up",
        record=interactive[0],
        cap=max_output_tokens,
        release="after-cancellation",
        depends_on=long_entry.key,
    )
    return ArrivalPlan(
        scenario=scenario,
        concurrency=concurrency,
        entries=(long_entry, follow_up),
        cancel_target_key=long_entry.key,
    )


def _records_by_id(workload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in _records(workload):
        workload_id = str(record["workload_id"])
        if workload_id in indexed:
            raise ArrivalLoadError(f"public workload repeats {workload_id}")
        indexed[workload_id] = record
    return indexed


def _response_summary(response: Any) -> dict[str, Any]:
    text = str(getattr(response, "text", ""))
    return {
        "request_id": str(getattr(response, "request_id", "")),
        "completion_tokens": int(getattr(response, "completion_tokens", 0)),
        "prefill_cache_hit": bool(getattr(response, "prefill_cache_hit", False)),
        "generation_tps": float(getattr(response, "generation_tps", 0.0)),
        "peak_memory_gb": float(getattr(response, "peak_memory_gb", 0.0)),
        "finish_reason": str(getattr(response, "finish_reason", "")),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _error_summary(error: Exception) -> dict[str, str | None]:
    return {
        "type": type(error).__name__,
        "code": getattr(error, "code", None),
        "message": str(error),
    }


def _has_prefill_request(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    requests = status.get("requests")
    if not isinstance(requests, list):
        return False
    return any(isinstance(request, dict) and request.get("phase") == "prefill" for request in requests)


def _engine_lifecycle_snapshot(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {}
    requests = status.get("requests")
    prefix_cache = status.get("prefix_cache_stats")
    reservation_trace = status.get("snapshot_reservation_trace")
    prefix_cache = prefix_cache if isinstance(prefix_cache, dict) else {}
    reservation_trace = reservation_trace if isinstance(reservation_trace, dict) else {}
    trace_events = reservation_trace.get("events")
    return {
        "active_requests": len(requests) if isinstance(requests, list) else 0,
        "pending_requests": status.get("pending_requests"),
        "prefill_requests": status.get("prefill_requests"),
        "decode_requests": status.get("decode_requests"),
        "active_estimated_bytes": status.get("active_estimated_bytes"),
        "completed_requests": status.get("completed_requests"),
        "failed_requests": status.get("failed_requests"),
        "cancelled_requests": status.get("cancelled_requests"),
        "snapshot_preflight_skips": status.get("snapshot_preflight_skips"),
        "snapshot_entries": status.get("snapshot_entries"),
        "snapshot_bytes": status.get("snapshot_bytes"),
        "prefix_reuse_attempts": status.get("prefix_reuse_attempts"),
        "prefix_reuse_hits": status.get("prefix_reuse_hits"),
        "prefix_tokens_reused": status.get("prefix_tokens_reused"),
        "prefix_cache": {
            key: prefix_cache.get(key)
            for key in (
                "entries",
                "bytes",
                "pinned_entries",
                "pinned_bytes",
                "lookups",
                "hits",
                "stores",
                "evictions",
                "exact_hits",
                "prefix_hits",
                "lcp_hits",
            )
        },
        "snapshot_reservation_trace": {
            "enabled": reservation_trace.get("enabled"),
            "max_events": reservation_trace.get("max_events"),
            "retained_events": len(trace_events) if isinstance(trace_events, list) else 0,
            "dropped_events": reservation_trace.get("dropped_events"),
        },
    }


def _new_engine_lifecycle_sample_summary(*, interval_seconds: float) -> dict[str, Any]:
    if interval_seconds <= 0:
        raise ArrivalLoadError("engine lifecycle sample interval must be positive")
    return {
        "schema_version": 1,
        "interval_seconds": interval_seconds,
        "sample_count": 0,
        "maxima": {
            "active_requests": 0,
            "pending_requests": 0,
            "prefill_requests": 0,
            "decode_requests": 0,
            "active_estimated_bytes": 0,
            "snapshot_entries": 0,
            "snapshot_bytes": 0,
            "live_snapshot_bytes": 0,
            "pinned_entries": 0,
            "pinned_bytes": 0,
            "reservation_trace_events": 0,
            "reservation_trace_dropped_events": 0,
        },
    }


def _sample_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def _record_engine_lifecycle_sample(summary: dict[str, Any], status: Any) -> None:
    snapshot = _engine_lifecycle_snapshot(status)
    prefix_cache = snapshot.get("prefix_cache")
    reservation_trace = snapshot.get("snapshot_reservation_trace")
    prefix_cache = prefix_cache if isinstance(prefix_cache, dict) else {}
    reservation_trace = reservation_trace if isinstance(reservation_trace, dict) else {}
    active_bytes = _sample_int(snapshot.get("active_estimated_bytes"))
    snapshot_bytes = _sample_int(snapshot.get("snapshot_bytes"))
    values = {
        "active_requests": _sample_int(snapshot.get("active_requests")),
        "pending_requests": _sample_int(snapshot.get("pending_requests")),
        "prefill_requests": _sample_int(snapshot.get("prefill_requests")),
        "decode_requests": _sample_int(snapshot.get("decode_requests")),
        "active_estimated_bytes": active_bytes,
        "snapshot_entries": _sample_int(snapshot.get("snapshot_entries")),
        "snapshot_bytes": snapshot_bytes,
        "live_snapshot_bytes": active_bytes + snapshot_bytes,
        "pinned_entries": _sample_int(prefix_cache.get("pinned_entries")),
        "pinned_bytes": _sample_int(prefix_cache.get("pinned_bytes")),
        "reservation_trace_events": _sample_int(reservation_trace.get("retained_events")),
        "reservation_trace_dropped_events": _sample_int(
            reservation_trace.get("dropped_events")
        ),
    }
    maxima = summary["maxima"]
    for key, value in values.items():
        maxima[key] = max(_sample_int(maxima.get(key)), value)
    summary["sample_count"] = _sample_int(summary.get("sample_count")) + 1


async def _sample_engine_lifecycle(
    engine: Any,
    stop_event: asyncio.Event,
    summary: dict[str, Any],
) -> None:
    interval_seconds = float(summary["interval_seconds"])
    while True:
        _record_engine_lifecycle_sample(summary, engine.status())
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue
        return


async def _wait_for_prefill(engine: Any, task: asyncio.Task[dict[str, Any]], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _has_prefill_request(engine.status()):
            return
        if task.done():
            outcome = await task
            raise ArrivalLoadError(
                f"dependent request completed before reaching prefill: {outcome['key']}"
            )
        await asyncio.sleep(0.005)
    raise ArrivalLoadError("dependent request did not reach prefill before timeout")


async def execute_arrival_plan(
    engine: Any,
    plan: ArrivalPlan,
    workload: dict[str, Any],
    *,
    resolve_prompt: Callable[[dict[str, Any]], str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute one public arrival plan against an already-started Aster engine."""

    if timeout_seconds <= 0:
        raise ArrivalLoadError("timeout_seconds must be positive")
    records = _records_by_id(workload)
    entries_by_key = {entry.key: entry for entry in plan.entries}
    if len(entries_by_key) != len(plan.entries):
        raise ArrivalLoadError("arrival plan repeats an entry key")
    for entry in plan.entries:
        if entry.workload_id not in records:
            raise ArrivalLoadError(f"arrival plan references missing public record {entry.workload_id}")
        if entry.depends_on is not None and entry.depends_on not in entries_by_key:
            raise ArrivalLoadError(f"arrival plan has an unknown dependency {entry.depends_on}")

    started = time.perf_counter()
    tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def submit(entry: ArrivalEntry) -> dict[str, Any]:
        record = records[entry.workload_id]
        prompt = resolve_prompt(record)
        if entry.prompt_suffix is not None:
            prompt += entry.prompt_suffix
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=entry.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            trace_id=f"public-arrival:{entry.key}",
            request_aliases=(entry.key,),
            timeout_seconds=timeout_seconds,
            enable_thinking=False,
        )
        submitted_after_seconds = time.perf_counter() - started
        try:
            response = await engine.submit(request)
        except Exception as error:
            event = {
                "key": entry.key,
                "workload_id": entry.workload_id,
                "submitted_after_seconds": submitted_after_seconds,
                "response": None,
                "error": _error_summary(error),
            }
        else:
            event = {
                "key": entry.key,
                "workload_id": entry.workload_id,
                "submitted_after_seconds": submitted_after_seconds,
                "response": _response_summary(response),
                "error": None,
            }
        if plan.scenario in LIFECYCLE_SCENARIOS:
            event["engine_lifecycle"] = _engine_lifecycle_snapshot(engine.status())
        return event

    def launch(entry: ArrivalEntry) -> None:
        if entry.key in tasks:
            raise ArrivalLoadError(f"arrival entry was launched twice: {entry.key}")
        tasks[entry.key] = asyncio.create_task(submit(entry), name=f"public-arrival:{entry.key}")

    for entry in plan.entries:
        if entry.release == "at-start":
            launch(entry)
    if not tasks:
        if not plan.entries:
            final_status = engine.status()
            return {
                "scenario": plan.scenario,
                "concurrency": plan.concurrency,
                "elapsed_seconds": time.perf_counter() - started,
                "cancel_accepted": None,
                "events": [],
                "engine_status": final_status if isinstance(final_status, dict) else {},
            }
        raise ArrivalLoadError("arrival plan has no at-start request")

    after_prefill = [entry for entry in plan.entries if entry.release == "after-prefill"]
    if after_prefill:
        dependency_key = after_prefill[0].depends_on
        if dependency_key is None or dependency_key not in tasks:
            raise ArrivalLoadError("after-prefill entries require a launched dependency")
        if any(entry.depends_on != dependency_key for entry in after_prefill):
            raise ArrivalLoadError("after-prefill entries must share one dependency")
        await _wait_for_prefill(engine, tasks[dependency_key], timeout_seconds)
        previous_delay = 0.0
        for entry in sorted(after_prefill, key=lambda item: item.delay_seconds):
            await asyncio.sleep(max(entry.delay_seconds - previous_delay, 0.0))
            launch(entry)
            previous_delay = entry.delay_seconds

    cancel_accepted: bool | None = None
    if plan.cancel_target_key is not None:
        target_task = tasks.get(plan.cancel_target_key)
        if target_task is None:
            raise ArrivalLoadError("cancellation target was not launched")
        await _wait_for_prefill(engine, target_task, timeout_seconds)
        cancel_accepted = bool(await engine.cancel(plan.cancel_target_key))
        await target_task
        for entry in plan.entries:
            if entry.release == "after-cancellation":
                launch(entry)

    after_completion = [entry for entry in plan.entries if entry.release == "after-completion"]
    for entry in after_completion:
        if entry.depends_on is None or entry.depends_on not in tasks:
            raise ArrivalLoadError("after-completion entry requires a launched dependency")
        dependency = await tasks[entry.depends_on]
        if dependency["response"] is None:
            raise ArrivalLoadError(f"completion dependency failed: {entry.depends_on}")
        launch(entry)

    events = [await tasks[entry.key] for entry in plan.entries]
    final_status = engine.status()
    return {
        "scenario": plan.scenario,
        "concurrency": plan.concurrency,
        "elapsed_seconds": time.perf_counter() - started,
        "cancel_accepted": cancel_accepted,
        "events": events,
        "engine_status": final_status if isinstance(final_status, dict) else {},
    }


def _load_workload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArrivalLoadError(f"cannot load public workload {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArrivalLoadError("public workload root must be an object")
    return payload


def _plan_payload(plan: ArrivalPlan) -> dict[str, Any]:
    entries = []
    for entry in plan.entries:
        payload = asdict(entry)
        if payload["prompt_suffix"] is None:
            payload.pop("prompt_suffix")
        entries.append(payload)
    return {
        "schema_version": 1,
        "kind": "public-arrival-load-plan",
        "scenario": plan.scenario,
        "concurrency": plan.concurrency,
        "cancel_target_key": plan.cancel_target_key,
        "entries": entries,
    }


def _plan_max_output_tokens(plan: ArrivalPlan) -> int | None:
    return max((entry.max_tokens for entry in plan.entries), default=None)


def _apply_baseline_settings(
    settings: RuntimeSettings,
    *,
    concurrency: int,
    prefix_cache_enabled: bool,
    decode_active_prefill_token_budget: int | None,
    snapshot_budget_bytes: int | None,
    snapshot_max_entries: int | None,
    snapshot_reservation_trace_max_events: int | None = None,
    max_active_requests: int | None = None,
) -> RuntimeSettings:
    if decode_active_prefill_token_budget is not None and decode_active_prefill_token_budget < 1:
        raise ArrivalLoadError("decode_active_prefill_token_budget must be positive")
    if snapshot_budget_bytes is not None and snapshot_budget_bytes < 1:
        raise ArrivalLoadError("snapshot_budget_bytes must be positive")
    if snapshot_max_entries is not None and snapshot_max_entries < 1:
        raise ArrivalLoadError("snapshot_max_entries must be positive")
    if max_active_requests is not None and max_active_requests < 1:
        raise ArrivalLoadError("max_active_requests must be positive")
    if (
        snapshot_reservation_trace_max_events is not None
        and snapshot_reservation_trace_max_events < 0
    ):
        raise ArrivalLoadError("snapshot_reservation_trace_max_events must be non-negative")
    if (
        snapshot_reservation_trace_max_events is not None
        and snapshot_reservation_trace_max_events > 256
    ):
        raise ArrivalLoadError("snapshot_reservation_trace_max_events must be at most 256")
    engine_updates: dict[str, Any] = {
        "engine_type": "manual",
        "runtime_kernel": "manual",
        "max_active_requests": (
            max(settings.engine.max_active_requests, concurrency)
            if max_active_requests is None
            else max_active_requests
        ),
        "prefix_cache_enabled": prefix_cache_enabled,
        "prefix_cache_load_on_warmup": False,
        "prefix_cache_save_on_shutdown": False,
        "warm_prompts_path": None,
        "decode_active_prefill_token_budget": decode_active_prefill_token_budget,
    }
    if snapshot_budget_bytes is not None:
        engine_updates["snapshot_budget_bytes"] = snapshot_budget_bytes
    if snapshot_max_entries is not None:
        engine_updates["snapshot_max_entries"] = snapshot_max_entries
    if snapshot_reservation_trace_max_events is not None:
        engine_updates["snapshot_reservation_trace_max_events"] = (
            snapshot_reservation_trace_max_events
        )
    return settings.model_copy(
        update={
            "engine": settings.engine.model_copy(update=engine_updates)
        }
    )


def _resource_snapshot() -> dict[str, int | str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "swap_used_bytes": int(psutil.swap_memory().used),
    }


def _resource_delta(
    before: dict[str, int | str], after: dict[str, int | str]
) -> dict[str, int]:
    return {
        "process_rss_bytes": int(after["process_rss_bytes"])
        - int(before["process_rss_bytes"]),
        "swap_used_bytes": int(after["swap_used_bytes"])
        - int(before["swap_used_bytes"]),
    }


def _resource_summary(
    *,
    lifecycle: dict[str, dict[str, int | str]],
    rss_samples: list[int],
) -> dict[str, Any]:
    before_workload = lifecycle["before_workload"]
    after_workload = lifecycle["after_workload"]
    return {
        # Preserve the original workload-only fields for existing consumers.
        "before": before_workload,
        "after": after_workload,
        "peak_rss_bytes": max(rss_samples),
        "rss_delta_bytes": int(after_workload["process_rss_bytes"])
        - int(before_workload["process_rss_bytes"]),
        "swap_delta_bytes": int(after_workload["swap_used_bytes"])
        - int(before_workload["swap_used_bytes"]),
        "lifecycle": lifecycle,
        "stage_deltas": {
            "engine_create": _resource_delta(
                lifecycle["before_engine_create"], lifecycle["after_engine_create"]
            ),
            "engine_start": _resource_delta(
                lifecycle["after_engine_create"], lifecycle["after_engine_start"]
            ),
            "warmup": _resource_delta(
                lifecycle["after_engine_start"], lifecycle["after_warmup"]
            ),
            "workload": _resource_delta(
                before_workload, after_workload
            ),
            "close": _resource_delta(
                after_workload, lifecycle["after_close"]
            ),
            "total": _resource_delta(
                lifecycle["before_engine_create"], lifecycle["after_close"]
            ),
        },
    }


async def _sample_rss(stop_event: asyncio.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop_event.is_set():
        try:
            samples.append(int(process.memory_info().rss))
        except psutil.Error:
            return
        await asyncio.sleep(0.05)


def _attach_timelines(result: dict[str, Any]) -> None:
    engine_status = result.get("engine_status")
    timelines = engine_status.get("recent_request_timelines", []) if isinstance(engine_status, dict) else []
    by_request_id = {
        timeline.get("request_id"): timeline
        for timeline in timelines
        if isinstance(timeline, dict) and isinstance(timeline.get("request_id"), str)
    }
    events = result.get("events")
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        response = event.get("response")
        if not isinstance(response, dict):
            continue
        request_id = response.get("request_id")
        if isinstance(request_id, str) and request_id in by_request_id:
            event["timeline"] = by_request_id[request_id]


async def run_public_arrival_baseline(
    *,
    config_path: Path,
    workload_path: Path,
    lock_path: Path,
    data_root: Path,
    plan: ArrivalPlan,
    prefix_cache_enabled: bool,
    timeout_seconds: float,
    decode_active_prefill_token_budget: int | None = None,
    snapshot_budget_bytes: int | None = None,
    snapshot_max_entries: int | None = None,
    snapshot_reservation_trace_max_events: int | None = None,
    max_active_requests: int | None = None,
    sample_engine_lifecycle: bool = False,
    engine_lifecycle_sample_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    workload = _load_workload(workload_path)
    if workload.get("lock_sha256") != public.sha256_file(lock_path):
        raise ArrivalLoadError("workload source lock differs from the active source lock")
    lock = public.load_lock(lock_path)
    resolver = public.PublicWorkloadResolver(lock, data_root)
    settings = _apply_baseline_settings(
        load_settings(str(config_path)),
        concurrency=plan.concurrency,
        prefix_cache_enabled=prefix_cache_enabled,
        decode_active_prefill_token_budget=decode_active_prefill_token_budget,
        snapshot_budget_bytes=snapshot_budget_bytes,
        snapshot_max_entries=snapshot_max_entries,
        snapshot_reservation_trace_max_events=snapshot_reservation_trace_max_events,
        max_active_requests=max_active_requests,
    )
    lifecycle: dict[str, dict[str, int | str]] = {
        "before_engine_create": _resource_snapshot()
    }
    engine = InferenceEngine(settings, MetricsRegistry(settings.telemetry.metrics_namespace))
    engine_lifecycle_summary = (
        _new_engine_lifecycle_sample_summary(
            interval_seconds=engine_lifecycle_sample_interval_seconds
        )
        if sample_engine_lifecycle
        else None
    )
    lifecycle["after_engine_create"] = _resource_snapshot()
    rss_samples: list[int] = []
    rss_stop = asyncio.Event()
    rss_task: asyncio.Task[None] | None = None
    engine_lifecycle_task: asyncio.Task[None] | None = None
    try:
        await engine.start()
        lifecycle["after_engine_start"] = _resource_snapshot()
        await engine.warmup()
        lifecycle["after_warmup"] = _resource_snapshot()
        lifecycle["before_workload"] = _resource_snapshot()
        rss_samples.append(int(lifecycle["before_workload"]["process_rss_bytes"]))
        rss_task = asyncio.create_task(_sample_rss(rss_stop, rss_samples))
        if engine_lifecycle_summary is not None:
            engine_lifecycle_task = asyncio.create_task(
                _sample_engine_lifecycle(engine, rss_stop, engine_lifecycle_summary)
            )
        result = await execute_arrival_plan(
            engine,
            plan,
            workload,
            resolve_prompt=resolver.resolve,
            timeout_seconds=timeout_seconds,
        )
    finally:
        rss_stop.set()
        if rss_task is not None:
            await rss_task
        if engine_lifecycle_task is not None:
            await engine_lifecycle_task
        lifecycle["after_workload"] = _resource_snapshot()
        try:
            await engine.aclose()
        finally:
            lifecycle["after_close"] = _resource_snapshot()

    _attach_timelines(result)
    resources = _resource_summary(lifecycle=lifecycle, rss_samples=rss_samples)
    execution = {
        "engine": "aster-manual",
        "config": str(config_path.resolve()),
        "prefix_cache_enabled": prefix_cache_enabled,
        "decode_active_prefill_token_budget": (
            settings.engine.decode_active_prefill_token_budget
        ),
        "snapshot_budget_bytes": settings.engine.snapshot_budget_bytes,
        "snapshot_max_entries": settings.engine.snapshot_max_entries,
        "snapshot_reservation_trace_max_events": (
            settings.engine.snapshot_reservation_trace_max_events
        ),
        "max_active_requests": settings.engine.max_active_requests,
        "timeout_seconds": timeout_seconds,
        "max_output_tokens": _plan_max_output_tokens(plan),
    }
    if engine_lifecycle_summary is not None:
        engine_lifecycle_summary["final"] = _engine_lifecycle_snapshot(
            result.get("engine_status")
        )
        resources["engine_lifecycle_sampling"] = engine_lifecycle_summary
        execution["engine_lifecycle_sample_interval_seconds"] = (
            engine_lifecycle_sample_interval_seconds
        )
    return {
        "schema_version": 1,
        "kind": "public-arrival-load-result",
        "source": {
            "workload_path": str(workload_path.resolve()),
            "workload_sha256": public.sha256_file(workload_path),
            "source_lock_sha256": public.sha256_file(lock_path),
            "generation": workload.get("generation"),
        },
        "execution": execution,
        "plan": _plan_payload(plan),
        "resources": resources,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--stagger-delay-seconds", type=float, default=0.05)
    parser.add_argument("--qmsum-start-index", type=int, default=0)
    parser.add_argument("--exact-replay-count", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--lock", type=Path, default=public.DEFAULT_LOCK_PATH)
    parser.add_argument("--data-root", type=Path, default=public.DEFAULT_DATA_ROOT)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="on")
    parser.add_argument("--decode-active-prefill-budget", type=int)
    parser.add_argument("--snapshot-budget-bytes", type=int)
    parser.add_argument("--snapshot-max-entries", type=int)
    parser.add_argument("--snapshot-reservation-trace-max-events", type=int)
    parser.add_argument("--max-active-requests", type=int)
    parser.add_argument("--sample-engine-lifecycle", action="store_true")
    parser.add_argument("--engine-lifecycle-sample-interval-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_arrival_plan(
        _load_workload(args.workload),
        scenario=args.scenario,
        concurrency=args.concurrency,
        max_output_tokens=args.max_output_tokens,
        stagger_delay_seconds=args.stagger_delay_seconds,
        qmsum_start_index=args.qmsum_start_index,
        exact_replay_count=args.exact_replay_count,
    )
    payload = (
        asyncio.run(
            run_public_arrival_baseline(
                config_path=args.config.resolve(),
                workload_path=args.workload.resolve(),
                lock_path=args.lock.resolve(),
                data_root=args.data_root.resolve(),
                plan=plan,
                prefix_cache_enabled=args.prefix_cache == "on",
                timeout_seconds=args.timeout_seconds,
                decode_active_prefill_token_budget=args.decode_active_prefill_budget,
                snapshot_budget_bytes=args.snapshot_budget_bytes,
                snapshot_max_entries=args.snapshot_max_entries,
                snapshot_reservation_trace_max_events=(
                    args.snapshot_reservation_trace_max_events
                ),
                max_active_requests=args.max_active_requests,
                sample_engine_lifecycle=args.sample_engine_lifecycle,
                engine_lifecycle_sample_interval_seconds=(
                    args.engine_lifecycle_sample_interval_seconds
                ),
            )
        )
        if args.execute
        else _plan_payload(plan)
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
