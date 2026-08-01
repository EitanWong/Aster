#!/usr/bin/env python3
"""Map active-request cap behavior across exact, short, and mixed B8 workloads."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev import public_arrival_load as arrival  # noqa: E402

DEFAULT_WORKLOAD = PROJECT_ROOT / "run/loop-engineering/public-benchmarks/cross-engine-core.json"
WORKLOADS = ("exact-long", "short-simultaneous", "mixed")
CAPS = (2, 3, 4, 5, 6, 16)
DIAGNOSTIC_CAPS = (2, 3, 5, 16)
CONCURRENCY = 8
MAX_OUTPUT_TOKENS = 8
DECODE_ACTIVE_PREFILL_TOKEN_BUDGET = 512
SNAPSHOT_BUDGET_BYTES = 8 * 1024**3
SNAPSHOT_MAX_ENTRIES = 256
SNAPSHOT_TRACE_EVENTS = 64
THROUGHPUT_FLOOR = 0.97
NO_REGRESSION_RATIO = 1.03
EXACT_MEMORY_REDUCTION_GATE = 10.0


class BenchmarkError(RuntimeError):
    """Raised when a frontier record violates the frozen I088 contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise BenchmarkError("at least one value is required")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(percentile * len(ordered))))
    return ordered[rank - 1]


def build_frontier_plan(public_workload: dict[str, Any], *, workload: str) -> arrival.ArrivalPlan:
    if workload == "exact-long":
        return arrival.build_arrival_plan(
            public_workload,
            scenario="shared-prefix",
            concurrency=CONCURRENCY,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stagger_delay_seconds=0.0,
        )
    if workload == "short-simultaneous":
        return arrival.build_arrival_plan(
            public_workload,
            scenario="simultaneous",
            concurrency=CONCURRENCY,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            stagger_delay_seconds=0.0,
        )
    if workload != "mixed":
        raise BenchmarkError(f"unknown frontier workload: {workload}")

    records = arrival._records(public_workload)
    interactive = [
        record for record in records if arrival._scenario_family(record) == "interactive"
    ]
    qmsum = [record for record in records if arrival._source_dataset(record) == "qmsum"]
    if len(interactive) < 4 or len(qmsum) < 2:
        raise BenchmarkError("mixed frontier requires four interactive and two QMSUM records")
    long_entry = arrival._entry(
        key="long-primary",
        record=qmsum[0],
        cap=MAX_OUTPUT_TOKENS,
        release="at-start",
    )
    entries = [long_entry]
    for index in range(2):
        entries.append(
            arrival._entry(
                key=f"mixed-exact-{index}",
                record=qmsum[0],
                cap=MAX_OUTPUT_TOKENS,
                release="after-completion",
                depends_on=long_entry.key,
            )
        )
    entries.append(
        arrival._entry(
            key="mixed-distinct-0",
            record=qmsum[1],
            cap=MAX_OUTPUT_TOKENS,
            release="after-completion",
            depends_on=long_entry.key,
        )
    )
    for index, record in enumerate(interactive[:4]):
        entries.append(
            arrival._entry(
                key=f"mixed-short-{index}",
                record=record,
                cap=MAX_OUTPUT_TOKENS,
                release="after-completion",
                depends_on=long_entry.key,
            )
        )
    return arrival.ArrivalPlan(
        scenario="active-cap-mixed",
        concurrency=CONCURRENCY,
        entries=tuple(entries),
    )


def _measurement_events(events: list[dict[str, Any]], *, workload: str) -> list[dict[str, Any]]:
    measured = (
        events
        if workload == "short-simultaneous"
        else [event for event in events if event.get("key") != "long-primary"]
    )
    expected = 8 if workload == "short-simultaneous" else 7
    if len(measured) != expected:
        raise BenchmarkError(f"{workload} has an unexpected measured-request count")
    return measured


def compact_result(
    payload: dict[str, Any], *, workload: str, cap: int, sequence: int
) -> dict[str, Any]:
    if workload not in WORKLOADS:
        raise BenchmarkError(f"unknown frontier workload: {workload}")
    if cap not in CAPS:
        raise BenchmarkError(f"unsupported active cap: {cap}")
    if sequence < 1:
        raise BenchmarkError("sequence must be positive")

    events = payload["result"]["events"]
    if len(events) != 8:
        raise BenchmarkError("frontier workloads require exactly eight requests")
    measured = _measurement_events(events, workload=workload)
    measured_ids = {id(event) for event in measured}
    submissions: list[float] = []
    completions: list[float] = []
    ttft: list[float] = []
    latency: list[float] = []
    completion_tokens = 0
    peak_mlx_memory = 0.0
    fingerprints: dict[str, list[str | int]] = {}
    request_contract = True
    for event in events:
        response = event.get("response")
        timeline = event.get("timeline")
        if (
            event.get("error") is not None
            or not isinstance(response, dict)
            or not isinstance(timeline, dict)
        ):
            request_contract = False
            continue
        key = str(event["key"])
        fingerprints[key] = [
            str(timeline.get("output_token_ids_sha256")),
            str(response.get("text_sha256")),
            str(response.get("finish_reason")),
            int(response.get("completion_tokens", -1)),
        ]
        request_contract = request_contract and response.get("finish_reason") == "length"
        request_contract = request_contract and int(timeline.get("decode_steps", -1)) == 8
        peak_mlx_memory = max(peak_mlx_memory, float(response.get("peak_memory_gb", 0.0)))
        if id(event) in measured_ids:
            submitted = float(event["submitted_after_seconds"])
            request_latency = float(timeline["total_latency_s"])
            submissions.append(submitted)
            completions.append(submitted + request_latency)
            ttft.append(float(timeline["ttft_s"]))
            latency.append(request_latency)
            completion_tokens += int(timeline["completion_tokens"])

    if len(fingerprints) != 8:
        request_contract = False
    service_window = max(completions) - min(submissions)
    if service_window <= 0:
        raise BenchmarkError("measured service window must be positive")

    resources = payload["resources"]
    sampling = resources.get("engine_lifecycle_sampling")
    if not isinstance(sampling, dict):
        raise BenchmarkError("engine lifecycle sampling is required")
    final = sampling["final"]
    maxima = sampling["maxima"]
    prefix = final["prefix_cache"]
    trace = final["snapshot_reservation_trace"]
    terminal_clean = all(
        int(final.get(field, -1)) == 0
        for field in (
            "active_requests",
            "pending_requests",
            "prefill_requests",
            "decode_requests",
            "failed_requests",
            "cancelled_requests",
        )
    ) and all(
        int(prefix.get(field, -1)) == 0 for field in ("pinned_entries", "pinned_bytes", "evictions")
    )
    cache_contract = (
        int(final.get("snapshot_preflight_skips", -1)) == 0
        and int(trace.get("dropped_events", -1)) == 0
    )
    expected_exact_hits = {"exact-long": 7, "short-simultaneous": 0, "mixed": 2}
    cache_contract = (
        cache_contract and int(prefix.get("exact_hits", -1)) == expected_exact_hits[workload]
    )
    status = payload["result"]["engine_status"]
    scheduler = status["scheduler"]
    configured_cap = int(payload["execution"]["max_active_requests"])
    configured_contract = configured_cap == cap and int(scheduler["max_active_requests"]) == cap

    return {
        "workload": workload,
        "cap": cap,
        "sequence": sequence,
        "plan_sha256": _canonical_sha256(payload["plan"]),
        "workload_sha256": str(payload["source"]["workload_sha256"]),
        "source_lock_sha256": str(payload["source"]["source_lock_sha256"]),
        "model": str(status["model"]),
        "max_decode_batch": int(scheduler["max_decode_batch"]),
        "configured_max_active_requests": configured_cap,
        "observed_peak_submitted_requests": int(maxima["active_requests"]),
        "observed_peak_active_estimated_bytes": int(maxima["active_estimated_bytes"]),
        "peak_mlx_memory_gb": peak_mlx_memory,
        "peak_rss_bytes": int(resources["peak_rss_bytes"]),
        "aggregate_tps": completion_tokens / service_window,
        "service_window_seconds": service_window,
        "p95_ttft_seconds": _nearest_rank(ttft, 0.95),
        "p95_latency_seconds": _nearest_rank(latency, 0.95),
        "max_latency_seconds": max(latency),
        "completion_spread_seconds": max(completions) - min(completions),
        "output_fingerprints": fingerprints,
        "prefix_cache": {
            field: int(prefix.get(field, 0))
            for field in ("lookups", "hits", "exact_hits", "stores", "entries", "evictions")
        },
        "request_contract": request_contract,
        "cache_contract": cache_contract,
        "terminal_clean": terminal_clean,
        "configured_contract": configured_contract,
        "contract_passed": (
            request_contract and cache_contract and terminal_clean and configured_contract
        ),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_material_regression = (
        float(left["peak_mlx_memory_gb"]) <= float(right["peak_mlx_memory_gb"]) * 1.03
        and float(left["aggregate_tps"]) >= float(right["aggregate_tps"]) * 0.97
        and float(left["p95_ttft_seconds"]) <= float(right["p95_ttft_seconds"]) * 1.03
        and float(left["p95_latency_seconds"]) <= float(right["p95_latency_seconds"]) * 1.03
        and float(left["max_latency_seconds"]) <= float(right["max_latency_seconds"]) * 1.03
    )
    material_gain = (
        float(left["peak_mlx_memory_gb"]) < float(right["peak_mlx_memory_gb"]) * 0.97
        or float(left["aggregate_tps"]) > float(right["aggregate_tps"]) * 1.03
        or float(left["p95_ttft_seconds"]) < float(right["p95_ttft_seconds"]) * 0.97
        or float(left["p95_latency_seconds"]) < float(right["p95_latency_seconds"]) * 0.97
        or float(left["max_latency_seconds"]) < float(right["max_latency_seconds"]) * 0.97
    )
    return no_material_regression and material_gain


def _summarize_output_consistency(
    indexed: dict[tuple[str, int], dict[str, Any]],
    *,
    workload: str,
) -> dict[str, Any]:
    baseline_cap = 16
    baseline = indexed[(workload, baseline_cap)]["output_fingerprints"]
    divergent_caps: list[int] = []
    divergent_keys: dict[str, list[int]] = {}
    for cap in CAPS:
        if cap == baseline_cap:
            continue
        fingerprints = indexed[(workload, cap)]["output_fingerprints"]
        if fingerprints == baseline:
            continue
        divergent_caps.append(cap)
        for key in sorted(set(baseline) | set(fingerprints)):
            if fingerprints.get(key) != baseline.get(key):
                divergent_keys.setdefault(key, []).append(cap)
    return {
        "consistent": not divergent_caps,
        "baseline_cap": baseline_cap,
        "divergent_caps": divergent_caps,
        "divergent_keys": divergent_keys,
    }


def summarize_pilot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["workload"]), int(row["cap"]))
        if key in indexed:
            raise BenchmarkError("pilot repeats a workload/cap cell")
        indexed[key] = row
    expected = {(workload, cap) for workload in WORKLOADS for cap in CAPS}
    if set(indexed) != expected:
        raise BenchmarkError("pilot requires a complete 18-row grid")
    if not all(bool(row["contract_passed"]) for row in rows):
        raise BenchmarkError("pilot contains a failed request/cache/lifecycle contract")

    workload_summaries: dict[str, Any] = {}
    eligible_by_workload: dict[str, set[int]] = {}
    output_consistency: dict[str, Any] = {}
    for workload in WORKLOADS:
        workload_rows = [indexed[(workload, cap)] for cap in CAPS]
        for field in ("plan_sha256", "workload_sha256", "model", "max_decode_batch"):
            if len({json.dumps(row[field], sort_keys=True) for row in workload_rows}) != 1:
                raise BenchmarkError(f"{workload} {field} differs across caps")
        consistency = _summarize_output_consistency(indexed, workload=workload)
        output_consistency[workload] = consistency

        baseline = indexed[(workload, 16)]
        cells: dict[str, Any] = {}
        eligible: set[int] = set()
        for cap in CAPS:
            row = indexed[(workload, cap)]
            throughput_ratio = float(row["aggregate_tps"]) / float(baseline["aggregate_tps"])
            ttft_ratio = float(row["p95_ttft_seconds"]) / float(baseline["p95_ttft_seconds"])
            latency_ratio = float(row["p95_latency_seconds"]) / float(
                baseline["p95_latency_seconds"]
            )
            max_latency_ratio = float(row["max_latency_seconds"]) / float(
                baseline["max_latency_seconds"]
            )
            memory_ratio = float(row["peak_mlx_memory_gb"]) / float(baseline["peak_mlx_memory_gb"])
            memory_reduction_percent = 100.0 * (1.0 - memory_ratio)
            general_eligible = (
                throughput_ratio >= THROUGHPUT_FLOOR
                and ttft_ratio <= NO_REGRESSION_RATIO
                and latency_ratio <= NO_REGRESSION_RATIO
                and max_latency_ratio <= NO_REGRESSION_RATIO
                and memory_ratio <= NO_REGRESSION_RATIO
            )
            exact_memory_eligible = (
                workload != "exact-long" or memory_reduction_percent >= EXACT_MEMORY_REDUCTION_GATE
            )
            if cap != 16 and general_eligible and exact_memory_eligible:
                eligible.add(cap)
            cells[str(cap)] = {
                "peak_mlx_memory_gb": float(row["peak_mlx_memory_gb"]),
                "aggregate_tps": float(row["aggregate_tps"]),
                "p95_ttft_seconds": float(row["p95_ttft_seconds"]),
                "p95_latency_seconds": float(row["p95_latency_seconds"]),
                "max_latency_seconds": float(row["max_latency_seconds"]),
                "throughput_ratio_vs_16": throughput_ratio,
                "ttft_ratio_vs_16": ttft_ratio,
                "latency_ratio_vs_16": latency_ratio,
                "max_latency_ratio_vs_16": max_latency_ratio,
                "memory_ratio_vs_16": memory_ratio,
                "memory_reduction_percent_vs_16": memory_reduction_percent,
                "eligible": cap in eligible,
            }
        pareto_caps = [
            cap
            for cap in CAPS
            if not any(
                other != cap and _dominates(indexed[(workload, other)], indexed[(workload, cap)])
                for other in CAPS
            )
        ]
        eligible_by_workload[workload] = eligible
        workload_summaries[workload] = {
            "cells": cells,
            "eligible_caps": sorted(eligible),
            "pareto_caps": pareto_caps,
            "output_consistent": consistency["consistent"],
        }

    performance_global_eligible = sorted(
        set.intersection(*(eligible_by_workload[name] for name in WORKLOADS))
    )
    cross_cap_output_consistent = all(
        consistency["consistent"] for consistency in output_consistency.values()
    )
    diagnostic_caps: list[int] = []
    if not cross_cap_output_consistent:
        global_eligible: list[int] = []
        confirmation_caps: list[int] = []
        diagnostic_caps = sorted(
            {
                16,
                *(
                    cap
                    for consistency in output_consistency.values()
                    for cap in consistency["divergent_caps"]
                ),
            }
        )
        decision = "reject-output-drift"
    elif performance_global_eligible:
        global_eligible = performance_global_eligible
        selected = min(
            global_eligible,
            key=lambda cap: (
                -statistics.geometric_mean(
                    indexed[(workload, cap)]["aggregate_tps"]
                    / indexed[(workload, 16)]["aggregate_tps"]
                    for workload in WORKLOADS
                ),
                cap,
            ),
        )
        confirmation_caps = [selected, 16]
        decision = "confirm-global-candidate"
    else:
        global_eligible = []
        conditional_caps = sorted(set().union(*(eligible_by_workload[name] for name in WORKLOADS)))
        confirmation_caps = [*conditional_caps, 16]
        decision = (
            "confirm-conditional-candidates" if conditional_caps else "reject-low-cap-frontier"
        )

    return {
        "cell_contracts_passed": True,
        "cross_cap_output_consistent": cross_cap_output_consistent,
        "contracts_passed": cross_cap_output_consistent,
        "output_consistency": output_consistency,
        "workloads": workload_summaries,
        "performance_global_eligible_caps": performance_global_eligible,
        "global_eligible_caps": global_eligible,
        "confirmation_caps": confirmation_caps,
        "diagnostic_caps": diagnostic_caps,
        "decision": decision,
    }


def summarize_diagnostics(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        cap = int(payload["cap"])
        if cap in indexed:
            raise BenchmarkError("diagnostics repeat an active cap")
        indexed[cap] = payload
    if set(indexed) != set(DIAGNOSTIC_CAPS):
        raise BenchmarkError("diagnostics require caps 2/3/5/16")
    if any(
        payload.get("kind") != "active-cap-greedy-logit-diagnostic"
        or payload.get("performance_measurement_valid") is not False
        or payload.get("request_contract_passed") is not True
        for payload in payloads
    ):
        raise BenchmarkError("diagnostic metadata or request contract failed")
    for field in ("target_request_id", "candidate_token_ids"):
        if len({json.dumps(payload[field], sort_keys=True) for payload in payloads}) != 1:
            raise BenchmarkError(f"diagnostic {field} differs across caps")

    sequences: dict[int, list[int]] = {}
    cohort_steps: dict[int, dict[int, dict[str, Any]]] = {}
    trace_steps: dict[int, dict[int, dict[str, Any]]] = {}
    for cap, payload in indexed.items():
        trace = payload.get("trace")
        cohorts = payload.get("cohorts")
        if not isinstance(trace, list) or not isinstance(cohorts, list):
            raise BenchmarkError("diagnostic trace and cohorts are required")
        by_trace = {int(step["completion_tokens"]): step for step in trace}
        by_cohort = {int(step["completion_tokens"]): step for step in cohorts}
        if set(by_trace) != set(range(8)) or set(by_cohort) != set(range(8)):
            raise BenchmarkError("diagnostic requires eight unique decode steps")
        sequence = [int(by_trace[index]["selected_token"]) for index in range(8)]
        sequences[cap] = sequence
        trace_steps[cap] = by_trace
        cohort_steps[cap] = by_cohort

    first_divergent = next(
        (
            index
            for index in range(8)
            if len({sequences[cap][index] for cap in DIAGNOSTIC_CAPS}) != 1
        ),
        None,
    )
    if first_divergent != 6:
        raise BenchmarkError("diagnostic did not reproduce the frozen step-6 divergence")

    output_groups: dict[str, list[int]] = {}
    text_by_output: dict[str, set[str]] = {}
    for cap, payload in indexed.items():
        output_hash = str(payload["output_token_ids_sha256"])
        output_groups.setdefault(output_hash, []).append(cap)
        text_by_output.setdefault(output_hash, set()).add(str(payload["text_sha256"]))
    if len(output_groups) != 2 or any(len(texts) != 1 for texts in text_by_output.values()):
        raise BenchmarkError("diagnostic did not reproduce two stable output groups")

    divergent_step: dict[str, Any] = {}
    for cap in DIAGNOSTIC_CAPS:
        trace = trace_steps[cap][first_divergent]
        cohort = cohort_steps[cap][first_divergent]
        candidate_logits = {
            str(token_id): float(value) for token_id, value in trace["candidate_logits"].items()
        }
        if max(candidate_logits.values()) - min(candidate_logits.values()) > 0.125:
            raise BenchmarkError("diagnostic divergence is not a near tie")
        selected_token = int(trace["selected_token"])
        selected_logit = candidate_logits.get(str(selected_token))
        if selected_logit is None or selected_logit != max(candidate_logits.values()):
            raise BenchmarkError("diagnostic selected token is not a candidate-logit maximum")
        divergent_step[str(cap)] = {
            "mode": str(cohort["mode"]),
            "cohort_size": len(cohort["request_ids"]),
            "request_ids": list(cohort["request_ids"]),
            "selected_token": selected_token,
            "candidate_logits": candidate_logits,
        }

    if (
        {divergent_step[str(cap)]["mode"] for cap in (2, 5)} != {"single"}
        or {divergent_step[str(cap)]["mode"] for cap in (3, 16)} != {"batch"}
        or {sequences[cap][first_divergent] for cap in (2, 5)}
        == {sequences[cap][first_divergent] for cap in (3, 16)}
    ):
        raise BenchmarkError("diagnostic output groups do not follow the cohort-shape split")

    return {
        "contracts_passed": True,
        "caps": list(DIAGNOSTIC_CAPS),
        "output_groups": [
            {"caps": sorted(caps), "output_token_ids_sha256": output_hash}
            for output_hash, caps in sorted(output_groups.items())
        ],
        "shared_selected_prefix": sequences[DIAGNOSTIC_CAPS[0]][:first_divergent],
        "first_divergent_completion_index": first_divergent,
        "divergent_step": divergent_step,
        "diagnosis": "batch-shape-sensitive-near-tie",
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_retained_pilot(
    paths: list[Path], diagnostic_paths: list[Path] | None = None
) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    rows = [
        compact_result(
            payload,
            workload=str(payload["experiment"]["workload"]),
            cap=int(payload["experiment"]["cap"]),
            sequence=int(payload["experiment"]["sequence"]),
        )
        for payload in payloads
    ]
    pilot = summarize_pilot(rows)
    diagnostic_paths = diagnostic_paths or []
    diagnostics = (
        summarize_diagnostics([json.loads(path.read_text()) for path in diagnostic_paths])
        if diagnostic_paths
        else None
    )
    source_paths = (
        Path("scripts/dev/benchmark_active_cap_frontier.py"),
        Path("scripts/dev/diagnose_active_cap_logits.py"),
        Path("tests/test_active_cap_frontier_benchmark.py"),
        Path("scripts/dev/public_arrival_load.py"),
    )
    return {
        "schema_version": 2,
        "kind": "active-cap-workload-frontier-evidence",
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline_commit": _git_head(),
        "caps": list(CAPS),
        "workloads": list(WORKLOADS),
        "production_routing_changed": False,
        "references": {
            "feather": "https://arxiv.org/abs/2605.06046v1",
            "vllm_metal_main": "b6e35b6c642162dbf6f31009b81635426a91b64a",
            "sglang_main": "e1964da451ef9fbec04b326c729916281f90809b",
        },
        "rows": rows,
        "pilot": pilot,
        "diagnostics": diagnostics,
        "raw_sha256": {path.name: _sha256_file(path) for path in paths},
        "diagnostic_sha256": {path.name: _sha256_file(path) for path in diagnostic_paths},
        "source_sha256": {str(path): _sha256_file(PROJECT_ROOT / path) for path in source_paths},
    }


async def _run_cell(args: argparse.Namespace) -> dict[str, Any]:
    public_workload = arrival._load_workload(args.source_workload)
    plan = build_frontier_plan(public_workload, workload=args.workload)
    payload = await arrival.run_public_arrival_baseline(
        config_path=args.config.resolve(),
        workload_path=args.source_workload.resolve(),
        lock_path=args.lock.resolve(),
        data_root=args.data_root.resolve(),
        plan=plan,
        prefix_cache_enabled=True,
        timeout_seconds=args.timeout_seconds,
        decode_active_prefill_token_budget=DECODE_ACTIVE_PREFILL_TOKEN_BUDGET,
        snapshot_budget_bytes=SNAPSHOT_BUDGET_BYTES,
        snapshot_max_entries=SNAPSHOT_MAX_ENTRIES,
        snapshot_reservation_trace_max_events=SNAPSHOT_TRACE_EVENTS,
        max_active_requests=args.cap,
        sample_engine_lifecycle=True,
        engine_lifecycle_sample_interval_seconds=0.05,
    )
    payload["experiment"] = {
        "kind": "active-cap-workload-frontier",
        "workload": args.workload,
        "cap": args.cap,
        "sequence": args.sequence,
        "compact": compact_result(
            payload,
            workload=args.workload,
            cap=args.cap,
            sequence=args.sequence,
        ),
    }
    return payload


def _write_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workload", choices=WORKLOADS, required=True)
    run_parser.add_argument("--cap", type=int, choices=CAPS, required=True)
    run_parser.add_argument("--sequence", type=int, required=True)
    run_parser.add_argument("--source-workload", type=Path, default=DEFAULT_WORKLOAD)
    run_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    run_parser.add_argument("--lock", type=Path, default=arrival.public.DEFAULT_LOCK_PATH)
    run_parser.add_argument("--data-root", type=Path, default=arrival.public.DEFAULT_DATA_ROOT)
    run_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    run_parser.add_argument("--output", type=Path, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("inputs", nargs="+", type=Path)
    summarize_parser.add_argument("--diagnostics", nargs="*", type=Path, default=[])
    summarize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        payload = asyncio.run(_run_cell(args))
        _write_payload(payload, args.output)
        print(json.dumps(payload["experiment"]["compact"], indent=2, sort_keys=True))
        return

    summary = build_retained_pilot(args.inputs, args.diagnostics)
    _write_payload(summary, args.output)
    print(json.dumps(summary["pilot"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
