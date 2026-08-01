#!/usr/bin/env python3
"""Screen a B4 active-request cohort cap on one exact-prefix B8 arrival plan."""

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
CONCURRENCY = 8
CANDIDATE_ACTIVE_CAP = 4
MAX_OUTPUT_TOKENS = 8
DECODE_ACTIVE_PREFILL_TOKEN_BUDGET = 512
SNAPSHOT_BUDGET_BYTES = 8 * 1024**3
SNAPSHOT_MAX_ENTRIES = 256
SNAPSHOT_TRACE_EVENTS = 64
I084_ESTIMATED_BYTES_PER_ACTIVE_REPLAY = 390_397_952
MEMORY_REDUCTION_PERCENT_GATE = 10.0
MEMORY_REDUCTION_GB_GATE = 1.0
NO_REGRESSION_RATIO = 1.03
THROUGHPUT_RATIO_FLOOR = 0.97
REQUIRED_PAIRS = 5


class BenchmarkError(RuntimeError):
    """Raised when a cohort-cap record violates the frozen I087 contract."""


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


def compact_result(
    payload: dict[str, Any],
    *,
    lane: str,
    replicate: int,
    order: str,
) -> dict[str, Any]:
    if lane not in {"baseline", "candidate"}:
        raise BenchmarkError(f"unknown lane: {lane}")
    if replicate < 1:
        raise BenchmarkError("replicate must be positive")
    if order not in {"baseline-first", "candidate-first"}:
        raise BenchmarkError(f"unknown order: {order}")

    plan = payload["plan"]
    if plan.get("scenario") != "shared-prefix" or int(plan.get("concurrency", 0)) != 8:
        raise BenchmarkError("cohort-cap screen requires the locked shared-prefix B8 plan")
    events = payload["result"]["events"]
    replays = [event for event in events if str(event.get("key", "")).startswith("shared-prefix-")]
    if len(events) != 8 or len(replays) != 7:
        raise BenchmarkError("cohort-cap screen requires one cold request and seven replays")

    replay_submissions: list[float] = []
    replay_completions: list[float] = []
    replay_ttft: list[float] = []
    replay_latency: list[float] = []
    replay_tokens = 0
    output_token_hashes: set[str] = set()
    text_hashes: set[str] = set()
    finish_reasons: set[str] = set()
    peak_mlx_memory = 0.0
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
        output_token_hashes.add(str(timeline.get("output_token_ids_sha256")))
        text_hashes.add(str(response.get("text_sha256")))
        finish_reasons.add(str(response.get("finish_reason")))
        peak_mlx_memory = max(peak_mlx_memory, float(response.get("peak_memory_gb", 0.0)))
        if event in replays:
            submitted = float(event["submitted_after_seconds"])
            latency = float(timeline["total_latency_s"])
            replay_submissions.append(submitted)
            replay_completions.append(submitted + latency)
            replay_ttft.append(float(timeline["ttft_s"]))
            replay_latency.append(latency)
            replay_tokens += int(timeline["completion_tokens"])
            request_contract = request_contract and bool(response.get("prefill_cache_hit"))
            request_contract = request_contract and int(timeline["decode_steps"]) == 8

    service_window = max(replay_completions) - min(replay_submissions)
    if service_window <= 0:
        raise BenchmarkError("replay service window must be positive")

    resources = payload["resources"]
    sampling = resources.get("engine_lifecycle_sampling")
    if not isinstance(sampling, dict):
        raise BenchmarkError("engine lifecycle sampling is required")
    final = sampling["final"]
    maxima = sampling["maxima"]
    prefix = final["prefix_cache"]
    status = payload["result"]["engine_status"]
    scheduler = status["scheduler"]
    peak_active_estimated_bytes = int(maxima["active_estimated_bytes"])
    peak_active_cache_equivalents = math.ceil(
        peak_active_estimated_bytes / I084_ESTIMATED_BYTES_PER_ACTIVE_REPLAY
    )
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
        int(prefix.get("exact_hits", -1)) == 7
        and int(prefix.get("stores", -1)) == 1
        and int(prefix.get("entries", -1)) == 1
        and int(final.get("snapshot_preflight_skips", -1)) == 0
        and int(final["snapshot_reservation_trace"].get("dropped_events", -1)) == 0
    )
    output_contract = (
        len(output_token_hashes) == 1 and len(text_hashes) == 1 and finish_reasons == {"length"}
    )

    return {
        "lane": lane,
        "replicate": replicate,
        "order": order,
        "plan_sha256": _canonical_sha256(plan),
        "workload_sha256": str(payload["source"]["workload_sha256"]),
        "source_lock_sha256": str(payload["source"]["source_lock_sha256"]),
        "model": str(status["model"]),
        "max_decode_batch": int(scheduler["max_decode_batch"]),
        "configured_max_active_requests": int(payload["execution"]["max_active_requests"]),
        "observed_peak_submitted_requests": int(maxima["active_requests"]),
        "observed_peak_active_estimated_bytes": peak_active_estimated_bytes,
        "observed_peak_active_cache_equivalents": peak_active_cache_equivalents,
        "peak_mlx_memory_gb": peak_mlx_memory,
        "peak_rss_bytes": int(resources["peak_rss_bytes"]),
        "aggregate_replay_tps": replay_tokens / service_window,
        "replay_service_window_seconds": service_window,
        "replay_p95_ttft_seconds": _nearest_rank(replay_ttft, 0.95),
        "replay_p95_latency_seconds": _nearest_rank(replay_latency, 0.95),
        "replay_max_latency_seconds": max(replay_latency),
        "replay_completion_spread_seconds": max(replay_completions) - min(replay_completions),
        "request_count": len(events),
        "replay_count": len(replays),
        "output_token_ids_sha256": next(iter(output_token_hashes), None),
        "text_sha256": next(iter(text_hashes), None),
        "finish_reason": next(iter(finish_reasons), None),
        "request_contract": request_contract,
        "output_contract": output_contract,
        "cache_contract": cache_contract,
        "terminal_clean": terminal_clean,
        "contract_passed": (
            request_contract and output_contract and cache_contract and terminal_clean
        ),
    }


def _paired_ratio(candidate: dict[str, Any], baseline: dict[str, Any], field: str) -> float:
    denominator = float(baseline[field])
    if denominator <= 0:
        raise BenchmarkError(f"baseline {field} must be positive")
    return float(candidate[field]) / denominator


def _stratum_medians(pairs: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        order: statistics.median(float(pair[field]) for pair in pairs if pair["order"] == order)
        for order in ("baseline-first", "candidate-first")
    }


def summarize_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        replicate = int(row["replicate"])
        lane = str(row["lane"])
        if lane not in {"baseline", "candidate"} or lane in grouped.setdefault(replicate, {}):
            raise BenchmarkError("matrix repeats or mislabels a lane")
        grouped[replicate][lane] = row
    if set(grouped) != set(range(1, REQUIRED_PAIRS + 1)) or any(
        set(pair) != {"baseline", "candidate"} for pair in grouped.values()
    ):
        raise BenchmarkError("matrix requires five complete pairs")

    stable_fields = ("plan_sha256", "workload_sha256", "model", "max_decode_batch")
    for field in stable_fields:
        if len({str(row[field]) for row in rows}) != 1:
            label = "plan" if field == "plan_sha256" else field
            raise BenchmarkError(f"matrix {label} differs across lanes")

    pairs: list[dict[str, Any]] = []
    for replicate in sorted(grouped):
        baseline = grouped[replicate]["baseline"]
        candidate = grouped[replicate]["candidate"]
        if baseline["order"] != candidate["order"]:
            raise BenchmarkError("paired lanes disagree on execution order")
        baseline_memory = float(baseline["peak_mlx_memory_gb"])
        candidate_memory = float(candidate["peak_mlx_memory_gb"])
        pairs.append(
            {
                "replicate": replicate,
                "order": str(baseline["order"]),
                "memory_reduction_gb": baseline_memory - candidate_memory,
                "memory_reduction_percent": 100.0 * (1.0 - candidate_memory / baseline_memory),
                "aggregate_replay_tps_ratio": _paired_ratio(
                    candidate, baseline, "aggregate_replay_tps"
                ),
                "replay_p95_ttft_ratio": _paired_ratio(
                    candidate, baseline, "replay_p95_ttft_seconds"
                ),
                "replay_p95_latency_ratio": _paired_ratio(
                    candidate, baseline, "replay_p95_latency_seconds"
                ),
                "replay_max_latency_ratio": _paired_ratio(
                    candidate, baseline, "replay_max_latency_seconds"
                ),
            }
        )

    throughput_strata = _stratum_medians(pairs, "aggregate_replay_tps_ratio")
    ttft_strata = _stratum_medians(pairs, "replay_p95_ttft_ratio")
    latency_strata = _stratum_medians(pairs, "replay_p95_latency_ratio")
    max_latency_strata = _stratum_medians(pairs, "replay_max_latency_ratio")
    memory_percent = statistics.median(float(pair["memory_reduction_percent"]) for pair in pairs)
    memory_gb = statistics.median(float(pair["memory_reduction_gb"]) for pair in pairs)
    throughput = statistics.median(float(pair["aggregate_replay_tps_ratio"]) for pair in pairs)
    ttft = statistics.median(float(pair["replay_p95_ttft_ratio"]) for pair in pairs)
    latency = statistics.median(float(pair["replay_p95_latency_ratio"]) for pair in pairs)
    max_latency = statistics.median(float(pair["replay_max_latency_ratio"]) for pair in pairs)

    contracts = all(bool(row["contract_passed"]) for row in rows)
    cap_observed = all(
        int(row["configured_max_active_requests"]) == CANDIDATE_ACTIVE_CAP
        and int(row["observed_peak_active_cache_equivalents"]) <= CANDIDATE_ACTIVE_CAP
        for row in rows
        if row["lane"] == "candidate"
    )
    memory_repeatable = (
        sum(
            float(pair["memory_reduction_percent"]) >= MEMORY_REDUCTION_PERCENT_GATE
            and float(pair["memory_reduction_gb"]) >= MEMORY_REDUCTION_GB_GATE
            for pair in pairs
        )
        >= 4
    )
    gates = {
        "all_request_output_cache_and_cleanup_contracts": contracts,
        "candidate_active_cap_observed": cap_observed,
        "peak_mlx_reduction_at_least_10_percent": (
            memory_percent >= MEMORY_REDUCTION_PERCENT_GATE and memory_repeatable
        ),
        "peak_mlx_reduction_at_least_1_gb": (
            memory_gb >= MEMORY_REDUCTION_GB_GATE and memory_repeatable
        ),
        "aggregate_replay_tps_no_regression_3_percent": (
            throughput >= THROUGHPUT_RATIO_FLOOR
            and min(throughput_strata.values()) >= THROUGHPUT_RATIO_FLOOR
        ),
        "replay_p95_ttft_no_regression_3_percent": (
            ttft <= NO_REGRESSION_RATIO and max(ttft_strata.values()) <= NO_REGRESSION_RATIO
        ),
        "replay_p95_latency_no_regression_3_percent": (
            latency <= NO_REGRESSION_RATIO and max(latency_strata.values()) <= NO_REGRESSION_RATIO
        ),
        "replay_max_latency_no_regression_3_percent": (
            max_latency <= NO_REGRESSION_RATIO
            and max(max_latency_strata.values()) <= NO_REGRESSION_RATIO
        ),
    }
    lane_statistics: dict[str, dict[str, float | int]] = {}
    for lane in ("baseline", "candidate"):
        lane_rows = [row for row in rows if row["lane"] == lane]
        lane_statistics[lane] = {
            "peak_mlx_memory_gb_median": statistics.median(
                float(row["peak_mlx_memory_gb"]) for row in lane_rows
            ),
            "aggregate_replay_tps_median": statistics.median(
                float(row["aggregate_replay_tps"]) for row in lane_rows
            ),
            "replay_p95_ttft_seconds_median": statistics.median(
                float(row["replay_p95_ttft_seconds"]) for row in lane_rows
            ),
            "replay_p95_latency_seconds_median": statistics.median(
                float(row["replay_p95_latency_seconds"]) for row in lane_rows
            ),
            "replay_max_latency_seconds_median": statistics.median(
                float(row["replay_max_latency_seconds"]) for row in lane_rows
            ),
            "active_cache_equivalents_max": max(
                int(row["observed_peak_active_cache_equivalents"]) for row in lane_rows
            ),
        }
    return {
        "gates": gates,
        "decision": "screen-passed" if all(gates.values()) else "screen-rejected",
        "lanes": lane_statistics,
        "paired": {
            "pairs": pairs,
            "memory_reduction_percent_median": memory_percent,
            "memory_reduction_gb_median": memory_gb,
            "aggregate_replay_tps_ratio_median": throughput,
            "replay_p95_ttft_ratio_median": ttft,
            "replay_p95_latency_ratio_median": latency,
            "replay_max_latency_ratio_median": max_latency,
            "order_strata": {
                "aggregate_replay_tps_ratio": throughput_strata,
                "replay_p95_ttft_ratio": ttft_strata,
                "replay_p95_latency_ratio": latency_strata,
                "replay_max_latency_ratio": max_latency_strata,
            },
        },
    }


def compact_cancellation(payload: dict[str, Any]) -> dict[str, Any]:
    events = {str(event.get("key")): event for event in payload["result"]["events"]}
    cancelled = events.get("long-primary", {})
    follow_up = events.get("cancel-follow-up", {})
    response = follow_up.get("response")
    timeline = follow_up.get("timeline")
    final = payload["resources"]["engine_lifecycle_sampling"]["final"]
    prefix = final["prefix_cache"]
    terminal_clean = all(
        int(final.get(field, -1)) == expected
        for field, expected in (
            ("active_requests", 0),
            ("pending_requests", 0),
            ("prefill_requests", 0),
            ("decode_requests", 0),
            ("failed_requests", 0),
            ("cancelled_requests", 1),
            ("completed_requests", 1),
        )
    ) and all(int(prefix.get(field, -1)) == 0 for field in ("pinned_entries", "pinned_bytes"))
    follow_up_complete = (
        follow_up.get("error") is None
        and isinstance(response, dict)
        and isinstance(timeline, dict)
        and response.get("finish_reason") == "length"
        and int(response.get("completion_tokens", -1)) == 8
        and int(timeline.get("decode_steps", -1)) == 8
    )
    cancellation_accepted = bool(payload["result"].get("cancel_accepted"))
    target_cancelled = cancelled.get("error", {}).get("code") == "request_cancelled"
    configured_cap = int(payload["execution"]["max_active_requests"])
    swap_delta_bytes = int(payload["resources"]["swap_delta_bytes"])
    return {
        "configured_max_active_requests": configured_cap,
        "cancel_accepted": cancellation_accepted,
        "target_cancelled": target_cancelled,
        "follow_up_complete": follow_up_complete,
        "terminal_clean": terminal_clean,
        "swap_delta_bytes": swap_delta_bytes,
        "passed": (
            configured_cap == CANDIDATE_ACTIVE_CAP
            and cancellation_accepted
            and target_cancelled
            and follow_up_complete
            and terminal_clean
            and swap_delta_bytes == 0
        ),
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


def build_retained_summary(
    paths: list[Path], *, cancellation_path: Path | None = None
) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    rows = [
        compact_result(
            payload,
            lane=str(payload["experiment"]["lane"]),
            replicate=int(payload["experiment"]["replicate"]),
            order=str(payload["experiment"]["order"]),
        )
        for payload in payloads
    ]
    matrix = summarize_matrix(rows)
    cancellation = (
        compact_cancellation(json.loads(cancellation_path.read_text()))
        if cancellation_path is not None
        else None
    )
    if matrix["decision"] == "screen-passed" and cancellation is None:
        raise BenchmarkError("passing matrix requires cancellation evidence")
    post_pass_gate = bool(cancellation is not None and cancellation["passed"])
    source_paths = (
        Path("scripts/dev/benchmark_exact_prefix_cohort_cap.py"),
        Path("scripts/dev/public_arrival_load.py"),
        Path("tests/test_exact_prefix_cohort_cap_benchmark.py"),
        Path("tests/test_public_arrival_load.py"),
    )
    summary = {
        "schema_version": 1,
        "kind": "exact-prefix-active-cohort-cap-summary",
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline_commit": _git_head(),
        "candidate": {
            "arrival_plan": "unchanged shared-prefix B8",
            "baseline_max_active_requests": 16,
            "candidate_max_active_requests": CANDIDATE_ACTIVE_CAP,
            "max_decode_batch_both_lanes": 4,
            "production_routing_changed": False,
        },
        "references": {
            "feather": "https://arxiv.org/abs/2605.06046v1",
            "vllm_metal_main": "b6e35b6c642162dbf6f31009b81635426a91b64a",
            "sglang_main": "e1964da451ef9fbec04b326c729916281f90809b",
        },
        "rows": rows,
        "matrix": matrix,
        "post_pass": {"candidate_cancellation_cleanup": cancellation},
        "raw_sha256": {path.name: _sha256_file(path) for path in paths},
        "source_sha256": {str(path): _sha256_file(PROJECT_ROOT / path) for path in source_paths},
        "decision": {
            "status": (
                "screen-passed"
                if matrix["decision"] == "screen-passed" and post_pass_gate
                else "screen-rejected"
            ),
            "change_production_defaults": False,
            "run_conditional_scheduler_implementation": (
                matrix["decision"] == "screen-passed" and post_pass_gate
            ),
        },
    }
    if cancellation_path is not None:
        summary["raw_sha256"][cancellation_path.name] = _sha256_file(cancellation_path)
    return summary


async def _run_lane(args: argparse.Namespace) -> dict[str, Any]:
    workload = arrival._load_workload(args.workload)
    plan = arrival.build_arrival_plan(
        workload,
        scenario="shared-prefix",
        concurrency=CONCURRENCY,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        stagger_delay_seconds=0.0,
    )
    active_cap = None if args.lane == "baseline" else CANDIDATE_ACTIVE_CAP
    payload = await arrival.run_public_arrival_baseline(
        config_path=args.config.resolve(),
        workload_path=args.workload.resolve(),
        lock_path=args.lock.resolve(),
        data_root=args.data_root.resolve(),
        plan=plan,
        prefix_cache_enabled=True,
        timeout_seconds=args.timeout_seconds,
        decode_active_prefill_token_budget=DECODE_ACTIVE_PREFILL_TOKEN_BUDGET,
        snapshot_budget_bytes=SNAPSHOT_BUDGET_BYTES,
        snapshot_max_entries=SNAPSHOT_MAX_ENTRIES,
        snapshot_reservation_trace_max_events=SNAPSHOT_TRACE_EVENTS,
        max_active_requests=active_cap,
        sample_engine_lifecycle=True,
        engine_lifecycle_sample_interval_seconds=0.05,
    )
    payload["experiment"] = {
        "kind": "exact-prefix-active-cohort-cap",
        "lane": args.lane,
        "replicate": args.replicate,
        "order": args.order,
        "compact": compact_result(
            payload,
            lane=args.lane,
            replicate=args.replicate,
            order=args.order,
        ),
    }
    return payload


def _write_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--lane", choices=("baseline", "candidate"), required=True)
    run_parser.add_argument("--replicate", type=int, required=True)
    run_parser.add_argument("--order", choices=("baseline-first", "candidate-first"), required=True)
    run_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    run_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    run_parser.add_argument("--lock", type=Path, default=arrival.public.DEFAULT_LOCK_PATH)
    run_parser.add_argument("--data-root", type=Path, default=arrival.public.DEFAULT_DATA_ROOT)
    run_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    run_parser.add_argument("--output", type=Path, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("inputs", nargs="+", type=Path)
    summary_parser.add_argument("--cancellation", type=Path)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        payload = asyncio.run(_run_lane(args))
        _write_payload(payload, args.output)
        print(json.dumps(payload["experiment"]["compact"], indent=2, sort_keys=True))
        return

    summary = build_retained_summary(args.inputs, cancellation_path=args.cancellation)
    _write_payload(summary, args.output)
    print(json.dumps(summary["matrix"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
