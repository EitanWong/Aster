#!/usr/bin/env python3
"""Run an adjacent, state-balanced decode-observer matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUNDATION_PATH = PROJECT_ROOT / "scripts/dev/benchmark_foundation_parity.py"
DEFAULT_ITERATION = "ITER-20260821-093-low-overhead-decode-stage-attribution"
CELLS = ("b4-short", "b4-mixed")
ENGINES = ("aster", "mlx-lm")
STATES = ("observer-off", "observer-on")
METRICS = (
    "decode_driver_tps",
    "decode_driver_seconds",
    "aggregate_generation_tps",
    "ttft_p95_seconds",
    "end_to_end_p95_seconds",
    "peak_mlx_memory_gb",
    "peak_rss_bytes",
    "swap_delta_bytes",
)


def load_foundation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_foundation_parity", FOUNDATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FOUNDATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_order(cell: str, repetition: int) -> tuple[str, str]:
    if cell not in CELLS or repetition < 1:
        raise ValueError("invalid observer matrix cell or repetition")
    off_first = (repetition + CELLS.index(cell)) % 2 == 0
    return STATES if off_first else ("observer-on", "observer-off")


def _runner_args(args: argparse.Namespace, config: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        workload=args.workload,
        lock=args.lock,
        data_root=args.data_root,
        timeout_seconds=args.timeout_seconds,
        memory_sample_interval=args.memory_sample_interval,
        max_output_tokens=args.max_output_tokens,
    )


def _run_row(
    foundation: ModuleType,
    args: argparse.Namespace,
    *,
    state: str,
    cell: str,
    repetition: int,
    engine: str,
    pair_order: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = args.off_config if state == "observer-off" else args.on_config
    output = args.run_root / f"r{repetition}-{cell}-{state}-{engine}.json"
    command = foundation._cell_command(
        _runner_args(args, config),
        cell=cell,
        engine=engine,
        repetition=repetition,
        pair_order=pair_order,
        output=output,
        fingerprint=args.fingerprint,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=args.process_timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"observer row failed: {cell}/{repetition}/{state}/{engine}\n{completed.stderr[-4000:]}"
        )
    result = json.loads(output.read_text())
    return (
        {
            "cell": cell,
            "repetition": repetition,
            "state": state,
            "engine": engine,
            "source_path": str(output),
            "source_file_sha256": sha256(output),
            "result": result,
        },
        {
            "cell": cell,
            "repetition": repetition,
            "state": state,
            "engine": engine,
            "status": completed.returncode,
        },
    )


def _load_row(
    args: argparse.Namespace,
    *,
    state: str,
    cell: str,
    repetition: int,
    engine: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = args.run_root / f"r{repetition}-{cell}-{state}-{engine}.json"
    result = json.loads(output.read_text())
    return (
        {
            "cell": cell,
            "repetition": repetition,
            "state": state,
            "engine": engine,
            "source_path": str(output),
            "source_file_sha256": sha256(output),
            "result": result,
        },
        {
            "cell": cell,
            "repetition": repetition,
            "state": state,
            "engine": engine,
            "status": 0,
        },
    )


def _relative(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0 if baseline else 0.0


def _output_fingerprint(foundation: ModuleType, row: dict[str, Any]) -> Any:
    return foundation._request_output_fingerprint(row["result"])


def summarize(
    foundation: ModuleType,
    rows: list[dict[str, Any]],
    collection: list[dict[str, Any]],
    *,
    repetitions: int,
    expected_sample_interval: int,
    expected_max_output_tokens: int = 8,
) -> dict[str, Any]:
    expected = {
        (cell, repetition, state, engine)
        for cell in CELLS
        for repetition in range(1, repetitions + 1)
        for state in STATES
        for engine in ENGINES
    }
    indexed = {
        (row["cell"], int(row["repetition"]), row["state"], row["engine"]): row for row in rows
    }
    if set(indexed) != expected:
        raise ValueError(f"expected {len(expected)} observer rows")
    if any(entry["status"] != 0 for entry in collection):
        raise ValueError("observer matrix contains a failed process")
    if any(
        int(row["result"]["execution"].get("max_output_tokens", -1))
        != expected_max_output_tokens
        for row in rows
    ):
        raise ValueError("observer matrix output cap differs from the requested contract")

    source_fields = (
        "workload_sha256",
        "source_lock_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "common_source_sha256",
    )
    source_comparable = all(
        len({str(row["result"]["source"].get(field)) for row in rows}) == 1
        for field in source_fields
    )
    engine_sources_comparable = all(
        len(
            {
                str(row["result"]["source"].get("engine_source_sha256"))
                for row in rows
                if row["engine"] == engine
            }
        )
        == 1
        for engine in ENGINES
    )
    cell_summaries: dict[str, Any] = {}
    exact_output = True
    terminal_clean = True
    zero_fallbacks = True
    for cell in CELLS:
        state_engine: dict[str, dict[str, list[dict[str, Any]]]] = {
            state: {
                engine: [
                    indexed[(cell, repetition, state, engine)]
                    for repetition in range(1, repetitions + 1)
                ]
                for engine in ENGINES
            }
            for state in STATES
        }
        medians = {
            state: {
                engine: {
                    metric: statistics.median(
                        float(row["result"]["metrics"][metric])
                        for row in state_engine[state][engine]
                    )
                    for metric in METRICS
                }
                for engine in ENGINES
            }
            for state in STATES
        }
        paired: dict[str, list[dict[str, Any]]] = {engine: [] for engine in ENGINES}
        for repetition in range(1, repetitions + 1):
            for engine in ENGINES:
                off = indexed[(cell, repetition, "observer-off", engine)]
                on = indexed[(cell, repetition, "observer-on", engine)]
                off_result = off["result"]
                on_result = on["result"]
                exact_output = exact_output and _output_fingerprint(
                    foundation, off
                ) == _output_fingerprint(foundation, on)
                terminal_clean = (
                    terminal_clean
                    and bool(off_result["lifecycle"]["terminal_clean"])
                    and bool(on_result["lifecycle"]["terminal_clean"])
                )
                zero_fallbacks = zero_fallbacks and all(
                    int(
                        result["lifecycle"]
                        .get("decode_batch_diagnostics", {})
                        .get("batch_fallbacks", 0)
                    )
                    == 0
                    for result in (off_result, on_result)
                )
                paired[engine].append(
                    {
                        "repetition": repetition,
                        "state_first": state_order(cell, repetition)[0],
                        "relative_delta": {
                            metric: _relative(
                                float(on_result["metrics"][metric]),
                                float(off_result["metrics"][metric]),
                            )
                            for metric in METRICS
                        },
                    }
                )
        state_observer: dict[str, Any] = {}
        for state in STATES:
            observer_rows = [
                row["result"]["lifecycle"]["decode_stage_observer"]
                for row in state_engine[state]["aster"]
            ]
            state_observer[state] = {
                "sampled_steps": [int(observer["sampled_steps"]) for observer in observer_rows],
                "event_counts": [len(observer["events"]) for observer in observer_rows],
                "batch_steps": [int(observer["batch_steps"]) for observer in observer_rows],
                "single_steps": [int(observer["single_steps"]) for observer in observer_rows],
                "dropped_events": [int(observer["dropped_events"]) for observer in observer_rows],
            }
        cell_summaries[cell] = {
            "sample_count_per_state_per_engine": repetitions,
            "medians": medians,
            "paired_deltas": paired,
            "observer": state_observer,
            "state_order_balance": {
                state: sum(
                    state_order(cell, repetition)[0] == state
                    for repetition in range(1, repetitions + 1)
                )
                for state in STATES
            },
        }

    # The diagnostic no-op gate is intentionally absolute and applies to both
    # engines, so a host-side state effect cannot be mistaken for an Aster gain.
    no_op_metrics = METRICS[:-1]
    order_strata: dict[str, Any] = {}
    aster_no_op = True
    control_no_op = True
    for cell, summary in cell_summaries.items():
        order_strata[cell] = {}
        for engine in ENGINES:
            order_strata[cell][engine] = {}
            for metric in no_op_metrics:
                values = summary["paired_deltas"][engine]
                strata = {
                    first: statistics.median(
                        float(item["relative_delta"][metric])
                        for item in values
                        if item["state_first"] == first
                    )
                    for first in STATES
                }
                order_strata[cell][engine][metric] = strata
                if engine == "aster":
                    aster_no_op = aster_no_op and all(
                        abs(value) < 0.01 for value in strata.values()
                    )
                else:
                    control_no_op = control_no_op and all(
                        abs(value) < 0.01 for value in strata.values()
                    )

    observer_rows = [
        row["result"]["lifecycle"]["decode_stage_observer"]
        for row in rows
        if row["engine"] == "aster" and row["state"] == "observer-on"
    ]
    observer_contract = all(
        (
            int(row["result"]["execution"]["decode_stage_observer_max_events"]) == 0
            if row["state"] == "observer-off"
            else (
                int(row["result"]["execution"]["decode_stage_observer_max_events"]) > 0
                and int(row["result"]["execution"]["decode_stage_observer_sample_interval"])
                == expected_sample_interval
            )
        )
        for row in rows
        if row["engine"] == "aster"
    )
    observer_bounded = all(len(observer["events"]) <= 64 for observer in observer_rows)
    observer_dropped = all(int(observer["dropped_events"]) == 0 for observer in observer_rows)
    observer_off_empty = all(
        not row["result"]["lifecycle"]["decode_stage_observer"]["events"]
        for row in rows
        if row["engine"] == "aster" and row["state"] == "observer-off"
    )
    return {
        "measurement_status": "valid",
        "candidate_admitted": False,
        "decision": "reject-observer-unproven-or-noop-gate",
        "primary_metric": "decode_driver_tps",
        "no_op_threshold_ratio": 0.01,
        "source_comparable": source_comparable and engine_sources_comparable,
        "exact_output_identity_off_vs_on": exact_output,
        "terminal_clean": terminal_clean,
        "zero_decode_fallbacks": zero_fallbacks,
        "observer_contract": observer_contract,
        "observer_off_empty": observer_off_empty,
        "observer_on_bounded": observer_bounded,
        "observer_on_zero_drops": observer_dropped,
        "aster_no_op_gate_all_metrics_and_strata": aster_no_op,
        "control_engine_stable_all_metrics_and_strata": control_no_op,
        "measurement_confounded_by_control_variance": not control_no_op,
        "no_op_gate_all_metrics_and_strata": aster_no_op,
        "order_strata": order_strata,
        "cell_summaries": cell_summaries,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    foundation = load_foundation()
    args.run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    collection: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for cell in CELLS:
            pair_order = foundation.engine_order_for_pair(cell, repetition)
            for engine in pair_order:
                for state in state_order(cell, repetition):
                    if args.reuse_existing:
                        row, record = _load_row(
                            args,
                            state=state,
                            cell=cell,
                            repetition=repetition,
                            engine=engine,
                        )
                    else:
                        row, record = _run_row(
                            foundation,
                            args,
                            state=state,
                            cell=cell,
                            repetition=repetition,
                            engine=engine,
                            pair_order=pair_order,
                        )
                    rows.append(row)
                    collection.append(record)
                    if not args.reuse_existing:
                        time.sleep(args.cooldown_seconds)
    summary = summarize(
        foundation,
        rows,
        collection,
        repetitions=args.repetitions,
        expected_sample_interval=args.expected_sample_interval,
        expected_max_output_tokens=args.max_output_tokens,
    )
    first = rows[0]["result"]
    return {
        "schema_version": 1,
        "kind": "decode-stage-observer-sampled-matrix",
        "iteration": getattr(args, "iteration", DEFAULT_ITERATION),
        "created_at": datetime.now(UTC).isoformat(),
        "execution": {
            "cells": list(CELLS),
            "engines": list(ENGINES),
            "states": list(STATES),
            "repetitions": args.repetitions,
            "state_order": "alternating per cell/repetition",
            "collection_statuses": collection,
            "off_config": str(args.off_config),
            "on_config": str(args.on_config),
            "expected_sample_interval": args.expected_sample_interval,
            "max_output_tokens": args.max_output_tokens,
        },
        "source": first["source"],
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-config", type=Path, required=True)
    parser.add_argument("--on-config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--expected-sample-interval", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", default=DEFAULT_ITERATION)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    args.off_config = args.off_config.resolve()
    args.on_config = args.on_config.resolve()
    args.workload = args.workload.resolve()
    args.lock = args.lock.resolve()
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    args.output = args.output.resolve()
    if args.max_output_tokens <= 0:
        raise ValueError("max output tokens must be positive")
    args.fingerprint = {
        "model_sha256": args.model_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
    }
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
