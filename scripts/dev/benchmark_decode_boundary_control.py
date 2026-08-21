#!/usr/bin/env python3
"""Quantify fresh-process control variance at the common decode boundary."""

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
OBSERVER_PATH = PROJECT_ROOT / "scripts/dev/benchmark_decode_observer.py"
DEFAULT_ITERATION = "ITER-20260823-095-decode-boundary-control"
CELLS = ("b4-short", "b4-mixed")
ENGINES = ("aster", "mlx-lm")
STATES = ("observer-off", "control-off")
PRIMARY_METRIC = "decode_driver_tps"
CONTROL_THRESHOLD = 0.01


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_foundation() -> ModuleType:
    return _load_module(FOUNDATION_PATH, "benchmark_foundation_parity_for_control")


def load_observer() -> ModuleType:
    return _load_module(OBSERVER_PATH, "benchmark_decode_observer_for_control")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_order(cell: str, repetition: int) -> tuple[str, str]:
    """Alternate the baseline/control process position within each cell."""

    if cell not in CELLS or repetition < 1:
        raise ValueError("invalid control cell or repetition")
    control_first = (repetition + CELLS.index(cell)) % 2 == 0
    return STATES if control_first else ("control-off", "observer-off")


def _runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.off_config,
        workload=args.workload,
        lock=args.lock,
        data_root=args.data_root,
        timeout_seconds=args.timeout_seconds,
        memory_sample_interval=args.memory_sample_interval,
        max_output_tokens=args.max_output_tokens,
    )


def _unwrap(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    return result if isinstance(result, dict) else row


def _index_base(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        # The retained observer matrix contains both off and on rows; the
        # control comparison is anchored exclusively to its observer-off rows.
        if row.get("state") not in (None, "observer-off"):
            continue
        result = _unwrap(row)
        key = (str(result["cell"]), int(result["repetition"]), str(result["engine"]))
        if key in indexed:
            raise ValueError(f"duplicate baseline row {key}")
        indexed[key] = result
    return indexed


def _index_control(rows: list[dict[str, Any]], repetitions: int) -> dict[tuple[str, int, str], dict[str, Any]]:
    expected = {
        (cell, repetition, engine)
        for cell in CELLS
        for repetition in range(1, repetitions + 1)
        for engine in ENGINES
    }
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("state")) != "control-off":
            raise ValueError("control rows must be labeled control-off")
        key = (str(row["cell"]), int(row["repetition"]), str(row["engine"]))
        if key in indexed:
            raise ValueError(f"duplicate control row {key}")
        indexed[key] = _unwrap(row)
    if set(indexed) != expected:
        raise ValueError(f"expected {len(expected)} control rows")
    return indexed


def _relative(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0 if baseline else 0.0


def _source_comparable(rows: list[dict[str, Any]]) -> bool:
    fields = (
        "workload_sha256",
        "source_lock_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "common_source_sha256",
    )
    return all(len({str(_unwrap(row)["source"].get(field)) for row in rows}) == 1 for field in fields)


def _engine_source_comparable(rows: list[dict[str, Any]]) -> bool:
    return all(
        len(
            {
                str(_unwrap(row)["source"].get("engine_source_sha256"))
                for row in rows
                if str(_unwrap(row).get("engine")) == engine
            }
        )
        == 1
        for engine in ENGINES
    )


def _output_fingerprint(foundation: ModuleType, row: dict[str, Any]) -> Any:
    return foundation._request_output_fingerprint(_unwrap(row))


def _terminal_fingerprint(foundation: ModuleType, row: dict[str, Any]) -> Any:
    return foundation._request_terminal_fingerprint(_unwrap(row))


def _zero_fallbacks(row: dict[str, Any]) -> bool:
    lifecycle = _unwrap(row).get("lifecycle", {})
    diagnostics = lifecycle.get("decode_batch_diagnostics", {})
    return int(diagnostics.get("batch_fallbacks", 0)) == 0


def _state_first(row: dict[str, Any]) -> str:
    value = row.get("state_first")
    if value in STATES:
        return str(value)
    return control_order(str(row["cell"]), int(row["repetition"]))[0]


def _median_strata(values: list[dict[str, Any]], metric: str) -> dict[str, float]:
    return {
        first: float(
            statistics.median(
                float(item["relative_delta"][metric])
                for item in values
                if item["state_first"] == first
            )
        )
        for first in STATES
    }


def _observer_effect(
    foundation: ModuleType,
    baseline_rows: list[dict[str, Any]],
    *,
    repetitions: int,
) -> dict[str, Any]:
    """Extract off/on effects from the retained observer matrix when present."""

    wrapped = [row for row in baseline_rows if row.get("state") in ("observer-off", "observer-on")]
    if not wrapped:
        return {}
    indexed = {
        (str(row["cell"]), int(row["repetition"]), str(row["engine"]), str(row["state"])): row
        for row in wrapped
    }
    metrics = tuple(load_observer().METRICS)
    effects: dict[str, Any] = {}
    for cell in CELLS:
        effects[cell] = {}
        for engine in ENGINES:
            values: list[dict[str, Any]] = []
            for repetition in range(1, repetitions + 1):
                off = indexed[(cell, repetition, engine, "observer-off")]
                on = indexed[(cell, repetition, engine, "observer-on")]
                off_result = _unwrap(off)
                on_result = _unwrap(on)
                values.append(
                    {
                        "repetition": repetition,
                        "state_first": load_observer().state_order(cell, repetition)[0],
                        "relative_delta": {
                            metric: _relative(
                                float(on_result["metrics"][metric]),
                                float(off_result["metrics"][metric]),
                            )
                            for metric in metrics
                        },
                    }
                )
            effects[cell][engine] = {
                "paired_deltas": values,
                "order_strata": {
                    first: {
                        metric: float(
                            statistics.median(
                                float(item["relative_delta"][metric])
                                for item in values
                                if item["state_first"] == first
                            )
                        )
                        for metric in metrics
                    }
                    for first in ("observer-off", "observer-on")
                },
            }
    return effects


def summarize_control(
    foundation: ModuleType,
    baseline_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    *,
    repetitions: int,
    expected_max_output_tokens: int,
) -> dict[str, Any]:
    """Validate off/off controls and separate control noise from observer deltas."""

    if repetitions < 2 or repetitions % 2:
        raise ValueError("control repetitions must be an even count of at least two")
    expected = {
        (cell, repetition, engine)
        for cell in CELLS
        for repetition in range(1, repetitions + 1)
        for engine in ENGINES
    }
    baseline_index = _index_base(baseline_rows)
    control_index = _index_control(control_rows, repetitions)
    if not expected.issubset(set(baseline_index)):
        raise ValueError(f"expected baseline rows for {len(expected)} controls")

    metrics = tuple(load_observer().METRICS)
    exact_output = True
    terminal_clean = True
    zero_fallbacks = True
    cap_ok = True
    contract_ok = True
    prewarm_ok = True
    comparable_rows: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, list[dict[str, Any]]]] = {
        cell: {engine: [] for engine in ENGINES} for cell in CELLS
    }
    for cell, repetition, engine in sorted(expected):
        baseline = baseline_index[(cell, repetition, engine)]
        control = control_index[(cell, repetition, engine)]
        comparable_rows.extend((baseline, control))
        exact_output = exact_output and _output_fingerprint(foundation, baseline) == _output_fingerprint(
            foundation, control
        )
        terminal_clean = terminal_clean and bool(baseline.get("lifecycle", {}).get("terminal_clean")) and bool(
            control.get("lifecycle", {}).get("terminal_clean")
        )
        zero_fallbacks = zero_fallbacks and _zero_fallbacks(baseline) and _zero_fallbacks(control)
        cap_ok = cap_ok and int(control.get("execution", {}).get("max_output_tokens", -1)) == expected_max_output_tokens
        contract_ok = contract_ok and bool(control.get("contract", {}).get("passed", False))
        prewarm_ok = prewarm_ok and int(control.get("execution", {}).get("warmup_requests", 0)) > 0
        if baseline.get("plan_sha256") != control.get("plan_sha256"):
            raise ValueError(f"control plan differs for {cell}/{repetition}/{engine}")
        if baseline.get("input_manifest_sha256") != control.get("input_manifest_sha256"):
            raise ValueError(f"control input differs for {cell}/{repetition}/{engine}")
        deltas[cell][engine].append(
            {
                "repetition": repetition,
                "state_first": _state_first(control_rows[
                    next(
                        index
                        for index, item in enumerate(control_rows)
                        if str(item["cell"]) == cell
                        and int(item["repetition"]) == repetition
                        and str(item["engine"]) == engine
                    )
                ]),
                "relative_delta": {
                    metric: _relative(
                        float(control["metrics"][metric]), float(baseline["metrics"][metric])
                    )
                    for metric in metrics
                },
            }
        )

    if not cap_ok:
        raise ValueError("control output cap differs from the requested contract")

    source_comparable = _source_comparable(comparable_rows) and _engine_source_comparable(comparable_rows)
    order_strata: dict[str, Any] = {}
    medians: dict[str, Any] = {}
    control_stable = True
    for cell in CELLS:
        order_strata[cell] = {}
        medians[cell] = {}
        for engine in ENGINES:
            values = deltas[cell][engine]
            order_strata[cell][engine] = _median_strata(values, PRIMARY_METRIC)
            medians[cell][engine] = {
                metric: float(
                    statistics.median(float(item["relative_delta"][metric]) for item in values)
                )
                for metric in metrics
            }
            control_stable = control_stable and all(
                abs(value) <= CONTROL_THRESHOLD
                for value in order_strata[cell][engine].values()
            )

    structural = (
        source_comparable
        and exact_output
        and terminal_clean
        and zero_fallbacks
        and cap_ok
        and contract_ok
        and prewarm_ok
    )
    observer_effect = _observer_effect(foundation, baseline_rows, repetitions=repetitions)
    return {
        "measurement_status": "valid" if structural else "invalid-contract",
        "candidate_admitted": False,
        "decision": (
            "classify-boundary-no-production-change"
            if control_stable and structural
            else "reject-control-variance"
        ),
        "primary_metric": PRIMARY_METRIC,
        "control_threshold_ratio": CONTROL_THRESHOLD,
        "control_contract": structural,
        "source_comparable": source_comparable,
        "exact_output_identity_off_vs_control": exact_output,
        "terminal_clean": terminal_clean,
        "zero_decode_fallbacks": zero_fallbacks,
        "prewarm_contract": prewarm_ok,
        "control_stable_primary_and_strata": control_stable,
        "measurement_confounded_by_control_variance": not control_stable,
        "control_medians": medians,
        "control_order_strata": order_strata,
        "control_deltas": deltas,
        "observer_effect_from_retained_matrix": observer_effect,
    }


def _run_control_row(
    foundation: ModuleType,
    args: argparse.Namespace,
    *,
    cell: str,
    repetition: int,
    engine: str,
    pair_order: tuple[str, str],
    state_first: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = args.run_root / f"r{repetition}-{cell}-control-off-{engine}.json"
    command = foundation._cell_command(
        _runner_args(args),
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
            f"control row failed: {cell}/{repetition}/control-off/{engine}\n"
            f"{completed.stderr[-4000:]}"
        )
    result = json.loads(output.read_text())
    row = {
        "cell": cell,
        "repetition": repetition,
        "state": "control-off",
        "engine": engine,
        "state_first": state_first,
        "control_protocol": {
            "config_state": "observer-off",
            "process_isolation": "fresh-process",
            "prewarm": "foundation-declared-warmup",
            "order": "alternating-observer-off-control-off",
        },
        "source_path": str(output),
        "source_file_sha256": sha256(output),
        "result": result,
    }
    return row, {
        "cell": cell,
        "repetition": repetition,
        "state": "control-off",
        "engine": engine,
        "status": completed.returncode,
        "source_path": str(output),
    }


def _load_control_row(args: argparse.Namespace, *, cell: str, repetition: int, engine: str, state_first: str) -> tuple[dict[str, Any], dict[str, Any]]:
    output = args.run_root / f"r{repetition}-{cell}-control-off-{engine}.json"
    result = json.loads(output.read_text())
    row = {
        "cell": cell,
        "repetition": repetition,
        "state": "control-off",
        "engine": engine,
        "state_first": state_first,
        "control_protocol": {
            "config_state": "observer-off",
            "process_isolation": "fresh-process",
            "prewarm": "foundation-declared-warmup",
            "order": "alternating-observer-off-control-off",
        },
        "source_path": str(output),
        "source_file_sha256": sha256(output),
        "result": result,
    }
    return row, {"cell": cell, "repetition": repetition, "state": "control-off", "engine": engine, "status": 0, "source_path": str(output)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    foundation = load_foundation()
    observer = load_observer()
    observer_payload = json.loads(args.observer_matrix.read_text())
    baseline_rows = observer_payload["rows"]
    expected_cap = int(observer_payload["execution"]["max_output_tokens"])
    if expected_cap != args.max_output_tokens:
        raise ValueError("observer matrix output cap differs from control cap")
    if int(observer_payload["execution"]["repetitions"]) != args.repetitions:
        raise ValueError("observer matrix repetitions differ from control repetitions")
    observer.summarize(
        foundation,
        baseline_rows,
        observer_payload["execution"]["collection_statuses"],
        repetitions=args.repetitions,
        expected_sample_interval=int(observer_payload["execution"]["expected_sample_interval"]),
        expected_max_output_tokens=args.max_output_tokens,
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    collection: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for cell in CELLS:
            pair_order = foundation.engine_order_for_pair(cell, repetition)
            state_first, _ = control_order(cell, repetition)
            for engine in pair_order:
                if args.reuse_existing:
                    row, record = _load_control_row(
                        args,
                        cell=cell,
                        repetition=repetition,
                        engine=engine,
                        state_first=state_first,
                    )
                else:
                    row, record = _run_control_row(
                        foundation,
                        args,
                        cell=cell,
                        repetition=repetition,
                        engine=engine,
                        pair_order=pair_order,
                        state_first=state_first,
                    )
                rows.append(row)
                collection.append(record)
                if not args.reuse_existing:
                    time.sleep(args.cooldown_seconds)
    summary = summarize_control(
        foundation,
        baseline_rows,
        rows,
        repetitions=args.repetitions,
        expected_max_output_tokens=args.max_output_tokens,
    )
    first = _unwrap(rows[0])
    return {
        "schema_version": 1,
        "kind": "decode-boundary-control-evidence",
        "iteration": args.iteration,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_matrix_path": str(args.observer_matrix),
        "baseline_matrix_sha256": sha256(args.observer_matrix),
        "execution": {
            "cells": list(CELLS),
            "engines": list(ENGINES),
            "states": list(STATES),
            "repetitions": args.repetitions,
            "max_output_tokens": args.max_output_tokens,
            "observer_matrix_expected_sample_interval": int(
                observer_payload["execution"]["expected_sample_interval"]
            ),
            "control_order": "alternating observer-off/control-off first",
            "process_isolation": "fresh-process",
            "prewarm": "foundation-declared-warmup",
            "off_config": str(args.off_config),
            "cooldown_seconds": args.cooldown_seconds,
            "collection_statuses": collection,
        },
        "source": first["source"],
        "summary": summary,
        "observer_rows": baseline_rows,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-matrix", type=Path, required=True)
    parser.add_argument("--off-config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--process-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02)
    parser.add_argument("--cooldown-seconds", type=float, default=1.0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", default=DEFAULT_ITERATION)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    for name in ("observer_matrix", "off_config", "workload", "lock", "data_root", "run_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.max_output_tokens <= 0 or args.repetitions < 2 or args.repetitions % 2:
        raise ValueError("max output tokens must be positive and repetitions must be even")
    args.fingerprint = {"model_sha256": args.model_sha256, "tokenizer_sha256": args.tokenizer_sha256}
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
