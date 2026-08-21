#!/usr/bin/env python3
"""Quantify fresh-process control variance at the common decode boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
TELEMETRY_PATH = PROJECT_ROOT / "scripts/dev/host_state_telemetry.py"
DEFAULT_ITERATION = "ITER-20260824-096-host-state-trace"
CELLS = ("b4-short", "b4-mixed")
ENGINES = ("aster", "mlx-lm")
STATES = ("observer-off", "control-off")
PRIMARY_METRIC = "decode_driver_tps"
CONTROL_THRESHOLD = 0.01
HOST_STATE_FEATURES = (
    "child_cpu_percent_avg",
    "child_cpu_percent_max",
    "system_cpu_percent_avg",
    "system_cpu_percent_max",
    "system_available_memory_min_bytes",
    "system_available_memory_min_percent",
    "system_swap_used_max_bytes",
    "load_average_one_min_max",
    "peak_rss_bytes",
    "host_memory_available_before_bytes",
    "host_memory_available_after_bytes",
    "mlx_active_memory_before_bytes",
    "mlx_active_memory_after_bytes",
    "mlx_cache_memory_before_bytes",
    "mlx_cache_memory_after_bytes",
    "mlx_peak_memory_after_bytes",
)


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


def load_telemetry() -> ModuleType:
    return _load_module(TELEMETRY_PATH, "host_state_telemetry_for_control")


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
        indexed[key] = row
    return indexed


def _index_control(
    rows: list[dict[str, Any]], repetitions: int
) -> dict[tuple[str, int, str], dict[str, Any]]:
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
        indexed[key] = row
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
    return all(
        len({str(_unwrap(row)["source"].get(field)) for row in rows}) == 1 for field in fields
    )


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


def _telemetry_contract(row: dict[str, Any]) -> bool:
    telemetry = row.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("enabled") is not True:
        return False
    before = telemetry.get("host_state_before")
    after = telemetry.get("host_state_after")
    process = telemetry.get("process")
    thermal = telemetry.get("thermal_power")
    if not all(isinstance(value, dict) for value in (before, after, process, thermal)):
        return False
    for snapshot in (before, after):
        if snapshot.get("schema_version") != 1:
            return False
        system = snapshot.get("system")
        process_state = snapshot.get("process")
        if not isinstance(system, dict) or not isinstance(process_state, dict):
            return False
        if int(process_state.get("pid", 0)) <= 0:
            return False
        if int(system.get("memory_total_bytes", 0)) <= 0:
            return False
        if int(system.get("memory_available_bytes", 0)) <= 0:
            return False
        available_percent = float(system.get("memory_available_percent", -1.0))
        if not 0.0 < available_percent <= 100.0:
            return False
        if int(system.get("swap_used_bytes", -1)) < 0:
            return False
    if process.get("status") != "complete" or int(process.get("sample_count", 0)) < 1:
        return False
    if int(process.get("peak_rss_bytes", 0)) < int(process.get("rss_before_bytes", 0)):
        return False
    child_cpu_avg = float(process.get("cpu_percent_avg", -1.0))
    child_cpu_max = float(process.get("cpu_percent_max", -1.0))
    if child_cpu_avg < 0.0 or child_cpu_max < child_cpu_avg:
        return False
    system_cpu_avg = float(process.get("system_cpu_percent_avg", -1.0))
    system_cpu_max = float(process.get("system_cpu_percent_max", -1.0))
    if system_cpu_avg < 0.0 or system_cpu_max < system_cpu_avg:
        return False
    if int(process.get("system_available_memory_min_bytes", 0)) <= 0:
        return False
    if int(process.get("system_swap_used_max_bytes", -1)) < 0:
        return False
    available_percent = float(process.get("system_available_memory_min_percent", -1.0))
    if not 0.0 < available_percent <= 100.0:
        return False
    probes = thermal.get("probes")
    if thermal.get("schema_version") != 1 or not isinstance(probes, dict):
        return False
    if set(probes) != {"powermetrics", "pmset_thermal", "memory_pressure"}:
        return False
    return all(
        isinstance(probe, dict) and probe.get("status") in {"available", "unavailable"}
        for probe in probes.values()
    )


def _allocator_contract(row: dict[str, Any]) -> bool:
    lifecycle = _unwrap(row).get("lifecycle")
    if not isinstance(lifecycle, dict):
        return False
    allocator = lifecycle.get("mlx_allocator")
    if not isinstance(allocator, dict):
        return False
    for point in ("before_timed", "after_timed"):
        snapshot = allocator.get(point)
        if not isinstance(snapshot, dict):
            return False
        if any(
            int(snapshot.get(field, -1)) < 0
            for field in (
                "active_memory_bytes",
                "cache_memory_bytes",
                "peak_memory_bytes",
            )
        ):
            return False
    return True


def _host_state_features(row: dict[str, Any]) -> dict[str, float]:
    telemetry = row["telemetry"]
    process = telemetry["process"]
    host_before = telemetry["host_state_before"]["system"]
    host_after = telemetry["host_state_after"]["system"]
    allocator = _unwrap(row)["lifecycle"]["mlx_allocator"]
    before_allocator = allocator["before_timed"]
    after_allocator = allocator["after_timed"]
    return {
        "child_cpu_percent_avg": float(process["cpu_percent_avg"]),
        "child_cpu_percent_max": float(process["cpu_percent_max"]),
        "system_cpu_percent_avg": float(process["system_cpu_percent_avg"]),
        "system_cpu_percent_max": float(process["system_cpu_percent_max"]),
        "system_available_memory_min_bytes": float(process["system_available_memory_min_bytes"]),
        "system_available_memory_min_percent": float(
            process["system_available_memory_min_percent"]
        ),
        "system_swap_used_max_bytes": float(process["system_swap_used_max_bytes"]),
        "load_average_one_min_max": float(process.get("load_average_one_min_max") or 0.0),
        "peak_rss_bytes": float(process["peak_rss_bytes"]),
        "host_memory_available_before_bytes": float(host_before["memory_available_bytes"]),
        "host_memory_available_after_bytes": float(host_after["memory_available_bytes"]),
        "mlx_active_memory_before_bytes": float(before_allocator["active_memory_bytes"]),
        "mlx_active_memory_after_bytes": float(after_allocator["active_memory_bytes"]),
        "mlx_cache_memory_before_bytes": float(before_allocator["cache_memory_bytes"]),
        "mlx_cache_memory_after_bytes": float(after_allocator["cache_memory_bytes"]),
        "mlx_peak_memory_after_bytes": float(after_allocator["peak_memory_bytes"]),
    }


def _pearson(values: list[tuple[float, float]]) -> float | None:
    finite = [(x, y) for x, y in values if math.isfinite(x) and math.isfinite(y)]
    if len(finite) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in finite)
    mean_y = statistics.fmean(y for _, y in finite)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in finite)
    scale_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in finite))
    scale_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in finite))
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    return numerator / (scale_x * scale_y)


def _host_state_diagnostics(
    baseline_index: dict[tuple[str, int, str], dict[str, Any]],
    control_index: dict[tuple[str, int, str], dict[str, Any]],
    expected: set[tuple[str, int, str]],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    probe_status_counts: dict[str, dict[str, int]] = {}
    for key in sorted(expected):
        cell, repetition, engine = key
        baseline_row = baseline_index[key]
        control_row = control_index[key]
        baseline = _host_state_features(baseline_row)
        control = _host_state_features(control_row)
        decode_delta = _relative(
            float(_unwrap(control_row)["metrics"][PRIMARY_METRIC]),
            float(_unwrap(baseline_row)["metrics"][PRIMARY_METRIC]),
        )
        pairs.append(
            {
                "cell": cell,
                "repetition": repetition,
                "engine": engine,
                "state_first": _state_first(control_row),
                "decode_driver_tps_relative_delta": decode_delta,
                "baseline": baseline,
                "control": control,
                "control_minus_baseline": {
                    field: control[field] - baseline[field] for field in HOST_STATE_FEATURES
                },
            }
        )
        for row in (baseline_row, control_row):
            probes = row["telemetry"]["thermal_power"]["probes"]
            for name, probe in probes.items():
                status = str(probe["status"])
                counts = probe_status_counts.setdefault(str(name), {})
                counts[status] = counts.get(status, 0) + 1

    scopes = {
        "all": pairs,
        **{engine: [pair for pair in pairs if pair["engine"] == engine] for engine in ENGINES},
    }
    correlations: dict[str, Any] = {}
    for scope, scoped_pairs in scopes.items():
        correlations[scope] = {
            field: {
                "sample_count": len(scoped_pairs),
                "pearson_r": _pearson(
                    [
                        (
                            float(pair["control_minus_baseline"][field]),
                            float(pair["decode_driver_tps_relative_delta"]),
                        )
                        for pair in scoped_pairs
                    ]
                ),
            }
            for field in HOST_STATE_FEATURES
        }
    return {
        "status": "complete",
        "interpretation": "diagnostic-only-small-n-noncausal",
        "pair_count": len(pairs),
        "probe_status_counts": probe_status_counts,
        "correlations_to_decode_driver_tps_delta": correlations,
        "pairs": pairs,
    }


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
    if not wrapped or not any(row.get("state") == "observer-on" for row in wrapped):
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
    require_telemetry: bool = False,
    observer_reference_rows: list[dict[str, Any]] | None = None,
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
    zero_swap_growth = True
    cap_ok = True
    contract_ok = True
    prewarm_ok = True
    telemetry_ok = True
    allocator_ok = True
    comparable_rows: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, list[dict[str, Any]]]] = {
        cell: {engine: [] for engine in ENGINES} for cell in CELLS
    }
    for cell, repetition, engine in sorted(expected):
        baseline_row = baseline_index[(cell, repetition, engine)]
        baseline = _unwrap(baseline_row)
        control_row = control_index[(cell, repetition, engine)]
        control = _unwrap(control_row)
        comparable_rows.extend((baseline_row, control_row))
        exact_output = exact_output and _output_fingerprint(
            foundation, baseline
        ) == _output_fingerprint(foundation, control)
        terminal_clean = (
            terminal_clean
            and bool(baseline.get("lifecycle", {}).get("terminal_clean"))
            and bool(control.get("lifecycle", {}).get("terminal_clean"))
        )
        zero_fallbacks = zero_fallbacks and _zero_fallbacks(baseline) and _zero_fallbacks(control)
        zero_swap_growth = (
            zero_swap_growth
            and int(baseline.get("metrics", {}).get("swap_delta_bytes", -1)) == 0
            and int(control.get("metrics", {}).get("swap_delta_bytes", -1)) == 0
        )
        cap_ok = (
            cap_ok
            and int(baseline.get("execution", {}).get("max_output_tokens", -1))
            == expected_max_output_tokens
        )
        cap_ok = (
            cap_ok
            and int(control.get("execution", {}).get("max_output_tokens", -1))
            == expected_max_output_tokens
        )
        contract_ok = (
            contract_ok
            and bool(baseline.get("contract", {}).get("passed", False))
            and bool(control.get("contract", {}).get("passed", False))
        )
        prewarm_ok = (
            prewarm_ok
            and int(baseline.get("execution", {}).get("warmup_requests", 0)) > 0
            and int(control.get("execution", {}).get("warmup_requests", 0)) > 0
        )
        telemetry_ok = (
            telemetry_ok and _telemetry_contract(baseline_row) and _telemetry_contract(control_row)
        )
        allocator_ok = (
            allocator_ok and _allocator_contract(baseline_row) and _allocator_contract(control_row)
        )
        if baseline.get("plan_sha256") != control.get("plan_sha256"):
            raise ValueError(f"control plan differs for {cell}/{repetition}/{engine}")
        if baseline.get("input_manifest_sha256") != control.get("input_manifest_sha256"):
            raise ValueError(f"control input differs for {cell}/{repetition}/{engine}")
        deltas[cell][engine].append(
            {
                "repetition": repetition,
                "state_first": _state_first(control_row),
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

    source_comparable = _source_comparable(comparable_rows) and _engine_source_comparable(
        comparable_rows
    )
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
                abs(value) <= CONTROL_THRESHOLD for value in order_strata[cell][engine].values()
            )

    structural = (
        source_comparable
        and exact_output
        and terminal_clean
        and zero_fallbacks
        and (zero_swap_growth if require_telemetry else True)
        and cap_ok
        and contract_ok
        and prewarm_ok
        and (telemetry_ok if require_telemetry else True)
        and (allocator_ok if require_telemetry else True)
    )
    observer_effect = _observer_effect(
        foundation,
        observer_reference_rows if observer_reference_rows is not None else baseline_rows,
        repetitions=repetitions,
    )
    summary = {
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
    if require_telemetry:
        summary["telemetry_contract"] = telemetry_ok
        summary["allocator_contract"] = allocator_ok
        summary["host_state_diagnostics"] = (
            _host_state_diagnostics(baseline_index, control_index, expected)
            if telemetry_ok and allocator_ok
            else {"status": "invalid-contract"}
        )
    return summary


def _run_control_row(
    foundation: ModuleType,
    args: argparse.Namespace,
    *,
    cell: str,
    repetition: int,
    engine: str,
    pair_order: tuple[str, str],
    state_first: str,
    state: str = "control-off",
    telemetry_module: ModuleType | None = None,
    thermal_power: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state not in STATES:
        raise ValueError(f"unsupported control state: {state}")
    output = args.run_root / f"r{repetition}-{cell}-{state}-{engine}.json"
    command = foundation._cell_command(
        _runner_args(args),
        cell=cell,
        engine=engine,
        repetition=repetition,
        pair_order=pair_order,
        output=output,
        fingerprint=args.fingerprint,
    )
    telemetry: dict[str, Any] | None = None
    idle_started = time.monotonic()
    if args.idle_seconds > 0:
        time.sleep(args.idle_seconds)
    idle_elapsed = time.monotonic() - idle_started
    if telemetry_module is None:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=args.process_timeout_seconds,
        )
        returncode = completed.returncode
        stderr = completed.stderr
    else:
        host_before = telemetry_module.capture_host_state()
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        sampler = telemetry_module.ProcessTelemetrySampler(
            child.pid, interval_seconds=args.telemetry_interval_seconds
        )
        sampler.start()
        try:
            stdout, stderr = child.communicate(timeout=args.process_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            child.kill()
            stdout, stderr = child.communicate()
            sampler.finish()
            raise RuntimeError(
                f"control row timed out: {cell}/{repetition}/{state}/{engine}\n{stderr[-4000:]}"
            ) from error
        process_telemetry = sampler.finish()
        host_after = telemetry_module.capture_host_state()
        returncode = child.returncode
        telemetry = {
            "enabled": True,
            "host_state_before": host_before,
            "host_state_after": host_after,
            "process": process_telemetry,
            "thermal_power": thermal_power
            or telemetry_module.probe_thermal_power(
                timeout_seconds=args.telemetry_probe_timeout_seconds
            ),
        }
    if returncode != 0:
        raise RuntimeError(
            f"control row failed: {cell}/{repetition}/{state}/{engine}\n{stderr[-4000:]}"
        )
    result = json.loads(output.read_text())
    row = {
        "cell": cell,
        "repetition": repetition,
        "state": state,
        "engine": engine,
        "state_first": state_first,
        "control_protocol": {
            "config_state": "observer-off",
            "process_isolation": "fresh-process",
            "prewarm": "foundation-declared-warmup",
            "order": "alternating-observer-off-control-off",
            "idle_seconds_requested": args.idle_seconds,
            "idle_seconds_observed": idle_elapsed,
        },
        **({"telemetry": telemetry} if telemetry is not None else {}),
        "source_path": str(output),
        "source_file_sha256": sha256(output),
        "result": result,
    }
    return row, {
        "cell": cell,
        "repetition": repetition,
        "state": state,
        "engine": engine,
        "status": returncode,
        "source_path": str(output),
    }


def _load_control_row(
    args: argparse.Namespace, *, cell: str, repetition: int, engine: str, state_first: str
) -> tuple[dict[str, Any], dict[str, Any]]:
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
            "idle_seconds_requested": args.idle_seconds,
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
        "status": 0,
        "source_path": str(output),
    }


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
    telemetry_module = load_telemetry() if args.enable_telemetry else None
    thermal_power = (
        telemetry_module.probe_thermal_power(timeout_seconds=args.telemetry_probe_timeout_seconds)
        if telemetry_module is not None
        else None
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    paired_baseline_rows: list[dict[str, Any]] = []
    paired_control_rows: list[dict[str, Any]] = []
    collection: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for cell in CELLS:
            pair_order = foundation.engine_order_for_pair(cell, repetition)
            state_sequence = (
                control_order(cell, repetition) if args.paired_host_state else ("control-off",)
            )
            state_first = control_order(cell, repetition)[0]
            for engine in pair_order:
                for state in state_sequence:
                    if args.reuse_existing:
                        if args.paired_host_state:
                            raise ValueError(
                                "reuse-existing is not supported for paired host-state rows"
                            )
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
                            state=state,
                            telemetry_module=telemetry_module,
                            thermal_power=thermal_power,
                        )
                    rows.append(row)
                    collection.append(record)
                    if state == "observer-off":
                        paired_baseline_rows.append(row)
                    else:
                        paired_control_rows.append(row)
                    time.sleep(args.cooldown_seconds)
    summary_baseline_rows = paired_baseline_rows if args.paired_host_state else baseline_rows
    summary_control_rows = paired_control_rows if args.paired_host_state else rows
    summary = summarize_control(
        foundation,
        summary_baseline_rows,
        summary_control_rows,
        repetitions=args.repetitions,
        expected_max_output_tokens=args.max_output_tokens,
        require_telemetry=args.enable_telemetry,
        observer_reference_rows=baseline_rows if args.paired_host_state else None,
    )
    first = _unwrap(rows[0])
    return {
        "schema_version": 1,
        "kind": (
            "decode-boundary-host-state-evidence"
            if args.paired_host_state
            else "decode-boundary-control-evidence"
        ),
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
            "idle_seconds": args.idle_seconds,
            "telemetry_enabled": args.enable_telemetry,
            "telemetry_interval_seconds": args.telemetry_interval_seconds,
            "telemetry_probe_timeout_seconds": args.telemetry_probe_timeout_seconds,
            "paired_host_state": args.paired_host_state,
            "thermal_power_capability": thermal_power,
            "collection_statuses": collection,
        },
        "source": first["source"],
        "summary": summary,
        "observer_reference_rows": baseline_rows,
        "paired_baseline_rows": paired_baseline_rows,
        "paired_control_rows": paired_control_rows,
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
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=0.05)
    parser.add_argument("--telemetry-probe-timeout-seconds", type=float, default=1.5)
    parser.add_argument("--enable-telemetry", action="store_true")
    parser.add_argument("--paired-host-state", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", default=DEFAULT_ITERATION)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    for name in (
        "observer_matrix",
        "off_config",
        "workload",
        "lock",
        "data_root",
        "run_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.max_output_tokens <= 0 or args.repetitions < 2 or args.repetitions % 2:
        raise ValueError("max output tokens must be positive and repetitions must be even")
    if args.idle_seconds < 0 or args.telemetry_interval_seconds <= 0:
        raise ValueError("idle seconds must be non-negative and telemetry interval positive")
    if args.telemetry_probe_timeout_seconds <= 0:
        raise ValueError("telemetry probe timeout must be positive")
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
