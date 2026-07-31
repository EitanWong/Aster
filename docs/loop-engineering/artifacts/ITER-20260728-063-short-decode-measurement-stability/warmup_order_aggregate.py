#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
RECORD_NAME = re.compile(
    r"^warmup-(?P<warmup>sp|ps)-measure-(?P<measurement>sp|ps)-(?P<replicate>[1-9][0-9]*)\.json$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _quantile(values, 0.95),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "environment": payload["environment"],
            "settings": payload["settings"],
            "source_sha256": payload["source_sha256"],
            "model_input_sha256": payload["model_input_sha256"],
        },
        sort_keys=True,
    )


def _order(code: str) -> list[str]:
    return ["serial", "pipeline"] if code == "sp" else ["pipeline", "serial"]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)


def _host_observability(records: list[dict[str, Any]]) -> dict[str, Any]:
    hosts = [record["host_after_warmup"] for record in records]
    stream_counts = [host["mlx_stream_count"] for host in hosts]
    load_averages = [host["host_load_average"] for host in hosts]
    cpu_frequencies = [host["cpu_frequency_mhz"] for host in hosts]
    return {
        "mlx_stream_count_available_records": sum(value is not None for value in stream_counts),
        "mlx_stream_count_values": sorted({value for value in stream_counts if value is not None}),
        "host_load_average_1m": _summary(
            [float(value[0]) for value in load_averages if value is not None]
        ),
        "cpu_frequency_current_mhz": _summary(
            [float(value["current"]) for value in cpu_frequencies if value is not None]
        ),
        "process_thread_count": _summary(
            [float(host["process_thread_count"]) for host in hosts]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the I063 prewarm-order measurement-stability screen."
    )
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.resolve().read_text()) for path in args.record]
    failures: list[str] = []
    fingerprints: set[str] = set()
    pids: set[int] = set()
    cells: dict[str, list[float]] = defaultdict(list)
    by_measurement_order: dict[str, list[float]] = defaultdict(list)
    by_warmup_terminal: dict[str, list[float]] = defaultdict(list)
    paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    gains: list[float] = []

    for index, (path, record) in enumerate(zip(args.record, records, strict=True), start=1):
        match = RECORD_NAME.fullmatch(path.name)
        if match is None:
            failures.append(f"record {index} has an unsupported filename: {path.name}")
            continue
        warmup_code = match["warmup"]
        measurement_code = match["measurement"]
        replicate = int(match["replicate"])
        expected_warmup = _order(warmup_code)
        expected_measurement = _order(measurement_code)
        if record["warmup_order"] != expected_warmup:
            failures.append(f"record {index} warmup order does not match its filename")
        if record["measurement_order"] != expected_measurement:
            failures.append(f"record {index} measurement order does not match its filename")
        if not record["comparable"] or not all(record["gates"].values()):
            failures.append(f"record {index} failed exactness or swap gates")
        if record["pid"] in pids:
            failures.append(f"record {index} reused PID {record['pid']}")
        pids.add(record["pid"])
        fingerprints.add(_fingerprint(record))
        gain = float(record["pipeline_elapsed_gain_percent"])
        terminal = record["warmup_terminal_variant"]
        measurement_first = record["measurement_first_variant"]
        expected_cell = f"warmup_{warmup_code}_measure_{measurement_code}"
        cells[expected_cell].append(gain)
        by_measurement_order[f"{measurement_first}_first"].append(gain)
        by_warmup_terminal[terminal].append(gain)
        paired[(replicate, warmup_code)][measurement_first] = gain
        gains.append(gain)

    if len(records) != 8:
        failures.append(f"expected eight records, found {len(records)}")
    if len(pids) != len(records):
        failures.append("independent-process PID gate failed")
    if len(fingerprints) != 1:
        failures.append("model, environment, settings, or source drifted")
    expected_cells = {
        "warmup_ps_measure_ps",
        "warmup_ps_measure_sp",
        "warmup_sp_measure_ps",
        "warmup_sp_measure_sp",
    }
    if set(cells) != expected_cells or any(len(values) != 2 for values in cells.values()):
        failures.append("2x2 cross and two-repeat balance failed")

    contrasts: list[dict[str, Any]] = []
    for (replicate, warmup_code), paired_gains in sorted(paired.items()):
        if set(paired_gains) != {"pipeline", "serial"}:
            failures.append(
                f"replicate {replicate} / warmup {warmup_code} lacks a paired measurement order"
            )
            continue
        contrasts.append(
            {
                "replicate": replicate,
                "warmup_order": _order(warmup_code),
                "pipeline_first_gain_percent": paired_gains["pipeline"],
                "serial_first_gain_percent": paired_gains["serial"],
                "pipeline_first_minus_serial_first_percent": (
                    paired_gains["pipeline"] - paired_gains["serial"]
                ),
            }
        )

    contrast_values = [
        float(item["pipeline_first_minus_serial_first_percent"]) for item in contrasts
    ]
    gates = {
        "eight_records": len(records) == 8,
        "independent_processes": len(pids) == len(records),
        "stable_fingerprint": len(fingerprints) == 1,
        "crossed_two_repeat_design": not any(
            "2x2 cross" in failure for failure in failures
        ),
        "exactness_and_swap": not any("exactness or swap" in failure for failure in failures),
        "four_matched_measurement_order_pairs": len(contrasts) == 4,
        "all_matched_measurement_order_contrasts_positive": bool(contrast_values)
        and min(contrast_values) > 0.0,
    }
    decision = "reject" if all(gates.values()) and not failures else "invalid"
    payload = {
        "schema_version": 1,
        "record_paths": [_display_path(path) for path in args.record],
        "gates": gates,
        "failures": failures,
        "decision": decision,
        "reason": (
            "Crossing the terminal prewarm variant did not remove the measurement-order "
            "interaction: every matched pipeline-first minus serial-first contrast was "
            "positive. The fixed serial-to-pipeline prewarm sequence is therefore not a "
            "sufficient explanation, and the asynchronous candidate remains rejected."
            if decision == "reject"
            else "The screen did not meet its predeclared evidence-integrity gates."
        ),
        "pipeline_elapsed_gain_percent": _summary(gains) if gains else None,
        "by_measurement_order": {
            name: _summary(values) for name, values in sorted(by_measurement_order.items())
        },
        "by_warmup_terminal": {
            name: _summary(values) for name, values in sorted(by_warmup_terminal.items())
        },
        "by_cell": {name: _summary(values) for name, values in sorted(cells.items())},
        "matched_measurement_order_contrasts": contrasts,
        "host_observability": _host_observability(records),
        "source_sha256": {
            "warmup_order_aggregate.py": _sha256(Path(__file__).resolve())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
