#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


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


def _bootstrap_median_interval(values: list[float]) -> dict[str, float | int]:
    generator = random.Random(62062)
    count = len(values)
    medians = sorted(
        statistics.median([values[generator.randrange(count)] for _ in range(count)])
        for _ in range(20000)
    )
    return {
        "confidence": 0.95,
        "samples": 20000,
        "lower": _quantile(medians, 0.025),
        "upper": _quantile(medians, 0.975),
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


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the I062 paired MLX-LM pipeline screen."
    )
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.resolve().read_text()) for path in args.record]
    failures: list[str] = []
    gains: list[float] = []
    by_order: dict[str, list[float]] = {"serial_first": [], "pipeline_first": []}
    pids: set[int] = set()
    fingerprints: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not record["comparable"] or not all(record["gates"].values()):
            failures.append(f"pair {index} failed exactness or swap gates")
        if record["pid"] in pids:
            failures.append(f"pair {index} reused PID {record['pid']}")
        pids.add(record["pid"])
        fingerprints.add(_fingerprint(record))
        order = record["order"]
        if order == ["serial", "pipeline"]:
            order_name = "serial_first"
        elif order == ["pipeline", "serial"]:
            order_name = "pipeline_first"
        else:
            failures.append(f"pair {index} has invalid order {order}")
            continue
        gain = float(record["pipeline_elapsed_gain_percent"])
        gains.append(gain)
        by_order[order_name].append(gain)

    if len(records) != 6:
        failures.append(f"expected six records, found {len(records)}")
    if len(pids) != len(records):
        failures.append("independent-process PID gate failed")
    if len(fingerprints) != 1:
        failures.append("model, environment, settings, or source drifted")
    if len(by_order["serial_first"]) != 3 or len(by_order["pipeline_first"]) != 3:
        failures.append("AB/BA order balance failed")

    interval = _bootstrap_median_interval(gains) if gains else None
    order_medians = {
        name: statistics.median(values) for name, values in by_order.items() if values
    }
    gates = {
        "six_records": len(records) == 6,
        "independent_processes": len(pids) == len(records),
        "stable_fingerprint": len(fingerprints) == 1,
        "balanced_order": len(by_order["serial_first"]) == 3
        and len(by_order["pipeline_first"]) == 3,
        "exactness_and_swap": not any("exactness or swap" in failure for failure in failures),
        "median_interval_above_3_percent": bool(interval and interval["lower"] >= 3.0),
        "both_order_medians_above_3_percent": len(order_medians) == 2
        and all(value >= 3.0 for value in order_medians.values()),
        "no_pair_regression": bool(gains and min(gains) >= 0.0),
    }
    payload = {
        "schema_version": 1,
        "record_paths": [_display_path(path) for path in args.record],
        "gates": gates,
        "failures": failures,
        "pipeline_elapsed_gain_percent": _summary(gains) if gains else None,
        "bootstrap_median_interval": interval,
        "by_order": {
            name: _summary(values) for name, values in by_order.items() if values
        },
        "decision": "advance"
        if all(gates.values()) and not failures
        else "inconclusive",
        "source_sha256": {
            "pipeline_pair_aggregate.py": _sha256(Path(__file__).resolve())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
