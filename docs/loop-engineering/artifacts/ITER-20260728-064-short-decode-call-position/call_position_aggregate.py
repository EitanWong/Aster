#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
RECORD_NAME = re.compile(
    r"^(?P<variant>serial|pipeline)-(?P<warmup>sp|ps)-r(?P<replicate>[1-9][0-9]*)\.json$"
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


def _bootstrap_median_interval(values: list[float]) -> dict[str, float | int]:
    generator = random.Random(64064)
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


def _order(code: str) -> list[str]:
    return ["serial", "pipeline"] if code == "sp" else ["pipeline", "serial"]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the I064 same-variant short-decode call-position screen."
    )
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.resolve().read_text()) for path in args.record]
    failures: list[str] = []
    fingerprints: set[str] = set()
    pids: set[int] = set()
    by_variant: dict[str, list[float]] = defaultdict(list)
    by_terminal: dict[str, list[float]] = defaultdict(list)
    by_cell: dict[str, list[float]] = defaultdict(list)
    paired: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    gains: list[float] = []

    for index, (path, record) in enumerate(zip(args.record, records, strict=True), start=1):
        match = RECORD_NAME.fullmatch(path.name)
        if match is None:
            failures.append(f"record {index} has an unsupported filename: {path.name}")
            continue
        variant = match["variant"]
        warmup_code = match["warmup"]
        replicate = int(match["replicate"])
        if record["variant"] != variant:
            failures.append(f"record {index} variant does not match its filename")
        if record["warmup_order"] != _order(warmup_code):
            failures.append(f"record {index} warmup order does not match its filename")
        if record["replicate"] != replicate:
            failures.append(f"record {index} repeat identifier does not match its filename")
        if not record["comparable"] or not all(record["gates"].values()):
            failures.append(f"record {index} failed exactness or swap gates")
        if record["pid"] in pids:
            failures.append(f"record {index} reused PID {record['pid']}")
        pids.add(record["pid"])
        fingerprints.add(_fingerprint(record))
        gain = float(record["second_vs_first_elapsed_gain_percent"])
        by_variant[variant].append(gain)
        by_terminal[record["warmup_terminal_variant"]].append(gain)
        by_cell[f"{variant}_warmup_{warmup_code}"].append(gain)
        paired[(replicate, warmup_code)][variant] = gain
        gains.append(gain)

    if len(records) != 8:
        failures.append(f"expected eight records, found {len(records)}")
    if len(pids) != len(records):
        failures.append("independent-process PID gate failed")
    if len(fingerprints) != 1:
        failures.append("model, environment, settings, or source drifted")
    expected_cells = {
        "pipeline_warmup_ps",
        "pipeline_warmup_sp",
        "serial_warmup_ps",
        "serial_warmup_sp",
    }
    if set(by_cell) != expected_cells or any(len(values) != 2 for values in by_cell.values()):
        failures.append("2x2 variant/warmup and two-repeat balance failed")

    paired_deltas: list[dict[str, Any]] = []
    for (replicate, warmup_code), paired_gains in sorted(paired.items()):
        if set(paired_gains) != {"serial", "pipeline"}:
            failures.append(
                f"replicate {replicate} / warmup {warmup_code} lacks a paired variant"
            )
            continue
        paired_deltas.append(
            {
                "replicate": replicate,
                "warmup_order": _order(warmup_code),
                "serial_second_vs_first_percent": paired_gains["serial"],
                "pipeline_second_vs_first_percent": paired_gains["pipeline"],
                "pipeline_minus_serial_percent": (
                    paired_gains["pipeline"] - paired_gains["serial"]
                ),
            }
        )

    variant_medians = {
        name: statistics.median(values) for name, values in by_variant.items() if values
    }
    slower_count = sum(gain < 0.0 for gain in gains)
    gates = {
        "eight_records": len(records) == 8,
        "independent_processes": len(pids) == len(records),
        "stable_fingerprint": len(fingerprints) == 1,
        "crossed_two_repeat_design": not any(
            "2x2 variant/warmup" in failure for failure in failures
        ),
        "exactness_and_swap": not any("exactness or swap" in failure for failure in failures),
        "four_matched_variant_pairs": len(paired_deltas) == 4,
        "both_variant_medians_below_minus_3_percent": len(variant_medians) == 2
        and all(value <= -3.0 for value in variant_medians.values()),
        "at_least_seven_second_calls_slower": slower_count >= 7,
    }
    decision = "reject" if all(gates.values()) and not failures else "invalid"
    payload = {
        "schema_version": 1,
        "record_paths": [_display_path(path) for path in args.record],
        "gates": gates,
        "failures": failures,
        "decision": decision,
        "reason": (
            "Same-variant fresh-cache calls show a material second-call slowdown for "
            "both serial and pipeline paths. Adjacent mixed serial/pipeline pairs are "
            "therefore position-confounded and cannot authorize an asynchronous candidate."
            if decision == "reject"
            else "The screen did not meet its predeclared evidence-integrity gates."
        ),
        "second_call_slower_count": slower_count,
        "second_vs_first_elapsed_gain_percent": _summary(gains) if gains else None,
        "by_variant": {name: _summary(values) for name, values in sorted(by_variant.items())},
        "by_variant_bootstrap_median_interval": {
            name: _bootstrap_median_interval(values)
            for name, values in sorted(by_variant.items())
        },
        "by_warmup_terminal": {
            name: _summary(values) for name, values in sorted(by_terminal.items())
        },
        "by_cell": {name: _summary(values) for name, values in sorted(by_cell.items())},
        "paired_variant_deltas": paired_deltas,
        "source_sha256": {
            "call_position_aggregate.py": _sha256(Path(__file__).resolve())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
