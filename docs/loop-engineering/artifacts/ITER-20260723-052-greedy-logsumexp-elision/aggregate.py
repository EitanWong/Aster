#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _speed_percent(payload: dict[str, Any]) -> float:
    baseline = sum(payload["timings"]["baseline_step_seconds"])
    candidate = sum(payload["timings"]["production_step_seconds"])
    if baseline <= 0 or candidate <= 0:
        raise ValueError("timing totals must be positive")
    return (baseline / candidate - 1.0) * 100.0


def summarize(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[(str(payload["workload"]), int(payload["batch_size"]))].append(payload)

    cells: dict[str, Any] = {}
    for (workload, batch_size), records in sorted(grouped.items()):
        speeds = [_speed_percent(record) for record in records]
        key = f"{workload}-b{batch_size}"
        cells[key] = {
            "workload": workload,
            "batch_size": batch_size,
            "records": len(records),
            "speed_percent": {
                "median": statistics.median(speeds),
                "min": min(speeds),
                "max": max(speeds),
                "values": speeds,
            },
            "all_exact_token_text_cache": all(
                bool(record["parity"]["exact_token_text_cache"])
                for record in records
            ),
            "direct_logit_rows": [
                int(record["policy_metrics"]["production"].get("direct_logit_rows", 0))
                for record in records
            ],
            "normalized_rows": [
                int(record["policy_metrics"]["production"].get("normalized_rows", 0))
                for record in records
            ],
            "swap_deltas": [
                int(record["memory"]["swap_after_bytes"])
                - int(record["memory"]["swap_before_bytes"])
                for record in records
            ],
        }

    return {
        "schema_version": 1,
        "record_count": sum(len(records) for records in grouped.values()),
        "cells": cells,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.input]
    result = summarize(payloads)
    result["inputs"] = [str(path) for path in args.input]
    result["aggregate_source_sha256"] = _sha256(Path(__file__).resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
