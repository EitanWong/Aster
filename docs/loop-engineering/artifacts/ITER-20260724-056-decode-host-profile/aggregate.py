#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
HOST_FILES = tuple(ARTIFACT_DIR / f"screen-b{batch}-run-1.json" for batch in (2, 4, 8))
PENALTY_FILES = tuple(
    ARTIFACT_DIR / f"penalty-screen-b{batch}-run-1.json" for batch in (2, 4, 8)
)
NORMALIZATION_FILES = tuple(
    ARTIFACT_DIR / f"normalization-screen-top-p-b{batch}-run-1.json"
    for batch in (2, 4, 8)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _gain(payload: dict[str, Any]) -> float:
    baseline = statistics.median(payload["timings"]["baseline_step_seconds"])
    candidate = statistics.median(payload["timings"]["production_step_seconds"])
    return 100.0 * (baseline - candidate) / baseline


def _exact(payload: dict[str, Any]) -> bool:
    return bool(payload["parity"]["exact_token_text_cache"])


def _zero_swap(payload: dict[str, Any]) -> bool:
    memory = payload["memory"]
    return int(memory["swap_after_bytes"]) <= int(memory["swap_before_bytes"])


def build_summary() -> dict[str, Any]:
    host = [_load(path) for path in HOST_FILES]
    penalties = [_load(path) for path in PENALTY_FILES]
    normalization = [_load(path) for path in NORMALIZATION_FILES]
    all_payloads = [*host, *penalties, *normalization]

    host_rows = []
    for payload in host:
        metrics = payload["policy_metrics"]["production"]
        batch_total = float(metrics["batch_seconds"]["total"])
        materialize_total = float(metrics["materialize_seconds"]["total"])
        result_total = float(metrics["decode_result_seconds"]["total"])
        host_rows.append(
            {
                "batch_size": payload["batch_size"],
                "materialize_pct": 100.0 * materialize_total / batch_total,
                "host_post_eval_pct": 100.0
                * (materialize_total + result_total)
                / batch_total,
            }
        )

    penalty_rows = [
        {
            "batch_size": payload["batch_size"],
            "median_gain_pct": _gain(payload),
            "vectorized_batches": payload["policy_metrics"]["production"]["vectorized_batches"],
            "vectorized_rows": payload["policy_metrics"]["production"]["vectorized_rows"],
        }
        for payload in penalties
    ]
    normalization_rows = [
        {
            "batch_size": payload["batch_size"],
            "median_gain_pct": _gain(payload),
            "batched_normalization_batches": payload["policy_metrics"]["production"][
                "batched_normalization_batches"
            ],
            "batched_normalization_rows": payload["policy_metrics"]["production"][
                "batched_normalization_rows"
            ],
        }
        for payload in normalization
    ]

    exact_all = all(_exact(payload) for payload in all_payloads)
    zero_swap_all = all(_zero_swap(payload) for payload in all_payloads)
    host_materialization_below_one_pct = max(
        row["host_post_eval_pct"] for row in host_rows
    ) < 1.0
    penalty_below_gate = max(row["median_gain_pct"] for row in penalty_rows) < 3.0
    normalization_below_gate = max(
        row["median_gain_pct"] for row in normalization_rows
    ) < 3.0
    source = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "source_sha256": {source.name: _sha256(source)},
        "payload_sha256": {
            path.name: _sha256(path)
            for path in (*HOST_FILES, *PENALTY_FILES, *NORMALIZATION_FILES)
        },
        "host_profile": host_rows,
        "batched_penalties": penalty_rows,
        "batched_normalization": normalization_rows,
        "gates": {
            "exact_all": exact_all,
            "zero_swap_all": zero_swap_all,
            "host_materialization_below_one_pct": host_materialization_below_one_pct,
            "penalty_below_three_pct_gate": penalty_below_gate,
            "normalization_below_three_pct_gate": normalization_below_gate,
        },
        "decision": {
            "admitted": False,
            "host_materialization": "rejected: insufficient addressable share",
            "batched_penalties": "rejected: below end-to-end gain gate",
            "batched_normalization": "rejected: below gate and regressed B2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "summary.json")
    args = parser.parse_args()
    payload = build_summary()
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
