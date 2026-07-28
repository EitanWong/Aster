#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
BASE_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import strict_aggregate as base  # noqa: E402

SPEED_FLOOR_PERCENT = 3.0


def aggregate(manifest_path: Path) -> dict[str, object]:
    original_artifact_dir = base.legacy.ARTIFACT_DIR
    original_speed_floor = base._speed_floor
    base.legacy.ARTIFACT_DIR = ARTIFACT_DIR
    base._speed_floor = lambda workload: SPEED_FLOOR_PERCENT
    try:
        payload = base.aggregate(manifest_path)
    finally:
        base.legacy.ARTIFACT_DIR = original_artifact_dir
        base._speed_floor = original_speed_floor

    for cell in payload["cells"].values():
        intervals = cell["intervals"]
        replicates = cell["replicates"]
        gates = cell["gates"]
        gates["baseline_first_median_interval_clears_floor"] = bool(
            intervals["baseline_first"]["lower_meets_speed_floor"]
        )
        gates["production_first_median_interval_clears_floor"] = bool(
            intervals["production_first"]["lower_meets_speed_floor"]
        )
        gates["order_strata_speed_floor_required"] = (
            gates["baseline_first_median_interval_clears_floor"]
            and gates["production_first_median_interval_clears_floor"]
        )
        gates["within_replicate_stability"] = (
            int(cell["stable_replicates"]) >= len(replicates) - 1
        )
        cell["order_strata_inference_required"] = True
        cell["within_replicate_stability_required"] = True
        cell["passed"] = all(gates.values())

    payload["speed_floor_percent"] = SPEED_FLOOR_PERCENT
    payload["all_cells_passed"] = all(
        cell["passed"] for cell in payload["cells"].values()
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
