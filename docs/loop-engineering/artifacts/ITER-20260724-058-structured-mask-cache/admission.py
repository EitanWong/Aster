#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
REFERENCE_DIR = ARTIFACT_DIR.parent / "ITER-20260724-057-structured-python-tokens"
PRODUCTION_SOURCE = PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py"
RESULTS = (
    ("short", "strict-short-b4-r18", "structured-b4", 16 * 1024**2),
    ("long", "strict-long-b2-r18", "structured-b2", 8 * 1024**2),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payloads(root: Path, result_name: str) -> list[dict[str, Any]]:
    result_dir = root / "results" / result_name
    manifest = _load(result_dir / "execution-manifest.json")
    return [_load(root / record["output"]) for record in manifest["records"]]


def build_admission() -> dict[str, Any]:
    source_hash = _sha256(PRODUCTION_SOURCE)
    formal: dict[str, Any] = {}
    model_signatures: list[dict[str, str]] = []
    for label, result_name, cell_name, peak_delta_limit in RESULTS:
        result_dir = ARTIFACT_DIR / "results" / result_name
        aggregate = _load(result_dir / "aggregate.json")
        payloads = _payloads(ARTIFACT_DIR, result_name)
        reference_payloads = _payloads(REFERENCE_DIR, result_name)
        model_signatures.append(payloads[0]["model_input_sha256"])
        settings = payloads[0]["settings"]
        expected_misses = (
            int(settings["steps"]) + int(settings["pair_warmup_steps"])
        ) * int(payloads[0]["batch_size"])
        current_peak = statistics.median(
            payload["memory"]["mlx_peak_bytes"] for payload in payloads
        )
        reference_peak = statistics.median(
            payload["memory"]["mlx_peak_bytes"] for payload in reference_payloads
        )
        formal[label] = {
            "records": len(payloads),
            "unique_pids": len({payload["pid"] for payload in payloads}),
            "aggregate_passed": aggregate["all_cells_passed"],
            "all_exact": all(
                payload["parity"]["exact_token_text_cache"] is True
                for payload in payloads
            ),
            "all_swap_non_growth": all(
                payload["memory"]["swap_after_bytes"]
                <= payload["memory"]["swap_before_bytes"]
                for payload in payloads
            ),
            "all_current_production_source": all(
                payload["source_sha256"][
                    "aster/inference/constrained/json_schema_processor.py"
                ]
                == source_hash
                for payload in payloads
            ),
            "all_baseline_calls_forced_miss": all(
                payload["policy_metrics"]["baseline"]["forced_mask_misses"]
                == expected_misses
                and payload["policy_metrics"]["production"]["forced_mask_misses"]
                == 0
                for payload in payloads
            ),
            "dual_runner_mlx_peak_median_delta_bytes": current_peak
            - reference_peak,
            "dual_runner_mlx_peak_delta_limit_bytes": peak_delta_limit,
            "dual_runner_mlx_peak_within_limit": (
                current_peak - reference_peak <= peak_delta_limit
            ),
            "balanced_interval_percent": aggregate["cells"][cell_name]["intervals"][
                "balanced"
            ],
        }

    validation = _load(ARTIFACT_DIR / "structured-validation-b4-run-1.json")
    screening_evidence = (
        "screen-b4-run-1.json",
        "screen-cap1-b4-run-2.json",
        "screen-cap1-long-b2-run-1.json",
    )
    gates = {
        "short_formal_admission": formal["short"]["aggregate_passed"],
        "long_formal_admission": formal["long"]["aggregate_passed"],
        "formal_process_count": all(
            result["records"] == 18 and result["unique_pids"] == 18
            for result in formal.values()
        ),
        "formal_exactness": all(result["all_exact"] for result in formal.values()),
        "formal_swap_non_growth": all(
            result["all_swap_non_growth"] for result in formal.values()
        ),
        "current_production_source_bound": all(
            result["all_current_production_source"] for result in formal.values()
        ),
        "baseline_forced_miss_counts": all(
            result["all_baseline_calls_forced_miss"] for result in formal.values()
        ),
        "mlx_peak_delta_within_limits": all(
            result["dual_runner_mlx_peak_within_limit"]
            for result in formal.values()
        ),
        "model_signature_equal": model_signatures[0] == model_signatures[1],
        "structured_schema_valid": validation["all_schema_valid"],
        "structured_stop_valid": validation["all_stopped_before_limit"],
        "dynamic_membership_shrinks": (
            validation["membership_sizes"][0] == 4
            and validation["membership_sizes"][-1] == 1
        ),
        "screening_evidence_retained": all(
            (ARTIFACT_DIR / relative).is_file() for relative in screening_evidence
        ),
    }
    return {
        "schema_version": 1,
        "production_source_sha256": source_hash,
        "formal": formal,
        "structured_validation": {
            "lanes": len(validation["lanes"]),
            "membership_start": validation["membership_sizes"][0],
            "membership_end": validation["membership_sizes"][-1],
        },
        "screening_evidence": list(screening_evidence),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "final-admission.json",
    )
    args = parser.parse_args()
    rendered = json.dumps(build_admission(), indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
