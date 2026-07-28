#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
PRODUCTION_SOURCE = PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py"
RESULTS = (
    ("short", "strict-short-b4-r18", "structured-b4"),
    ("long", "strict-long-b2-r18", "structured-b2"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_admission() -> dict[str, Any]:
    source_hash = _sha256(PRODUCTION_SOURCE)
    formal: dict[str, Any] = {}
    model_signatures: list[dict[str, str]] = []
    for label, result_name, cell_name in RESULTS:
        result_dir = ARTIFACT_DIR / "results" / result_name
        manifest = _load(result_dir / "execution-manifest.json")
        aggregate = _load(result_dir / "aggregate.json")
        payloads = [
            _load(ARTIFACT_DIR / record["output"])
            for record in manifest["records"]
        ]
        model_signatures.append(payloads[0]["model_input_sha256"])
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
            "balanced_interval_percent": aggregate["cells"][cell_name]["intervals"][
                "balanced"
            ],
        }

    validation = _load(ARTIFACT_DIR / "structured-validation-b4-run-1.json")
    failed_evidence = (
        "screen-b4-run-1.json",
        "long-b2-run-1.json",
        "optimized-screen-b4-run-1.json",
        "optimized-long-b2-run-1.json",
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
        "model_signature_equal": model_signatures[0] == model_signatures[1],
        "structured_schema_valid": validation["all_schema_valid"],
        "structured_stop_valid": validation["all_stopped_before_limit"],
        "dynamic_membership_shrinks": (
            validation["membership_sizes"][0] == 4
            and validation["membership_sizes"][-1] == 1
        ),
        "failed_candidates_retained": all(
            (ARTIFACT_DIR / relative).is_file() for relative in failed_evidence
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
        "failed_evidence": list(failed_evidence),
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
