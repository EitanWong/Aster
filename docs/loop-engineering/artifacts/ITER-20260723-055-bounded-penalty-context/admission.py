#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
RESULTS = ARTIFACT_DIR / "results"
LONG = RESULTS / "strict-long-window-r18"
SHORT_B2 = RESULTS / "short-b2-no-regression-r18"
SHORT_B4 = RESULTS / "short-b4-no-regression-r18"
FAILED_SHORT_WINDOW = RESULTS / "strict-ultra-long-r18"
NO_REGRESSION_FLOOR = -1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _cell(payload: dict[str, Any]) -> dict[str, Any]:
    cells = payload["cells"]
    if len(cells) != 1:
        raise ValueError("admission inputs must contain exactly one cell")
    return next(iter(cells.values()))


def _intervals_clear_floor(cell: dict[str, Any], floor: float) -> bool:
    return all(
        interval["confidence_met"] is True and float(interval["lower"]) >= floor
        for interval in cell["intervals"].values()
    )


def _payload_signature(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    record = manifest["records"][0]
    payload = _load(ARTIFACT_DIR / record["output"])
    return payload["source_sha256"], payload["model_input_sha256"]


def build() -> dict[str, Any]:
    directories = {
        "long": LONG,
        "short_b2": SHORT_B2,
        "short_b4": SHORT_B4,
        "failed_short_window": FAILED_SHORT_WINDOW,
    }
    manifests = {
        name: _load(directory / "execution-manifest.json")
        for name, directory in directories.items()
    }
    aggregates = {
        name: _load(directory / "strict-aggregate.json")
        for name, directory in directories.items()
    }
    summaries = {
        name: _load(directory / "descriptive-summary.json")
        for name, directory in directories.items()
        if name != "failed_short_window"
    }
    signatures = {
        name: _payload_signature(manifest)
        for name, manifest in manifests.items()
    }
    source_signatures = [signature[0] for signature in signatures.values()]
    model_signatures = [signature[1] for signature in signatures.values()]
    long_cell = _cell(aggregates["long"])
    short_cells = {
        name: _cell(aggregates[name]) for name in ("short_b2", "short_b4")
    }
    gates = {
        "long_strict_admission": aggregates["long"]["all_cells_passed"] is True,
        "long_exact_and_swap_non_growth": (
            long_cell["gates"]["all_exact_token_text_cache_parity"] is True
            and long_cell["gates"]["all_swap_non_growth"] is True
        ),
        "short_b2_no_regression": _intervals_clear_floor(
            short_cells["short_b2"], NO_REGRESSION_FLOOR
        ),
        "short_b4_no_regression": _intervals_clear_floor(
            short_cells["short_b4"], NO_REGRESSION_FLOOR
        ),
        "short_exact_and_swap_non_growth": all(
            cell["gates"]["all_exact_token_text_cache_parity"] is True
            and cell["gates"]["all_swap_non_growth"] is True
            for cell in short_cells.values()
        ),
        "current_source_signatures_match": all(
            signature == source_signatures[0] for signature in source_signatures
        ),
        "model_signatures_match": all(
            signature == model_signatures[0] for signature in model_signatures
        ),
        "failed_short_window_retained": (
            aggregates["failed_short_window"]["all_cells_passed"] is False
            and len(manifests["failed_short_window"]["records"]) == 18
        ),
        "formal_process_count": sum(
            len(manifests[name]["records"])
            for name in ("long", "short_b2", "short_b4")
        )
        == 54,
    }
    input_paths: list[Path] = []
    for name, directory in directories.items():
        input_paths.extend(
            (
                directory / "execution-manifest.json",
                directory / "strict-aggregate.json",
            )
        )
        if name != "failed_short_window":
            input_paths.append(directory / "descriptive-summary.json")
    return {
        "schema_version": 1,
        "candidate": "bounded built-in penalty processor context",
        "no_regression_floor_percent": NO_REGRESSION_FLOOR,
        "gates": gates,
        "passed": all(gates.values()),
        "long": summaries["long"]["cells"],
        "short_b2": summaries["short_b2"]["cells"],
        "short_b4": summaries["short_b4"]["cells"],
        "inputs_sha256": {
            str(path.relative_to(ARTIFACT_DIR)): _sha256(path)
            for path in input_paths
        },
        "admission_source_sha256": _sha256(Path(__file__).resolve()),
    }


def main() -> None:
    payload = build()
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output = ARTIFACT_DIR / "final-admission.json"
    output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
