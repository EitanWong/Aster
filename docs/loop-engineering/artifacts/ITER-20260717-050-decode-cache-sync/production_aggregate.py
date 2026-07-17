#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from token_budget_long_aggregate import aggregate as aggregate_token_budget_long

ARTIFACT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _load_manifest_payloads(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    payloads: list[dict[str, Any]] = []
    for record in manifest["records"]:
        output = _resolve(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        payloads.append(payload)
    return manifest, payloads


def _cell(payload: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(payload["cache_kind"]),
        int(payload["batch_size"]),
        int(payload["context_words"]),
    )


def aggregate(
    production_manifest_path: Path,
    confirmation_manifest_path: Path,
    token_budget_manifest_path: Path,
    token_budget_long_manifest_path: Path,
    token_budget_long_aggregate_path: Path,
) -> dict[str, Any]:
    production_manifest, production_payloads = _load_manifest_payloads(
        production_manifest_path
    )
    confirmation_manifest, confirmation_payloads = _load_manifest_payloads(
        confirmation_manifest_path
    )
    token_manifest, token_payloads = _load_manifest_payloads(token_budget_manifest_path)
    token_long_manifest, token_long_payloads = _load_manifest_payloads(
        token_budget_long_manifest_path
    )
    token_long_aggregate = json.loads(token_budget_long_aggregate_path.read_text())
    recomputed_token_long_aggregate = aggregate_token_budget_long(
        token_budget_long_manifest_path
    )
    expected = len(production_manifest["matrix"]["cells"]) * int(
        production_manifest["matrix"]["runs"]
    )
    if len(production_payloads) != expected:
        raise ValueError(f"expected {expected} production records")
    if len({int(payload["pid"]) for payload in production_payloads}) != expected:
        raise ValueError("production records do not have unique PIDs")

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    baseline_grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    candidate_grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in production_payloads:
        grouped[_cell(payload)].append(payload)
    for payload in confirmation_payloads:
        if payload["policy"] == "baseline":
            baseline_grouped[_cell(payload)].append(payload)
        elif payload["policy"] == "periodic-512":
            candidate_grouped[_cell(payload)].append(payload)
    for payload in token_payloads:
        if payload["policy"] == "periodic-token-512":
            candidate_grouped[_cell(payload)] = [
                *candidate_grouped[_cell(payload)],
                payload,
            ]

    cells: dict[str, Any] = {}
    all_pass = True
    for descriptor in production_manifest["matrix"]["cells"]:
        key = (
            str(descriptor["cache_kind"]),
            int(descriptor["batch_size"]),
            int(descriptor["context_words"]),
        )
        production = grouped[key]
        baseline = baseline_grouped[key]
        if key[1] > 1:
            token_candidates = [
                payload
                for payload in token_payloads
                if _cell(payload) == key and payload["policy"] == "periodic-token-512"
            ]
            reference = token_candidates
        else:
            reference = candidate_grouped[key]
        if not production or not baseline or not reference:
            raise ValueError(f"incomplete bridge cell: {key}")

        production_tps = statistics.median(
            float(payload["decode"]["tokens_per_second"]) for payload in production
        )
        baseline_tps = statistics.median(
            float(payload["decode"]["tokens_per_second"]) for payload in baseline
        )
        reference_tps = statistics.median(
            float(payload["decode"]["tokens_per_second"]) for payload in reference
        )
        speedup = (production_tps / baseline_tps - 1.0) * 100.0
        reference_delta = (production_tps / reference_tps - 1.0) * 100.0
        baseline_rss = statistics.median(
            int(payload["memory"]["rss_peak_bytes"]) for payload in baseline
        )
        production_rss = statistics.median(
            int(payload["memory"]["rss_peak_bytes"]) for payload in production
        )
        rss_change = (production_rss / baseline_rss - 1.0) * 100.0
        reference_payload = reference[0]
        parity = all(
            payload["decode"]["token_ids"] == reference_payload["decode"]["token_ids"]
            and payload["decode"]["text_sha256"]
            == reference_payload["decode"]["text_sha256"]
            and payload["decode"]["cache_digest"]
            == reference_payload["decode"]["cache_digest"]
            for payload in production
        )
        expected_clears = int(production_manifest["matrix"]["max_tokens"]) * key[1] // 512
        policy_counts = all(
            int(payload["policy_metrics"]["clear_executed"]) == expected_clears
            and int(payload["policy_metrics"]["clear_failures"]) == 0
            for payload in production
        )
        swap_zero = all(
            int(payload["memory"]["swap_after_bytes"])
            - int(payload["memory"]["swap_before_bytes"])
            == 0
            for payload in production
        )
        gate = {
            "exact_reference_parity": parity,
            "policy_counts": policy_counts,
            "swap_zero": swap_zero,
            "speedup_over_archived_baseline_at_least_3_percent": speedup >= 3.0,
            "no_more_than_3_percent_below_experimental_candidate": (
                reference_delta >= -3.0
            ),
            "rss_peak_regression_at_most_1_percent": rss_change <= 1.0,
        }
        passed = all(gate.values())
        all_pass &= passed
        name = f"{key[0]}-b{key[1]}-{key[2]}w"
        cells[name] = {
            "production_decode_tps_median": production_tps,
            "archived_baseline_decode_tps_median": baseline_tps,
            "experimental_candidate_decode_tps_median": reference_tps,
            "speedup_over_archived_baseline_percent": speedup,
            "delta_from_experimental_candidate_percent": reference_delta,
            "rss_peak_regression_percent": rss_change,
            "expected_clears": expected_clears,
            "gate": gate,
            "passed": passed,
        }

    long_gate = token_long_aggregate.get("gate")
    long_stress_passed = (
        token_long_aggregate == recomputed_token_long_aggregate
        and token_long_aggregate.get("manifest_sha256")
        == _sha256(token_budget_long_manifest_path)
        and len(token_long_payloads) == 2
        and len({int(payload["pid"]) for payload in token_long_payloads}) == 2
        and token_long_aggregate.get("passed") is True
        and token_long_aggregate.get("production_integration_ready") is True
        and isinstance(long_gate, dict)
        and bool(long_gate)
        and all(value is True for value in long_gate.values())
    )
    admission = {
        "production_bridge_passed": all_pass,
        "token_budget_long_stress_passed": long_stress_passed,
    }
    return {
        "schema_version": 1,
        "production_manifest_sha256": _sha256(production_manifest_path),
        "confirmation_manifest_sha256": _sha256(confirmation_manifest_path),
        "token_budget_manifest_sha256": _sha256(token_budget_manifest_path),
        "token_budget_long_manifest_sha256": _sha256(token_budget_long_manifest_path),
        "token_budget_long_aggregate_sha256": _sha256(token_budget_long_aggregate_path),
        "records": len(production_payloads),
        "unique_pids": len({int(payload["pid"]) for payload in production_payloads}),
        "source_manifests": {
            "production_records": len(production_manifest["records"]),
            "confirmation_records": len(confirmation_manifest["records"]),
            "token_budget_records": len(token_manifest["records"]),
            "token_budget_long_records": len(token_long_manifest["records"]),
        },
        "cells": cells,
        "all_production_bridge_cells_passed": all_pass,
        "admission": admission,
        "integration_approved": all(admission.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/production/execution-manifest.json",
    )
    parser.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/confirmation/execution-manifest.json",
    )
    parser.add_argument(
        "--token-budget-manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/token-budget-confirmation/execution-manifest.json",
    )
    parser.add_argument(
        "--token-budget-long-manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json",
    )
    parser.add_argument(
        "--token-budget-long-aggregate",
        type=Path,
        default=ARTIFACT_DIR / "results/token-budget-long/aggregate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/production/aggregate.json",
    )
    args = parser.parse_args()
    payload = aggregate(
        args.production_manifest.resolve(),
        args.confirmation_manifest.resolve(),
        args.token_budget_manifest.resolve(),
        args.token_budget_long_manifest.resolve(),
        args.token_budget_long_aggregate.resolve(),
    )
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
