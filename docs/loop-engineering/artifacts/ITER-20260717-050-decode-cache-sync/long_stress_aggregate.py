#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _speedup(candidate: float, baseline: float) -> float:
    return (baseline / candidate - 1.0) * 100.0


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    expected = len(manifest["matrix"]["cells"]) * len(manifest["matrix"]["policies"])
    records = manifest["records"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    if len({int(record["pid"]) for record in records}) != expected:
        raise ValueError("stress records do not have unique PIDs")

    payloads: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        output = _resolve(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        if payload["environment"]["git_commit"] != manifest["environment"]["git_commit"]:
            raise ValueError(f"commit mismatch: {output}")
        observed = payload["source_sha256"] | payload["model_input_sha256"]
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in observed.items()
        ):
            raise ValueError(f"source hash mismatch: {output}")
        key = (
            str(payload["cache_kind"]),
            int(payload["batch_size"]),
            int(payload["settings"]["max_tokens"]),
        )
        payloads.setdefault(key, {})[str(payload["policy"])] = payload

    cells: dict[str, Any] = {}
    all_pass = True
    for cell in manifest["matrix"]["cells"]:
        key = (
            str(cell["cache_kind"]),
            int(cell["batch_size"]),
            int(cell["max_tokens"]),
        )
        pair = payloads[key]
        if set(pair) != {"baseline", "periodic-512"}:
            raise ValueError(f"incomplete stress pair: {key}")
        baseline = pair["baseline"]
        candidate = pair["periodic-512"]
        parity = (
            baseline["decode"]["token_ids"] == candidate["decode"]["token_ids"]
            and baseline["decode"]["text_sha256"] == candidate["decode"]["text_sha256"]
            and baseline["decode"]["cache_digest"] == candidate["decode"]["cache_digest"]
        )
        speed = _speedup(
            float(candidate["decode"]["elapsed_seconds"]),
            float(baseline["decode"]["elapsed_seconds"]),
        )
        rss = _change(
            float(candidate["memory"]["rss_peak_bytes"]),
            float(baseline["memory"]["rss_peak_bytes"]),
        )
        active = _change(
            float(candidate["memory"]["mlx_after_decode"]["active_bytes"]),
            float(baseline["memory"]["mlx_after_decode"]["active_bytes"]),
        )
        peak = _change(
            float(candidate["memory"]["mlx_after_decode"]["peak_bytes"]),
            float(baseline["memory"]["mlx_after_decode"]["peak_bytes"]),
        )
        max_tokens = int(cell["max_tokens"])
        policy_counts = (
            int(baseline["policy_metrics"]["cache_eval_executed"]) == max_tokens
            and int(baseline["policy_metrics"]["clear_executed"]) == max_tokens
            and int(candidate["policy_metrics"]["cache_eval_skipped"]) == max_tokens
            and int(candidate["policy_metrics"]["clear_executed"]) == max_tokens // 512
        )
        swap_zero = all(
            int(payload["memory"]["swap_after_bytes"])
            - int(payload["memory"]["swap_before_bytes"])
            == 0
            for payload in pair.values()
        )
        candidate_cache_max = max(
            int(sample["cache_bytes"]) for sample in candidate["memory"]["curve"]
        )
        gate = {
            "exact_token_text_cache_parity": parity,
            "policy_counts": policy_counts,
            "swap_zero": swap_zero,
            "speedup_at_least_3_percent": speed >= 3.0,
            "rss_peak_regression_at_most_1_percent": rss <= 1.0,
            "mlx_active_regression_at_most_1_percent": active <= 1.0,
            "mlx_peak_regression_at_most_1_percent": peak <= 1.0,
            "candidate_allocator_cache_below_256_mib": candidate_cache_max <= 256 * 1024 * 1024,
        }
        passed = all(gate.values())
        all_pass &= passed
        name = f"{key[0]}-b{key[1]}-{key[2]}t"
        cells[name] = {
            "decode_speedup_percent": speed,
            "rss_peak_regression_percent": rss,
            "mlx_active_regression_percent": active,
            "mlx_peak_regression_percent": peak,
            "candidate_allocator_cache_max_bytes": candidate_cache_max,
            "gate": gate,
            "passed": passed,
        }

    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": len(records),
        "unique_pids": len({int(record["pid"]) for record in records}),
        "cells": cells,
        "admission": {
            "all_long_stress_cells_passed": all_pass,
            "production_integration_ready": all_pass,
            "integration_approved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/long-stress/execution-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/long-stress/aggregate.json",
    )
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
