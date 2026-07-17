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


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["records"]) != 2:
        raise ValueError("token-budget long manifest requires one A/B pair")
    payloads: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        output = _resolve(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        observed = payload["source_sha256"] | payload["model_input_sha256"]
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in observed.items()
        ):
            raise ValueError(f"source hash mismatch: {output}")
        payloads[str(record["role"])] = payload
    if set(payloads) != {"baseline", "candidate"}:
        raise ValueError("long manifest roles are incomplete")

    baseline = payloads["baseline"]
    candidate = payloads["candidate"]
    parity = (
        baseline["decode"]["token_ids"] == candidate["decode"]["token_ids"]
        and baseline["decode"]["text_sha256"] == candidate["decode"]["text_sha256"]
        and baseline["decode"]["cache_digest"] == candidate["decode"]["cache_digest"]
    )
    speedup = (
        float(baseline["decode"]["elapsed_seconds"])
        / float(candidate["decode"]["elapsed_seconds"])
        - 1.0
    ) * 100.0
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
    first_clear_step = 512 // int(candidate["batch_size"])
    post_clear_curve = [
        sample
        for sample in candidate["memory"]["curve"]
        if int(sample["step"]) >= first_clear_step
    ]
    post_clear_cache_max = max(int(sample["cache_bytes"]) for sample in post_clear_curve)
    max_tokens = int(candidate["settings"]["max_tokens"])
    expected_clears = max_tokens * int(candidate["batch_size"]) // 512
    policy_counts = (
        int(candidate["policy_metrics"]["cache_eval_skipped"]) == max_tokens
        and int(candidate["policy_metrics"]["clear_executed"]) == expected_clears
    )
    swap_zero = all(
        int(payload["memory"]["swap_after_bytes"])
        - int(payload["memory"]["swap_before_bytes"])
        == 0
        for payload in payloads.values()
    )
    gate = {
        "exact_token_text_cache_parity": parity,
        "policy_counts": policy_counts,
        "swap_zero": swap_zero,
        "speedup_at_least_3_percent": speedup >= 3.0,
        "rss_peak_regression_at_most_1_percent": rss <= 1.0,
        "mlx_active_regression_at_most_1_percent": active <= 1.0,
        "mlx_peak_regression_at_most_1_percent": peak <= 1.0,
        "post_first_clear_allocator_cache_below_16_mib": (
            post_clear_cache_max <= 16 * 1024 * 1024
        ),
    }
    passed = all(gate.values())
    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": 2,
        "unique_pids": len({int(payload["pid"]) for payload in payloads.values()}),
        "decode_speedup_percent": speedup,
        "rss_peak_regression_percent": rss,
        "mlx_active_regression_percent": active,
        "mlx_peak_regression_percent": peak,
        "first_clear_step": first_clear_step,
        "expected_clears": expected_clears,
        "post_first_clear_allocator_cache_max_bytes": post_clear_cache_max,
        "gate": gate,
        "passed": passed,
        "production_integration_ready": passed,
        "integration_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/token-budget-long/execution-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/token-budget-long/aggregate.json",
    )
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
