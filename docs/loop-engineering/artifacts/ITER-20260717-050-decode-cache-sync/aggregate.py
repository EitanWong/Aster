#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_median_interval(
    values: list[float],
    *,
    seed: int = 20260717,
    samples: int = 20_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _resolve_output(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _load_records(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    matrix = manifest["matrix"]
    expected = len(matrix["policies"]) * len(matrix["contexts"]) * int(matrix["runs"])
    records = manifest["records"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    if len({int(record["pid"]) for record in records}) != expected:
        raise ValueError("benchmark records do not have unique process IDs")

    payloads: list[dict[str, Any]] = []
    cells: set[tuple[int, str, int]] = set()
    for record in records:
        output = _resolve_output(record["output"])
        if _sha256(output) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        payload = json.loads(output.read_text())
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"PID mismatch: {output}")
        if payload["environment"]["git_commit"] != manifest["environment"]["git_commit"]:
            raise ValueError(f"commit mismatch: {output}")
        observed_hashes = payload["source_sha256"] | payload["model_input_sha256"]
        if any(
            manifest["source_sha256"].get(path) != digest
            for path, digest in observed_hashes.items()
        ):
            raise ValueError(f"source hash mismatch: {output}")
        cell = (int(payload["context_words"]), str(payload["policy"]), int(payload["run_id"]))
        if cell in cells:
            raise ValueError(f"duplicate cell: {cell}")
        cells.add(cell)
        payloads.append(payload)
    return manifest, payloads


def _paired_percent(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _cell_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    tps = [float(payload["decode"]["tokens_per_second"]) for payload in payloads]
    step_ms = [float(payload["decode"]["step_seconds"]["median"]) * 1000 for payload in payloads]
    rss = [int(payload["memory"]["rss_peak_bytes"]) for payload in payloads]
    active = [int(payload["memory"]["mlx_after_decode"]["active_bytes"]) for payload in payloads]
    cache = [int(payload["memory"]["mlx_after_decode"]["cache_bytes"]) for payload in payloads]
    peak = [int(payload["memory"]["mlx_after_decode"]["peak_bytes"]) for payload in payloads]
    return {
        "runs": len(payloads),
        "decode_tps_median": statistics.median(tps),
        "decode_tps_min": min(tps),
        "decode_tps_max": max(tps),
        "step_ms_median": statistics.median(step_ms),
        "rss_peak_bytes_median": statistics.median(rss),
        "mlx_active_bytes_median": statistics.median(active),
        "mlx_cache_bytes_median": statistics.median(cache),
        "mlx_peak_bytes_median": statistics.median(peak),
        "swap_delta_max_bytes": max(
            int(payload["memory"]["swap_after_bytes"])
            - int(payload["memory"]["swap_before_bytes"])
            for payload in payloads
        ),
    }


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest, payloads = _load_records(manifest_path)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_run: dict[tuple[int, str, int], dict[str, Any]] = {}
    for payload in payloads:
        context = int(payload["context_words"])
        policy = str(payload["policy"])
        run_id = int(payload["run_id"])
        grouped[(context, policy)].append(payload)
        by_run[(context, policy, run_id)] = payload

    contexts: dict[str, Any] = {}
    all_parity = True
    candidate_passes: dict[str, bool] = {}
    for context in manifest["matrix"]["contexts"]:
        context = int(context)
        baseline_payloads = grouped[(context, "baseline")]
        baseline_by_run = {int(payload["run_id"]): payload for payload in baseline_payloads}
        policies: dict[str, Any] = {}
        for policy in manifest["matrix"]["policies"]:
            cell = sorted(grouped[(context, policy)], key=lambda payload: int(payload["run_id"]))
            summary = _cell_summary(cell)
            if policy == "baseline":
                summary["paired_decode_tps_change_percent"] = [0.0] * len(cell)
                summary["paired_decode_tps_change_median_percent"] = 0.0
                summary["paired_decode_tps_bootstrap95_percent"] = [0.0, 0.0]
                summary["token_text_cache_parity"] = True
                policies[policy] = summary
                continue

            paired = []
            parity = True
            for candidate in cell:
                run_id = int(candidate["run_id"])
                baseline = baseline_by_run[run_id]
                paired.append(
                    _paired_percent(
                        float(candidate["decode"]["tokens_per_second"]),
                        float(baseline["decode"]["tokens_per_second"]),
                    )
                )
                parity &= candidate["decode"]["token_ids"] == baseline["decode"]["token_ids"]
                parity &= candidate["decode"]["text_sha256"] == baseline["decode"]["text_sha256"]
                parity &= candidate["decode"]["cache_digest"] == baseline["decode"]["cache_digest"]
            interval = _bootstrap_median_interval(paired, seed=20260717 + context)
            summary["paired_decode_tps_change_percent"] = paired
            summary["paired_decode_tps_change_median_percent"] = statistics.median(paired)
            summary["paired_decode_tps_bootstrap95_percent"] = list(interval)
            summary["token_text_cache_parity"] = parity
            policies[policy] = summary
            all_parity &= parity
        contexts[str(context)] = policies

    for policy in manifest["matrix"]["policies"]:
        if policy == "baseline":
            continue
        policy_cells = [contexts[str(context)][policy] for context in manifest["matrix"]["contexts"]]
        candidate_passes[policy] = all(
            cell["token_text_cache_parity"]
            and cell["paired_decode_tps_change_median_percent"] >= 3.0
            and cell["paired_decode_tps_bootstrap95_percent"][0] > 0.0
            and cell["swap_delta_max_bytes"] == 0
            for cell in policy_cells
        )

    selected = "periodic-512" if candidate_passes.get("periodic-512", False) else None
    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "all_token_text_cache_parity": all_parity,
        "contexts": contexts,
        "screen_gate": {
            "requirements": {
                "paired_decode_tps_median_percent": 3.0,
                "paired_bootstrap_lower_percent": 0.0,
                "swap_delta_bytes": 0,
                "exact_token_text_cache_parity": True,
            },
            "candidate_passes": candidate_passes,
            "selected_for_confirmation": selected,
            "integration_approved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ARTIFACT_DIR / "results/screen/execution-manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "results/screen/aggregate.json",
    )
    args = parser.parse_args()
    payload = aggregate(args.manifest.resolve())
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
