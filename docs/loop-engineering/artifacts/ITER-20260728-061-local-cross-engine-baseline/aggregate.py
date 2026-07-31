#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified(root: Path, descriptor: dict[str, str]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if _sha256(path) != descriptor["sha256"]:
        raise RuntimeError(f"record hash mismatch: {descriptor['path']}")
    return json.loads(path.read_text())


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _quantile(values, 0.95),
    }


def _bootstrap_median_interval(
    values: list[float],
    *,
    seed: int,
    samples: int = 20000,
) -> dict[str, float | int]:
    generator = random.Random(seed)
    count = len(values)
    medians = sorted(
        statistics.median([values[generator.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    return {
        "confidence": 0.95,
        "samples": samples,
        "lower": _quantile(medians, 0.025),
        "upper": _quantile(medians, 0.975),
    }


def _source_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "environment": payload["environment"],
            "source_sha256": payload["source_sha256"],
        },
        sort_keys=True,
    )


def _swap_non_growth(payload: dict[str, Any]) -> bool:
    memory = payload["memory"]
    return memory["swap_after_bytes"] <= memory["swap_before_bytes"]


def _scenario_result(root: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    settings = scenario["settings"]
    pairs = scenario["pairs"]
    failures: list[str] = []
    aster_tps: list[float] = []
    reference_tps: list[float] = []
    aster_rss_growth: list[float] = []
    reference_rss_growth: list[float] = []
    paired_change: list[float] = []
    by_order: dict[str, list[float]] = {"aster_first": [], "mlx_lm_first": []}
    source_fingerprints: dict[str, set[str]] = {"aster": set(), "mlx-lm": set()}
    model_fingerprints: set[str] = set()
    pids: set[int] = set()

    expected_pairs = 6
    if len(pairs) != expected_pairs:
        failures.append(f"expected {expected_pairs} pairs, found {len(pairs)}")
    for pair in pairs:
        order = pair["order"]
        if order not in (["aster", "mlx-lm"], ["mlx-lm", "aster"]):
            failures.append(f"invalid order for pair {pair['pair_number']}")
            continue
        comparison = _load_verified(root, pair["comparison"])
        if not comparison["comparable"] or not all(comparison["gates"].values()):
            failures.append(f"comparison gate failed for pair {pair['pair_number']}")
            continue
        aster = _load_verified(root, pair["records"]["aster"])
        reference = _load_verified(root, pair["records"]["mlx-lm"])
        for engine, payload in (("aster", aster), ("mlx-lm", reference)):
            if payload["engine"] != engine:
                failures.append(f"engine label drift in pair {pair['pair_number']}: {engine}")
            if payload["pid"] in pids:
                failures.append(f"PID reused in pair {pair['pair_number']}: {payload['pid']}")
            pids.add(payload["pid"])
            source_fingerprints[engine].add(_source_fingerprint(payload))
            if not _swap_non_growth(payload):
                failures.append(f"swap grew in pair {pair['pair_number']}: {engine}")
            result = payload["result"]
            if (
                len(result["output_token_ids"]) != settings["max_tokens"]
                or result["finish_reason"] != "length"
            ):
                failures.append(f"fixed-length gate failed in pair {pair['pair_number']}: {engine}")
        if aster["model_input_sha256"] != reference["model_input_sha256"]:
            failures.append(f"model input drift in pair {pair['pair_number']}")
        model_fingerprints.add(json.dumps(aster["model_input_sha256"], sort_keys=True))
        aster_value = float(aster["result"]["output_tokens_per_second"])
        reference_value = float(reference["result"]["output_tokens_per_second"])
        if aster_value <= 0 or reference_value <= 0:
            failures.append(f"non-positive throughput in pair {pair['pair_number']}")
            continue
        aster_tps.append(aster_value)
        reference_tps.append(reference_value)
        aster_rss_growth.append(
            aster["memory"]["rss_after_bytes"] - aster["memory"]["rss_before_bytes"]
        )
        reference_rss_growth.append(
            reference["memory"]["rss_after_bytes"]
            - reference["memory"]["rss_before_bytes"]
        )
        change = (aster_value / reference_value - 1.0) * 100.0
        paired_change.append(change)
        by_order["aster_first" if order[0] == "aster" else "mlx_lm_first"].append(change)

    for engine, fingerprints in source_fingerprints.items():
        if len(fingerprints) != 1:
            failures.append(f"source/environment drift across {engine} records")
    if len(model_fingerprints) != 1:
        failures.append("model input drift across pairs")
    if len(by_order["aster_first"]) != 3 or len(by_order["mlx_lm_first"]) != 3:
        failures.append("AB/BA balance gate failed")
    if len(aster_tps) != expected_pairs or len(reference_tps) != expected_pairs:
        failures.append("missing valid observations")

    result: dict[str, Any] = {
        "name": settings["name"],
        "settings": settings,
        "gates": {
            "independent_processes": len(pids) == expected_pairs * 2,
            "balanced_order": len(by_order["aster_first"]) == 3
            and len(by_order["mlx_lm_first"]) == 3,
            "exact_equivalence": not any("comparison gate" in item for item in failures),
            "fixed_length": not any("fixed-length" in item for item in failures),
            "zero_swap_growth": not any("swap grew" in item for item in failures),
            "stable_sources": not any("drift" in item for item in failures),
            "complete_observations": len(aster_tps) == expected_pairs,
        },
        "failures": failures,
    }
    if aster_tps and reference_tps:
        result["aster_output_tokens_per_second"] = _summary(aster_tps)
        result["mlx_lm_output_tokens_per_second"] = _summary(reference_tps)
        result["paired_change_percent"] = {
            "p50": statistics.median(paired_change),
            "bootstrap_median_interval": _bootstrap_median_interval(
                paired_change,
                seed=61000 + len(settings["name"]),
            ),
            "by_order": {
                name: _summary(values) for name, values in by_order.items() if values
            },
        }
        result["rss_growth_bytes"] = {
            "aster": _summary(aster_rss_growth),
            "mlx_lm": _summary(reference_rss_growth),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen I061 local cross-engine matrix."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["matrix"]["repetitions"] != 6:
        raise SystemExit("formal aggregate requires the predeclared six repetitions")
    scenario_results = [
        _scenario_result(manifest_path.parent, scenario) for scenario in manifest["scenarios"]
    ]
    gates = {
        "manifest_hashable": bool(_sha256(manifest_path)),
        "two_scenarios": len(scenario_results) == 2,
        "all_scenario_gates": all(
            all(result["gates"].values()) and not result["failures"]
            for result in scenario_results
        ),
    }
    payload = {
        "schema_version": 1,
        "manifest_sha256": _sha256(manifest_path),
        "measurement_scope": (
            "Same-host, single-request, raw-token greedy decode comparison between "
            "Aster's manual runtime and direct mlx_lm.stream_generate. It excludes "
            "server APIs, batching, structured decoding, prefix caching, and power."
        ),
        "gates": gates,
        "scenarios": scenario_results,
        "decision": "admit" if all(gates.values()) else "reject",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["decision"] != "admit":
        raise SystemExit("formal matrix rejected")


if __name__ == "__main__":
    main()
