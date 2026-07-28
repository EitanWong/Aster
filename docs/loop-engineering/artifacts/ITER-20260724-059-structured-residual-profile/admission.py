#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]

_AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "iter059_strict_aggregate",
    ARTIFACT_DIR / "strict_aggregate.py",
)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise ImportError("could not load Iteration 059 strict aggregate")
strict_aggregate = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(strict_aggregate)

BUNDLE = ARTIFACT_DIR / "formal-evidence.tar.gz"
RUN_ROOT = "run/loop-engineering/ITER-20260724-059-structured-residual-profile"
PRODUCTION_SOURCE = PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py"
REFERENCE_DIR = ARTIFACT_DIR.parent / "ITER-20260724-058-structured-mask-cache"
FORMAL = {
    "short": ("strict-short-b4-r18", "structured-b4", 18),
    "long": ("strict-long-b2-r18", "structured-b2", 18),
}
MEMORY_CONFIRMATION = {
    "short": (
        "memory-confirm-short-b4-r2",
        "structured-b4",
        4 * 1024**3,
        16 * 1024**2,
        "strict-short-b4-r18",
    ),
    "long": (
        "memory-confirm-long-b2-r2",
        "structured-b2",
        2 * 1024**3,
        8 * 1024**2,
        "strict-long-b2-r18",
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceBundle:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = tarfile.open(path, "r:gz")
        members = [member.name for member in self.archive.getmembers() if member.isfile()]
        if len(members) != len(set(members)):
            raise ValueError("evidence bundle contains duplicate file members")
        self.file_members = frozenset(members)

    def close(self) -> None:
        self.archive.close()

    def read_bytes(self, name: str) -> bytes:
        if name not in self.file_members:
            raise FileNotFoundError(f"missing evidence member: {name}")
        handle = self.archive.extractfile(name)
        if handle is None:
            raise FileNotFoundError(f"unreadable evidence member: {name}")
        return handle.read()

    def read_json(self, name: str) -> dict[str, Any]:
        value = json.loads(self.read_bytes(name))
        if not isinstance(value, dict):
            raise ValueError(f"evidence member must contain a JSON object: {name}")
        return value


def _matrix_evidence(
    bundle: EvidenceBundle,
    result_name: str,
) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    result_root = f"{RUN_ROOT}/{result_name}"
    manifest_name = f"{result_root}/execution-manifest.json"
    manifest_bytes = bundle.read_bytes(manifest_name)
    manifest = json.loads(manifest_bytes)
    payloads: list[dict[str, Any]] = []
    for record in manifest["records"]:
        output = str(record["output"])
        raw = bundle.read_bytes(output)
        if _sha256_bytes(raw) != record["sha256"]:
            raise ValueError(f"evidence payload hash mismatch: {output}")
        payload = json.loads(raw)
        if int(payload["pid"]) != int(record["pid"]):
            raise ValueError(f"evidence payload PID mismatch: {output}")
        payloads.append(payload)
    return manifest_bytes, manifest, payloads


def _sources_match(source_hashes: dict[str, str]) -> bool:
    return all(
        (path := PROJECT_ROOT / relative).is_file() and _sha256(path) == expected
        for relative, expected in source_hashes.items()
    )


def _recompute_aggregate(
    manifest_bytes: bytes,
    manifest: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    original_load = strict_aggregate.base.legacy._load
    strict_aggregate.base.legacy._load = lambda _path: (manifest, payloads)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "execution-manifest.json"
            manifest_path.write_bytes(manifest_bytes)
            return strict_aggregate.aggregate(manifest_path)
    finally:
        strict_aggregate.base.legacy._load = original_load


def _expected_forced_misses(payload: dict[str, Any]) -> int:
    settings = payload["settings"]
    return (int(settings["steps"]) + int(settings["pair_warmup_steps"])) * int(
        payload["batch_size"]
    )


def _forced_misses_match(payloads: list[dict[str, Any]]) -> bool:
    return all(
        payload["policy_metrics"]["baseline"]["forced_eos_membership_misses"]
        == _expected_forced_misses(payload)
        and payload["policy_metrics"]["production"]["forced_eos_membership_misses"] == 0
        for payload in payloads
    )


def _reference_mlx_median(result_name: str) -> float:
    paths = sorted((REFERENCE_DIR / "results" / result_name).glob("structured-*.json"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 Iteration 058 reference payloads: {result_name}")
    peaks = [int(json.loads(path.read_text())["memory"]["mlx_peak_bytes"]) for path in paths]
    return statistics.median(peaks)


def _formal_result(
    bundle: EvidenceBundle,
    *,
    result_name: str,
    cell_name: str,
    expected_records: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_bytes, manifest, payloads = _matrix_evidence(bundle, result_name)
    saved = bundle.read_json(f"{RUN_ROOT}/{result_name}/aggregate.json")
    recomputed = _recompute_aggregate(manifest_bytes, manifest, payloads)
    cell = recomputed["cells"][cell_name]
    result = {
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "aggregate_recomputed_exactly": recomputed == saved,
        "aggregate_passed": recomputed["all_cells_passed"],
        "all_exact_token_text_cache": all(
            payload["parity"]["exact_token_text_cache"] is True for payload in payloads
        ),
        "all_swap_non_growth": all(
            payload["memory"]["swap_after_bytes"] <= payload["memory"]["swap_before_bytes"]
            for payload in payloads
        ),
        "all_source_bound": _sources_match(manifest["source_sha256"]),
        "all_baseline_calls_forced_miss": _forced_misses_match(payloads),
        "stable_replicates": cell["stable_replicates"],
        "balanced_interval_percent": cell["intervals"]["balanced"],
        "baseline_first_interval_percent": cell["intervals"]["baseline_first"],
        "production_first_interval_percent": cell["intervals"]["production_first"],
    }
    result["passed"] = (
        result["records"] == expected_records
        and result["unique_pids"] == expected_records
        and result["aggregate_recomputed_exactly"] is True
        and result["aggregate_passed"] is True
        and result["all_exact_token_text_cache"] is True
        and result["all_swap_non_growth"] is True
        and result["all_source_bound"] is True
        and result["all_baseline_calls_forced_miss"] is True
    )
    return result, payloads


def _memory_result(
    bundle: EvidenceBundle,
    *,
    result_name: str,
    rss_limit_bytes: int,
    mlx_delta_limit_bytes: int,
    reference_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _manifest_bytes, manifest, payloads = _matrix_evidence(bundle, result_name)
    rss_deltas = [
        int(payload["memory"]["rss_after_bytes"]) - int(payload["memory"]["rss_before_bytes"])
        for payload in payloads
    ]
    mlx_peaks = [int(payload["memory"]["mlx_peak_bytes"]) for payload in payloads]
    reference_median = _reference_mlx_median(reference_name)
    observed_median = statistics.median(mlx_peaks)
    result = {
        "records": len(payloads),
        "unique_pids": len({int(payload["pid"]) for payload in payloads}),
        "rss_limit_bytes": rss_limit_bytes,
        "rss_delta_bytes": rss_deltas,
        "all_rss_within_limit": all(delta <= rss_limit_bytes for delta in rss_deltas),
        "mlx_reference_iteration": "ITER-20260724-058-structured-mask-cache",
        "mlx_reference_median_bytes": reference_median,
        "mlx_observed_median_bytes": observed_median,
        "mlx_median_delta_bytes": observed_median - reference_median,
        "mlx_delta_limit_bytes": mlx_delta_limit_bytes,
        "mlx_median_within_limit": (observed_median - reference_median <= mlx_delta_limit_bytes),
        "all_swap_non_growth": all(
            payload["memory"]["swap_after_bytes"] <= payload["memory"]["swap_before_bytes"]
            for payload in payloads
        ),
        "all_exact_token_text_cache": all(
            payload["parity"]["exact_token_text_cache"] is True for payload in payloads
        ),
        "all_source_bound": _sources_match(manifest["source_sha256"]),
        "all_baseline_calls_forced_miss": _forced_misses_match(payloads),
    }
    result["passed"] = (
        result["records"] == 2
        and result["unique_pids"] == 2
        and result["all_rss_within_limit"] is True
        and result["mlx_median_within_limit"] is True
        and result["all_swap_non_growth"] is True
        and result["all_exact_token_text_cache"] is True
        and result["all_source_bound"] is True
        and result["all_baseline_calls_forced_miss"] is True
    )
    return result, payloads


def build_admission() -> dict[str, Any]:
    bundle = EvidenceBundle(BUNDLE)
    try:
        formal: dict[str, Any] = {}
        confirmations: dict[str, Any] = {}
        all_payloads: list[dict[str, Any]] = []
        for label, (result_name, cell_name, expected_records) in FORMAL.items():
            formal[label], payloads = _formal_result(
                bundle,
                result_name=result_name,
                cell_name=cell_name,
                expected_records=expected_records,
            )
            all_payloads.extend(payloads)
        for label, (
            result_name,
            _cell_name,
            rss_limit_bytes,
            mlx_delta_limit_bytes,
            reference_name,
        ) in MEMORY_CONFIRMATION.items():
            confirmations[label], payloads = _memory_result(
                bundle,
                result_name=result_name,
                rss_limit_bytes=rss_limit_bytes,
                mlx_delta_limit_bytes=mlx_delta_limit_bytes,
                reference_name=reference_name,
            )
            all_payloads.extend(payloads)

        validation = bundle.read_json(f"{RUN_ROOT}/structured-validation-b4-run-1.json")
        production_screens = {
            label: bundle.read_json(f"{RUN_ROOT}/production-screen-{name}-run-1.json")
            for label, name in (("short", "short-b4"), ("long", "long-b2"))
        }
        screen_summary = {
            label: {
                "baseline_tokens_per_second": payload["timings"]["baseline_tokens_per_second"],
                "production_tokens_per_second": payload["timings"]["production_tokens_per_second"],
                "speedup_percent": (
                    payload["timings"]["production_tokens_per_second"]
                    / payload["timings"]["baseline_tokens_per_second"]
                    - 1.0
                )
                * 100.0,
                "exact_token_text_cache": payload["parity"]["exact_token_text_cache"],
            }
            for label, payload in production_screens.items()
        }
        model_signatures = [payload["model_input_sha256"] for payload in all_payloads]
        gates = {
            "short_formal_admission": formal["short"]["passed"],
            "long_formal_admission": formal["long"]["passed"],
            "short_memory_confirmation": confirmations["short"]["passed"],
            "long_memory_confirmation": confirmations["long"]["passed"],
            "model_signature_equal": all(
                signature == model_signatures[0] for signature in model_signatures
            ),
            "structured_schema_valid": validation["all_schema_valid"] is True,
            "structured_stop_valid": validation["all_stopped_before_limit"] is True,
            "dynamic_membership_shrinks": (
                validation["membership_sizes"][0] == 4 and validation["membership_sizes"][-1] == 1
            ),
            "validation_source_bound": _sources_match(validation["source_sha256"]),
            "validation_swap_non_growth": (
                validation["memory"]["swap_after_bytes"]
                <= validation["memory"]["swap_before_bytes"]
            ),
            "production_screens_clear_floor": all(
                summary["speedup_percent"] >= 3.0 and summary["exact_token_text_cache"] is True
                for summary in screen_summary.values()
            ),
            "compact_evidence_budget": (
                len(bundle.file_members) == 50 and BUNDLE.stat().st_size <= 5 * 1024**2
            ),
        }
        return {
            "schema_version": 1,
            "decision": "admit" if all(gates.values()) else "reject",
            "production_source_sha256": _sha256(PRODUCTION_SOURCE),
            "evidence": {
                "path": str(BUNDLE.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(BUNDLE),
                "bytes": BUNDLE.stat().st_size,
                "file_members": len(bundle.file_members),
            },
            "formal": formal,
            "memory_confirmation": confirmations,
            "structured_validation": {
                "lanes": len(validation["lanes"]),
                "membership_start": validation["membership_sizes"][0],
                "membership_end": validation["membership_sizes"][-1],
            },
            "production_screens": screen_summary,
            "protocol_deviation": (
                "The first formal matrix discovered memory behavior because numeric "
                "ceilings were absent from CURRENT.json. Admission uses the fresh "
                "confirmation executed after explicit ceilings were recorded."
            ),
            "gates": gates,
            "passed": all(gates.values()),
        }
    finally:
        bundle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_DIR / "final-admission.json",
    )
    args = parser.parse_args()
    payload = build_admission()
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
