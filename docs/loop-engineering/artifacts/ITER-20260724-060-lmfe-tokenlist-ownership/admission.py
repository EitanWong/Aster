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
BUNDLE = ARTIFACT_DIR / "formal-evidence.tar.gz"
RUN_ROOT = "run/loop-engineering/ITER-20260724-060-lmfe-tokenlist-ownership"
PRODUCTION_SOURCE = PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py"
FORMAL = {
    "short": ("strict-short-b4-r18", "structured-b4", 4),
    "long": ("strict-long-b2-r18", "structured-b2", 2),
}
MEMORY = {
    "short": {
        "batch_size": 4,
        "context_words": 128,
        "steps": 256,
        "prompt_tokens": 409,
        "baseline": ("memory-short-b4-baseline-r1.json", "memory-short-b4-baseline-r2.json"),
        "production": (
            "memory-short-b4-production-r1.json",
            "memory-short-b4-production-r2.json",
        ),
    },
    "long": {
        "batch_size": 2,
        "context_words": 8192,
        "steps": 128,
        "prompt_tokens": 24601,
        "baseline": ("memory-long-b2-baseline-r1.json", "memory-long-b2-baseline-r2.json"),
        "production": (
            "memory-long-b2-production-r1.json",
            "memory-long-b2-production-r2.json",
        ),
    },
}
VALIDATION = "structured-validation-b4-run-1.json"

_AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "iter060_strict_aggregate",
    ARTIFACT_DIR / "strict_aggregate.py",
)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise ImportError("could not load Iteration 060 strict aggregate")
strict_aggregate = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(strict_aggregate)


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


def expected_members() -> frozenset[str]:
    members: set[str] = {f"{RUN_ROOT}/{VALIDATION}"}
    for result_name, cell_name, _batch_size in FORMAL.values():
        result_root = f"{RUN_ROOT}/{result_name}"
        members.add(f"{result_root}/execution-manifest.json")
        members.add(f"{result_root}/aggregate.json")
        for run_id in range(1, 19):
            members.add(f"{result_root}/{cell_name}-run-{run_id}.json")
    for settings in MEMORY.values():
        for group in ("baseline", "production"):
            members.update(f"{RUN_ROOT}/{name}" for name in settings[group])
    return frozenset(members)


def _sources_match(source_hashes: dict[str, str]) -> bool:
    return all(
        (path := PROJECT_ROOT / relative).is_file() and _sha256(path) == expected
        for relative, expected in source_hashes.items()
    )


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


def _formal_result(
    bundle: EvidenceBundle,
    *,
    result_name: str,
    cell_name: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_bytes, manifest, payloads = _matrix_evidence(bundle, result_name)
    saved = bundle.read_json(f"{RUN_ROOT}/{result_name}/aggregate.json")
    recomputed = _recompute_aggregate(manifest_bytes, manifest, payloads)
    cell = recomputed["cells"][cell_name]
    baseline_prefix_states = [
        int(payload["policy_metrics"]["baseline"]["max_prefix_state_count"])
        for payload in payloads
    ]
    production_prefix_states = [
        int(payload["policy_metrics"]["production"]["max_prefix_state_count"])
        for payload in payloads
    ]
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
        "all_batch_sizes_match": all(int(payload["batch_size"]) == batch_size for payload in payloads),
        "stable_replicates": cell["stable_replicates"],
        "balanced_interval_percent": cell["intervals"]["balanced"],
        "baseline_first_interval_percent": cell["intervals"]["baseline_first"],
        "production_first_interval_percent": cell["intervals"]["production_first"],
        "baseline_prefix_states": {
            "minimum": min(baseline_prefix_states),
            "maximum": max(baseline_prefix_states),
        },
        "production_prefix_states": {
            "minimum": min(production_prefix_states),
            "maximum": max(production_prefix_states),
        },
    }
    result["passed"] = (
        result["records"] == 18
        and result["unique_pids"] == 18
        and result["aggregate_recomputed_exactly"] is True
        and result["aggregate_passed"] is True
        and result["all_exact_token_text_cache"] is True
        and result["all_swap_non_growth"] is True
        and result["all_source_bound"] is True
        and result["all_batch_sizes_match"] is True
        and result["stable_replicates"] == 9
        and result["balanced_interval_percent"]["lower"] >= 3.0
        and result["baseline_first_interval_percent"]["lower"] >= 3.0
        and result["production_first_interval_percent"]["lower"] >= 3.0
        and result["production_prefix_states"]["maximum"] <= batch_size
    )
    return result, payloads


def _profile_matches(
    payload: dict[str, Any],
    *,
    mode: str,
    batch_size: int,
    context_words: int,
    steps: int,
    prompt_tokens: int,
) -> bool:
    workload = payload["workload"]
    return (
        workload["name"] == "structured"
        and workload["freetext_allowlist_mode"] == mode
        and bool(workload["bounded_prefix_states"]) == (mode == "reused_list_backing")
        and int(workload["batch_size"]) == batch_size
        and int(workload["context_words"]) == context_words
        and int(workload["steps"]) == steps
        and int(workload["prompt_tokens"]) == prompt_tokens
    )


def _memory_result(
    bundle: EvidenceBundle,
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    baseline = [bundle.read_json(f"{RUN_ROOT}/{name}") for name in settings["baseline"]]
    production = [bundle.read_json(f"{RUN_ROOT}/{name}") for name in settings["production"]]
    batch_size = int(settings["batch_size"])
    steps = int(settings["steps"])
    profile_args = {
        "batch_size": batch_size,
        "context_words": int(settings["context_words"]),
        "steps": steps,
        "prompt_tokens": int(settings["prompt_tokens"]),
    }
    baseline_rss = [int(payload["ownership_summary"]["rss_delta_bytes"]) for payload in baseline]
    production_rss = [int(payload["ownership_summary"]["rss_delta_bytes"]) for payload in production]
    baseline_median = statistics.median(baseline_rss)
    production_median = statistics.median(production_rss)
    reduction_percent = (1.0 - production_median / baseline_median) * 100.0
    all_profiles = [*baseline, *production]
    output_hashes = {
        tuple(str(value) for value in payload["decode"]["output_text_sha256"])
        for payload in all_profiles
    }
    result = {
        "records": len(all_profiles),
        "unique_pids": len({int(payload["pid"]) for payload in all_profiles}),
        "baseline_rss_delta_bytes": baseline_rss,
        "production_rss_delta_bytes": production_rss,
        "baseline_rss_median_bytes": baseline_median,
        "production_rss_median_bytes": production_median,
        "rss_reduction_percent": reduction_percent,
        "all_source_bound": all(_sources_match(payload["source_sha256"]) for payload in all_profiles),
        "all_output_hashes_equal": len(output_hashes) == 1,
        "all_post_release_request_token_lists_gone": all(
            int(payload["post_release"]["request_token_lists_still_live"]) == 0
            for payload in all_profiles
        ),
        "baseline_workload_matches": all(
            _profile_matches(payload, mode="native", **profile_args) for payload in baseline
        ),
        "production_workload_matches": all(
            _profile_matches(payload, mode="reused_list_backing", **profile_args)
            for payload in production
        ),
        "baseline_prefix_state_counts": [
            int(payload["ownership_summary"]["prefix_states_last"]) for payload in baseline
        ],
        "production_prefix_state_counts": [
            int(payload["ownership_summary"]["prefix_states_last"]) for payload in production
        ],
        "production_working_list_counts": [
            int(payload["ownership_summary"]["working_freetext_lists_last"])
            for payload in production
        ],
    }
    result["passed"] = (
        result["records"] == 4
        and result["unique_pids"] == 4
        and result["all_source_bound"] is True
        and result["all_output_hashes_equal"] is True
        and result["all_post_release_request_token_lists_gone"] is True
        and result["baseline_workload_matches"] is True
        and result["production_workload_matches"] is True
        and result["baseline_prefix_state_counts"] == [batch_size * steps] * 2
        and result["production_prefix_state_counts"] == [batch_size] * 2
        and result["production_working_list_counts"] == [batch_size] * 2
        and result["rss_reduction_percent"] >= 25.0
    )
    return result


def build_admission() -> dict[str, Any]:
    bundle = EvidenceBundle(BUNDLE)
    try:
        formal: dict[str, Any] = {}
        formal_payloads: list[dict[str, Any]] = []
        for label, (result_name, cell_name, batch_size) in FORMAL.items():
            formal[label], payloads = _formal_result(
                bundle,
                result_name=result_name,
                cell_name=cell_name,
                batch_size=batch_size,
            )
            formal_payloads.extend(payloads)
        memory = {
            label: _memory_result(bundle, settings=settings)
            for label, settings in MEMORY.items()
        }
        validation = bundle.read_json(f"{RUN_ROOT}/{VALIDATION}")
        model_signatures = [
            json.dumps(payload["model_input_sha256"], sort_keys=True)
            for payload in formal_payloads
        ]
        model_signatures.append(json.dumps(validation["model_input_sha256"], sort_keys=True))
        expected = expected_members()
        gates = {
            "short_formal_admission": formal["short"]["passed"],
            "long_formal_admission": formal["long"]["passed"],
            "short_memory_admission": memory["short"]["passed"],
            "long_memory_admission": memory["long"]["passed"],
            "model_signature_equal": len(set(model_signatures)) == 1,
            "structured_schema_valid": validation["all_schema_valid"] is True,
            "structured_stop_valid": validation["all_stopped_before_limit"] is True,
            "dynamic_membership_shrinks": (
                validation["membership_sizes"][0] == 4
                and validation["membership_sizes"][-1] < validation["membership_sizes"][0]
            ),
            "validation_source_bound": _sources_match(validation["source_sha256"]),
            "validation_swap_non_growth": (
                validation["memory"]["swap_after_bytes"]
                <= validation["memory"]["swap_before_bytes"]
            ),
            "compact_evidence_exact_members": bundle.file_members == expected,
            "compact_evidence_budget": BUNDLE.stat().st_size <= 5 * 1024**2,
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
            "memory": memory,
            "structured_validation": {
                "lanes": len(validation["lanes"]),
                "membership_start": validation["membership_sizes"][0],
                "membership_end": validation["membership_sizes"][-1],
            },
            "gates": gates,
            "passed": all(gates.values()),
        }
    finally:
        bundle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "final-admission.json")
    args = parser.parse_args()
    payload = build_admission()
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
