from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any

SUMMARY_KEYS = ("median_ms", "p95_ms", "min_ms", "max_ms", "stdev_ms")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_repo_root(artifact_root: Path) -> Path:
    for candidate in (artifact_root, *artifact_root.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Cannot locate repository root")


def verify_manifest(artifact_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["artifact_sources"].items():
        actual = sha256(artifact_root / relative_path)
        if actual != expected:
            raise ValueError(f"Artifact source hash mismatch: {relative_path}")

    repo_root = find_repo_root(artifact_root)
    for relative_path, expected in manifest["repository_sources"].items():
        actual = sha256(repo_root / relative_path)
        if actual != expected:
            raise ValueError(f"Repository source hash mismatch: {relative_path}")

    reference = manifest["reference"]
    reference_repo = repo_root / reference["repo_path"]
    head = subprocess.run(
        ["git", "-C", str(reference_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != reference["commit"]:
        raise ValueError("Reference repository commit does not match manifest")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(reference_repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("Reference repository tracked files are dirty")
    return manifest, sha256(manifest_path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
    }


def validate_measurements(
    record: dict[str, Any],
    methods: frozenset[str],
    *,
    expected_warmups: int,
    expected_iterations: int,
) -> None:
    if record.get("warmups") != expected_warmups:
        raise ValueError("Benchmark warmup count does not match")
    if record.get("iterations") != expected_iterations:
        raise ValueError("Benchmark iteration count does not match")
    if set(record.get("samples_ms", {})) != methods:
        raise ValueError("Benchmark sample methods do not match")
    if set(record.get("methods", {})) != methods:
        raise ValueError("Benchmark summary methods do not match")

    for method in methods:
        samples = record["samples_ms"][method]
        if not isinstance(samples, list) or len(samples) != expected_iterations:
            raise ValueError("Benchmark sample count does not match")
        if any(not math.isfinite(sample) or sample <= 0.0 for sample in samples):
            raise ValueError("Benchmark samples must be finite and positive")
        actual_summary = record["methods"][method]
        expected_summary = summarize(samples)
        if set(actual_summary) != set(SUMMARY_KEYS):
            raise ValueError("Benchmark summary fields do not match")
        for key, expected in expected_summary.items():
            actual = actual_summary[key]
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("Benchmark summary does not match raw samples")


def validate_cells(actual: list[Any], expected: frozenset[Any]) -> None:
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate benchmark cells are not allowed")
    if set(actual) != expected:
        raise ValueError("Benchmark cells do not match")


def process_delta(record: dict[str, Any], baseline: str, candidate: str) -> float:
    baseline_value = statistics.median(record["samples_ms"][baseline])
    candidate_value = statistics.median(record["samples_ms"][candidate])
    return 100.0 * (candidate_value / baseline_value - 1.0)


def paired_point_delta(records: list[dict[str, Any]], baseline: str, candidate: str) -> float:
    return statistics.median(process_delta(record, baseline, candidate) for record in records)


def method_point(records: list[dict[str, Any]], method: str) -> float:
    return statistics.median(statistics.median(record["samples_ms"][method]) for record in records)


def validate_recorded_delta(
    record: dict[str, Any], field: str, baseline: str, candidate: str
) -> None:
    actual = record.get(field)
    expected = process_delta(record, baseline, candidate)
    if not isinstance(actual, (int, float)) or not math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Recorded delta does not match raw samples: {field}")


def bootstrap_delta(
    records: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    *,
    resamples: int,
    generator: random.Random,
) -> list[float]:
    deltas: list[float] = []
    for _ in range(resamples):
        selected = generator.choices(records, k=len(records))
        process_deltas: list[float] = []
        for record in selected:
            baseline_samples = record["samples_ms"][baseline]
            candidate_samples = record["samples_ms"][candidate]
            block_size = max(2, math.isqrt(len(baseline_samples)))
            indices: list[int] = []
            while len(indices) < len(baseline_samples):
                start = generator.randrange(len(baseline_samples))
                indices.extend(
                    (start + offset) % len(baseline_samples) for offset in range(block_size)
                )
            indices = indices[: len(baseline_samples)]
            baseline_value = statistics.median(baseline_samples[index] for index in indices)
            candidate_value = statistics.median(candidate_samples[index] for index in indices)
            process_deltas.append(100.0 * (candidate_value / baseline_value - 1.0))
        deltas.append(statistics.median(process_deltas))
    return [percentile(deltas, 0.025), percentile(deltas, 0.975)]
