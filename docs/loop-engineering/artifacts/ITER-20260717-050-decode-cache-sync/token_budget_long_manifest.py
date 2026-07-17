#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
LONG_DIR = ARTIFACT_DIR / "results/long-stress"
OUTPUT_DIR = ARTIFACT_DIR / "results/token-budget-long"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ARTIFACT_DIR / candidate


def _find_baseline_record(manifest: dict[str, Any]) -> dict[str, Any]:
    for record in manifest["records"]:
        payload = json.loads(_resolve(record["output"]).read_text())
        if (
            payload["cache_kind"] == "native"
            and int(payload["batch_size"]) == 4
            and int(payload["settings"]["max_tokens"]) == 4096
            and payload["policy"] == "baseline"
        ):
            return record
    raise ValueError("native batch-4 long baseline record is missing")


def main() -> None:
    source_manifest_path = LONG_DIR / "execution-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    baseline_record = _find_baseline_record(source_manifest)
    baseline_path = _resolve(baseline_record["output"])
    if _sha256(baseline_path) != baseline_record["sha256"]:
        raise ValueError("baseline output hash mismatch")

    candidate_path = LONG_DIR / "native-b4-128w-4096t-periodic-token-512.json"
    candidate = json.loads(candidate_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if baseline["environment"]["git_commit"] != candidate["environment"]["git_commit"]:
        raise ValueError("baseline and candidate commits differ")
    baseline_sources = baseline["source_sha256"] | baseline["model_input_sha256"]
    candidate_sources = candidate["source_sha256"] | candidate["model_input_sha256"]
    for path in set(baseline_sources) & set(candidate_sources):
        if baseline_sources[path] != candidate_sources[path]:
            raise ValueError(f"shared source hash mismatch: {path}")

    records = [
        {
            **baseline_record,
            "role": "baseline",
            "source_manifest": str(source_manifest_path.relative_to(ARTIFACT_DIR)),
            "source_manifest_sha256": _sha256(source_manifest_path),
        },
        {
            "output": str(candidate_path.relative_to(ARTIFACT_DIR)),
            "sha256": _sha256(candidate_path),
            "pid": int(candidate["pid"]),
            "role": "candidate",
            "command": [
                str(PROJECT_ROOT / ".venv/bin/python"),
                str(ARTIFACT_DIR / "token_budget_benchmark.py"),
                "--policy",
                "periodic-token-512",
                "--cache-kind",
                "native",
                "--batch-size",
                "4",
                "--context-words",
                "128",
                "--max-tokens",
                "4096",
                "--warmup-tokens",
                "16",
                "--memory-sample-interval",
                "128",
                "--run-id",
                "1",
                "--output",
                str(candidate_path),
            ],
        },
    ]
    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": candidate["environment"],
        "matrix": {
            "cache_kind": "native",
            "batch_size": 4,
            "context_words": 128,
            "max_tokens_per_lane": 4096,
            "policies": ["baseline", "periodic-token-512"],
            "decode_cache_clear_token_budget": 512,
        },
        "source_sha256": baseline_sources | candidate_sources,
        "records": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "execution-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}))


if __name__ == "__main__":
    main()
