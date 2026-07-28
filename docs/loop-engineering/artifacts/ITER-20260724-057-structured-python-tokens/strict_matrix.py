#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
for path in (PROJECT_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paired_matrix as base  # noqa: E402

BENCHMARK = ARTIFACT_DIR / "allowed_tokens_benchmark.py"
AGGREGATE = ARTIFACT_DIR / "strict_aggregate.py"


def _source_hashes(config: Path) -> dict[str, str]:
    iter050_dir = ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync"
    iter055_dir = ARTIFACT_DIR.parent / "ITER-20260723-055-bounded-penalty-context"
    paths = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        config,
        BASE_DIR / "paired_benchmark.py",
        BASE_DIR / "production_benchmark.py",
        BASE_DIR / "sampling_benchmark.py",
        iter050_dir / "benchmark.py",
        iter055_dir / "candidate_benchmark.py",
        BENCHMARK,
        AGGREGATE,
        Path(__file__).resolve(),
    )
    return {
        str(path.relative_to(PROJECT_ROOT)): base._sha256(path)
        for path in paths
    }


def _payload_source_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    analysis_paths = {
        str(AGGREGATE.relative_to(PROJECT_ROOT)),
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
    }
    return {
        path: digest
        for path, digest in source_hashes.items()
        if path not in analysis_paths
    }


def main() -> None:
    original_artifact_dir = base.ARTIFACT_DIR
    original_benchmark = base.BENCHMARK
    original_source_hashes = base._source_hashes
    original_payload_source_hashes = base._payload_source_hashes
    original_validate_payload = base._validate_payload

    def validate_payload(payload: dict[str, Any], **kwargs: Any) -> None:
        original_validate_payload(payload, **kwargs)
        if payload["parity"]["exact_token_text_cache"] is not True:
            raise RuntimeError("paired payload lost exact token/text/cache parity")

    base.ARTIFACT_DIR = ARTIFACT_DIR
    base.BENCHMARK = BENCHMARK
    base._source_hashes = _source_hashes
    base._payload_source_hashes = _payload_source_hashes
    base._validate_payload = validate_payload
    try:
        base.main()
    finally:
        base.ARTIFACT_DIR = original_artifact_dir
        base.BENCHMARK = original_benchmark
        base._source_hashes = original_source_hashes
        base._payload_source_hashes = original_payload_source_hashes
        base._validate_payload = original_validate_payload


if __name__ == "__main__":
    main()
