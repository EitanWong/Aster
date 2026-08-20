from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/diagnose_greedy_batch_shape.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260801-089-greedy-batch-shape-determinism"
    / "greedy-batch-shape-evidence.json"
)


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("diagnose_greedy_batch_shape", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe(token: int) -> dict[str, object]:
    return {
        "selected_token": token,
        "candidate_scores": {"364": 21.0, "421": 20.875, "8574": 20.875},
        "top_token_ids": [token, 8574],
    }


def _record(
    sequence: int,
    *,
    single: int = 364,
    merged_single: int = 364,
    aster_batch: int = 364,
    native_batch: int = 364,
    paired_history_single: int = 8574,
    paired_history_duplicate: int = 364,
    paired_history_actual: int = 421,
) -> dict[str, object]:
    order = "single-first" if sequence % 2 else "batch-first"
    return {
        "schema_version": 1,
        "kind": "greedy-batch-shape-probe",
        "performance_measurement_valid": False,
        "process": {"pid": 1000 + sequence, "sequence": sequence, "order": order},
        "source_binding": {
            "workload_id": "mt-bench:84:turn-1",
            "workload_sha256": "w" * 64,
            "source_lock_sha256": "l" * 64,
            "record_sha256": "r" * 64,
            "prompt_sha256": "p" * 64,
            "model_fingerprint": {
                "model_sha256": "m" * 64,
                "tokenizer_sha256": "t" * 64,
            },
            "runtime_source_sha256": {
                "aster/inference/model_runner.py": "a" * 64,
                "mlx_lm/generate.py": "g" * 64,
            },
        },
        "frozen_state": {
            "prompt_token_ids_sha256": "i" * 64,
            "prompt_token_count": 39,
            "selected_prefix": [271, 12646, 25, 357, 2526, 2923],
            "completion_index": 6,
            "input_token": 2923,
            "candidate_token_ids": [364, 421, 8574],
            "serial_cache_sha256": "c" * 64,
            "aster_paired_cache_sha256": "b" * 64,
            "mlx_lm_paired_cache_sha256": "b" * 64,
            "cache_layers": 32,
        },
        "cache_integrity": {
            "serial": {"before_sha256": "c" * 64, "after_sha256": "c" * 64},
            "aster_paired": {
                "before_sha256": "b" * 64,
                "after_sha256": "b" * 64,
            },
            "mlx_lm_paired": {
                "before_sha256": "b" * 64,
                "after_sha256": "b" * 64,
            },
        },
        "probes": {
            "aster_single": {"rows": [_probe(single)]},
            "aster_merge_extract_single": {"rows": [_probe(merged_single)]},
            "aster_duplicate_batch": {"rows": [_probe(aster_batch), _probe(aster_batch)]},
            "mlx_lm_generation_batch": {"rows": [_probe(native_batch), _probe(native_batch)]},
            "aster_paired_history_single": {"rows": [_probe(paired_history_single)]},
            "aster_paired_history_merge_extract_single": {"rows": [_probe(paired_history_single)]},
            "aster_paired_history_duplicate_batch": {
                "rows": [
                    _probe(paired_history_duplicate),
                    _probe(paired_history_duplicate),
                ]
            },
            "mlx_lm_paired_history_generation_batch": {
                "rows": [
                    _probe(paired_history_duplicate),
                    _probe(paired_history_duplicate),
                ]
            },
            "aster_paired_history_actual_batch": {
                "target_row": 1,
                "rows": [_probe(999), _probe(paired_history_actual)],
            },
            "mlx_lm_paired_history_actual_batch": {
                "target_row": 1,
                "rows": [_probe(999), _probe(paired_history_actual)],
            },
        },
    }


def test_classifies_reference_shared_batched_history_cohort_arithmetic() -> None:
    tool = load_tool()

    summary = tool.classify_records([_record(sequence) for sequence in range(1, 5)])

    assert summary["contracts_passed"] is True
    assert summary["diagnosis"] == "reference-shared-batched-history-cohort-arithmetic"
    assert summary["production_decision"] == "no-production-change"
    assert summary["selected_tokens"]["aster_single"] == [364]
    assert summary["selected_tokens"]["aster_duplicate_batch"] == [364]
    assert summary["selected_tokens"]["mlx_lm_generation_batch"] == [364]
    assert summary["selected_tokens"]["aster_paired_history_single"] == [8574]
    assert summary["selected_tokens"]["aster_paired_history_duplicate_batch"] == [364]
    assert summary["selected_tokens"]["aster_paired_history_actual_batch"] == [421]
    assert summary["selected_tokens"]["mlx_lm_paired_history_actual_batch"] == [421]
    assert all(summary["gates"].values())


def test_classifies_merge_extract_sensitive_state() -> None:
    tool = load_tool()

    summary = tool.classify_records(
        [_record(sequence, merged_single=421) for sequence in range(1, 5)]
    )

    assert summary["contracts_passed"] is False
    assert summary["diagnosis"] == "cache-merge-extract-sensitive"
    assert summary["gates"]["merge_extract_matches_single"] is False
    assert summary["production_decision"] == "no-production-change"


def test_rejects_source_or_frozen_state_drift() -> None:
    tool = load_tool()
    records = [_record(sequence) for sequence in range(1, 5)]
    records[-1]["source_binding"]["prompt_sha256"] = "x" * 64

    with pytest.raises(tool.DiagnosticError, match="source binding"):
        tool.classify_records(records)

    records = [_record(sequence) for sequence in range(1, 5)]
    records[-1]["frozen_state"]["serial_cache_sha256"] = "x" * 64
    with pytest.raises(tool.DiagnosticError, match="frozen state"):
        tool.classify_records(records)


def test_requires_independent_balanced_processes() -> None:
    tool = load_tool()
    records = [_record(sequence) for sequence in range(1, 5)]
    records[-1]["process"]["pid"] = records[0]["process"]["pid"]

    with pytest.raises(tool.DiagnosticError, match="independent process"):
        tool.classify_records(records)

    records = [_record(sequence) for sequence in range(1, 5)]
    for record in records:
        record["process"]["order"] = "single-first"
    with pytest.raises(tool.DiagnosticError, match="balanced probe order"):
        tool.classify_records(records)


def test_retained_evidence_binds_recorded_sources_processes_and_cache_states() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    classification = artifact["classification"]
    records = artifact["records"]

    assert artifact["kind"] == "greedy-batch-shape-determinism-evidence"
    assert classification["contracts_passed"] is True
    assert classification["diagnosis"] == ("reference-shared-batched-history-cohort-arithmetic")
    assert classification["production_decision"] == "no-production-change"
    assert len(artifact["raw_sha256"]) == len(records) == 4
    assert len({record["process"]["pid"] for record in records}) == 4
    assert sorted(record["process"]["order"] for record in records) == [
        "batch-first",
        "batch-first",
        "single-first",
        "single-first",
    ]
    assert classification["selected_tokens"]["aster_single"] == [364]
    assert classification["selected_tokens"]["aster_paired_history_single"] == [8574]
    assert classification["selected_tokens"]["aster_paired_history_actual_batch"] == [421]
    for record in records:
        frozen = record["frozen_state"]
        assert frozen["serial_cache_sha256"] != frozen["aster_paired_cache_sha256"]
        assert frozen["aster_paired_cache_sha256"] == frozen["mlx_lm_paired_cache_sha256"]
    assert set(artifact["source_sha256"]) == {
        "aster/inference/model_runner.py",
        "scripts/dev/diagnose_greedy_batch_shape.py",
        "tests/test_greedy_batch_shape_diagnostic.py",
    }
    assert all(len(value) == 64 for value in artifact["source_sha256"].values())
    assert {
        record["source_binding"]["runtime_source_sha256"]["aster/inference/model_runner.py"]
        for record in records
    } == {artifact["source_sha256"]["aster/inference/model_runner.py"]}
