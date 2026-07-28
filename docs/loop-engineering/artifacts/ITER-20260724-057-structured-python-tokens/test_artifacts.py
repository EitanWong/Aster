from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
FORMAL_RESULTS = (
    ("strict-short-b4-r18", "structured-b4", 409),
    ("strict-long-b2-r18", "structured-b2", 24_601),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_formal_matrices_pass_source_bound_admission() -> None:
    for result_name, cell_name, prompt_tokens in FORMAL_RESULTS:
        result_dir = ARTIFACT_DIR / "results" / result_name
        manifest_path = result_dir / "execution-manifest.json"
        aggregate = _load(result_dir / "aggregate.json")
        manifest = _load(manifest_path)
        cell = aggregate["cells"][cell_name]

        assert aggregate["manifest_sha256"] == _sha256(manifest_path)
        assert aggregate["records"] == 18
        assert aggregate["unique_pids"] == 18
        assert aggregate["all_cells_passed"] is True
        assert cell["independent_processes"] == 18
        assert cell["independent_replicates"] == 9
        assert cell["stable_replicates"] == 9
        assert cell["speed_floor_percent"] == 3.0
        assert all(cell["gates"].values())

        for relative, expected_hash in manifest["source_sha256"].items():
            assert _sha256(PROJECT_ROOT / relative) == expected_hash

        records = manifest["records"]
        assert len(records) == 18
        assert len({record["pid"] for record in records}) == 18
        for record in records:
            payload_path = ARTIFACT_DIR / record["output"]
            payload = _load(payload_path)
            assert _sha256(payload_path) == record["sha256"]
            assert payload["settings"]["actual_prompt_tokens"] == prompt_tokens
            assert payload["parity"]["exact_token_text_cache"] is True
            assert payload["memory"]["swap_after_bytes"] <= payload["memory"]["swap_before_bytes"]
            assert payload["policy_metrics"]["baseline"]["legacy_allowed_list_copies"] > 0
            assert payload["policy_metrics"]["production"]["legacy_allowed_list_copies"] == 0


def test_stop_aware_structured_validation_passes() -> None:
    payload = _load(ARTIFACT_DIR / "structured-validation-b4-run-1.json")

    assert payload["all_schema_valid"] is True
    assert payload["all_stopped_before_limit"] is True
    assert len(payload["lanes"]) == 4
    assert all(lane["schema_valid"] for lane in payload["lanes"])
    assert payload["membership_sizes"][0] == 4
    assert payload["membership_sizes"][-1] == 1
    assert payload["memory"]["swap_after_bytes"] <= payload["memory"]["swap_before_bytes"]
    for relative, expected_hash in payload["source_sha256"].items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash


def test_composite_admission_passes() -> None:
    payload = _load(ARTIFACT_DIR / "final-admission.json")

    assert payload["passed"] is True
    assert all(payload["gates"].values())
    assert payload["production_source_sha256"] == _sha256(
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py"
    )
