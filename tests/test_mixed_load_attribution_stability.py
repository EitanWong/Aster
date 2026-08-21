from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = PROJECT_ROOT / "scripts/dev/benchmark_decode_observer.py"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "docs/loop-engineering/artifacts/ITER-20260822-094-mixed-load-attribution-stability"
    / "mixed-load-attribution-stability.json"
)


def load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_decode_observer", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_i094_artifact_recomputes_long_window_summary() -> None:
    harness = load_harness()
    foundation = harness.load_foundation()
    payload = json.loads(ARTIFACT_PATH.read_text())

    assert payload["iteration"] == "ITER-20260822-094-mixed-load-attribution-stability"
    assert payload["candidate"]["window_tokens"] == 32
    assert payload["execution"]["max_output_tokens"] == 32
    assert len(payload["rows"]) == 32
    assert payload["summary"]["measurement_status"] == "valid"
    assert payload["summary"]["candidate_admitted"] is False
    assert payload["summary"]["exact_output_identity_off_vs_on"] is True
    assert payload["summary"]["control_engine_stable_all_metrics_and_strata"] is False

    recomputed = harness.summarize(
        foundation,
        payload["rows"],
        payload["execution"]["collection_statuses"],
        repetitions=4,
        expected_sample_interval=8,
        expected_max_output_tokens=32,
    )
    for key, value in recomputed.items():
        assert payload["summary"][key] == value

    assert {row["result"]["execution"]["max_output_tokens"] for row in payload["rows"]} == {32}
    assert {
        request["completion_tokens"]
        for row in payload["rows"]
        for request in row["result"]["requests"]
    } == {32}
