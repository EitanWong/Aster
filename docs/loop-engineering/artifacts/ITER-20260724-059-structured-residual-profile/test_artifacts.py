from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent


def _load_admission_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "iter059_admission",
        ARTIFACT_DIR / "admission.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load Iteration 059 admission module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_composite_admission_recomputes_from_compact_evidence() -> None:
    admission = _load_admission_module()
    stored = json.loads((ARTIFACT_DIR / "final-admission.json").read_text())
    recomputed = admission.build_admission()

    assert recomputed == stored
    assert recomputed["decision"] == "admit"
    assert recomputed["passed"] is True
    assert all(recomputed["gates"].values())
    for result in recomputed["formal"].values():
        assert result["records"] == 18
        assert result["unique_pids"] == 18
        assert result["stable_replicates"] == 9
        assert result["aggregate_recomputed_exactly"] is True
        assert result["balanced_interval_percent"]["lower"] >= 3.0
        assert result["baseline_first_interval_percent"]["lower"] >= 3.0
        assert result["production_first_interval_percent"]["lower"] >= 3.0
    assert all(result["passed"] is True for result in recomputed["memory_confirmation"].values())


def test_compact_evidence_members_are_bounded_and_relative() -> None:
    admission = _load_admission_module()
    bundle = admission.EvidenceBundle(admission.BUNDLE)
    try:
        assert len(bundle.file_members) == 50
        assert admission.BUNDLE.stat().st_size <= 5 * 1024**2
        for member in bundle.file_members:
            path = PurePosixPath(member)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert path.parts[:2] == ("run", "loop-engineering")
    finally:
        bundle.close()
