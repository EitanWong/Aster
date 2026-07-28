from __future__ import annotations

import json
from pathlib import Path

import aggregate

ARTIFACT_DIR = Path(__file__).resolve().parent


def test_summary_matches_current_payloads() -> None:
    stored = json.loads((ARTIFACT_DIR / "summary.json").read_text())
    assert stored == aggregate.build_summary()


def test_all_screens_preserve_outputs_and_swap() -> None:
    summary = aggregate.build_summary()
    assert summary["gates"]["exact_all"] is True
    assert summary["gates"]["zero_swap_all"] is True


def test_host_materialization_has_insufficient_addressable_share() -> None:
    summary = aggregate.build_summary()
    shares = [row["host_post_eval_pct"] for row in summary["host_profile"]]
    assert max(shares) < 1.0
    assert summary["gates"]["host_materialization_below_one_pct"] is True


def test_tensorized_candidates_remain_below_admission_gate() -> None:
    summary = aggregate.build_summary()
    penalty_gains = [row["median_gain_pct"] for row in summary["batched_penalties"]]
    normalization_gains = [
        row["median_gain_pct"] for row in summary["batched_normalization"]
    ]
    assert max(penalty_gains) < 3.0
    assert max(normalization_gains) < 3.0
    assert min(normalization_gains) < 0.0
    assert summary["decision"]["admitted"] is False
