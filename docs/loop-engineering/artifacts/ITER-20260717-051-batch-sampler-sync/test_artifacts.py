from __future__ import annotations

# Import order pins this iteration's same-named modules before the prior artifact path.
# ruff: noqa: I001

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ARTIFACT_DIR = Path(__file__).resolve().parent
if str(ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_DIR))

import aggregate as aggregate_module  # noqa: E402
import run_matrix as matrix  # noqa: E402
import production_matrix  # noqa: E402
import paired_aggregate  # noqa: E402
import paired_matrix  # noqa: E402
import strict_aggregate  # noqa: E402
import focus_manifest  # noqa: E402
import admission  # noqa: E402
import paired_benchmark  # noqa: E402
import production_benchmark  # noqa: E402
import sampling_benchmark as benchmark  # noqa: E402
import structured_validation  # noqa: E402


def test_warmup_uses_greedy_lane_zero() -> None:
    assert benchmark._lane_workload("iter050-warmup", "structured") == (
        "greedy",
        0,
    )


def test_numbered_lane_preserves_requested_workload() -> None:
    assert benchmark._lane_workload("iter050-lane-3", "mixed") == ("mixed", 3)


def test_sampling_metrics_count_clear_outside_decode_scope() -> None:
    class FakeMLX:
        clear_calls = 0

        def clear_cache(self) -> None:
            self.clear_calls += 1

    mlx = FakeMLX()
    metrics = benchmark.SamplingMetrics(mlx, "baseline")

    metrics.clear_cache()

    assert metrics.clear_requests == 1
    assert mlx.clear_calls == 1


def test_screen_policy_order_rotates_every_candidate() -> None:
    orders = [matrix._policy_order(run_id, 0) for run_id in range(1, 4)]

    assert all(set(order) == set(matrix.POLICIES) for order in orders)
    assert [order[0] for order in orders] == [
        "baseline",
        "grouped-eager",
        "grouped-lazy",
    ]


def test_screen_manifest_and_aggregate_recompute() -> None:
    manifest_path = ARTIFACT_DIR / "results/screen/execution-manifest.json"
    output_path = ARTIFACT_DIR / "results/screen/aggregate.json"
    manifest = aggregate_module._load(manifest_path)[0]
    archived = json.loads(output_path.read_text())

    assert len(manifest["records"]) == 48
    assert aggregate_module.aggregate(manifest_path) == archived
    assert archived["screen_winner"] == "grouped-async"


def test_exact_paired_bootstrap_and_workload_speed_floor() -> None:
    assert aggregate_module._bootstrap_median_ci([2.0, 4.0, 6.0]) == (2.0, 6.0)
    assert aggregate_module._speed_floor("structured") == 0.0
    assert aggregate_module._speed_floor("mixed") == 3.0


def test_production_matrix_rotates_legacy_and_current_paths() -> None:
    assert production_benchmark.POLICIES == ("baseline", "production")
    assert production_matrix._policy_order(1, 0) == ("baseline", "production")
    assert production_matrix._policy_order(2, 0) == ("production", "baseline")


def test_paired_policy_order_alternates_adjacent_measurements() -> None:
    assert paired_benchmark._policy_order(0) == ("baseline", "production")
    assert paired_benchmark._policy_order(1) == ("production", "baseline")
    assert paired_benchmark._policy_order(0, 1) == ("production", "baseline")
    assert paired_benchmark._policy_runner_assignment(1) == {
        "baseline": "runner_a",
        "production": "runner_b",
    }
    assert paired_benchmark._policy_runner_assignment(2) == {
        "baseline": "runner_b",
        "production": "runner_a",
    }
    assert [paired_benchmark._replicate_id(run_id) for run_id in (1, 2, 3, 4)] == [
        1,
        1,
        2,
        2,
    ]
    assert paired_benchmark._structured_valid('{"answer":"ok","score":1}') is True
    assert paired_benchmark._structured_valid('{"answer":"ok","score":true}') is False
    assert paired_benchmark._contains_structured_document(
        ['{"answer":', '"ok","score":1}']
    ) is True
    assert paired_aggregate._speed_floor("structured") == 0.0
    assert paired_aggregate._speed_floor("mixed") == 3.0
    assert structured_validation._schema_valid('{"answer":"ok","score":1}') is True


def test_structured_validation_binds_the_json_processor_source() -> None:
    paths = structured_validation._source_paths(
        benchmark.PROJECT_ROOT / "configs/config.yaml"
    )

    assert (
        benchmark.PROJECT_ROOT
        / "aster/inference/constrained/json_schema_processor.py"
        in paths
    )


def test_paired_block_speedups_use_equal_step_windows() -> None:
    speedups = paired_aggregate._block_speedups(
        baseline=[2.0, 2.0, 4.0, 4.0],
        production=[1.0, 1.0, 2.0, 2.0],
        block_size=2,
    )

    assert speedups == [100.0, 100.0]
    assert paired_aggregate._order_speedup(
        baseline=[2.0, 4.0],
        production=[1.0, 2.0],
        first_policies=["baseline", "production"],
        first_policy="production",
    ) == 100.0
    assert paired_matrix._run_order(3, 0) == (1, 2, 3)
    assert paired_matrix._run_order(4, 1) == (2, 3, 4, 1)
    assert paired_matrix._parse_cell("mixed:8") == ("mixed", 8)
    assert paired_matrix._execution_order(
        (("greedy", 2), ("mixed", 4)),
        3,
    ) == (
        ("greedy", 2, 1),
        ("mixed", 4, 1),
        ("mixed", 4, 2),
        ("greedy", 2, 2),
        ("greedy", 2, 3),
        ("mixed", 4, 3),
    )
    assert strict_aggregate._expected_record_order(
        (("greedy", 2), ("mixed", 4)), 3
    ) == paired_matrix._execution_order(
        (("greedy", 2), ("mixed", 4)), 3
    )
    assert focus_manifest._parse_run_id(Path("run-2.json")) == 2
    assert paired_aggregate._exact_process_median_resample_ci([2.0, 4.0, 6.0]) == (
        2.0,
        6.0,
    )


def test_strict_process_interval_is_distribution_free_and_requires_enough_runs() -> None:
    interval = strict_aggregate.distribution_free_median_interval(
        [float(value) for value in range(1, 10)]
    )

    assert interval == {
        "confidence_target": 0.95,
        "coverage": 0.9609375,
        "lower": 2.0,
        "upper": 8.0,
        "order_statistic_rank": 2,
        "observations": 9,
        "confidence_met": True,
    }
    assert strict_aggregate.distribution_free_median_interval(
        [2.0, 4.0, 6.0]
    )["confidence_met"] is False


def _strict_payload(run_id: int) -> dict[str, object]:
    first_policies = (
        ["baseline", "production"]
        if run_id % 2 == 1
        else ["production", "baseline"]
    )
    return {
        "pid": 1000 + run_id,
        "run_id": run_id,
        "workload": "greedy",
        "batch_size": 2,
        "context_words": 128,
        "comparison_design": {
            "policy_runner_assignment": paired_benchmark._policy_runner_assignment(
                run_id
            ),
            "runner_assignment_alternates_by_run": True,
            "assignment_balanced_replicate_id": paired_benchmark._replicate_id(
                run_id
            ),
        },
        "settings": {
            "steps": 2,
            "pair_warmup_steps": 1,
            "block_size": 1,
            "seed": 2000 + paired_benchmark._replicate_id(run_id),
        },
        "source_sha256": {"source.py": "digest"},
        "model_input_sha256": {"model": "digest"},
        "timings": {
            "baseline_step_seconds": [2.0, 2.0],
            "production_step_seconds": [1.0, 1.0],
            "first_policy_by_step": first_policies,
        },
        "parity": {"exact_token_text_cache": True},
        "memory": {"swap_before_bytes": 0, "swap_after_bytes": 0},
    }


def _strict_manifest() -> dict[str, object]:
    return {
        "matrix": {
            "cells": [{"workload": "greedy", "batch_size": 2}],
            "runs": 2,
            "context_words": 128,
            "steps": 2,
            "pair_warmup_steps": 1,
            "block_size": 1,
            "base_seed": 2000,
            "fresh_processes": True,
            "within_process_adjacent_pairing": True,
            "alternating_ab_ba_order": True,
            "round_robin_cell_execution": True,
            "alternating_policy_runner_assignment": True,
            "paired_runner_assignment_replicates": True,
            "independent_replicates": 1,
        },
        "records": [
            {
                "pid": 1000 + run_id,
                "run_id": run_id,
                "workload": "greedy",
                "batch_size": 2,
                "output": f"run-{run_id}.json",
            }
            for run_id in (1, 2)
        ],
    }


def test_strict_evidence_validation_rejects_bad_order_and_duplicate_runs() -> None:
    payloads = [_strict_payload(1), _strict_payload(2)]
    strict_aggregate.validate_evidence(_strict_manifest(), payloads)

    bad_order = _strict_payload(1)
    bad_order["timings"]["first_policy_by_step"] = ["baseline", "unknown"]
    with pytest.raises(ValueError, match="first-policy order"):
        strict_aggregate.validate_evidence(
            _strict_manifest(),
            [bad_order, _strict_payload(2)],
        )

    duplicate_manifest = _strict_manifest()
    duplicate_manifest["records"][1]["run_id"] = 1
    duplicate_payload = _strict_payload(1)
    duplicate_payload["pid"] = 1002
    with pytest.raises(ValueError, match="run IDs"):
        strict_aggregate.validate_evidence(
            duplicate_manifest,
            [_strict_payload(1), duplicate_payload],
        )

    duplicate_pid_manifest = _strict_manifest()
    duplicate_pid_manifest["records"][1]["pid"] = 1001
    duplicate_pid_payload = _strict_payload(2)
    duplicate_pid_payload["pid"] = 1001
    with pytest.raises(ValueError, match="fresh process"):
        strict_aggregate.validate_evidence(
            duplicate_pid_manifest,
            [_strict_payload(1), duplicate_pid_payload],
        )


def test_strict_evidence_validation_rejects_record_payload_mismatch() -> None:
    manifest = _strict_manifest()
    manifest["records"][0]["batch_size"] = 4

    with pytest.raises(ValueError, match="record metadata"):
        strict_aggregate.validate_evidence(
            manifest,
            [_strict_payload(1), _strict_payload(2)],
        )

    bad_assignment = _strict_payload(2)
    bad_assignment["comparison_design"]["policy_runner_assignment"] = {
        "baseline": "runner_a",
        "production": "runner_b",
    }
    with pytest.raises(ValueError, match="runner assignment"):
        strict_aggregate.validate_evidence(
            _strict_manifest(),
            [_strict_payload(1), bad_assignment],
        )


def test_paired_resume_validation_requires_exact_sources_and_metadata() -> None:
    payload = _strict_payload(1)
    args = SimpleNamespace(
        context_words=128,
        steps=2,
        pair_warmup_steps=1,
        block_size=1,
        seed=2000,
    )
    paired_matrix._validate_payload(
        payload,
        expected_source_hashes={"source.py": "digest"},
        expected_model_hashes={"model": "digest"},
        workload="greedy",
        batch_size=2,
        run_id=1,
        args=args,
    )

    stale = deepcopy(payload)
    stale["source_sha256"] = {}
    with pytest.raises(RuntimeError, match="source key set"):
        paired_matrix._validate_payload(
            stale,
            expected_source_hashes={"source.py": "digest"},
            expected_model_hashes={"model": "digest"},
            workload="greedy",
            batch_size=2,
            run_id=1,
            args=args,
        )

    stale_model = deepcopy(payload)
    stale_model["model_input_sha256"] = {"model": "stale"}
    with pytest.raises(RuntimeError, match="model input hash"):
        paired_matrix._validate_payload(
            stale_model,
            expected_source_hashes={"source.py": "digest"},
            expected_model_hashes={"model": "digest"},
            workload="greedy",
            batch_size=2,
            run_id=1,
            args=args,
        )


def _strict_payloads(count: int = 18) -> list[dict[str, object]]:
    return [_strict_payload(run_id) for run_id in range(1, count + 1)]


def _make_marginal(payload: dict[str, object]) -> dict[str, object]:
    marginal = deepcopy(payload)
    marginal["timings"]["production_step_seconds"] = [1.95, 1.95]
    return marginal


def test_strict_cell_gate_requires_nine_replicates_and_tolerates_one_outlier() -> None:
    valid = _strict_payloads()
    assert strict_aggregate.aggregate_cell(
        "greedy", 2, valid, min_processes=18, min_replicates=9
    )["passed"] is True
    assert strict_aggregate.aggregate_cell(
        "greedy", 2, valid[:16], min_processes=18, min_replicates=9
    )["gates"]["minimum_independent_processes"] is False

    one_outlier = [*valid[:-2], *map(_make_marginal, valid[-2:])]
    one_result = strict_aggregate.aggregate_cell(
        "greedy", 2, one_outlier, min_processes=18, min_replicates=9
    )
    assert one_result["stable_replicates"] == 8
    assert one_result["passed"] is True

    two_outliers = [*valid[:-4], *map(_make_marginal, valid[-4:])]
    two_result = strict_aggregate.aggregate_cell(
        "greedy", 2, two_outliers, min_processes=18, min_replicates=9
    )
    assert two_result["stable_replicates"] == 7
    assert two_result["passed"] is False


def test_strict_cell_gate_rejects_one_order_stratum_even_when_balanced_passes() -> None:
    payloads = _strict_payloads()
    for payload in payloads:
        labels = payload["timings"]["first_policy_by_step"]
        payload["timings"]["production_step_seconds"] = [
            1.0 if label == "baseline" else 2.2 for label in labels
        ]

    result = strict_aggregate.aggregate_cell(
        "greedy", 2, payloads, min_processes=18, min_replicates=9
    )

    assert result["intervals"]["balanced"]["lower_meets_speed_floor"] is True
    assert (
        result["intervals"]["production_first"]["lower_meets_speed_floor"]
        is False
    )
    assert result["passed"] is False


def test_strict_structured_gate_accepts_neutral_fallback_but_rejects_regression() -> None:
    neutral = _strict_payloads()
    for payload in neutral:
        payload["workload"] = "structured"
        payload["timings"]["production_step_seconds"] = [2.0, 2.0]

    neutral_result = strict_aggregate.aggregate_cell(
        "structured", 2, neutral, min_processes=18, min_replicates=9
    )
    assert neutral_result["speed_floor_percent"] == -1.0
    assert neutral_result["passed"] is True

    regressed = deepcopy(neutral)
    for payload in regressed:
        payload["timings"]["production_step_seconds"] = [2.04, 2.04]
    assert strict_aggregate.aggregate_cell(
        "structured", 2, regressed, min_processes=18, min_replicates=9
    )["passed"] is False


def test_assignment_replicate_cancels_process_scale_and_runner_bias() -> None:
    run_a = _strict_payload(1)
    run_b = _strict_payload(2)
    for payload in (run_a, run_b):
        payload["workload"] = "structured"
    run_a["timings"]["baseline_step_seconds"] = [4.0, 4.0]
    run_a["timings"]["production_step_seconds"] = [2.0, 2.0]
    run_b["timings"]["baseline_step_seconds"] = [20.0, 20.0]
    run_b["timings"]["production_step_seconds"] = [40.0, 40.0]

    replicate = strict_aggregate._replicate_metrics([run_a, run_b])

    assert replicate["end_to_end_speedup_percent"] == pytest.approx(0.0)
    assert replicate["baseline_first_speedup_percent"] == pytest.approx(0.0)
    assert replicate["production_first_speedup_percent"] == pytest.approx(0.0)
    assert replicate["block_speedup_percent_median"] == pytest.approx(0.0)


def test_swap_gate_rejects_growth_but_accepts_system_swap_reclamation() -> None:
    reclaimed = _strict_payload(1)
    reclaimed["memory"] = {
        "swap_before_bytes": 16 * 1024 * 1024,
        "swap_after_bytes": 8 * 1024 * 1024,
    }
    reclaimed_run = strict_aggregate._timing_metrics(reclaimed)
    assert reclaimed_run["swap_zero"] is False
    assert reclaimed_run["swap_non_growth"] is True
    assert strict_aggregate._replicate_metrics(
        [reclaimed, _strict_payload(2)]
    )["swap_non_growth"] is True

    growth = deepcopy(reclaimed)
    growth["memory"]["swap_after_bytes"] = 24 * 1024 * 1024
    assert strict_aggregate._timing_metrics(growth)["swap_non_growth"] is False
    assert strict_aggregate._replicate_metrics(
        [growth, _strict_payload(2)]
    )["swap_non_growth"] is False


def test_structured_gate_uses_balanced_estimate_for_order_neutral_fallback() -> None:
    payloads = _strict_payloads()
    for payload in payloads:
        payload["workload"] = "structured"
        labels = payload["timings"]["first_policy_by_step"]
        payload["timings"]["baseline_step_seconds"] = [
            0.97 if label == "baseline" else 1.03 for label in labels
        ]
        payload["timings"]["production_step_seconds"] = [1.0, 1.0]

    result = strict_aggregate.aggregate_cell(
        "structured", 2, payloads, min_processes=18, min_replicates=9
    )

    assert result["intervals"]["balanced"]["lower_meets_speed_floor"] is True
    assert (
        result["intervals"]["baseline_first"]["lower_meets_speed_floor"]
        is False
    )
    assert result["gates"]["order_strata_speed_floor_required"] is True
    assert result["passed"] is True


def test_structured_neutral_fallback_reports_but_does_not_gate_block_stability() -> None:
    payloads = _strict_payloads()
    for payload in payloads:
        payload["workload"] = "structured"
        payload["timings"]["baseline_step_seconds"] = [1.0, 1.0]
        payload["timings"]["production_step_seconds"] = [1.02, 0.98]

    result = strict_aggregate.aggregate_cell(
        "structured", 2, payloads, min_processes=18, min_replicates=9
    )

    assert result["stable_replicates"] == 0
    assert result["intervals"]["balanced"]["lower_meets_speed_floor"] is True
    assert result["within_replicate_stability_required"] is False
    assert result["gates"]["within_replicate_stability"] is True
    assert result["passed"] is True


def _assert_current_sources(manifest: dict[str, object]) -> None:
    source_hashes = manifest["source_sha256"]
    assert isinstance(source_hashes, dict)
    for relative_path, expected in source_hashes.items():
        path = benchmark.PROJECT_ROOT / relative_path
        assert path.is_file()
        assert aggregate_module._sha256(path) == expected


def test_noisy_production_archives_recompute_as_historical_evidence() -> None:
    for name, records in (
        ("production-confirmation", 100),
        ("production-long-confirmation", 24),
    ):
        root = ARTIFACT_DIR / "results" / name
        manifest_path = root / "execution-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        archived = json.loads((root / "aggregate.json").read_text())

        assert len(manifest["records"]) == records
        assert aggregate_module.aggregate(manifest_path) == archived


def test_legacy_paired_archives_recompute_as_historical_evidence() -> None:
    for name, records in (
        ("paired-final2-short", 30),
        ("paired-final2-greedy-b2-512", 3),
        ("paired-final2-penalties-b2-512", 3),
        ("paired-final2-long", 12),
        ("paired-final2-long-greedy-b2-1024", 3),
        ("paired-final2-stress", 3),
    ):
        root = ARTIFACT_DIR / "results" / name
        manifest_path = root / "execution-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        archived = json.loads((root / "aggregate.json").read_text())

        assert len(manifest["records"]) == records
        assert paired_aggregate.aggregate(manifest_path) == archived
        assert archived["all_runs_passed"] is True


def test_strict_archives_recompute_with_expected_screen_failure() -> None:
    expectations = {
        "strict-final-v7-short-r18": (180, True),
        "strict-final-v7-long-r18": (72, False),
        "strict-final-v7-long-greedy-b2-r18-s1024": (18, True),
        "strict-final-v7-stress-mixed-b8-r18-s1024": (18, True),
    }
    for name, (records, expected_pass) in expectations.items():
        root = ARTIFACT_DIR / "results" / name
        manifest_path = root / "execution-manifest.json"
        archived = json.loads((root / "strict-aggregate.json").read_text())

        assert strict_aggregate.aggregate(manifest_path) == archived
        assert archived["records"] == records
        assert archived["all_cells_passed"] is expected_pass


def test_final_admission_recomputes_and_passes_every_component() -> None:
    archived = json.loads(
        (ARTIFACT_DIR / "results/final-admission.json").read_text()
    )

    assert admission.build() == archived
    assert archived["admitted"] is True
    assert all(archived["gates"].values())
    assert archived["components"]["long_screen"]["all_cells_passed"] is False
    assert archived["components"]["long_greedy_b2_focus"]["all_cells_passed"] is True
    assert archived["gates"]["long_screen_failure_resolved"] is True
