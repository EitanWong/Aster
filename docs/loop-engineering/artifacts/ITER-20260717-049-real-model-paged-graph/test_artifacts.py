from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from aggregate import (
    aggregate,
    paired_percent_deltas,
    validate_manifest_membership,
    validate_records,
)
from benchmark import _model_query_tokens, _resolve_runner
from profile_lib import TimingCollector, patch_method, summarize_samples
from run_matrix import build_cells, preserve_executable_path
from turboquant_aggregate import (
    SOURCE_NAMES,
    aggregate_turboquant,
    clears_stable_improvement_gate,
    paired_method_improvements,
    validate_turboquant_records,
)
from turboquant_bench import build_method_orders
from turboquant_model_aggregate import (
    aggregate_model_records,
    compare_model_pair,
    model_latency_metrics,
    validate_model_records,
)
from turboquant_model_bench import _encode_required_prefix
from turboquant_model_run_matrix import build_model_cells, model_cell_offset

from aster.core.config import RuntimeSettings
from aster.inference.model_runner import ModelRunner

ARTIFACT_DIR = Path(__file__).resolve().parent


def test_summarize_samples_groups_phase_and_reports_tail() -> None:
    collector = TimingCollector()
    collector.record("attention", 1_000_000, phase="decode", kv_tokens=128)
    collector.record("attention", 3_000_000, phase="decode", kv_tokens=256)
    collector.record("attention", 2_000_000, phase="prefill", kv_tokens=256)

    summary = summarize_samples(collector.samples)

    assert summary["attention:decode"] == {
        "count": 2,
        "total_ms": 4.0,
        "median_ms": 2.0,
        "p95_ms": 3.0,
        "min_ms": 1.0,
        "max_ms": 3.0,
    }
    assert summary["attention:prefill"]["median_ms"] == 2.0


def test_patch_method_records_result_and_restores_original() -> None:
    collector = TimingCollector()

    @dataclass
    class Worker:
        offset: int = 7

        def run(self, value: int) -> int:
            return self.offset + value

    original = Worker.run
    restore = patch_method(
        Worker,
        "run",
        collector,
        "worker",
        metadata=lambda args, _kwargs: {"phase": "decode", "kv_tokens": args[0].offset},
        capture_result=lambda result: collector.record_token(result),
    )

    assert Worker().run(5) == 12
    restore()

    assert Worker.run is original
    assert collector.tokens == (12,)
    assert collector.samples[0].name == "worker"
    assert collector.samples[0].kv_tokens == 7


def _record(variant: str, run_id: int, elapsed: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "control",
        "variant": variant,
        "run_id": run_id,
        "context_words": 1024,
        "response": {
            "elapsed_seconds": elapsed,
            "completion_tokens": 32,
            "generation_tps": 32.0 / elapsed,
            "mlx_peak_memory_bytes": 1000,
            "token_ids": list(range(32)),
            "text_sha256": "a" * 64,
        },
    }


def test_validate_records_rejects_duplicate_process_cells() -> None:
    record = _record("native", 1, 1.0)

    with pytest.raises(ValueError, match="duplicate process cell"):
        validate_records([record, dict(record)])


def test_validate_records_strict_mode_rejects_source_drift() -> None:
    native = {
        **_record("native", 1, 1.0),
        "pid": 101,
        "source_sha256": {"benchmark.py": "a" * 64},
        "settings": {"temperature": 0.0},
        "environment": {"python": "3.14", "mlx": "0.32.0"},
    }
    direct = {
        **_record("direct", 1, 1.0),
        "pid": 102,
        "source_sha256": {"benchmark.py": "b" * 64},
        "settings": {"temperature": 0.0},
        "environment": {"python": "3.14", "mlx": "0.32.0"},
    }

    with pytest.raises(ValueError, match="source provenance"):
        validate_records([native, direct], strict=True)


def test_paired_percent_deltas_pairs_by_run_id() -> None:
    records = [
        _record("native", 1, 2.0),
        _record("direct", 1, 1.0),
        _record("native", 2, 4.0),
        _record("direct", 2, 3.0),
    ]

    assert paired_percent_deltas(records, metric="elapsed_seconds") == [-50.0, -25.0]


def test_aggregate_keeps_profile_totals_separate_by_variant() -> None:
    native = {**_record("native", 1, 2.0), "timings": {"sync:decode": {"total_ms": 10.0}}}
    direct = {**_record("direct", 1, 1.0), "timings": {"sync:decode": {"total_ms": 20.0}}}

    group = aggregate([native, direct])["groups"]["control:1024"]

    assert group["phase_totals_median_ms_by_variant"] == {
        "native": {"sync:decode": 10.0},
        "direct": {"sync:decode": 20.0},
    }


def test_resolve_runner_uses_engine_model_runner_not_runtime_adapter() -> None:
    runner = ModelRunner(RuntimeSettings.model_validate({"embeddings": {"enabled": False}}))

    assert _resolve_runner(SimpleNamespace(model_runner=runner)) is runner

    with pytest.raises(TypeError, match="model_runner"):
        _resolve_runner(SimpleNamespace(runtime_kernel=SimpleNamespace(runner=runner)))


def test_model_query_tokens_uses_sequence_axis_for_token_ids() -> None:
    assert _model_query_tokens(SimpleNamespace(shape=(1, 1024))) == 1024
    assert _model_query_tokens(SimpleNamespace(shape=(1, 8, 1, 128))) == 128


def test_matrix_cells_are_deterministic_and_keep_pairs_adjacent() -> None:
    first = build_cells(modes=("control", "profile"), contexts=(2048,), runs=(1, 2), seed=49)
    second = build_cells(modes=("control", "profile"), contexts=(2048,), runs=(1, 2), seed=49)

    assert first == second
    assert len(first) == 8
    for index in range(0, len(first), 2):
        left, right = first[index : index + 2]
        assert (left.mode, left.context_words, left.run_id) == (
            right.mode,
            right.context_words,
            right.run_id,
        )
        assert {left.variant, right.variant} == {"native", "direct"}


def test_matrix_preserves_virtualenv_symlink_path() -> None:
    path = preserve_executable_path(Path(".venv/bin/python"), cwd=Path("/repo"))

    assert path == Path("/repo/.venv/bin/python")


def test_turboquant_method_orders_are_balanced_and_deterministic() -> None:
    methods = ("mlx_fp16", "aster_paged_fp16", "tq_fused", "tq_dequant")
    first = build_method_orders(methods, iterations=8, seed=49)
    second = build_method_orders(methods, iterations=8, seed=49)

    assert first == second
    assert len(first) == 8
    assert all(set(order) == set(methods) for order in first)
    assert {order[0] for order in first} == set(methods)


def test_model_benchmark_encodes_only_required_corpus_prefix() -> None:
    class Tokenizer:
        calls: list[int] = []

        def encode(self, text: str) -> list[int]:
            self.calls.append(len(text))
            return list(range(len(text) // 2))

    tokenizer = Tokenizer()
    token_ids = _encode_required_prefix(tokenizer, "x" * 10_000, required=100)

    assert len(token_ids) >= 100
    assert tokenizer.calls == [4096]


def test_model_pair_comparison_reports_quality_and_greedy_prefix() -> None:
    baseline = {
        "greedy": {"token_ids": [1, 2, 3, 4]},
        "teacher_forced": {
            "target_ids": [5, 6, 7],
            "top1_ids": [5, 8, 7],
            "target_logprobs": [-1.0, -2.0, -3.0],
            "perplexity": 7.38905609893065,
        },
    }
    candidate = {
        "greedy": {"token_ids": [1, 2, 9, 4]},
        "teacher_forced": {
            "target_ids": [5, 6, 7],
            "top1_ids": [5, 8, 9],
            "target_logprobs": [-1.1, -2.0, -3.0],
            "perplexity": 7.639505905129983,
        },
    }

    comparison = compare_model_pair(baseline, candidate)

    assert comparison["greedy_exact"] is False
    assert comparison["greedy_matching_prefix_tokens"] == 2
    assert comparison["teacher_top1_agreement_percent"] == pytest.approx(200 / 3)
    assert comparison["perplexity_delta_percent"] == pytest.approx(3.389469546)
    assert comparison["target_logprob_mean_abs_delta"] == pytest.approx(1 / 30)


def test_model_matrix_keeps_fp16_turboquant_pairs_adjacent() -> None:
    cells = build_model_cells(contexts=(2048, 8192), runs=(1, 2), seed=49)

    assert len(cells) == 8
    for index in range(0, len(cells), 2):
        left, right = cells[index : index + 2]
        assert (left.context_tokens, left.run_id) == (
            right.context_tokens,
            right.run_id,
        )
        assert {left.variant, right.variant} == {"fp16", "turboquant"}


def test_model_latency_metrics_do_not_double_count_first_token() -> None:
    record = {
        "greedy_prepare": {
            "prefill_seconds": 0.2,
            "conversion_seconds": 0.1,
        },
        "greedy": {
            "first_token_ms": 50.0,
            "elapsed_seconds": 0.4,
        },
    }

    assert model_latency_metrics(record) == {
        "ttft_ms": pytest.approx(350.0),
        "end_to_end_seconds": pytest.approx(0.7),
    }


def test_model_cell_offset_varies_quality_window_by_run_only() -> None:
    assert model_cell_offset(base=1024, stride=16384, run_id=1) == 1024
    assert model_cell_offset(base=1024, stride=16384, run_id=5) == 66560


def test_archived_real_paged_matrix_matches_execution_manifests() -> None:
    results = ARTIFACT_DIR / "results"
    paths = [
        *sorted(results.glob("control-*.json")),
        *sorted(results.glob("profile-*.json")),
        *sorted((results / "confirmation").glob("control-*.json")),
    ]
    records = [json.loads(path.read_text()) for path in paths]

    evidence = validate_manifest_membership(
        paths,
        records,
        [
            results / "execution-manifest.json",
            results / "confirmation/execution-manifest.json",
        ],
    )
    validate_records(records, strict=True)
    archived = json.loads((results / "aggregate.json").read_text())
    recomputed = aggregate(records, strict=True)
    recomputed["evidence"] = {
        **evidence,
        "source_sha256": records[0]["source_sha256"],
        "aggregator_sha256": hashlib.sha256(
            (ARTIFACT_DIR / "aggregate.py").read_bytes()
        ).hexdigest(),
    }

    assert evidence["input_records"] == 60
    assert evidence["fresh_processes"] == 60
    assert archived == recomputed


def test_archived_turboquant_matrices_pass_strict_validation() -> None:
    kernel_results = ARTIFACT_DIR / "results/turboquant"
    kernel_records = [
        json.loads(path.read_text())
        for path in sorted(kernel_results.glob("run-*.json"))
    ]
    validate_turboquant_records(
        kernel_records,
        expected_runs=5,
        expected_tokens=(2048, 8192, 32768, 65536),
        expected_iterations=200,
        expected_source_names=SOURCE_NAMES,
    )
    kernel_archived = json.loads((kernel_results / "aggregate.json").read_text())
    kernel_recomputed = aggregate_turboquant(
        kernel_records,
        expected_runs=5,
        expected_tokens=(2048, 8192, 32768, 65536),
        expected_iterations=200,
    )
    kernel_recomputed["evidence"]["aggregator_sha256"] = hashlib.sha256(
        (ARTIFACT_DIR / "turboquant_aggregate.py").read_bytes()
    ).hexdigest()
    assert kernel_archived == kernel_recomputed

    model_results = ARTIFACT_DIR / "results/turboquant-model"
    model_records = [
        json.loads(path.read_text())
        for path in sorted(model_results.glob("*-run-*.json"))
    ]
    validate_model_records(
        model_records,
        expected_runs=5,
        expected_contexts=(2048, 8192),
        expected_teacher_tokens=64,
        expected_generation_tokens=64,
    )
    model_archived = json.loads((model_results / "aggregate.json").read_text())
    model_recomputed = aggregate_model_records(
        model_records,
        expected_runs=5,
        expected_contexts=(2048, 8192),
        expected_teacher_tokens=64,
        expected_generation_tokens=64,
    )
    assert model_archived == model_recomputed


def test_archived_aggregate_hashes_match_manifests() -> None:
    for result_name in ("turboquant", "turboquant-model"):
        results = ARTIFACT_DIR / "results" / result_name
        manifest = json.loads((results / "execution-manifest.json").read_text())
        digest = hashlib.sha256((results / "aggregate.json").read_bytes()).hexdigest()

        assert manifest["aggregate_sha256"] == digest


def _latency(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": sum(ordered) / len(ordered),
        "p95_ms": ordered[-1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": 0.5,
        "samples": samples,
    }


def _turboquant_record(run_id: int, pid: int, *, candidate_ms: float) -> dict[str, object]:
    latencies = {
        "mlx_fp16": _latency([1.0, 2.0]),
        "aster_paged_fp16": _latency([2.0, 3.0]),
        "tq_fused": _latency([candidate_ms - 0.5, candidate_ms + 0.5]),
        "tq_dequant": _latency([3.0, 4.0]),
    }
    return {
        "schema_version": 1,
        "benchmark": "turboquant_decode_attention",
        "run_id": run_id,
        "pid": pid,
        "methods": ["mlx_fp16", "aster_paged_fp16", "tq_fused", "tq_dequant"],
        "provenance": {
            "sources": {"benchmark": {"sha256": "a" * 64}},
        },
        "contexts": [
            {
                "tokens": 2048,
                "iterations": 2,
                "latency_ms": latencies,
                "correctness": {
                    "all_finite": {name: True for name in latencies},
                    "aster_vs_mlx_max_abs": 0.0,
                    "tq_fused_vs_dequant_max_abs": 0.001,
                },
                "storage": {
                    "fp16_bytes": 400,
                    "turboquant_bytes": 100,
                    "compression_ratio": 4.0,
                },
                "swap_delta_bytes": 0,
            }
        ],
    }


def test_turboquant_validation_rejects_reused_process() -> None:
    records = [
        _turboquant_record(1, 100, candidate_ms=2.0),
        _turboquant_record(2, 100, candidate_ms=2.0),
    ]

    with pytest.raises(ValueError, match="fresh process"):
        validate_turboquant_records(
            records,
            expected_runs=2,
            expected_tokens=(2048,),
            expected_iterations=2,
        )


def test_turboquant_validation_recomputes_sample_statistics() -> None:
    record = _turboquant_record(1, 100, candidate_ms=2.0)
    record["contexts"][0]["latency_ms"]["tq_fused"]["median_ms"] = 99.0

    with pytest.raises(ValueError, match="median_ms"):
        validate_turboquant_records(
            [record],
            expected_runs=1,
            expected_tokens=(2048,),
            expected_iterations=2,
        )


def test_turboquant_improvements_are_paired_by_process() -> None:
    records = [
        _turboquant_record(1, 100, candidate_ms=2.0),
        _turboquant_record(2, 200, candidate_ms=1.5),
    ]

    improvements = paired_method_improvements(
        records,
        tokens=2048,
        candidate="tq_fused",
        baseline="aster_paged_fp16",
    )

    assert improvements == pytest.approx([20.0, 40.0])


def test_stable_improvement_gate_requires_median_and_interval() -> None:
    assert clears_stable_improvement_gate([4.0, 5.0, 6.0, 7.0, 8.0], seed=1)
    assert not clears_stable_improvement_gate([1.0, 2.0, 4.0, 5.0, 6.0], seed=1)
