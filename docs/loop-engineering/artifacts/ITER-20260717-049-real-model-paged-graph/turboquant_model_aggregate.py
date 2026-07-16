#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from turboquant_aggregate import (
    _bootstrap_median_interval,
    clears_stable_improvement_gate,
)

VARIANTS = ("fp16", "turboquant")


def model_latency_metrics(record: dict[str, Any]) -> dict[str, float]:
    preparation = record["greedy_prepare"]
    greedy = record["greedy"]
    prepare_seconds = float(preparation["prefill_seconds"]) + float(
        preparation["conversion_seconds"]
    )
    return {
        "ttft_ms": prepare_seconds * 1000 + float(greedy["first_token_ms"]),
        "end_to_end_seconds": prepare_seconds + float(greedy["elapsed_seconds"]),
    }


def compare_model_pair(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | int | bool]:
    baseline_greedy = baseline["greedy"]["token_ids"]
    candidate_greedy = candidate["greedy"]["token_ids"]
    if len(baseline_greedy) != len(candidate_greedy):
        raise ValueError("greedy token lengths differ")
    matching_prefix = 0
    for left, right in zip(baseline_greedy, candidate_greedy, strict=True):
        if left != right:
            break
        matching_prefix += 1

    baseline_teacher = baseline["teacher_forced"]
    candidate_teacher = candidate["teacher_forced"]
    if baseline_teacher["target_ids"] != candidate_teacher["target_ids"]:
        raise ValueError("teacher target ids differ")
    baseline_top1 = baseline_teacher["top1_ids"]
    candidate_top1 = candidate_teacher["top1_ids"]
    baseline_logprobs = baseline_teacher["target_logprobs"]
    candidate_logprobs = candidate_teacher["target_logprobs"]
    if not (
        len(baseline_top1)
        == len(candidate_top1)
        == len(baseline_logprobs)
        == len(candidate_logprobs)
        > 0
    ):
        raise ValueError("teacher evidence lengths differ")
    top1_matches = sum(
        left == right for left, right in zip(baseline_top1, candidate_top1, strict=True)
    )
    logprob_deltas = [
        abs(float(left) - float(right))
        for left, right in zip(baseline_logprobs, candidate_logprobs, strict=True)
    ]
    baseline_ppl = float(baseline_teacher["perplexity"])
    candidate_ppl = float(candidate_teacher["perplexity"])
    return {
        "greedy_exact": baseline_greedy == candidate_greedy,
        "greedy_matching_prefix_tokens": matching_prefix,
        "teacher_top1_agreement_percent": top1_matches / len(baseline_top1) * 100.0,
        "perplexity_delta_percent": (candidate_ppl - baseline_ppl) / baseline_ppl * 100.0,
        "target_logprob_mean_abs_delta": statistics.fmean(logprob_deltas),
        "target_logprob_max_abs_delta": max(logprob_deltas),
    }


def _cell_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return int(record["context_tokens"]), int(record["run_id"]), str(record["variant"])


def validate_model_records(
    records: Sequence[dict[str, Any]],
    *,
    expected_runs: int,
    expected_contexts: Sequence[int],
    expected_teacher_tokens: int,
    expected_generation_tokens: int,
) -> None:
    expected_cells = {
        (context, run_id, variant)
        for context in expected_contexts
        for run_id in range(1, expected_runs + 1)
        for variant in VARIANTS
    }
    cells = [_cell_key(record) for record in records]
    if len(cells) != len(set(cells)) or set(cells) != expected_cells:
        raise ValueError("model benchmark matrix is incomplete or duplicated")
    pids = [record.get("pid") for record in records]
    if len(pids) != len(set(pids)):
        raise ValueError("every model cell must use a fresh process")

    reference_dataset = None
    reference_model = None
    reference_provenance = None
    prompt_hashes: dict[tuple[int, int], str] = {}
    offsets_by_run: dict[int, int] = {}
    for record in records:
        if record.get("schema_version") != 1 or record.get("benchmark") != "turboquant_qwen35_model":
            raise ValueError("unexpected model benchmark schema")
        if record.get("teacher_tokens") != expected_teacher_tokens:
            raise ValueError("unexpected teacher token count")
        if record.get("generation_tokens") != expected_generation_tokens:
            raise ValueError("unexpected generation token count")
        dataset = record.get("dataset", {})
        model = record.get("model", {})
        provenance = record.get("provenance", {})
        dataset_identity = (dataset.get("sha256"), dataset.get("source"))
        if reference_dataset is None:
            reference_dataset = dataset_identity
            reference_model = model
            reference_provenance = provenance
        elif (
            dataset_identity != reference_dataset
            or model != reference_model
            or provenance != reference_provenance
        ):
            raise ValueError("model, dataset, or source provenance changed")
        context = int(record["context_tokens"])
        run_id = int(record["run_id"])
        offset = dataset.get("offset")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("invalid dataset offset")
        if run_id in offsets_by_run and offsets_by_run[run_id] != offset:
            raise ValueError("paired cells use different dataset offsets")
        offsets_by_run[run_id] = offset
        prompt_hash = dataset.get("prompt_ids_sha256")
        prompt_key = (context, run_id)
        if prompt_key in prompt_hashes and prompt_hashes[prompt_key] != prompt_hash:
            raise ValueError("prompt tokens changed between paired cells")
        prompt_hashes[prompt_key] = prompt_hash

        greedy = record.get("greedy", {})
        teacher = record.get("teacher_forced", {})
        if len(greedy.get("token_ids", ())) != expected_generation_tokens:
            raise ValueError("incomplete greedy tokens")
        if not (
            len(teacher.get("target_ids", ()))
            == len(teacher.get("top1_ids", ()))
            == len(teacher.get("target_logprobs", ()))
            == expected_teacher_tokens
        ):
            raise ValueError("incomplete teacher evidence")
        numeric_values = (
            greedy.get("elapsed_seconds"),
            greedy.get("generation_tps"),
            greedy.get("first_token_ms"),
            greedy.get("decode_median_ms"),
            teacher.get("perplexity"),
            record.get("mlx_peak_memory_bytes"),
            record.get("peak_rss_bytes"),
        )
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in numeric_values
        ):
            raise ValueError("invalid model benchmark metric")
        preparation = record.get("greedy_prepare", {})
        if record["variant"] == "fp16":
            if preparation.get("converted_full_attention_layers") != 0:
                raise ValueError("FP16 baseline unexpectedly converted cache layers")
        elif preparation.get("converted_full_attention_layers", 0) <= 0:
            raise ValueError("TurboQuant candidate converted no cache layers")
        swap_delta = record.get("swap_delta_bytes")
        if swap_delta is not None and not isinstance(swap_delta, int):
            raise ValueError("invalid model swap delta")

    if len(set(offsets_by_run.values())) != expected_runs:
        raise ValueError("quality runs must use distinct dataset windows")
    model_files = reference_model.get("files") if isinstance(reference_model, dict) else None
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("missing model file provenance")
    for digest in model_files.values():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid model file provenance hash")


def _latency_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def _median_and_interval(values: Sequence[float], *, seed: int) -> dict[str, object]:
    lower, upper = _bootstrap_median_interval(values, seed=seed)
    return {
        "values": list(values),
        "median_percent": statistics.median(values),
        "bootstrap95_median_percent": [lower, upper],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_model_records(
    records: Sequence[dict[str, Any]],
    *,
    expected_runs: int,
    expected_contexts: Sequence[int],
    expected_teacher_tokens: int,
    expected_generation_tokens: int,
    fallback_reference_tests_passed: bool = False,
) -> dict[str, object]:
    validate_model_records(
        records,
        expected_runs=expected_runs,
        expected_contexts=expected_contexts,
        expected_teacher_tokens=expected_teacher_tokens,
        expected_generation_tokens=expected_generation_tokens,
    )
    by_cell = {_cell_key(record): record for record in records}
    contexts: dict[str, object] = {}
    all_quality = True
    all_no_decode_regression = True
    all_no_rss_regression = True
    all_speed_gates = True
    all_zero_swap = True
    for context in expected_contexts:
        comparisons: list[dict[str, float | int | bool]] = []
        ttft_reductions: list[float] = []
        decode_reductions: list[float] = []
        end_to_end_reductions: list[float] = []
        compression_ratios: list[float] = []
        peak_reductions: list[float] = []
        rss_reductions: list[float] = []
        baseline_tps: list[float] = []
        candidate_tps: list[float] = []
        swap_deltas: list[int | None] = []
        for run_id in range(1, expected_runs + 1):
            baseline = by_cell[(context, run_id, "fp16")]
            candidate = by_cell[(context, run_id, "turboquant")]
            comparisons.append(compare_model_pair(baseline, candidate))
            baseline_latency = model_latency_metrics(baseline)
            candidate_latency = model_latency_metrics(candidate)
            ttft_reductions.append(
                _latency_reduction(
                    baseline_latency["ttft_ms"], candidate_latency["ttft_ms"]
                )
            )
            decode_reductions.append(
                _latency_reduction(
                    float(baseline["greedy"]["decode_median_ms"]),
                    float(candidate["greedy"]["decode_median_ms"]),
                )
            )
            end_to_end_reductions.append(
                _latency_reduction(
                    baseline_latency["end_to_end_seconds"],
                    candidate_latency["end_to_end_seconds"],
                )
            )
            compression_ratios.append(
                float(candidate["greedy_prepare"]["compression_ratio"])
            )
            peak_reductions.append(
                _latency_reduction(
                    float(baseline["mlx_peak_memory_bytes"]),
                    float(candidate["mlx_peak_memory_bytes"]),
                )
            )
            rss_reductions.append(
                _latency_reduction(
                    float(baseline["peak_rss_bytes"]),
                    float(candidate["peak_rss_bytes"]),
                )
            )
            baseline_tps.append(float(baseline["greedy"]["generation_tps"]))
            candidate_tps.append(float(candidate["greedy"]["generation_tps"]))
            swap_deltas.extend(
                [baseline.get("swap_delta_bytes"), candidate.get("swap_delta_bytes")]
            )

        quality_gate = (
            all(bool(item["greedy_exact"]) for item in comparisons)
            and min(float(item["teacher_top1_agreement_percent"]) for item in comparisons)
            >= 99.0
            and max(abs(float(item["perplexity_delta_percent"])) for item in comparisons)
            <= 0.5
        )
        decode_summary = _median_and_interval(
            decode_reductions, seed=49_217 + context
        )
        ttft_summary = _median_and_interval(
            ttft_reductions, seed=49_317 + context
        )
        end_to_end_summary = _median_and_interval(
            end_to_end_reductions, seed=49_417 + context
        )
        no_decode_regression = (
            decode_summary["bootstrap95_median_percent"][0] >= -1.0
        )
        speed_gate = clears_stable_improvement_gate(
            decode_reductions, seed=49_517 + context
        ) and clears_stable_improvement_gate(
            end_to_end_reductions, seed=49_617 + context
        )
        zero_swap = all(value is not None and value <= 0 for value in swap_deltas)
        rss_summary = _median_and_interval(
            rss_reductions, seed=49_717 + context
        )
        no_rss_regression = rss_summary["bootstrap95_median_percent"][0] >= -1.0
        all_quality = all_quality and quality_gate
        all_no_decode_regression = all_no_decode_regression and no_decode_regression
        all_no_rss_regression = all_no_rss_regression and no_rss_regression
        all_speed_gates = all_speed_gates and speed_gate
        all_zero_swap = all_zero_swap and zero_swap
        contexts[str(context)] = {
            "quality": {
                "greedy_exact_all": all(bool(item["greedy_exact"]) for item in comparisons),
                "greedy_matching_prefix_min": min(
                    int(item["greedy_matching_prefix_tokens"]) for item in comparisons
                ),
                "teacher_top1_agreement_min_percent": min(
                    float(item["teacher_top1_agreement_percent"]) for item in comparisons
                ),
                "perplexity_delta_median_percent": statistics.median(
                    float(item["perplexity_delta_percent"]) for item in comparisons
                ),
                "perplexity_delta_max_abs_percent": max(
                    abs(float(item["perplexity_delta_percent"])) for item in comparisons
                ),
                "target_logprob_max_abs_delta": max(
                    float(item["target_logprob_max_abs_delta"]) for item in comparisons
                ),
                "gate_passed": quality_gate,
            },
            "performance": {
                "ttft_latency_reduction": ttft_summary,
                "decode_latency_reduction": decode_summary,
                "end_to_end_latency_reduction": end_to_end_summary,
                "baseline_generation_tps_median": statistics.median(baseline_tps),
                "candidate_generation_tps_median": statistics.median(candidate_tps),
                "no_decode_regression_gate_passed": no_decode_regression,
                "stable_3_percent_speed_gate_passed": speed_gate,
            },
            "resources": {
                "cache_compression_ratio_median": statistics.median(compression_ratios),
                "peak_memory_reduction_percent_median": statistics.median(
                    peak_reductions
                ),
                "rss_reduction": rss_summary,
                "no_rss_regression_gate_passed": no_rss_regression,
                "material_cache_reduction_gate_passed": statistics.median(
                    compression_ratios
                )
                >= 1.5,
                "swap_delta_max_bytes": max(
                    value for value in swap_deltas if value is not None
                )
                if any(value is not None for value in swap_deltas)
                else None,
                "zero_swap_growth_gate_passed": zero_swap,
            },
            "pair_comparisons": comparisons,
        }

    memory_gate = all(
        contexts[str(context)]["resources"]["material_cache_reduction_gate_passed"]
        for context in expected_contexts
    )
    benefit_gate = all_speed_gates or memory_gate
    admit = (
        all_quality
        and benefit_gate
        and all_no_decode_regression
        and all_no_rss_regression
        and all_zero_swap
        and fallback_reference_tests_passed
    )
    first = records[0]
    dataset_windows = {
        f"{context}:{run_id}": {
            "offset": by_cell[(context, run_id, "fp16")]["dataset"]["offset"],
            "prompt_ids_sha256": by_cell[(context, run_id, "fp16")]["dataset"][
                "prompt_ids_sha256"
            ],
        }
        for context in expected_contexts
        for run_id in range(1, expected_runs + 1)
    }
    return {
        "schema_version": 1,
        "benchmark": "turboquant_qwen35_model_aggregate",
        "evidence": {
            "fresh_processes": len(records),
            "runs_per_variant_context": expected_runs,
            "contexts": list(expected_contexts),
            "teacher_tokens_per_cell": expected_teacher_tokens,
            "generation_tokens_per_cell": expected_generation_tokens,
            "pids": sorted(int(record["pid"]) for record in records),
        },
        "dataset": {
            "sha256": first["dataset"]["sha256"],
            "source": first["dataset"]["source"],
            "windows": dataset_windows,
        },
        "model": first["model"],
        "environment": first["environment"],
        "provenance": {
            **first["provenance"],
            "aggregator_sha256": _sha256(Path(__file__)),
        },
        "contexts": contexts,
        "decision": {
            "quality_gate_all_contexts": all_quality,
            "stable_3_percent_speed_gate_all_contexts": all_speed_gates,
            "material_cache_reduction_all_contexts": memory_gate,
            "benefit_gate_all_contexts": benefit_gate,
            "no_decode_regression_all_contexts": all_no_decode_regression,
            "no_rss_regression_all_contexts": all_no_rss_regression,
            "zero_swap_growth_all_contexts": all_zero_swap,
            "fallback_reference_tests_passed": fallback_reference_tests_passed,
            "aster_runtime_integration": admit,
            "reason": (
                "Quality, benefit, latency/RSS, swap, and fallback gates passed."
                if admit
                else "At least one quality, benefit, latency/RSS, swap, or fallback gate failed."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--contexts", nargs="+", type=int, default=(2048, 8192))
    parser.add_argument("--teacher-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--fallback-reference-tests-passed", action="store_true")
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.inputs]
    payload = aggregate_model_records(
        records,
        expected_runs=args.runs,
        expected_contexts=args.contexts,
        expected_teacher_tokens=args.teacher_tokens,
        expected_generation_tokens=args.generation_tokens,
        fallback_reference_tests_passed=args.fallback_reference_tests_passed,
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
