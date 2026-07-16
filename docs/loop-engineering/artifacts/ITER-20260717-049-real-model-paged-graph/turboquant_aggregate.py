#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

METHODS = ("mlx_fp16", "aster_paged_fp16", "tq_fused", "tq_dequant")
SOURCE_NAMES = (
    "benchmark",
    "aster_paged_attention",
    "mlx_vlm_turboquant_runtime",
    "omlx_turboquant_patch",
    "vllm_metal_turboquant",
    "gemma4metal_turboquant",
)


def _context(record: dict[str, Any], tokens: int) -> dict[str, Any]:
    matches = [item for item in record["contexts"] if item["tokens"] == tokens]
    if len(matches) != 1:
        raise ValueError(f"expected one {tokens}-token context per record")
    return matches[0]


def _sample_summary(samples: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.pstdev(ordered),
    }


def _require_close(actual: Any, expected: float, *, field: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"invalid recomputed {field}: {actual!r} != {expected!r}")


def _source_hashes(record: dict[str, Any]) -> dict[str, str]:
    sources = record.get("provenance", {}).get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("missing source provenance")
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        digest = source.get("sha256") if isinstance(source, dict) else None
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid source provenance hash")
        hashes[name] = digest
    return hashes


def validate_turboquant_records(
    records: Sequence[dict[str, Any]],
    *,
    expected_runs: int,
    expected_tokens: Sequence[int],
    expected_iterations: int,
    expected_source_names: Sequence[str] | None = None,
) -> None:
    if len(records) != expected_runs:
        raise ValueError(f"expected {expected_runs} records, found {len(records)}")
    run_ids = [record.get("run_id") for record in records]
    if len(set(run_ids)) != len(run_ids) or set(run_ids) != set(range(1, expected_runs + 1)):
        raise ValueError("run ids must be unique and contiguous")
    pids = [record.get("pid") for record in records]
    if len(set(pids)) != len(pids):
        raise ValueError("each record must come from a fresh process")

    expected_token_set = set(expected_tokens)
    reference_hashes: dict[str, str] | None = None
    for record in records:
        if record.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        if record.get("benchmark") != "turboquant_decode_attention":
            raise ValueError("unexpected benchmark name")
        if tuple(record.get("methods", ())) != METHODS:
            raise ValueError("unexpected method set or order")
        contexts = record.get("contexts")
        if not isinstance(contexts, list):
            raise ValueError("contexts must be a list")
        tokens = [context.get("tokens") for context in contexts]
        if len(tokens) != len(set(tokens)) or set(tokens) != expected_token_set:
            raise ValueError("context token matrix is incomplete or duplicated")

        hashes = _source_hashes(record)
        if expected_source_names is not None and set(hashes) != set(expected_source_names):
            raise ValueError("source provenance set does not match expectation")
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError("source provenance changed between processes")

        for context in contexts:
            if context.get("iterations") != expected_iterations:
                raise ValueError("unexpected iteration count")
            latencies = context.get("latency_ms")
            if not isinstance(latencies, dict) or set(latencies) != set(METHODS):
                raise ValueError("latency method matrix is incomplete")
            for method in METHODS:
                summary = latencies[method]
                samples = summary.get("samples") if isinstance(summary, dict) else None
                if (
                    not isinstance(samples, list)
                    or len(samples) != expected_iterations
                    or any(
                        not isinstance(sample, (int, float))
                        or not math.isfinite(float(sample))
                        or sample <= 0
                        for sample in samples
                    )
                ):
                    raise ValueError(f"invalid latency samples for {method}")
                recomputed = _sample_summary([float(sample) for sample in samples])
                for field, value in recomputed.items():
                    if field == "count":
                        if summary.get(field) != value:
                            raise ValueError(f"invalid recomputed {field}")
                    else:
                        _require_close(summary.get(field), float(value), field=field)

            correctness = context.get("correctness", {})
            all_finite = correctness.get("all_finite", {})
            if set(all_finite) != set(METHODS) or not all(all_finite.values()):
                raise ValueError("non-finite attention output")
            if float(correctness.get("aster_vs_mlx_max_abs", math.inf)) > 5e-3:
                raise ValueError("Aster paged attention parity failure")
            if float(correctness.get("tq_fused_vs_dequant_max_abs", math.inf)) > 5e-3:
                raise ValueError("TurboQuant fused/dequant parity failure")

            storage = context.get("storage", {})
            fp_bytes = storage.get("fp16_bytes")
            tq_bytes = storage.get("turboquant_bytes")
            if not isinstance(fp_bytes, int) or not isinstance(tq_bytes, int) or min(fp_bytes, tq_bytes) <= 0:
                raise ValueError("invalid cache storage evidence")
            _require_close(
                storage.get("compression_ratio"),
                fp_bytes / tq_bytes,
                field="compression_ratio",
            )
            swap_delta = context.get("swap_delta_bytes")
            if swap_delta is not None and not isinstance(swap_delta, int):
                raise ValueError("invalid swap delta")


def paired_method_improvements(
    records: Sequence[dict[str, Any]],
    *,
    tokens: int,
    candidate: str,
    baseline: str,
) -> list[float]:
    if candidate not in METHODS or baseline not in METHODS or candidate == baseline:
        raise ValueError("candidate and baseline must be distinct known methods")
    improvements: list[float] = []
    for record in sorted(records, key=lambda item: int(item["run_id"])):
        context = _context(record, tokens)
        candidate_ms = float(context["latency_ms"][candidate]["median_ms"])
        baseline_ms = float(context["latency_ms"][baseline]["median_ms"])
        improvements.append((baseline_ms - candidate_ms) / baseline_ms * 100.0)
    return improvements


def _bootstrap_median_interval(
    values: Sequence[float], *, seed: int, samples: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    population = tuple(values)
    medians = sorted(
        statistics.median(rng.choices(population, k=len(population)))
        for _ in range(samples)
    )
    lower = medians[max(0, math.floor(samples * 0.025) - 1)]
    upper = medians[min(samples - 1, math.ceil(samples * 0.975) - 1)]
    return lower, upper


def clears_stable_improvement_gate(
    values: Sequence[float], *, seed: int, threshold_percent: float = 3.0
) -> bool:
    lower, _ = _bootstrap_median_interval(values, seed=seed)
    return (
        statistics.median(values) >= threshold_percent
        and lower >= threshold_percent
    )


def _median(values: Iterable[float]) -> float:
    return statistics.median(tuple(values))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_turboquant(
    records: Sequence[dict[str, Any]],
    *,
    expected_runs: int,
    expected_tokens: Sequence[int],
    expected_iterations: int,
) -> dict[str, object]:
    validate_turboquant_records(
        records,
        expected_runs=expected_runs,
        expected_tokens=expected_tokens,
        expected_iterations=expected_iterations,
    )
    contexts: dict[str, object] = {}
    all_aster_paged_gates = True
    all_mlx_default_gates = True
    for tokens in expected_tokens:
        token_contexts = [_context(record, tokens) for record in records]
        method_summary: dict[str, object] = {}
        for method in METHODS:
            process_medians = [
                float(context["latency_ms"][method]["median_ms"])
                for context in token_contexts
            ]
            process_p95s = [
                float(context["latency_ms"][method]["p95_ms"])
                for context in token_contexts
            ]
            method_summary[method] = {
                "process_median_ms": _median(process_medians),
                "process_p95_median_ms": _median(process_p95s),
                "process_medians_ms": process_medians,
                "process_p95s_ms": process_p95s,
            }

        comparisons: dict[str, object] = {}
        for baseline in ("aster_paged_fp16", "mlx_fp16", "tq_dequant"):
            improvements = paired_method_improvements(
                records,
                tokens=tokens,
                candidate="tq_fused",
                baseline=baseline,
            )
            lower, upper = _bootstrap_median_interval(
                improvements, seed=49_117 + tokens + len(baseline)
            )
            comparisons[baseline] = {
                "paired_latency_reduction_percent": improvements,
                "median_latency_reduction_percent": _median(improvements),
                "bootstrap95_median_percent": [lower, upper],
            }

        aster_paged_gate = clears_stable_improvement_gate(
            comparisons["aster_paged_fp16"]["paired_latency_reduction_percent"],
            seed=49_117 + tokens + len("aster_paged_fp16"),
        )
        mlx_default_gate = clears_stable_improvement_gate(
            comparisons["mlx_fp16"]["paired_latency_reduction_percent"],
            seed=49_117 + tokens + len("mlx_fp16"),
        )
        all_aster_paged_gates = all_aster_paged_gates and aster_paged_gate
        all_mlx_default_gates = all_mlx_default_gates and mlx_default_gate
        swap_deltas = [
            context["swap_delta_bytes"]
            for context in token_contexts
            if context["swap_delta_bytes"] is not None
        ]
        contexts[str(tokens)] = {
            "methods": method_summary,
            "tq_fused_comparisons": comparisons,
            "correctness": {
                "aster_vs_mlx_max_abs": max(
                    float(context["correctness"]["aster_vs_mlx_max_abs"])
                    for context in token_contexts
                ),
                "tq_fused_vs_dequant_max_abs": max(
                    float(context["correctness"]["tq_fused_vs_dequant_max_abs"])
                    for context in token_contexts
                ),
                "tq_fused_vs_mlx_max_abs_median": _median(
                    float(context["correctness"]["tq_fused_vs_mlx_max_abs"])
                    for context in token_contexts
                ),
                "tq_fused_vs_mlx_mse_median": _median(
                    float(context["correctness"]["tq_fused_vs_mlx_mse"])
                    for context in token_contexts
                ),
                "key_dequant_mse_median": _median(
                    float(context["correctness"]["key_dequant_mse"])
                    for context in token_contexts
                ),
                "key_dequant_cosine_min": min(
                    float(context["correctness"]["key_dequant_cosine"])
                    for context in token_contexts
                ),
                "value_dequant_mse_median": _median(
                    float(context["correctness"]["value_dequant_mse"])
                    for context in token_contexts
                ),
                "value_dequant_cosine_min": min(
                    float(context["correctness"]["value_dequant_cosine"])
                    for context in token_contexts
                ),
            },
            "storage": {
                "fp16_bytes": int(_median(context["storage"]["fp16_bytes"] for context in token_contexts)),
                "turboquant_bytes": int(
                    _median(context["storage"]["turboquant_bytes"] for context in token_contexts)
                ),
                "compression_ratio_median": _median(
                    float(context["storage"]["compression_ratio"])
                    for context in token_contexts
                ),
            },
            "swap_delta_max_bytes": max(swap_deltas) if swap_deltas else None,
            "beats_aster_paged_gate_passed": aster_paged_gate,
            "beats_mlx_default_gate_passed": mlx_default_gate,
        }

    first = records[0]
    return {
        "schema_version": 1,
        "benchmark": "turboquant_decode_attention_aggregate",
        "evidence": {
            "fresh_processes": expected_runs,
            "iterations_per_method_per_process": expected_iterations,
            "tokens": list(expected_tokens),
            "total_timed_calls": expected_runs
            * expected_iterations
            * len(expected_tokens)
            * len(METHODS),
            "run_ids": sorted(int(record["run_id"]) for record in records),
            "pids": sorted(int(record["pid"]) for record in records),
            "source_hashes": _source_hashes(first),
        },
        "environment": first.get("environment"),
        "provenance": first.get("provenance"),
        "contexts": contexts,
        "decision": {
            "beats_aster_paged_gate_all_contexts": all_aster_paged_gates,
            "beats_mlx_default_gate_all_contexts": all_mlx_default_gates,
            "kernel_parity_gate": True,
            "full_model_quality_validated": False,
            "aster_runtime_integration": False,
            "reason": (
                "The candidate beats Aster paged but not the MLX default at every context; "
                "full-model quality is also unresolved."
                if all_aster_paged_gates and not all_mlx_default_gates
                else "The candidate did not clear every required kernel and quality gate."
            ),
        },
    }


def _load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", nargs="+", type=int, default=(2048, 8192, 32768, 65536))
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    records = _load_records(args.inputs)
    payload = aggregate_turboquant(
        records,
        expected_runs=args.runs,
        expected_tokens=args.tokens,
        expected_iterations=args.iterations,
    )
    validate_turboquant_records(
        records,
        expected_runs=args.runs,
        expected_tokens=args.tokens,
        expected_iterations=args.iterations,
        expected_source_names=SOURCE_NAMES,
    )
    payload["evidence"]["aggregator_sha256"] = _sha256(Path(__file__))
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
