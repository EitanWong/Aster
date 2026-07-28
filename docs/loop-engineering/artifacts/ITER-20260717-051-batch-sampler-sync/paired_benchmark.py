#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import production_benchmark as production
import psutil

from aster.inference.model_runner import DecodeResult, ModelRunner

sampling = production.sampling
base = sampling.base
ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _policy_order(step: int, phase: int = 0) -> tuple[str, str]:
    if (step + phase) % 2 == 0:
        return "baseline", "production"
    return "production", "baseline"


def _policy_runner_assignment(run_id: int) -> dict[str, str]:
    if run_id % 2 == 1:
        return {"baseline": "runner_a", "production": "runner_b"}
    return {"baseline": "runner_b", "production": "runner_a"}


def _replicate_id(run_id: int) -> int:
    if run_id < 1:
        raise ValueError("run_id must be positive")
    return (run_id + 1) // 2


def _structured_valid(text: str) -> bool:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"answer", "score"}
        and isinstance(value["answer"], str)
        and isinstance(value["score"], int)
        and not isinstance(value["score"], bool)
    )


def _contains_structured_document(segments: list[str]) -> bool:
    text = ""
    for segment in segments:
        text += segment
        if _structured_valid(text):
            return True
    return False


def _clone_runner(runner: ModelRunner) -> ModelRunner:
    cloned = copy.copy(runner)
    cloned._chat_prompt_cache = copy.copy(runner._chat_prompt_cache)
    cloned._decode_batch_cache_state = None
    cloned._decode_batch_cache_reuses = 0
    cloned._decode_batch_cache_rebuilds = 0
    cloned._decode_batch_post_sample_failures = 0
    cloned._decode_tokens_since_cache_clear = 0
    cloned._decode_cache_clear_attempts = 0
    cloned._decode_cache_clears = 0
    cloned._decode_cache_clear_failures = 0
    return cloned


def _prepare_lanes(
    runner: ModelRunner,
    *,
    label: str,
    args: argparse.Namespace,
    total_tokens: int,
) -> list[base.Lane]:
    lanes: list[base.Lane] = []
    prompt = base._prompt(args.context_words)
    for lane_index in range(args.batch_size):
        lane, _ = sampling._prepare_lane(
            runner,
            request_id=f"paired-{label}-lane-{lane_index}",
            prompt=prompt,
            max_tokens=total_tokens,
            prefill_step=args.prefill_step,
        )
        lanes.append(lane)
    return lanes


def _advance(
    runner: ModelRunner,
    lanes: list[base.Lane],
    *,
    max_tokens: int,
) -> None:
    results = runner.decode_batch_step(
        [base._work_item(lane, max_tokens) for lane in lanes]
    )
    if len(results) != len(lanes):
        raise RuntimeError("paired decode result count mismatch")
    for lane, result in zip(lanes, results, strict=True):
        if not isinstance(result, DecodeResult) or result.token_id is None:
            raise RuntimeError(f"paired decode failed: {result!r}")
        lane.prompt_cache = result.prompt_cache
        lane.input_token = result.token_id
        lane.output_tokens.append(result.token_id)
        lane.text_segments.append(result.text)


def _finalize_texts(runner: ModelRunner, lanes: list[base.Lane]) -> list[str]:
    texts: list[str] = []
    for lane in lanes:
        lane.text_segments.append(runner.finalize_detokenizer(lane.detokenizer))
        texts.append("".join(lane.text_segments))
    return texts


def run(args: argparse.Namespace) -> dict[str, object]:
    settings = base._settings(
        args.config,
        args.model,
        cache_kind="native",
        batch_size=args.batch_size,
    )
    runner_a = ModelRunner(settings)
    runner_a.warmup()
    base_mx = runner_a._mx
    if base_mx is None:
        raise RuntimeError("MLX failed to load")
    base._warmup(
        runner_a,
        tokens=args.model_warmup_tokens,
        prefill_step=args.prefill_step,
    )
    runner_b = _clone_runner(runner_a)
    runner_a._decode_batch_cache_state = None
    runner_a._decode_tokens_since_cache_clear = 0

    total_tokens = args.pair_warmup_steps + args.steps
    original_active_args = sampling._ACTIVE_ARGS
    sampling._ACTIVE_ARGS = args
    try:
        runner_a_lanes = _prepare_lanes(
            runner_a,
            label="runner-a",
            args=args,
            total_tokens=total_tokens,
        )
        runner_b_lanes = _prepare_lanes(
            runner_b,
            label="runner-b",
            args=args,
            total_tokens=total_tokens,
        )
        physical_runners = {
            "runner_a": (runner_a, runner_a_lanes),
            "runner_b": (runner_b, runner_b_lanes),
        }
        runner_assignment = _policy_runner_assignment(args.run_id)
        baseline_runner, baseline_lanes = physical_runners[
            runner_assignment["baseline"]
        ]
        production_runner, production_lanes = physical_runners[
            runner_assignment["production"]
        ]
        baseline_metrics, restore_baseline = production._install_policy(
            baseline_runner, "baseline"
        )
        production_metrics, restore_production = production._install_policy(
            production_runner, "production"
        )
    except Exception:
        sampling._ACTIVE_ARGS = original_active_args
        raise

    runners = {
        "baseline": (baseline_runner, baseline_lanes),
        "production": (production_runner, production_lanes),
    }
    phase = args.run_id - 1
    for step in range(args.pair_warmup_steps):
        for policy in _policy_order(step, phase):
            base_mx.random.seed(args.seed + step)
            runner, lanes = runners[policy]
            _advance(runner, lanes, max_tokens=total_tokens)
    base_mx.clear_cache()

    reset_peak = getattr(base_mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    timings: dict[str, list[float]] = {"baseline": [], "production": []}
    first_policies: list[str] = []
    try:
        for step in range(args.steps):
            policy_order = _policy_order(step, phase)
            first_policies.append(policy_order[0])
            for policy in policy_order:
                base_mx.random.seed(args.seed + args.pair_warmup_steps + step)
                runner, lanes = runners[policy]
                started = time.perf_counter()
                _advance(runner, lanes, max_tokens=total_tokens)
                timings[policy].append(time.perf_counter() - started)
    finally:
        restore_baseline()
        restore_production()
        sampling._ACTIVE_ARGS = original_active_args

    swap_after = int(psutil.swap_memory().used)
    baseline_cache = base._cache_digest(base_mx, baseline_runner, baseline_lanes)
    production_cache = base._cache_digest(base_mx, production_runner, production_lanes)
    baseline_texts = _finalize_texts(baseline_runner, baseline_lanes)
    production_texts = _finalize_texts(production_runner, production_lanes)
    baseline_tokens = [lane.output_tokens for lane in baseline_lanes]
    production_tokens = [lane.output_tokens for lane in production_lanes]
    baseline_text_hashes = [
        hashlib.sha256(text.encode()).hexdigest() for text in baseline_texts
    ]
    production_text_hashes = [
        hashlib.sha256(text.encode()).hexdigest() for text in production_texts
    ]
    structured_schema_valid = (
        all(
            _contains_structured_document(lane.text_segments)
            for lane in (*baseline_lanes, *production_lanes)
        )
        if args.workload == "structured"
        else None
    )
    exact = (
        baseline_tokens == production_tokens
        and baseline_text_hashes == production_text_hashes
        and baseline_cache == production_cache
    )

    sources = (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
        args.config,
        Path(__file__).resolve(),
        ARTIFACT_DIR / "production_benchmark.py",
        ARTIFACT_DIR / "sampling_benchmark.py",
        ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync/benchmark.py",
    )
    tokens = args.steps * args.batch_size
    payload: dict[str, object] = {
        "schema_version": 1,
        "pid": os.getpid(),
        "run_id": args.run_id,
        "workload": args.workload,
        "batch_size": args.batch_size,
        "context_words": args.context_words,
        "comparison_design": {
            "policy_runner_assignment": runner_assignment,
            "runner_assignment_alternates_by_run": True,
            "assignment_balanced_replicate_id": _replicate_id(args.run_id),
        },
        "settings": {
            "steps": args.steps,
            "pair_warmup_steps": args.pair_warmup_steps,
            "model_warmup_tokens": args.model_warmup_tokens,
            "prefill_step": args.prefill_step,
            "seed": args.seed,
            "block_size": args.block_size,
            "actual_prompt_tokens": len(baseline_lanes[0].prompt_tokens),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): base._sha256(path)
            for path in sources
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): base._sha256(path)
            for path in base._model_inputs(args.model)
        },
        "timings": {
            "baseline_step_seconds": timings["baseline"],
            "production_step_seconds": timings["production"],
            "first_policy_by_step": first_policies,
            "baseline_tokens_per_second": tokens / sum(timings["baseline"]),
            "production_tokens_per_second": tokens / sum(timings["production"]),
        },
        "parity": {
            "exact_token_text_cache": exact,
            "baseline_token_ids": baseline_tokens,
            "production_token_ids": production_tokens,
            "baseline_text_sha256": baseline_text_hashes,
            "production_text_sha256": production_text_hashes,
            "baseline_cache_digest": baseline_cache,
            "production_cache_digest": production_cache,
            "structured_schema_valid": structured_schema_valid,
        },
        "policy_metrics": {
            "baseline": baseline_metrics.metrics(),
            "production": production_metrics.metrics(),
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "mlx_peak_bytes": int(base_mx.get_peak_memory()),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
        },
    }

    for runner, lanes in runners.values():
        for lane in lanes:
            base._release_lane(runner, lane)
    base_mx.clear_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--workload", choices=sampling.WORKLOADS, required=True)
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--pair-warmup-steps", type=int, default=32)
    parser.add_argument("--model-warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if min(args.steps, args.pair_warmup_steps, args.model_warmup_tokens) < 1:
        raise ValueError("step counts must be positive")
    if args.block_size < 1 or args.steps % args.block_size:
        raise ValueError("steps must be divisible by block size")

    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
