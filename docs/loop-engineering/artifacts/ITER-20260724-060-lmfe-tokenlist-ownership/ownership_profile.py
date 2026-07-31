#!/usr/bin/env python3
"""Profile source-bound LMFE TokenList ownership and post-release lifetime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
ITER050_DIR = PROJECT_ROOT / "docs/loop-engineering/artifacts/ITER-20260717-050-decode-cache-sync"
ITER051_DIR = PROJECT_ROOT / "docs/loop-engineering/artifacts/ITER-20260717-051-batch-sampler-sync"
for path in (PROJECT_ROOT, ITER050_DIR, ITER051_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark as base  # noqa: E402
import sampling_benchmark as sampling  # noqa: E402
from lmformatenforcer import TokenEnforcer  # noqa: E402
from lmformatenforcer.tokenlist import TokenList  # noqa: E402

from aster.inference.constrained.json_schema_processor import (  # noqa: E402
    JSONSchemaLogitsProcessor,
)
from aster.inference.model_runner import DecodeResult, ModelRunner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _target_processor(processor: Any) -> JSONSchemaLogitsProcessor | None:
    current = processor
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, JSONSchemaLogitsProcessor):
            return current
        visited.add(id(current))
        current = getattr(current, "_inner", None)
    return None


def _target_processors(lanes: list[base.Lane]) -> list[JSONSchemaLogitsProcessor]:
    processors: list[JSONSchemaLogitsProcessor] = []
    for lane in lanes:
        for processor in lane.logits_processors:
            target = _target_processor(processor)
            if target is not None:
                processors.append(target)
                break
    return processors


def _token_list_stats(values: list[Any], *, large_threshold: int) -> tuple[dict[str, int], set[int]]:
    token_lists = {id(value): value for value in values if isinstance(value, TokenList)}
    backings = {
        id(value.allowed_tokens): value.allowed_tokens
        for value in token_lists.values()
        if hasattr(value, "allowed_tokens")
    }
    cardinalities = [
        len(backing)
        for backing in backings.values()
        if hasattr(backing, "__len__")
    ]
    return (
        {
            "references": len(values),
            "unique_token_lists": len(token_lists),
            "unique_backings": len(backings),
            "cardinality_total": sum(cardinalities),
            "largest_cardinality": max(cardinalities, default=0),
            "large_backings": sum(size >= large_threshold for size in cardinalities),
            "backing_shallow_bytes": sum(sys.getsizeof(backing) for backing in backings.values()),
        },
        set(token_lists),
    )


class OwnershipProfiler:
    def __init__(self, *, large_threshold: int) -> None:
        self.large_threshold = large_threshold
        self._previous_sequences: dict[int, tuple[int, ...]] = {}
        self._sequence_counts: dict[int, Counter[str]] = {}
        self._sequence_max_lengths: dict[int, int] = {}
        self.enforcer_ids: set[int] = set()
        self.request_token_list_ids: set[int] = set()
        self.static_token_list_ids: set[int] = set()

    def record_sequence(self, enforcer: TokenEnforcer, token_sequence: list[int]) -> None:
        enforcer_id = id(enforcer)
        sequence = tuple(int(token) for token in token_sequence)
        previous = self._previous_sequences.get(enforcer_id)
        counts = self._sequence_counts.setdefault(enforcer_id, Counter())
        counts["calls"] += 1
        if previous is not None:
            if sequence == previous:
                counts["repeats"] += 1
            elif len(sequence) == len(previous) + 1 and sequence[:-1] == previous:
                counts["append_one"] += 1
            else:
                counts["non_monotonic"] += 1
        self._previous_sequences[enforcer_id] = sequence
        self._sequence_max_lengths[enforcer_id] = max(
            self._sequence_max_lengths.get(enforcer_id, 0), len(sequence)
        )
        self.enforcer_ids.add(enforcer_id)

    def _enforcer_snapshot(self, enforcer: TokenEnforcer) -> dict[str, Any]:
        prefix_states = list(enforcer.prefix_states.items())
        state_values = [state.allowed_tokens for _key, state in prefix_states]
        cache_values = list(enforcer.allowed_token_cache.values())
        freetext_cache = list(enforcer.tokenizer_tree.json_freetext_tokens.allowlist_cache.values())
        working_values = [
            token_list
            for token_list, _static_length in getattr(
                enforcer, "_aster_working_freetext_lists", {}
            ).values()
        ]
        state_stats, state_ids = _token_list_stats(
            state_values, large_threshold=self.large_threshold
        )
        cache_stats, cache_ids = _token_list_stats(
            cache_values, large_threshold=self.large_threshold
        )
        static_stats, static_ids = _token_list_stats(
            freetext_cache, large_threshold=self.large_threshold
        )
        working_stats, working_ids = _token_list_stats(
            working_values, large_threshold=self.large_threshold
        )
        self.request_token_list_ids.update(state_ids)
        self.request_token_list_ids.update(cache_ids)
        self.request_token_list_ids.update(working_ids)
        self.static_token_list_ids.update(static_ids)
        shared_state_cache = len(state_ids & cache_ids)
        return {
            "prefix_state_count": len(prefix_states),
            "prefix_key_shallow_bytes": sum(sys.getsizeof(key) for key, _state in prefix_states),
            "prefix_states": state_stats,
            "allowed_token_cache": cache_stats,
            "freetext_cache": static_stats,
            "working_freetext_lists": working_stats,
            "shared_token_lists_between_prefix_and_cache": shared_state_cache,
        }

    def snapshot(self, *, step: int, lanes: list[base.Lane], mx: Any) -> dict[str, Any]:
        processors = _target_processors(lanes)
        enforcers = [processor._enforcer for processor in processors]
        per_lane = [self._enforcer_snapshot(enforcer) for enforcer in enforcers]
        current_python, peak_python = tracemalloc.get_traced_memory()
        return {
            "step": step,
            "rss_bytes": int(psutil.Process().memory_info().rss),
            "python_tracemalloc_current_bytes": current_python,
            "python_tracemalloc_peak_bytes": peak_python,
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "processors": per_lane,
        }

    def sequence_summary(self) -> dict[str, Any]:
        counts = Counter()
        for entry in self._sequence_counts.values():
            counts.update(entry)
        return {
            "enforcers": len(self._sequence_counts),
            "calls": counts["calls"],
            "append_one": counts["append_one"],
            "repeats": counts["repeats"],
            "non_monotonic": counts["non_monotonic"],
            "max_sequence_length": max(self._sequence_max_lengths.values(), default=0),
        }


def _snapshot_summary(snapshots: list[dict[str, Any]]) -> dict[str, int]:
    if not snapshots:
        return {}

    def total(snapshot: dict[str, Any], key: str) -> int:
        return sum(
            int(processor["prefix_states"][key])
            for processor in snapshot["processors"]
        )

    def working_total(snapshot: dict[str, Any], key: str) -> int:
        return sum(
            int(processor["working_freetext_lists"][key])
            for processor in snapshot["processors"]
        )

    first = snapshots[0]
    last = snapshots[-1]
    return {
        "snapshots": len(snapshots),
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
        "prefix_states_first": total(first, "references"),
        "prefix_states_last": total(last, "references"),
        "unique_prefix_token_lists_first": total(first, "unique_token_lists"),
        "unique_prefix_token_lists_last": total(last, "unique_token_lists"),
        "prefix_backing_bytes_first": total(first, "backing_shallow_bytes"),
        "prefix_backing_bytes_last": total(last, "backing_shallow_bytes"),
        "large_prefix_backings_first": total(first, "large_backings"),
        "large_prefix_backings_last": total(last, "large_backings"),
        "working_freetext_lists_first": working_total(first, "references"),
        "working_freetext_lists_last": working_total(last, "references"),
        "working_freetext_backing_bytes_first": working_total(
            first, "backing_shallow_bytes"
        ),
        "working_freetext_backing_bytes_last": working_total(
            last, "backing_shallow_bytes"
        ),
        "rss_delta_bytes": int(last["rss_bytes"]) - int(first["rss_bytes"]),
        "python_tracemalloc_delta_bytes": (
            int(last["python_tracemalloc_current_bytes"])
            - int(first["python_tracemalloc_current_bytes"])
        ),
    }


def _update_lanes(lanes: list[base.Lane], results: list[Any]) -> None:
    if len(results) != len(lanes):
        raise RuntimeError("decode result count mismatch")
    for lane, result in zip(lanes, results, strict=True):
        if not isinstance(result, DecodeResult) or result.token_id is None:
            raise RuntimeError(f"structured decode failed: {result!r}")
        lane.prompt_cache = result.prompt_cache
        lane.input_token = result.token_id
        lane.output_tokens.append(result.token_id)
        lane.text_segments.append(result.text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = base._settings(args.config, args.model, cache_kind="native", batch_size=args.batch_size)
    runner = ModelRunner(settings)
    runner.warmup()
    mx = runner._mx
    if mx is None:
        raise RuntimeError("MLX failed to load")
    base._warmup(runner, tokens=args.model_warmup_tokens, prefill_step=args.prefill_step)

    original_active_args = sampling._ACTIVE_ARGS
    original_get_allowed_tokens = TokenEnforcer.get_allowed_tokens
    profiler = OwnershipProfiler(large_threshold=args.large_token_list_threshold)
    sampling._ACTIVE_ARGS = args

    def observed_get_allowed_tokens(
        enforcer: TokenEnforcer, token_sequence: list[int]
    ) -> TokenList:
        result = original_get_allowed_tokens(enforcer, token_sequence)
        profiler.record_sequence(enforcer, token_sequence)
        return result

    TokenEnforcer.get_allowed_tokens = observed_get_allowed_tokens
    lanes: list[base.Lane] = []
    snapshots: list[dict[str, Any]] = []
    decode_started = 0.0
    decode_elapsed = 0.0
    try:
        tracemalloc.start()
        tracemalloc.reset_peak()
        for lane_index in range(args.batch_size):
            lane, _ = sampling._prepare_lane(
                runner,
                request_id=f"iter060-ownership-lane-{lane_index}",
                prompt=base._prompt(args.context_words),
                max_tokens=args.steps,
                prefill_step=args.prefill_step,
            )
            lanes.append(lane)
        prompt_token_count = len(lanes[0].prompt_tokens)
        if len(_target_processors(lanes)) != args.batch_size:
            raise RuntimeError("each structured lane must own one JSON processor")
        if args.retain_prefix_states:
            for processor in _target_processors(lanes):
                processor._enforcer = TokenEnforcer(
                    processor._tokenizer_data,
                    processor._enforcer.root_parser,
                )
                processor._enforcer.get_allowed_tokens(())
                processor._enforcer_last_suffix = ()
                processor._reuse_freetext_token_lists = False
                processor._bounded_prefix_states = False
            del processor
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        snapshots.append(profiler.snapshot(step=0, lanes=lanes, mx=mx))
        mx.random.seed(args.seed)
        decode_started = time.perf_counter()
        for step in range(args.steps):
            results = runner.decode_batch_step([base._work_item(lane, args.steps) for lane in lanes])
            _update_lanes(lanes, results)
            if step + 1 == args.steps or (step + 1) % args.sample_interval == 0:
                snapshots.append(profiler.snapshot(step=step + 1, lanes=lanes, mx=mx))
        decode_elapsed = time.perf_counter() - decode_started
        texts = [
            "".join([*lane.text_segments, runner.finalize_detokenizer(lane.detokenizer)])
            for lane in lanes
        ]
        output_hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    finally:
        TokenEnforcer.get_allowed_tokens = original_get_allowed_tokens
        sampling._ACTIVE_ARGS = original_active_args

    for lane in lanes:
        base._release_lane(runner, lane)
    del lane
    lanes.clear()
    gc.collect()
    live_ids = {id(value) for value in gc.get_objects()}
    post_release = {
        "rss_bytes": int(psutil.Process().memory_info().rss),
        "profiled_enforcers_still_live": len(profiler.enforcer_ids & live_ids),
        "request_token_lists_still_live": len(profiler.request_token_list_ids & live_ids),
        "static_token_lists_still_live": len(profiler.static_token_list_ids & live_ids),
    }
    mx.clear_cache()
    tracemalloc.stop()

    token_enforcer_path = Path(sys.modules[TokenEnforcer.__module__].__file__ or "")
    token_list_path = Path(sys.modules[TokenList.__module__].__file__ or "")
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        PROJECT_ROOT / "aster/inference/constrained/tokenizer_cache.py",
        token_enforcer_path,
        token_list_path,
        args.config,
    )
    payload = {
        "schema_version": 1,
        "purpose": "source-bound ownership and lifetime profile; not a performance A/B claim",
        "git_head": _git_head(),
        "pid": os.getpid(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "lm_format_enforcer": _distribution_version("lm-format-enforcer"),
        },
        "workload": {
            "name": "structured",
            "bounded_prefix_states": not args.retain_prefix_states,
            "freetext_allowlist_mode": (
                "native" if args.retain_prefix_states else "reused_list_backing"
            ),
            "batch_size": args.batch_size,
            "context_words": args.context_words,
            "steps": args.steps,
            "prompt_tokens": prompt_token_count,
            "prefill_step": args.prefill_step,
            "seed": args.seed,
            "sample_interval": args.sample_interval,
        },
        "decode": {
            "elapsed_seconds": decode_elapsed,
            "tokens_per_second": (
                args.steps * args.batch_size / decode_elapsed if decode_elapsed else 0.0
            ),
            "output_text_sha256": output_hashes,
        },
        "sequence": profiler.sequence_summary(),
        "ownership_summary": _snapshot_summary(snapshots),
        "post_release": post_release,
        "snapshots": snapshots,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): _sha256(path)
            for path in source_paths
            if path.is_file()
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--context-words", type=int, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--model-warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--sample-interval", type=int, default=16)
    parser.add_argument("--large-token-list-threshold", type=int, default=100_000)
    parser.add_argument("--retain-prefix-states", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.workload = "structured"
    if min(args.steps, args.model_warmup_tokens, args.prefill_step, args.sample_interval) < 1:
        raise ValueError("step, warmup, prefill, and sampling counts must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
