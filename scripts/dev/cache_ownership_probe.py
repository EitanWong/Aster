#!/usr/bin/env python3
"""Attribute exact-hit fanout memory to cache cloning or batch merge.

The probe resolves one locked public workload, prefills all but its final token,
then measures ``copy.deepcopy`` construction and MLX-LM batch-cache merge in
separate allocator intervals. Output contains hashes and measurements only;
prompt text and token IDs are never persisted.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_CONFIG = PROJECT_ROOT / "configs/config.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models/qwen3.5-9b-mlx/Qwen3.5-9B-4bit"
DEFAULT_WORKLOAD = PROJECT_ROOT / "run/loop-engineering/public-benchmarks/cross-engine-core.json"
SUPPORTED_CACHE_TYPES = frozenset({"ArraysCache", "KVCache"})
SUMMARY_METRICS = (
    "clone_seconds",
    "clone_active_delta_bytes",
    "merge_seconds",
    "merge_active_delta_bytes",
    "merged_state_bytes",
    "release_active_delta_bytes",
)


class ProbeError(RuntimeError):
    """Raised when a measurement would not satisfy the probe contract."""


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ProbeError("batch sizes must be comma-separated integers") from error
    if not parsed or any(size < 2 for size in parsed):
        raise ProbeError("every batch size must be at least 2")
    if len(set(parsed)) != len(parsed):
        raise ProbeError("batch sizes must be unique")
    return tuple(sorted(parsed))


def classify_cache_layers(cache: list[Any]) -> dict[str, Any]:
    counts = Counter(type(layer).__name__ for layer in cache)
    present = set(counts)
    unsupported = sorted(present - SUPPORTED_CACHE_TYPES)
    return {
        "counts": dict(sorted(counts.items())),
        "supported_types": sorted(present & SUPPORTED_CACHE_TYPES),
        "unsupported_types": unsupported,
        "all_supported": not unsupported,
    }


def _percentile(values: list[int | float], percentile: float) -> int | float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _numeric_summary(values: list[int | float]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def summarize_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ProbeError("at least one measurement row is required")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["fanout"])].append(row)

    by_fanout = {
        str(fanout): {
            metric: _numeric_summary([row[metric] for row in group]) for metric in SUMMARY_METRICS
        }
        for fanout, group in sorted(grouped.items())
    }
    clone_zero = all(int(row["clone_active_delta_bytes"]) == 0 for row in rows)
    merge_matches = all(
        abs(int(row["merge_active_delta_bytes"]) - int(row["merged_state_bytes"]))
        <= max(4096, int(row["merged_state_bytes"]) // 1000)
        for row in rows
    )
    release_clean = all(abs(int(row["release_active_delta_bytes"])) <= 4096 for row in rows)
    gates = {
        "clone_construction_zero_active_growth": clone_zero,
        "merge_growth_matches_materialized_state": merge_matches,
        "release_returns_to_baseline": release_clean,
    }
    return {
        "by_fanout": by_fanout,
        "gates": gates,
        "physical_owner": (
            "batch_merge_materialization" if all(gates.values()) else "inconclusive"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_workload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot read workload {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ProbeError(f"invalid public workload: {path}")
    return payload


def _select_record(workload: dict[str, Any], workload_id: str | None) -> dict[str, Any]:
    records = workload["records"]
    if workload_id is not None:
        selected = [record for record in records if record.get("workload_id") == workload_id]
    else:
        selected = [
            record
            for record in records
            if isinstance(record.get("source"), dict) and record["source"].get("dataset") == "qmsum"
        ][:1]
    if len(selected) != 1:
        label = workload_id or "first LongBench QMSUM record"
        raise ProbeError(f"workload selection did not resolve exactly one record: {label}")
    record = selected[0]
    if not isinstance(record, dict):
        raise ProbeError("selected workload record is not an object")
    return record


def _probe_settings(config_path: Path, model_path: Path, max_input_tokens: int):
    from aster.core.config import load_settings

    base = load_settings(str(config_path))
    model = base.model.model_copy(
        update={
            "name": model_path.name,
            "path": str(model_path),
            "context_length": max(base.model.context_length, max_input_tokens + 1),
            "enable_thinking": False,
        }
    )
    engine = base.engine.model_copy(
        update={
            "engine_type": "manual",
            "runtime_kernel": "manual",
            "max_active_requests": 1,
            "max_decode_batch": 1,
            "prefix_cache_enabled": False,
            "prefix_cache_load_on_warmup": False,
            "prefix_cache_save_on_shutdown": False,
            "warm_prompts_path": None,
            "paged_cache_enabled": False,
            "paged_cache_direct_attention_enabled": False,
        }
    )
    return base.model_copy(
        update={
            "model": model,
            "engine": engine,
            "cache": base.cache.model_copy(update={"prefix_cache_enabled": False}),
            "speculative": base.speculative.model_copy(
                update={"enabled": False, "max_draft_tokens": 0}
            ),
            "embeddings": base.embeddings.model_copy(update={"enabled": False}),
        }
    )


def cache_nbytes_by_type(cache: list[Any]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for layer in cache:
        nbytes = getattr(layer, "nbytes", None)
        if callable(nbytes):
            nbytes = nbytes()
        if not isinstance(nbytes, int):
            raise ProbeError(f"cache layer has no integer nbytes: {type(layer).__name__}")
        totals[type(layer).__name__] += nbytes
    return dict(sorted(totals.items()))


def _cache_nbytes(cache: list[Any]) -> int:
    return sum(cache_nbytes_by_type(cache).values())


def _clear_allocator(mx: Any) -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def _eval_cache_strict(cache: list[Any], mx: Any) -> None:
    arrays: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    for layer in cache:
        collect(getattr(layer, "state", ()))
    if not arrays:
        raise ProbeError("merged cache exposes no MLX arrays for strict evaluation")
    mx.eval(*arrays)
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()


def _prefill_public_record(
    runner: Any,
    prompt: str,
    *,
    prefill_step: int,
    max_input_tokens: int,
) -> tuple[list[int], list[Any], float]:
    from aster.inference.contracts import InferenceRequest

    request = InferenceRequest(
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        enable_thinking=False,
        trace_id="cache-ownership-probe-tokenize",
    )
    prompt_tokens = runner.encode_request(request).prompt_tokens
    if len(prompt_tokens) < 2:
        raise ProbeError("selected public prompt must encode to at least two tokens")
    if len(prompt_tokens) > max_input_tokens:
        raise ProbeError(
            f"public prompt has {len(prompt_tokens)} tokens, above the configured "
            f"maximum {max_input_tokens}"
        )
    target = len(prompt_tokens) - 1
    cache = None
    cached_tokens = 0
    elapsed = 0.0
    while cached_tokens < target:
        next_target = min(target, cached_tokens + prefill_step)
        result = runner.prefill_to(
            prompt_tokens=prompt_tokens,
            prompt_cache=cache,
            cache_token_count=cached_tokens,
            target_cache_token_count=next_target,
        )
        cache = result.prompt_cache
        cached_tokens = result.cache_token_count
        elapsed += result.elapsed_seconds
    if not isinstance(cache, list):
        raise ProbeError("model did not produce a list prompt cache")
    return prompt_tokens, cache, elapsed


def _measure_fanout(
    runner: Any,
    prompt_cache: list[Any],
    mx: Any,
    *,
    batch_sizes: tuple[int, ...],
    repetitions: int,
) -> tuple[list[dict[str, Any]], int, int]:
    _clear_allocator(mx)
    baseline_active = int(mx.get_active_memory())
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        rotation = (repetition - 1) % len(batch_sizes)
        order = (*batch_sizes[rotation:], *batch_sizes[:rotation])
        for order_index, fanout in enumerate(order, start=1):
            _clear_allocator(mx)
            before = int(mx.get_active_memory())
            mx.reset_peak_memory()
            started = time.perf_counter()
            clones = [copy.deepcopy(prompt_cache) for _ in range(fanout)]
            mx.synchronize()
            clone_seconds = time.perf_counter() - started
            after_clone = int(mx.get_active_memory())
            clone_peak = int(mx.get_peak_memory())

            started = time.perf_counter()
            merged = runner._merge_prompt_caches(clones)
            _eval_cache_strict(merged, mx)
            merge_seconds = time.perf_counter() - started
            after_merge = int(mx.get_active_memory())
            merge_peak = int(mx.get_peak_memory())
            merged_state_bytes = _cache_nbytes(merged)
            merged_state_bytes_by_type = cache_nbytes_by_type(merged)

            del merged, clones
            _clear_allocator(mx)
            after_release = int(mx.get_active_memory())
            rows.append(
                {
                    "repetition": repetition,
                    "order_index": order_index,
                    "fanout": fanout,
                    "active_before_bytes": before,
                    "clone_seconds": clone_seconds,
                    "clone_active_delta_bytes": after_clone - before,
                    "clone_peak_active_bytes": clone_peak,
                    "merge_seconds": merge_seconds,
                    "merge_active_delta_bytes": after_merge - after_clone,
                    "merge_peak_active_bytes": merge_peak,
                    "merged_state_bytes": merged_state_bytes,
                    "merged_state_bytes_by_type": merged_state_bytes_by_type,
                    "release_active_delta_bytes": after_release - baseline_active,
                }
            )
    _clear_allocator(mx)
    return rows, baseline_active, int(mx.get_active_memory())


def _model_descriptor(model_path: Path) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"path": _artifact_path(model_path)}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        path = model_path / name
        if path.is_file():
            descriptor[f"{name}_sha256"] = _sha256_file(path)
    return descriptor


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import public_benchmark as public

    from aster.inference.model_runner import ModelRunner

    if args.repetitions < 1:
        raise ProbeError("repetitions must be at least 1")
    if args.prefill_step < 1:
        raise ProbeError("prefill step must be at least 1")
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    workload_path = args.workload.resolve()
    lock_path = args.lock.resolve()
    workload = _load_workload(workload_path)
    if workload.get("lock_sha256") != _sha256_file(lock_path):
        raise ProbeError("workload source lock hash differs from the active source lock")
    record = _select_record(workload, args.workload_id)
    lock = public.load_lock(lock_path)
    resolver = public.PublicWorkloadResolver(lock, args.data_root.resolve())
    prompt = resolver.resolve(record)

    runner = ModelRunner(
        _probe_settings(args.config.resolve(), args.model.resolve(), args.max_input_tokens)
    )
    prompt_tokens, prompt_cache, prefill_seconds = _prefill_public_record(
        runner,
        prompt,
        prefill_step=args.prefill_step,
        max_input_tokens=args.max_input_tokens,
    )
    inventory = classify_cache_layers(prompt_cache)
    if not inventory["all_supported"]:
        raise ProbeError(
            "unsupported cache layer types: " + ", ".join(inventory["unsupported_types"])
        )
    rows, baseline_active, final_active = _measure_fanout(
        runner,
        prompt_cache,
        mx,
        batch_sizes=batch_sizes,
        repetitions=args.repetitions,
    )
    summary = summarize_measurements(rows)
    prompt_descriptor = record.get("prompt")
    if not isinstance(prompt_descriptor, dict):
        raise ProbeError("selected workload record has no prompt descriptor")
    return {
        "schema_version": 1,
        "kind": "cache-ownership-probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "source": {
            "workload": _artifact_path(workload_path),
            "workload_sha256": _sha256_file(workload_path),
            "source_lock": _artifact_path(lock_path),
            "source_lock_sha256": _sha256_file(lock_path),
            "workload_id": record.get("workload_id"),
            "prompt_sha256": prompt_descriptor.get("sha256"),
            "prompt_characters": prompt_descriptor.get("characters"),
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_ids_sha256": public.canonical_json_sha256(prompt_tokens),
        },
        "model": _model_descriptor(args.model),
        "execution": {
            "prefill_step": args.prefill_step,
            "cache_token_count": len(prompt_tokens) - 1,
            "prefill_seconds": prefill_seconds,
            "batch_sizes": list(batch_sizes),
            "repetitions": args.repetitions,
            "order": "left-rotated",
        },
        "cache": {
            **inventory,
            "base_allocated_bytes": _cache_nbytes(prompt_cache),
            "base_allocated_bytes_by_type": cache_nbytes_by_type(prompt_cache),
        },
        "measurements": rows,
        "summary": summary,
        "terminal": {
            "baseline_active_bytes": baseline_active,
            "final_active_bytes": final_active,
            "active_delta_bytes": final_active - baseline_active,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    import public_benchmark as public

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--workload-id")
    parser.add_argument("--lock", type=Path, default=public.DEFAULT_LOCK_PATH)
    parser.add_argument("--data-root", type=Path, default=public.DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        payload = run_probe(args)
    except ProbeError as error:
        raise SystemExit(f"cache ownership probe failed: {error}") from error
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
