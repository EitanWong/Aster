#!/usr/bin/env python3
"""Attribute a greedy near tie to cache state or model batch arithmetic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.model_runner import DecodeWorkItem, ModelRunner  # noqa: E402
from scripts.dev import public_benchmark as public  # noqa: E402
from scripts.dev import public_engine_matrix as matrix  # noqa: E402

ITERATION = "ITER-20260801-089-greedy-batch-shape-determinism"
WORKLOAD_ID = "mt-bench:84:turn-1"
COMPANION_WORKLOAD_ID = "mt-bench:83:turn-1"
SELECTED_PREFIX = (271, 12646, 25, 357, 2526, 2923)
CANDIDATE_TOKEN_IDS = (364, 421, 8574)
PROBE_NAMES = (
    "aster_single",
    "aster_merge_extract_single",
    "aster_duplicate_batch",
    "mlx_lm_generation_batch",
    "aster_paired_history_single",
    "aster_paired_history_merge_extract_single",
    "aster_paired_history_duplicate_batch",
    "mlx_lm_paired_history_generation_batch",
    "aster_paired_history_actual_batch",
    "mlx_lm_paired_history_actual_batch",
)
EXPECTED_ROWS = {
    "aster_single": 1,
    "aster_merge_extract_single": 1,
    "aster_duplicate_batch": 2,
    "mlx_lm_generation_batch": 2,
    "aster_paired_history_single": 1,
    "aster_paired_history_merge_extract_single": 1,
    "aster_paired_history_duplicate_batch": 2,
    "mlx_lm_paired_history_generation_batch": 2,
    "aster_paired_history_actual_batch": 2,
    "mlx_lm_paired_history_actual_batch": 2,
}
ORDERS = ("single-first", "batch-first")
DEFAULT_WORKLOAD = PROJECT_ROOT / "run/loop-engineering/public-benchmarks/cross-engine-core.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/config.yaml"
DEFAULT_SOURCE_PATHS = (
    Path("aster/inference/model_runner.py"),
    Path("scripts/dev/diagnose_greedy_batch_shape.py"),
    Path("tests/test_greedy_batch_shape_diagnostic.py"),
)


class DiagnosticError(RuntimeError):
    """Raised when an I089 probe violates its frozen evidence contract."""


class _DiagnosticDetokenizer:
    def __init__(self) -> None:
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        self.last_segment = str(token)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise DiagnosticError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _probe_tokens(records: list[dict[str, Any]], name: str) -> list[int]:
    expected_rows = EXPECTED_ROWS[name]
    selected: set[int] = set()
    for record in records:
        probes = record.get("probes")
        if not isinstance(probes, dict):
            raise DiagnosticError("probe record has no probes object")
        probe = probes.get(name)
        if not isinstance(probe, dict):
            raise DiagnosticError(f"probe record has no {name} result")
        rows = probe.get("rows")
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise DiagnosticError(f"{name} must contain {expected_rows} rows")
        target_row = probe.get("target_row")
        selected_rows = rows
        if target_row is not None:
            if not isinstance(target_row, int) or not 0 <= target_row < len(rows):
                raise DiagnosticError(f"{name} has an invalid target row")
            selected_rows = [rows[target_row]]
        for row in selected_rows:
            if not isinstance(row, dict) or not isinstance(row.get("selected_token"), int):
                raise DiagnosticError(f"{name} contains an invalid selected token")
            selected.add(int(row["selected_token"]))
    return sorted(selected)


def classify_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify four source-identical, process-independent probe records."""

    if len(records) != 4:
        raise DiagnosticError("I089 requires exactly four independent process records")
    for record in records:
        if record.get("kind") != "greedy-batch-shape-probe":
            raise DiagnosticError("unexpected probe record kind")
        if record.get("performance_measurement_valid") is not False:
            raise DiagnosticError("diagnostic records must invalidate performance timing")

    processes = [record.get("process") for record in records]
    if not all(isinstance(process, dict) for process in processes):
        raise DiagnosticError("every record requires process metadata")
    pids = [process.get("pid") for process in processes]
    sequences = [process.get("sequence") for process in processes]
    if len(set(pids)) != 4 or len(set(sequences)) != 4:
        raise DiagnosticError("I089 requires four independent process identities")
    orders = [process.get("order") for process in processes]
    if sorted(orders) != ["batch-first", "batch-first", "single-first", "single-first"]:
        raise DiagnosticError("I089 requires balanced probe order")

    source_hashes = {_canonical_sha256(record.get("source_binding")) for record in records}
    if len(source_hashes) != 1:
        raise DiagnosticError("probe records do not share one source binding")
    frozen_hashes = {_canonical_sha256(record.get("frozen_state")) for record in records}
    if len(frozen_hashes) != 1:
        raise DiagnosticError("probe records do not share one frozen state")

    selected_tokens = {name: _probe_tokens(records, name) for name in PROBE_NAMES}
    cache_immutable = True
    for record in records:
        integrity = record.get("cache_integrity")
        frozen = record.get("frozen_state")
        if not isinstance(integrity, dict) or not isinstance(frozen, dict):
            cache_immutable = False
            break
        for key, frozen_key in (
            ("serial", "serial_cache_sha256"),
            ("aster_paired", "aster_paired_cache_sha256"),
            ("mlx_lm_paired", "mlx_lm_paired_cache_sha256"),
        ):
            state = integrity.get(key)
            if not isinstance(state, dict) or not (
                state.get("before_sha256") == state.get("after_sha256") == frozen.get(frozen_key)
            ):
                cache_immutable = False
                break

    stable = {name: len(tokens) == 1 for name, tokens in selected_tokens.items()}
    single_stable = stable["aster_single"]
    merged_stable = stable["aster_merge_extract_single"]
    aster_batch_stable = stable["aster_duplicate_batch"]
    native_batch_stable = stable["mlx_lm_generation_batch"]
    merge_matches = (
        single_stable
        and merged_stable
        and selected_tokens["aster_single"] == selected_tokens["aster_merge_extract_single"]
    )
    native_matches = (
        aster_batch_stable
        and native_batch_stable
        and selected_tokens["aster_duplicate_batch"] == selected_tokens["mlx_lm_generation_batch"]
    )
    serial_history_shape_invariant = (
        all(stable[name] for name in PROBE_NAMES[:4])
        and len({tuple(selected_tokens[name]) for name in PROBE_NAMES[:4]}) == 1
    )
    paired_history_stable = all(stable[name] for name in PROBE_NAMES[4:])
    paired_merge_matches = (
        stable["aster_paired_history_single"]
        and stable["aster_paired_history_merge_extract_single"]
        and selected_tokens["aster_paired_history_single"]
        == selected_tokens["aster_paired_history_merge_extract_single"]
    )
    paired_duplicate_reference_matches = (
        stable["aster_paired_history_duplicate_batch"]
        and stable["mlx_lm_paired_history_generation_batch"]
        and selected_tokens["aster_paired_history_duplicate_batch"]
        == selected_tokens["mlx_lm_paired_history_generation_batch"]
    )
    paired_actual_reference_matches = (
        stable["aster_paired_history_actual_batch"]
        and stable["mlx_lm_paired_history_actual_batch"]
        and selected_tokens["aster_paired_history_actual_batch"]
        == selected_tokens["mlx_lm_paired_history_actual_batch"]
    )
    cohort_shape_split_reproduced = (
        paired_history_stable
        and len(
            {
                tuple(selected_tokens[name])
                for name in (
                    "aster_paired_history_single",
                    "aster_paired_history_duplicate_batch",
                    "aster_paired_history_actual_batch",
                )
            }
        )
        == 3
    )
    i088_target_reproduced = selected_tokens["aster_paired_history_actual_batch"] == [421]
    history_shift_reproduced = (
        serial_history_shape_invariant
        and paired_actual_reference_matches
        and selected_tokens["aster_single"] != selected_tokens["aster_paired_history_actual_batch"]
    )
    frozen = records[0]["frozen_state"]
    paired_cache_identity = frozen.get("aster_paired_cache_sha256") == frozen.get(
        "mlx_lm_paired_cache_sha256"
    )
    history_cache_changed = frozen.get("serial_cache_sha256") != frozen.get(
        "aster_paired_cache_sha256"
    )
    gates = {
        "performance_timing_invalidated": True,
        "source_identity": True,
        "frozen_state_identity": True,
        "independent_processes": True,
        "balanced_probe_order": True,
        "canonical_cache_immutable": cache_immutable,
        "single_stable": single_stable,
        "merged_single_stable": merged_stable,
        "aster_batch_stable": aster_batch_stable,
        "model_native_batch_stable": native_batch_stable,
        "merge_extract_matches_single": merge_matches,
        "model_native_matches_aster_batch": native_matches,
        "serial_history_shape_invariant": serial_history_shape_invariant,
        "paired_history_stable": paired_history_stable,
        "paired_merge_extract_matches_single": paired_merge_matches,
        "paired_duplicate_reference_matches": paired_duplicate_reference_matches,
        "paired_actual_reference_matches": paired_actual_reference_matches,
        "cohort_shape_split_reproduced": cohort_shape_split_reproduced,
        "i088_target_reproduced": i088_target_reproduced,
        "paired_cache_identity": paired_cache_identity,
        "history_cache_changed": history_cache_changed,
        "history_shift_reproduced": history_shift_reproduced,
    }

    if not cache_immutable:
        diagnosis = "canonical-cache-mutated"
    elif not merge_matches:
        diagnosis = "cache-merge-extract-sensitive"
    elif not paired_merge_matches:
        diagnosis = "paired-cache-merge-extract-sensitive"
    elif (
        not native_matches
        or not paired_duplicate_reference_matches
        or not paired_actual_reference_matches
    ):
        diagnosis = "aster-batch-boundary-specific"
    elif not all(stable.values()):
        diagnosis = "inconclusive-unstable"
    elif not paired_cache_identity:
        diagnosis = "reference-history-cache-mismatch"
    elif not history_shift_reproduced or not cohort_shape_split_reproduced:
        diagnosis = "batched-history-drift-not-reproduced"
    else:
        diagnosis = "reference-shared-batched-history-cohort-arithmetic"

    return {
        "schema_version": 1,
        "kind": "greedy-batch-shape-classification",
        "performance_measurement_valid": False,
        "record_count": len(records),
        "selected_tokens": selected_tokens,
        "gates": gates,
        "contracts_passed": all(gates.values()),
        "diagnosis": diagnosis,
        "production_decision": "no-production-change",
    }


def _cache_fingerprint(cache: list[Any], mx: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    arrays = 0
    logical_bytes = 0

    def update_scalar(path: str, value: Any) -> None:
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")

    def visit(path: str, value: Any) -> None:
        nonlocal arrays, logical_bytes
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            mx.eval(value)
            header = (path, tuple(value.shape), str(value.dtype))
            update_scalar(path, header)
            array = np.asarray(value.view(mx.uint8))
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            digest.update(memoryview(array).cast("B"))
            arrays += 1
            logical_bytes += int(array.nbytes)
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(f"{path}.{key}", value[key])
            return
        if isinstance(value, (list, tuple)):
            update_scalar(path, type(value).__name__)
            for index, child in enumerate(value):
                visit(f"{path}[{index}]", child)
            return
        update_scalar(path, value)

    layers: list[dict[str, Any]] = []
    for index, layer in enumerate(cache):
        layer_type = type(layer).__name__
        size = layer.size() if callable(getattr(layer, "size", None)) else None
        offset = getattr(layer, "offset", None)
        if hasattr(offset, "tolist"):
            mx.eval(offset)
            offset = offset.tolist()
        metadata = {"index": index, "type": layer_type, "size": size, "offset": offset}
        layers.append(metadata)
        visit(f"layer[{index}].metadata", metadata)
        visit(f"layer[{index}].state", layer.state)
    return {
        "sha256": digest.hexdigest(),
        "arrays": arrays,
        "logical_bytes": logical_bytes,
        "layers": layers,
    }


def _score_row(mx: Any, row: Any, *, score_kind: str) -> dict[str, Any]:
    candidate_indices = mx.array(CANDIDATE_TOKEN_IDS, dtype=mx.uint32)
    candidate_scores = row[candidate_indices]
    top_indices = mx.argpartition(row, kth=-5)[-5:]
    top_scores = row[top_indices]
    mx.eval(candidate_scores, top_indices, top_scores)
    top = sorted(
        zip(top_indices.tolist(), top_scores.tolist(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return {
        "score_kind": score_kind,
        "candidate_scores": dict(
            zip(
                (str(token_id) for token_id in CANDIDATE_TOKEN_IDS),
                candidate_scores.tolist(),
                strict=True,
            )
        ),
        "top_token_ids": [int(token_id) for token_id, _ in top],
        "top_scores": [float(score) for _, score in top],
    }


def _work_item(
    runner: ModelRunner,
    cache: list[Any],
    decode_init: Any,
    prompt_tokens: list[int],
    *,
    request_id: str,
) -> DecodeWorkItem:
    cloned = runner.clone_cache(cache)
    if not isinstance(cloned, list):
        raise DiagnosticError("expected a list-backed prompt cache")
    return DecodeWorkItem(
        prompt_cache=cloned,
        input_token=SELECTED_PREFIX[-1],
        sampler=decode_init.sampler,
        detokenizer=_DiagnosticDetokenizer(),
        stop_token_ids=decode_init.stop_token_ids,
        logits_processors=decode_init.logits_processors,
        logits_processor_tokens=[*prompt_tokens, *SELECTED_PREFIX[:-1]],
        logits_processor_context_size=decode_init.logits_processor_context_size,
        completion_tokens=len(SELECTED_PREFIX),
        max_tokens=8,
        request_id=request_id,
    )


def _probe_aster(
    runner: ModelRunner,
    items: list[DecodeWorkItem],
    *,
    batch: bool,
) -> dict[str, Any]:
    mx = runner._mx
    if mx is None:
        raise DiagnosticError("MLX is not loaded")
    captured: dict[str, dict[str, Any]] = {}
    original_processors = runner._apply_logits_processors

    def capture(logits: Any, *, item: DecodeWorkItem) -> Any:
        processed = original_processors(logits, item=item)
        if item.request_id is None:
            raise DiagnosticError("diagnostic work item has no request ID")
        captured[item.request_id] = _score_row(mx, processed[0], score_kind="logits")
        return processed

    runner._apply_logits_processors = capture  # type: ignore[method-assign]
    runner._decode_batch_cache_state = None
    try:
        results = runner._decode_batch(items) if batch else [runner._decode_single(items[0])]
    finally:
        runner._apply_logits_processors = original_processors  # type: ignore[method-assign]
        runner._decode_batch_cache_state = None
    rows = []
    for item, result in zip(items, results, strict=True):
        if item.request_id not in captured or result.token_id is None:
            raise DiagnosticError("Aster probe did not produce a complete row")
        rows.append(
            {
                **captured[str(item.request_id)],
                "request_id": item.request_id,
                "selected_token": int(result.token_id),
            }
        )
    return {"boundary": "aster-model-runner", "batch_size": len(rows), "rows": rows}


def _probe_merge_extract_single(
    runner: ModelRunner,
    cache: list[Any],
    decode_init: Any,
    prompt_tokens: list[int],
) -> dict[str, Any]:
    first = runner.clone_cache(cache)
    second = runner.clone_cache(cache)
    if not isinstance(first, list) or not isinstance(second, list):
        raise DiagnosticError("expected cloneable list-backed prompt caches")
    merged = runner._merge_prompt_caches([first, second])
    runner._eval_cache(merged)
    extracted = runner._extract_prompt_cache(merged, 0)
    runner._eval_cache(extracted)
    item = _work_item(
        runner,
        extracted,
        decode_init,
        prompt_tokens,
        request_id="aster-merge-extract-single",
    )
    payload = _probe_aster(runner, [item], batch=False)
    payload["boundary"] = "aster-merge-extract-then-single"
    return payload


def _probe_generation_batch(
    runner: ModelRunner,
    cache: list[Any],
    decode_init: Any,
    prompt_tokens: list[int],
) -> dict[str, Any]:
    generation = importlib.import_module("mlx_lm.generate")
    generation_batch = generation.GenerationBatch
    state_machine = generation.SequenceStateMachine
    mx = runner._mx
    model = runner._model
    if mx is None or model is None:
        raise DiagnosticError("MLX model is not loaded")
    caches = [runner.clone_cache(cache), runner.clone_cache(cache)]
    if not all(isinstance(item, list) for item in caches):
        raise DiagnosticError("expected cloneable list-backed prompt caches")
    merged = runner._merge_prompt_caches(caches)
    context = [*prompt_tokens, *SELECTED_PREFIX[:-1]]
    batch = generation_batch(
        model=model,
        uids=[1, 2],
        inputs=mx.array([SELECTED_PREFIX[-1], SELECTED_PREFIX[-1]], dtype=mx.uint32),
        prompt_cache=merged,
        tokens=[list(context), list(context)],
        samplers=[decode_init.sampler, decode_init.sampler],
        fallback_sampler=decode_init.sampler,
        logits_processors=[[], []],
        state_machines=[state_machine(), state_machine()],
        max_tokens=[8, 8],
    )
    mx.eval(batch._next_tokens, batch._next_logprobs)
    selected = batch._next_tokens.tolist()
    rows = []
    for index, logprobs in enumerate(batch._next_logprobs):
        rows.append(
            {
                **_score_row(mx, logprobs, score_kind="logprobs"),
                "request_id": f"mlx-lm-generation-batch-{index}",
                "selected_token": int(selected[index]),
            }
        )
    return {"boundary": "mlx-lm-generation-batch", "batch_size": 2, "rows": rows}


def _probe_aster_actual_batch(
    runner: ModelRunner,
    *,
    target_cache: list[Any],
    companion_cache: list[Any],
    target_init: Any,
    companion_init: Any,
    target_tokens: list[int],
    companion_tokens: list[int],
    history: dict[str, Any],
) -> dict[str, Any]:
    target_prefix = [int(token) for token in history["target_prefix"]]
    companion_prefix = [int(token) for token in history["companion_prefix"]]
    cloned_companion = runner.clone_cache(companion_cache)
    cloned_target = runner.clone_cache(target_cache)
    if not isinstance(cloned_companion, list) or not isinstance(cloned_target, list):
        raise DiagnosticError("paired-history caches are not cloneable")
    items = [
        _history_work_item(
            prompt_cache=cloned_companion,
            input_token=companion_prefix[-1],
            decode_init=companion_init,
            prompt_tokens=companion_tokens,
            generated=companion_prefix[:-1],
            request_id="public-arrival:mixed-short-2",
            completion_tokens=len(companion_prefix),
        ),
        _history_work_item(
            prompt_cache=cloned_target,
            input_token=target_prefix[-1],
            decode_init=target_init,
            prompt_tokens=target_tokens,
            generated=target_prefix[:-1],
            request_id="public-arrival:mixed-short-3",
            completion_tokens=len(target_prefix),
        ),
    ]
    payload = _probe_aster(runner, items, batch=True)
    payload.update(
        {
            "boundary": "aster-paired-history-actual-cohort",
            "row_roles": ["companion", "target"],
            "target_row": 1,
        }
    )
    return payload


def _probe_generation_actual_batch(
    runner: ModelRunner,
    *,
    target_cache: list[Any],
    companion_cache: list[Any],
    target_init: Any,
    companion_init: Any,
    target_tokens: list[int],
    companion_tokens: list[int],
    history: dict[str, Any],
) -> dict[str, Any]:
    generation = importlib.import_module("mlx_lm.generate")
    mx = runner._mx
    model = runner._model
    if mx is None or model is None:
        raise DiagnosticError("MLX model is not loaded")
    caches = [runner.clone_cache(companion_cache), runner.clone_cache(target_cache)]
    if not all(isinstance(cache, list) for cache in caches):
        raise DiagnosticError("paired-history caches are not cloneable")
    companion_prefix = [int(token) for token in history["companion_prefix"]]
    target_prefix = [int(token) for token in history["target_prefix"]]
    merged = runner._merge_prompt_caches(caches)
    batch = generation.GenerationBatch(
        model=model,
        uids=[1, 2],
        inputs=mx.array([companion_prefix[-1], target_prefix[-1]], dtype=mx.uint32),
        prompt_cache=merged,
        tokens=[
            [*companion_tokens, *companion_prefix[:-1]],
            [*target_tokens, *target_prefix[:-1]],
        ],
        samplers=[companion_init.sampler, target_init.sampler],
        fallback_sampler=target_init.sampler,
        logits_processors=[[], []],
        state_machines=[generation.SequenceStateMachine(), generation.SequenceStateMachine()],
        max_tokens=[32, 32],
    )
    mx.eval(batch._next_tokens, batch._next_logprobs)
    selected = [int(token) for token in batch._next_tokens.tolist()]
    rows = [
        {
            **_score_row(mx, logprobs, score_kind="logprobs"),
            "request_id": request_id,
            "selected_token": selected[index],
        }
        for index, (logprobs, request_id) in enumerate(
            zip(
                batch._next_logprobs,
                (
                    "public-arrival:mixed-short-2",
                    "public-arrival:mixed-short-3",
                ),
                strict=True,
            )
        )
    ]
    return {
        "boundary": "mlx-lm-paired-history-actual-cohort",
        "batch_size": 2,
        "row_roles": ["companion", "target"],
        "target_row": 1,
        "rows": rows,
    }


def _settle(runner: ModelRunner) -> None:
    mx = runner._mx
    if mx is not None:
        mx.eval([])
        mx.clear_cache()
    gc.collect()


def _prepare_prompt(
    runner: ModelRunner, prompt: str, *, trace_id: str
) -> tuple[list[Any], Any, list[int]]:
    request = InferenceRequest(
        prompt=prompt,
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        trace_id=trace_id,
    )
    prepared = runner.encode_request(request)
    prompt_tokens = prepared.prompt_tokens
    target = len(prompt_tokens) - 1
    if target < 1:
        raise DiagnosticError("diagnostic prompt is too short")
    prefill = runner.prefill_to(
        prompt_tokens=prompt_tokens,
        prompt_cache=None,
        cache_token_count=0,
        target_cache_token_count=target,
    )
    decode_init = runner.initialize_decode(
        prompt_tokens=prompt_tokens,
        cache_token_count=prefill.cache_token_count,
        prompt_cache=prefill.prompt_cache,
        request=request,
    )
    if not isinstance(decode_init.prompt_cache, list):
        raise DiagnosticError("expected a list-backed prompt cache")
    return decode_init.prompt_cache, decode_init, prompt_tokens


def _history_work_item(
    *,
    prompt_cache: Any,
    input_token: int,
    decode_init: Any,
    prompt_tokens: list[int],
    generated: list[int],
    request_id: str,
    completion_tokens: int | None = None,
) -> DecodeWorkItem:
    return DecodeWorkItem(
        prompt_cache=prompt_cache,
        input_token=input_token,
        sampler=decode_init.sampler,
        detokenizer=decode_init.detokenizer,
        stop_token_ids=decode_init.stop_token_ids,
        logits_processors=decode_init.logits_processors,
        logits_processor_tokens=[*prompt_tokens, *generated],
        logits_processor_context_size=decode_init.logits_processor_context_size,
        completion_tokens=(len(generated) if completion_tokens is None else completion_tokens),
        max_tokens=32,
        request_id=request_id,
    )


def _prepare_serial_state(runner: ModelRunner, prompt: str) -> tuple[list[Any], Any, list[int]]:
    cache, decode_init, prompt_tokens = _prepare_prompt(
        runner, prompt, trace_id="i089-serial-history"
    )
    cache = decode_init.prompt_cache
    input_token = decode_init.next_input_token
    generated: list[int] = []
    for index, expected in enumerate(SELECTED_PREFIX):
        item = _history_work_item(
            prompt_cache=cache,
            input_token=input_token,
            decode_init=decode_init,
            prompt_tokens=prompt_tokens,
            generated=generated,
            request_id="i089-serial-target",
        )
        result = runner._decode_single(item)
        if result.token_id != expected:
            raise DiagnosticError(
                f"frozen prefix drifted at completion index {index}: "
                f"expected {expected}, observed {result.token_id}"
            )
        cache = result.prompt_cache
        input_token = expected
        generated.append(expected)
    if input_token != SELECTED_PREFIX[-1] or not isinstance(cache, list):
        raise DiagnosticError("failed to prepare the frozen divergence state")
    return cache, decode_init, prompt_tokens


def _prepare_aster_paired_state(
    runner: ModelRunner,
    *,
    target_prompt: str,
    companion_prompt: str,
) -> tuple[list[Any], list[Any], Any, Any, list[int], list[int], dict[str, Any]]:
    companion_cache, companion_init, companion_tokens = _prepare_prompt(
        runner, companion_prompt, trace_id="i089-aster-paired-companion"
    )
    target_cache, target_init, target_tokens = _prepare_prompt(
        runner, target_prompt, trace_id="i089-aster-paired-target"
    )
    caches: list[Any] = [companion_cache, target_cache]
    inputs = [companion_init.next_input_token, target_init.next_input_token]
    inits = [companion_init, target_init]
    prompts = [companion_tokens, target_tokens]
    generated: list[list[int]] = [[], []]
    request_ids = ["public-arrival:mixed-short-2", "public-arrival:mixed-short-3"]
    runner._decode_batch_cache_state = None
    for index, expected in enumerate(SELECTED_PREFIX):
        items = [
            _history_work_item(
                prompt_cache=caches[row],
                input_token=inputs[row],
                decode_init=inits[row],
                prompt_tokens=prompts[row],
                generated=generated[row],
                request_id=request_ids[row],
            )
            for row in range(2)
        ]
        results = runner._decode_batch(items)
        selected = [result.token_id for result in results]
        if any(token is None for token in selected):
            raise DiagnosticError("Aster paired history produced a terminal token")
        if selected[1] != expected:
            raise DiagnosticError(
                f"Aster paired history drifted at completion index {index}: "
                f"expected {expected}, observed {selected[1]}"
            )
        for row, result in enumerate(results):
            token = int(selected[row])
            generated[row].append(token)
            inputs[row] = token
            caches[row] = result.prompt_cache
    companion_cache = runner._resolve_decode_cache(caches[0])
    target_cache = runner._resolve_decode_cache(caches[1])
    if not isinstance(companion_cache, list) or not isinstance(target_cache, list):
        raise DiagnosticError("Aster paired caches are not extractable")
    runner._eval_cache(companion_cache)
    runner._eval_cache(target_cache)
    runner._decode_batch_cache_state = None
    return (
        target_cache,
        companion_cache,
        target_init,
        companion_init,
        target_tokens,
        companion_tokens,
        {
            "companion_prefix": generated[0],
            "companion_prompt_token_ids_sha256": _canonical_sha256(companion_tokens),
            "companion_prompt_token_count": len(companion_tokens),
            "target_prefix": generated[1],
        },
    )


def _prepare_mlx_lm_paired_state(
    runner: ModelRunner,
    *,
    target_prompt: str,
    companion_prompt: str,
) -> tuple[list[Any], list[Any], Any, Any, list[int], list[int], dict[str, Any]]:
    generation = importlib.import_module("mlx_lm.generate")
    mx = runner._mx
    model = runner._model
    if mx is None or model is None:
        raise DiagnosticError("MLX model is not loaded")
    companion_cache, companion_init, companion_tokens = _prepare_prompt(
        runner, companion_prompt, trace_id="i089-mlx-lm-paired-companion"
    )
    target_cache, target_init, target_tokens = _prepare_prompt(
        runner, target_prompt, trace_id="i089-mlx-lm-paired-target"
    )
    merged = runner._merge_prompt_caches([companion_cache, target_cache])
    batch = generation.GenerationBatch(
        model=model,
        uids=[1, 2],
        inputs=mx.array(
            [companion_init.next_input_token, target_init.next_input_token],
            dtype=mx.uint32,
        ),
        prompt_cache=merged,
        tokens=[companion_tokens[:-1], target_tokens[:-1]],
        samplers=[companion_init.sampler, target_init.sampler],
        fallback_sampler=target_init.sampler,
        logits_processors=[[], []],
        state_machines=[generation.SequenceStateMachine(), generation.SequenceStateMachine()],
        max_tokens=[32, 32],
    )
    generated: list[list[int]] = [[], []]
    for index, expected in enumerate(SELECTED_PREFIX):
        if index:
            batch.next()
        mx.eval(batch._next_tokens)
        selected = [int(token) for token in batch._next_tokens.tolist()]
        if selected[1] != expected:
            raise DiagnosticError(
                f"MLX-LM paired history drifted at completion index {index}: "
                f"expected {expected}, observed {selected[1]}"
            )
        for row, token in enumerate(selected):
            generated[row].append(token)
    companion_extracted = batch.extract_cache(0)
    target_extracted = batch.extract_cache(1)
    runner._eval_cache(companion_extracted)
    runner._eval_cache(target_extracted)
    return (
        target_extracted,
        companion_extracted,
        target_init,
        companion_init,
        target_tokens,
        companion_tokens,
        {
            "companion_prefix": generated[0],
            "companion_prompt_token_ids_sha256": _canonical_sha256(companion_tokens),
            "companion_prompt_token_count": len(companion_tokens),
            "target_prefix": generated[1],
        },
    )


def _load_model_fingerprint(path: Path, model_path: Path) -> dict[str, str]:
    payload = _read_json(path)
    if Path(str(payload.get("model_path"))).resolve() != model_path.resolve():
        raise DiagnosticError("model fingerprint points at a different model")
    fingerprint = payload.get("model_fingerprint")
    if not isinstance(fingerprint, dict):
        raise DiagnosticError("model fingerprint payload is incomplete")
    values = {key: fingerprint.get(key) for key in ("model_sha256", "tokenizer_sha256")}
    if not all(isinstance(value, str) and len(value) == 64 for value in values.values()):
        raise DiagnosticError("model fingerprint hashes are incomplete")
    return {key: str(value) for key, value in values.items()}


def create_model_fingerprint(config_path: Path) -> dict[str, Any]:
    settings = load_settings(str(config_path))
    model_path = Path(settings.model.path)
    return {
        "schema_version": 1,
        "kind": "i089-model-fingerprint",
        "created_utc": _now(),
        "model_path": str(model_path.resolve()),
        "model_fingerprint": matrix.model_fingerprint(model_path),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.sequence < 1:
        raise DiagnosticError("sequence must be positive")
    workload = _read_json(args.workload)
    lock = _read_json(args.lock)
    records = workload.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("public workload has no records")
    records_by_id = {
        str(item.get("workload_id")): item for item in records if isinstance(item, dict)
    }
    record = records_by_id.get(WORKLOAD_ID)
    companion_record = records_by_id.get(COMPANION_WORKLOAD_ID)
    if record is None or companion_record is None:
        raise DiagnosticError("public workload is missing the target or companion record")
    prompt = public.resolve_workload_prompt(record, lock, args.data_root)
    companion_prompt = public.resolve_workload_prompt(companion_record, lock, args.data_root)
    settings = load_settings(str(args.config))
    model_path = Path(settings.model.path)
    model_fingerprint = _load_model_fingerprint(args.model_fingerprint, model_path)
    runner = ModelRunner(settings)
    cache, decode_init, prompt_tokens = _prepare_serial_state(runner, prompt)
    _settle(runner)
    (
        aster_paired_cache,
        aster_companion_cache,
        aster_paired_init,
        aster_companion_init,
        aster_paired_tokens,
        aster_companion_tokens,
        aster_history,
    ) = _prepare_aster_paired_state(
        runner,
        target_prompt=prompt,
        companion_prompt=companion_prompt,
    )
    _settle(runner)
    (
        mlx_lm_paired_cache,
        mlx_lm_companion_cache,
        mlx_lm_paired_init,
        mlx_lm_companion_init,
        mlx_lm_paired_tokens,
        mlx_lm_companion_tokens,
        mlx_lm_history,
    ) = _prepare_mlx_lm_paired_state(
        runner,
        target_prompt=prompt,
        companion_prompt=companion_prompt,
    )
    _settle(runner)
    mx = runner._mx
    if mx is None:
        raise DiagnosticError("MLX did not initialize")
    cache_before = _cache_fingerprint(cache, mx)
    aster_paired_before = _cache_fingerprint(aster_paired_cache, mx)
    mlx_lm_paired_before = _cache_fingerprint(mlx_lm_paired_cache, mx)

    def single_probes() -> dict[str, dict[str, Any]]:
        single = _probe_aster(
            runner,
            [
                _work_item(
                    runner,
                    cache,
                    decode_init,
                    prompt_tokens,
                    request_id="aster-single",
                )
            ],
            batch=False,
        )
        _settle(runner)
        merged_single = _probe_merge_extract_single(runner, cache, decode_init, prompt_tokens)
        _settle(runner)
        paired_single = _probe_aster(
            runner,
            [
                _work_item(
                    runner,
                    aster_paired_cache,
                    aster_paired_init,
                    aster_paired_tokens,
                    request_id="aster-paired-history-single",
                )
            ],
            batch=False,
        )
        _settle(runner)
        paired_merged_single = _probe_merge_extract_single(
            runner,
            aster_paired_cache,
            aster_paired_init,
            aster_paired_tokens,
        )
        paired_merged_single["boundary"] = "aster-paired-merge-extract-then-single"
        _settle(runner)
        return {
            "aster_single": single,
            "aster_merge_extract_single": merged_single,
            "aster_paired_history_single": paired_single,
            "aster_paired_history_merge_extract_single": paired_merged_single,
        }

    def batch_probes() -> dict[str, dict[str, Any]]:
        aster_batch = _probe_aster(
            runner,
            [
                _work_item(
                    runner,
                    cache,
                    decode_init,
                    prompt_tokens,
                    request_id=f"aster-duplicate-{index}",
                )
                for index in range(2)
            ],
            batch=True,
        )
        _settle(runner)
        native_batch = _probe_generation_batch(runner, cache, decode_init, prompt_tokens)
        _settle(runner)
        paired_aster_batch = _probe_aster(
            runner,
            [
                _work_item(
                    runner,
                    aster_paired_cache,
                    aster_paired_init,
                    aster_paired_tokens,
                    request_id=f"aster-paired-history-duplicate-{index}",
                )
                for index in range(2)
            ],
            batch=True,
        )
        _settle(runner)
        paired_native_batch = _probe_generation_batch(
            runner,
            mlx_lm_paired_cache,
            mlx_lm_paired_init,
            mlx_lm_paired_tokens,
        )
        paired_native_batch["boundary"] = "mlx-lm-paired-history-generation-batch"
        _settle(runner)
        paired_actual_batch = _probe_aster_actual_batch(
            runner,
            target_cache=aster_paired_cache,
            companion_cache=aster_companion_cache,
            target_init=aster_paired_init,
            companion_init=aster_companion_init,
            target_tokens=aster_paired_tokens,
            companion_tokens=aster_companion_tokens,
            history=aster_history,
        )
        _settle(runner)
        paired_native_actual_batch = _probe_generation_actual_batch(
            runner,
            target_cache=mlx_lm_paired_cache,
            companion_cache=mlx_lm_companion_cache,
            target_init=mlx_lm_paired_init,
            companion_init=mlx_lm_companion_init,
            target_tokens=mlx_lm_paired_tokens,
            companion_tokens=mlx_lm_companion_tokens,
            history=mlx_lm_history,
        )
        _settle(runner)
        return {
            "aster_duplicate_batch": aster_batch,
            "mlx_lm_generation_batch": native_batch,
            "aster_paired_history_duplicate_batch": paired_aster_batch,
            "mlx_lm_paired_history_generation_batch": paired_native_batch,
            "aster_paired_history_actual_batch": paired_actual_batch,
            "mlx_lm_paired_history_actual_batch": paired_native_actual_batch,
        }

    probes: dict[str, dict[str, Any]] = {}
    groups = (single_probes, batch_probes)
    if args.order == "batch-first":
        groups = tuple(reversed(groups))
    for group in groups:
        probes.update(group())

    cache_after = _cache_fingerprint(cache, mx)
    aster_paired_after = _cache_fingerprint(aster_paired_cache, mx)
    mlx_lm_paired_after = _cache_fingerprint(mlx_lm_paired_cache, mx)
    if any(
        before["sha256"] != after["sha256"]
        for before, after in (
            (cache_before, cache_after),
            (aster_paired_before, aster_paired_after),
            (mlx_lm_paired_before, mlx_lm_paired_after),
        )
    ):
        raise DiagnosticError("diagnostic probes mutated the canonical cache")
    installed_generate = Path(importlib.import_module("mlx_lm.generate").__file__)
    installed_cache = Path(importlib.import_module("mlx_lm.models.cache").__file__)
    source = record.get("source")
    prompt_descriptor = record.get("prompt")
    companion_source = companion_record.get("source")
    companion_prompt_descriptor = companion_record.get("prompt")
    if not all(
        isinstance(value, dict)
        for value in (
            source,
            prompt_descriptor,
            companion_source,
            companion_prompt_descriptor,
        )
    ):
        raise DiagnosticError("public record has incomplete source metadata")
    runtime_sources = {
        "aster/inference/model_runner.py": _sha256_file(
            PROJECT_ROOT / "aster/inference/model_runner.py"
        ),
        "mlx_lm/generate.py": _sha256_file(installed_generate),
        "mlx_lm/models/cache.py": _sha256_file(installed_cache),
    }
    return {
        "schema_version": 1,
        "kind": "greedy-batch-shape-probe",
        "created_utc": _now(),
        "iteration": ITERATION,
        "performance_measurement_valid": False,
        "process": {"pid": os.getpid(), "sequence": args.sequence, "order": args.order},
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "source_binding": {
            "workload_ids": [WORKLOAD_ID, COMPANION_WORKLOAD_ID],
            "workload_sha256": _sha256_file(args.workload),
            "source_lock_sha256": _sha256_file(args.lock),
            "records": {
                WORKLOAD_ID: {
                    "record_sha256": str(source.get("record_sha256")),
                    "prompt_sha256": str(prompt_descriptor.get("sha256")),
                },
                COMPANION_WORKLOAD_ID: {
                    "record_sha256": str(companion_source.get("record_sha256")),
                    "prompt_sha256": str(companion_prompt_descriptor.get("sha256")),
                },
            },
            "model_fingerprint": model_fingerprint,
            "runtime_source_sha256": runtime_sources,
        },
        "frozen_state": {
            "prompt_token_ids_sha256": _canonical_sha256(prompt_tokens),
            "prompt_token_count": len(prompt_tokens),
            "companion_prompt_token_ids_sha256": aster_history["companion_prompt_token_ids_sha256"],
            "companion_prompt_token_count": aster_history["companion_prompt_token_count"],
            "selected_prefix": list(SELECTED_PREFIX),
            "completion_index": len(SELECTED_PREFIX),
            "input_token": SELECTED_PREFIX[-1],
            "candidate_token_ids": list(CANDIDATE_TOKEN_IDS),
            "serial_cache_sha256": cache_before["sha256"],
            "aster_paired_cache_sha256": aster_paired_before["sha256"],
            "mlx_lm_paired_cache_sha256": mlx_lm_paired_before["sha256"],
            "cache_layers": len(cache),
            "cache_arrays": cache_before["arrays"],
            "cache_logical_bytes": cache_before["logical_bytes"],
            "cache_layer_metadata": cache_before["layers"],
            "aster_paired_history": aster_history,
            "mlx_lm_paired_history": mlx_lm_history,
        },
        "cache_integrity": {
            "serial": {
                "before_sha256": cache_before["sha256"],
                "after_sha256": cache_after["sha256"],
            },
            "aster_paired": {
                "before_sha256": aster_paired_before["sha256"],
                "after_sha256": aster_paired_after["sha256"],
            },
            "mlx_lm_paired": {
                "before_sha256": mlx_lm_paired_before["sha256"],
                "after_sha256": mlx_lm_paired_after["sha256"],
            },
        },
        "probe_execution_order": list(probes),
        "probes": probes,
    }


def build_evidence(input_paths: list[Path]) -> dict[str, Any]:
    records = [_read_json(path) for path in input_paths]
    classification = classify_records(records)
    return {
        "schema_version": 1,
        "kind": "greedy-batch-shape-determinism-evidence",
        "created_utc": _now(),
        "iteration": ITERATION,
        "classification": classification,
        "raw_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "source_sha256": {
            str(path): _sha256_file(PROJECT_ROOT / path) for path in DEFAULT_SOURCE_PATHS
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    fingerprint.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    run.add_argument("--lock", type=Path, default=public.DEFAULT_LOCK_PATH)
    run.add_argument("--data-root", type=Path, default=public.DEFAULT_DATA_ROOT)
    run.add_argument("--model-fingerprint", type=Path, required=True)
    run.add_argument("--sequence", type=int, required=True)
    run.add_argument("--order", choices=ORDERS, required=True)
    run.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--input", type=Path, nargs=4, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--artifact", type=Path)

    args = parser.parse_args()
    if args.command == "fingerprint":
        payload = create_model_fingerprint(args.config)
        _write_json(args.output, payload)
    elif args.command == "run":
        payload = run_probe(args)
        _write_json(args.output, payload)
    else:
        payload = build_evidence(args.input)
        _write_json(args.output, payload["classification"])
        if args.artifact is not None:
            _write_json(args.artifact, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
