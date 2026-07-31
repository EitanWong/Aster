#!/usr/bin/env python3
"""Run source-bound Aster and direct-MLX-LM public workload matrices.

The public workload manifest stores record and prompt hashes rather than prompt
text.  Each worker resolves its assigned records from the pinned source files,
runs one engine in a fresh process, and returns only reproducible hashes and
measurements.  The parent alternates engine order by public workload shard and
aggregates the partial records into the result format checked by
``public_benchmark.py validate-results``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import public_benchmark as public  # noqa: E402

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/config.yaml"
ENGINE_NAMES = ("aster", "mlx-lm")
TRUNCATION_POLICY = "official-longbench-half-head-half-tail"
ENGINE_ORDER_MODES = ("alternating", "reversed")
STATE_TRACE_TIMING_BOUNDARY = "outside-timed-public-request-intervals"
STATE_TRACE_QMSUM_SHARD = "longbench-qmsum"
STATE_TRACE_ABBA_MODES = ("alternating", "reversed", "reversed", "alternating")
COMPONENT_TRACE_TIMING_BOUNDARY = "inside-timed-public-request-decode-driver"
COMPONENT_TRACE_MODE = "source-bound-observer"
LOWER_LEVEL_DECODE_TRACE_TIMING_BOUNDARY = (
    "source-post-prefill-decode-step-submit-and-host-materialization"
)
LOWER_LEVEL_DECODE_TRACE_MODE = "source-bound-observer"
LOWER_LEVEL_INITIAL_OUTPUT_POLICY = (
    "exclude-first-output-step-because-direct-first-generator-advance-includes-prefill"
)
ASTER_INTERNAL_COMPONENT_SECONDS = (
    "cache_resolution_seconds",
    "model_graph_dispatch_seconds",
    "processor_graph_dispatch_seconds",
    "sampling_completion_seconds",
    "result_delivery_seconds",
    "unattributed_driver_seconds",
)
LOWER_LEVEL_COMPONENT_SECONDS = (
    "model_submit_seconds",
    "sampler_submit_seconds",
    "completion_residual_seconds",
)
PAIRED_METRICS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "ttft_seconds",
    "end_to_end_seconds",
    "peak_rss_bytes",
)
TRACE_NO_OP_METRICS = (
    "decode_tokens_per_second",
    "end_to_end_seconds",
    "ttft_seconds",
    "peak_rss_bytes",
)
TRACE_NO_OP_PARITY_FIELDS = (
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "prompt_token_count",
    "output_token_ids_sha256",
    "output_token_count",
    "text_sha256",
    "finish_reason",
)
LENGTH_BINS = (
    (0, 512, "[0,512)"),
    (512, 2048, "[512,2048)"),
    (2048, 8192, "[2048,8192)"),
    (8192, 32769, "[8192,32769)"),
    (32769, None, "[32769,+)"),
)
NO_OP_PERCENT = 3.0
MIN_ORDER_STRATUM_RECORDS = 8


class MatrixError(RuntimeError):
    """Raised when an adapter result cannot be used in a comparable matrix."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _state_trace_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return validated measurement-only trace metadata for a child process."""

    if not getattr(args, "state_trace", False):
        return None
    block_id = getattr(args, "state_trace_block_id", None)
    block_index = getattr(args, "state_trace_block_index", None)
    if not isinstance(block_id, str) or not block_id:
        raise MatrixError("state trace requires a non-empty block ID")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 1:
        raise MatrixError("state trace requires a positive block index")
    return {
        "schema_version": 1,
        "timing_boundary": STATE_TRACE_TIMING_BOUNDARY,
        "block": {"id": block_id, "index": block_index},
    }


def _component_trace_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timing_boundary": COMPONENT_TRACE_TIMING_BOUNDARY,
        "mode": COMPONENT_TRACE_MODE,
    }


def _component_trace_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the opt-in decode observer contract for a public matrix run."""

    if getattr(args, "component_trace", False) and getattr(args, "lower_level_decode_trace", False):
        raise MatrixError("component trace and lower-level decode trace are mutually exclusive")
    if not getattr(args, "component_trace", False):
        return None
    return _component_trace_metadata()


def _lower_level_decode_trace_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timing_boundary": LOWER_LEVEL_DECODE_TRACE_TIMING_BOUNDARY,
        "mode": LOWER_LEVEL_DECODE_TRACE_MODE,
        "initial_output_policy": LOWER_LEVEL_INITIAL_OUTPUT_POLICY,
    }


def _lower_level_decode_trace_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return I070's opt-in source-aligned lower-level trace contract."""

    if getattr(args, "component_trace", False) and getattr(args, "lower_level_decode_trace", False):
        raise MatrixError("component trace and lower-level decode trace are mutually exclusive")
    if not getattr(args, "lower_level_decode_trace", False):
        return None
    return _lower_level_decode_trace_metadata()


def _host_state_snapshot() -> dict[str, Any]:
    """Capture host/process context outside a timed public request interval."""

    process = psutil.Process()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process_memory = process.memory_info()
    process_cpu = process.cpu_times()
    try:
        load_average: list[float] | None = [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load_average = None
    return {
        "captured_utc": _now(),
        "load_average": load_average,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "available_memory_bytes": int(memory.available),
        "system_swap_used_bytes": int(swap.used),
        "process_rss_bytes": int(process_memory.rss),
        "process_vms_bytes": int(process_memory.vms),
        "process_cpu_user_seconds": float(process_cpu.user),
        "process_cpu_system_seconds": float(process_cpu.system),
    }


def _sha256_file(path: Path) -> str:
    return public.sha256_file(path)


def _json_hash(value: Any) -> str:
    return public.canonical_json_sha256(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MatrixError(f"JSON object expected at {path}")
    return payload


def _load_workload(path: Path) -> dict[str, Any]:
    workload = _read_json(path)
    if workload.get("kind") != "public-cross-engine-workload":
        raise MatrixError(f"not a public cross-engine workload: {path}")
    records = workload.get("records")
    if not isinstance(records, list) or not records:
        raise MatrixError("public workload has no records")
    if not isinstance(workload.get("generation"), dict):
        raise MatrixError("public workload has no generation contract")
    return workload


def _require_even_positive(value: int, name: str) -> int:
    if value < 2 or value % 2:
        raise MatrixError(f"{name} must be a positive even integer of at least two")
    return value


def _workload_shard_key(record: dict[str, Any]) -> str:
    source = record.get("source")
    if not isinstance(source, dict):
        raise MatrixError(f"workload record has no source: {record.get('workload_id')}")
    source_id = source.get("id")
    if source_id == "mt-bench-question":
        return "mt-bench"
    if source_id == "longbench-v1-data":
        dataset = source.get("dataset")
        if isinstance(dataset, str) and dataset:
            return f"longbench-{dataset}"
    raise MatrixError(f"unsupported workload source for sharding: {source_id}")


def workload_shards(workload: dict[str, Any]) -> OrderedDict[str, list[dict[str, Any]]]:
    shards: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in workload["records"]:
        if not isinstance(record, dict):
            raise MatrixError("workload contains a non-object record")
        shards.setdefault(_workload_shard_key(record), []).append(record)
    return shards


def engine_order_for_shard(shard_index: int, mode: str) -> tuple[str, str]:
    """Return the declared engine order for one workload shard.

    ``reversed`` is deliberately the exact inverse of the I066 alternating
    schedule. It changes process order only; workload selection and every
    per-engine execution setting remain unchanged.
    """

    if mode not in ENGINE_ORDER_MODES:
        raise MatrixError(
            f"engine order mode must be one of {', '.join(ENGINE_ORDER_MODES)}"
        )
    alternating = ENGINE_NAMES if shard_index % 2 == 0 else ENGINE_NAMES[::-1]
    if mode == "reversed":
        return tuple(reversed(alternating))
    return alternating


def _request(max_tokens: int, trace_id: str) -> InferenceRequest:
    return InferenceRequest(
        prompt="",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        enable_thinking=False,
        trace_id=trace_id,
    )


def _aster_settings(
    config_path: Path,
    model_path: Path,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
):
    base = load_settings(str(config_path))
    model = base.model.model_copy(
        update={
            "name": model_path.name,
            "path": str(model_path),
            "context_length": max(base.model.context_length, max_input_tokens + max_output_tokens),
            "enable_thinking": False,
        }
    )
    engine = base.engine.model_copy(
        update={
            "engine_type": "manual",
            "runtime_kernel": "manual",
            "max_active_requests": 1,
            "max_decode_batch": 1,
            "prefill_token_budget": 1024,
            "idle_prefill_token_limit": 1024,
            "pressure_prefill_token_budget": 1024,
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


def _trim_input_tokens(tokens: list[int], max_input_tokens: int) -> tuple[list[int], bool]:
    if len(tokens) <= max_input_tokens:
        return tokens, False
    half = max_input_tokens // 2
    return [*tokens[:half], *tokens[-half:]], True


class _TimedModelProxy:
    """Time model graph dispatch without changing the model object it delegates to."""

    def __init__(self, model: Any, observer: Any, metric: str = "model_graph_dispatch_seconds") -> None:
        self._model = model
        self._observer = observer
        self._metric = metric

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._observer.call(self._metric, self._model, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class _AsterDecodeComponentObserver:
    """Attach timing wrappers only around one already-initialized Aster decode loop.

    MLX executes lazily.  The sampling/materialization boundary therefore includes
    the completion barrier for the preceding model and processor graph; splitting
    that barrier with an extra ``mx.eval`` would change the measured execution.
    """

    _SECONDS = (
        "cache_resolution_seconds",
        "model_graph_dispatch_seconds",
        "processor_graph_dispatch_seconds",
        "sampling_completion_seconds",
        "result_delivery_seconds",
    )

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner
        self.seconds = {name: 0.0 for name in self._SECONDS}
        self._original_model: Any | None = None
        self._original_resolve: Any | None = None
        self._original_processors: Any | None = None
        self._original_sample: Any | None = None
        self._original_result: Any | None = None

    def call(self, name: str, callback: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            self.seconds[name] += time.perf_counter() - started

    def __enter__(self) -> _AsterDecodeComponentObserver:
        model = self.runner._model
        if model is None:
            raise MatrixError("Aster component trace requires a loaded model")
        self._original_model = model
        self._original_resolve = self.runner._resolve_decode_cache
        self._original_processors = self.runner._apply_logits_processors
        self._original_sample = self.runner._sample_token
        self._original_result = self.runner._decode_result

        def resolve(prompt_cache: Any) -> Any:
            return self.call("cache_resolution_seconds", self._original_resolve, prompt_cache)

        def processors(logits: Any, *, item: DecodeWorkItem) -> Any:
            return self.call(
                "processor_graph_dispatch_seconds",
                self._original_processors,
                logits,
                item=item,
            )

        def sample(logprobs: Any, sampler: Any) -> int:
            return self.call("sampling_completion_seconds", self._original_sample, logprobs, sampler)

        def decode_result(**kwargs: Any) -> DecodeResult:
            return self.call("result_delivery_seconds", self._original_result, **kwargs)

        self.runner._model = _TimedModelProxy(model, self)
        self.runner._resolve_decode_cache = resolve
        self.runner._apply_logits_processors = processors
        self.runner._sample_token = sample
        self.runner._decode_result = decode_result
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_model is not None:
            self.runner._model = self._original_model
        if self._original_resolve is not None:
            self.runner._resolve_decode_cache = self._original_resolve
        if self._original_processors is not None:
            self.runner._apply_logits_processors = self._original_processors
        if self._original_sample is not None:
            self.runner._sample_token = self._original_sample
        if self._original_result is not None:
            self.runner._decode_result = self._original_result


class _LowerLevelStepObserver:
    """Accumulate source call-site submissions within one decode outer step."""

    _SECONDS = ("model_submit_seconds", "sampler_submit_seconds")
    _CALLS = ("model_submit_calls", "sampler_submit_calls")

    def __init__(self) -> None:
        self.seconds = {name: 0.0 for name in self._SECONDS}
        self.calls = {name: 0 for name in self._CALLS}
        self._step_seconds: dict[str, float] | None = None
        self._step_calls: dict[str, int] | None = None

    def call(self, metric: str, callback: Any, *args: Any, **kwargs: Any) -> Any:
        if metric not in self.seconds:
            raise MatrixError(f"unknown lower-level timing metric: {metric}")
        started = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            self.seconds[metric] += time.perf_counter() - started
            self.calls[metric.replace("_seconds", "_calls")] += 1

    def begin_step(self) -> None:
        self._step_seconds = dict(self.seconds)
        self._step_calls = dict(self.calls)

    def finish_step(self, outer_step_seconds: float) -> dict[str, float | int]:
        if self._step_seconds is None or self._step_calls is None:
            raise MatrixError("lower-level trace step was not started")
        if not math.isfinite(outer_step_seconds) or outer_step_seconds < 0:
            raise MatrixError("lower-level trace has invalid outer step duration")
        model_submit_seconds = self.seconds["model_submit_seconds"] - self._step_seconds[
            "model_submit_seconds"
        ]
        sampler_submit_seconds = self.seconds["sampler_submit_seconds"] - self._step_seconds[
            "sampler_submit_seconds"
        ]
        accounted = model_submit_seconds + sampler_submit_seconds
        if accounted > outer_step_seconds + 1e-6:
            raise MatrixError("lower-level submissions exceed their outer decode step")
        return {
            "outer_step_seconds": outer_step_seconds,
            "model_submit_seconds": model_submit_seconds,
            "sampler_submit_seconds": sampler_submit_seconds,
            "completion_residual_seconds": max(0.0, outer_step_seconds - accounted),
            "model_submit_calls": self.calls["model_submit_calls"]
            - self._step_calls["model_submit_calls"],
            "sampler_submit_calls": self.calls["sampler_submit_calls"]
            - self._step_calls["sampler_submit_calls"],
        }


class _AsterLowerLevelDecodeObserver(_LowerLevelStepObserver):
    """Observe Aster's already-existing single-request decode source calls."""

    def __init__(self, runner: ModelRunner) -> None:
        super().__init__()
        self.runner = runner
        self._original_model: Any | None = None
        self._original_sample: Any | None = None

    def __enter__(self) -> _AsterLowerLevelDecodeObserver:
        model = self.runner._model
        if model is None:
            raise MatrixError("Aster lower-level trace requires a loaded model")
        self._original_model = model
        self._original_sample = self.runner._sample_token

        def sample(logprobs: Any, sampler: Any) -> int:
            def timed_sampler(value: Any) -> Any:
                return self.call("sampler_submit_seconds", sampler, value)

            return self._original_sample(logprobs, timed_sampler)

        self.runner._model = _TimedModelProxy(model, self, "model_submit_seconds")
        self.runner._sample_token = sample
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_model is not None:
            self.runner._model = self._original_model
        if self._original_sample is not None:
            self.runner._sample_token = self._original_sample


class _DirectLowerLevelDecodeObserver(_LowerLevelStepObserver):
    """Observe direct MLX-LM calls without modifying ``stream_generate`` itself."""

    def __init__(self, model: Any, sampler: Any) -> None:
        super().__init__()
        self.model = _TimedModelProxy(model, self, "model_submit_seconds")
        self._sampler = sampler

    def sampler(self, logprobs: Any) -> Any:
        return self.call("sampler_submit_seconds", self._sampler, logprobs)


def _diagnostic_delta(
    before: dict[str, object],
    after: dict[str, object],
    field: str,
) -> int:
    previous = before.get(field)
    current = after.get(field)
    if (
        isinstance(previous, bool)
        or not isinstance(previous, int)
        or isinstance(current, bool)
        or not isinstance(current, int)
        or current < previous
    ):
        raise MatrixError(f"Aster decode diagnostic {field} is not monotonic")
    return current - previous


def _aster_component_trace(
    observer: _AsterDecodeComponentObserver,
    *,
    decode_steps: int,
    driver_seconds: float,
    caller_bookkeeping_seconds: float,
    post_decode_delivery_seconds: float,
    diagnostics_before: dict[str, object],
    diagnostics_after: dict[str, object],
) -> dict[str, Any]:
    component_seconds = dict(observer.seconds)
    accounted_seconds = sum(component_seconds.values())
    if accounted_seconds > driver_seconds + 1e-6:
        raise MatrixError("Aster component trace exceeds its decode driver boundary")
    component_seconds["unattributed_driver_seconds"] = max(
        0.0,
        driver_seconds - accounted_seconds,
    )
    cache_fields = (
        "batch_cache_reuses",
        "batch_cache_rebuilds",
        "single_steps",
        "cache_clear_attempts",
        "cache_clears",
        "cache_clear_failures",
    )
    return {
        "schema_version": 1,
        "timing_boundary": COMPONENT_TRACE_TIMING_BOUNDARY,
        "mode": COMPONENT_TRACE_MODE,
        "engine_boundary": "aster-manual-model-runner-single-decode-step",
        "decode": {
            "steps": decode_steps,
            "batch_size_min": 1,
            "batch_size_max": 1,
            "batch_size_total_items": decode_steps,
        },
        "seconds": {
            "decode_driver_seconds": driver_seconds,
            "caller_bookkeeping_seconds": caller_bookkeeping_seconds,
            "post_decode_delivery_seconds": post_decode_delivery_seconds,
            **component_seconds,
        },
        "cache": {
            "decode_mode": "single-request-no-batch-merge",
            **{
                field: _diagnostic_delta(diagnostics_before, diagnostics_after, field)
                for field in cache_fields
            },
        },
        "cross_engine_comparable_boundary": "decode_driver_seconds",
    }


def _mlx_lm_component_trace(
    *,
    decode_steps: int,
    driver_seconds: float,
    caller_bookkeeping_seconds: float,
    post_decode_delivery_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timing_boundary": COMPONENT_TRACE_TIMING_BOUNDARY,
        "mode": COMPONENT_TRACE_MODE,
        "engine_boundary": "mlx-lm-stream-generate-next",
        "decode": {
            "steps": decode_steps,
            "batch_size_min": 1,
            "batch_size_max": 1,
            "batch_size_total_items": decode_steps,
        },
        "seconds": {
            "decode_driver_seconds": driver_seconds,
            "caller_bookkeeping_seconds": caller_bookkeeping_seconds,
            "post_decode_delivery_seconds": post_decode_delivery_seconds,
        },
        "cross_engine_comparable_boundary": "decode_driver_seconds",
    }


def _lower_level_decode_trace(
    *,
    engine: str,
    output_token_count: int,
    steps: list[dict[str, float | int]],
) -> dict[str, Any]:
    expected_steps = max(output_token_count - 1, 0)
    if len(steps) != expected_steps:
        raise MatrixError("lower-level trace does not cover every post-prefill decode step")
    seconds = {
        field: sum(float(step[field]) for step in steps)
        for field in ("outer_step_seconds", *LOWER_LEVEL_COMPONENT_SECONDS)
    }
    calls = {
        field: sum(int(step[field]) for step in steps)
        for field in ("model_submit_calls", "sampler_submit_calls")
    }
    if not math.isclose(
        seconds["outer_step_seconds"],
        sum(seconds[field] for field in LOWER_LEVEL_COMPONENT_SECONDS),
        rel_tol=0.01,
        abs_tol=1e-6,
    ):
        raise MatrixError("lower-level trace does not reconcile to its outer decode boundary")
    return {
        **_lower_level_decode_trace_metadata(),
        "engine_boundary": {
            "aster": "aster-manual-single-decode-step",
            "mlx-lm": "mlx-lm-stream-generate-post-first-next",
        }[engine],
        "decode": {
            "generated_output_steps": output_token_count,
            "traced_post_prefill_steps": expected_steps,
            "excluded_initial_output_steps": 1 if output_token_count else 0,
        },
        "seconds": seconds,
        "calls": calls,
        "cross_engine_comparable_components": [
            "outer_step_seconds",
            *LOWER_LEVEL_COMPONENT_SECONDS,
        ],
    }


def _aster_generate(
    runner: ModelRunner,
    prompt_tokens: list[int],
    request: InferenceRequest,
    *,
    prefill_step: int,
    component_trace: bool = False,
    lower_level_decode_trace: bool = False,
) -> dict[str, Any]:
    if component_trace and lower_level_decode_trace:
        raise MatrixError("component trace and lower-level decode trace are mutually exclusive")
    if len(prompt_tokens) < 2:
        raise MatrixError("public prompt must encode to at least two tokens")
    # The production scheduler owns this counter across live requests.  A public
    # matrix record is one independent request, so scope the maintenance counter
    # to that record.  Otherwise a warmup can clear MLX's allocator mid-response
    # at a different token position than direct MLX-LM and invalidate parity.
    runner._decode_tokens_since_cache_clear = 0
    prompt_cache = None
    cache_token_count = 0
    target = len(prompt_tokens) - 1
    started = time.perf_counter()
    while cache_token_count < target:
        result = runner.prefill_to(
            prompt_tokens=prompt_tokens,
            prompt_cache=prompt_cache,
            cache_token_count=cache_token_count,
            target_cache_token_count=min(cache_token_count + prefill_step, target),
        )
        prompt_cache = result.prompt_cache
        cache_token_count = result.cache_token_count
    decode = runner.initialize_decode(
        prompt_tokens=prompt_tokens,
        cache_token_count=cache_token_count,
        prompt_cache=prompt_cache,
        request=request,
    )
    prefill_seconds = time.perf_counter() - started
    input_token = decode.next_input_token
    output_tokens: list[int] = []
    text_segments: list[str] = []
    decode_started = time.perf_counter()
    first_token_seconds: float | None = None
    finish_reason = "length"
    component_payload: dict[str, Any] | None = None
    lower_level_payload: dict[str, Any] | None = None
    if component_trace:
        observer = _AsterDecodeComponentObserver(runner)
        diagnostics_before = runner.decode_diagnostics()
        driver_seconds = 0.0
        caller_bookkeeping_seconds = 0.0
        with observer:
            for _ in range(request.max_tokens):
                step_started = time.perf_counter()
                processor_tokens = [*prompt_tokens, *output_tokens]
                if processor_tokens and processor_tokens[-1] == input_token:
                    processor_tokens.pop()
                item = DecodeWorkItem(
                    prompt_cache=decode.prompt_cache,
                    input_token=input_token,
                    sampler=decode.sampler,
                    detokenizer=decode.detokenizer,
                    stop_token_ids=decode.stop_token_ids,
                    logits_processors=decode.logits_processors,
                    logits_processor_tokens=processor_tokens,
                    completion_tokens=len(output_tokens),
                    max_tokens=request.max_tokens,
                    request_id=request.trace_id,
                    logits_processor_context_size=decode.logits_processor_context_size,
                )
                driver_started = time.perf_counter()
                result = runner.decode_batch_step([item])[0]
                driver_elapsed = time.perf_counter() - driver_started
                driver_seconds += driver_elapsed
                if not isinstance(result, DecodeResult) or result.token_id is None:
                    raise MatrixError(f"Aster decode failed: {result!r}")
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started
                decode.prompt_cache = result.prompt_cache
                input_token = result.token_id
                output_tokens.append(result.token_id)
                text_segments.append(result.text)
                caller_bookkeeping_seconds += max(
                    0.0,
                    time.perf_counter() - step_started - driver_elapsed,
                )
                if result.finish_reason is not None:
                    finish_reason = result.finish_reason
                    break
        decode_seconds = time.perf_counter() - decode_started
        post_decode_started = time.perf_counter()
        text_segments.append(runner.finalize_detokenizer(decode.detokenizer))
        post_decode_delivery_seconds = time.perf_counter() - post_decode_started
        component_payload = _aster_component_trace(
            observer,
            decode_steps=len(output_tokens),
            driver_seconds=driver_seconds,
            caller_bookkeeping_seconds=caller_bookkeeping_seconds,
            post_decode_delivery_seconds=post_decode_delivery_seconds,
            diagnostics_before=diagnostics_before,
            diagnostics_after=runner.decode_diagnostics(),
        )
    elif lower_level_decode_trace:
        observer = _AsterLowerLevelDecodeObserver(runner)
        lower_level_steps: list[dict[str, float | int]] = []
        with observer:
            for decode_index in range(request.max_tokens):
                processor_tokens = [*prompt_tokens, *output_tokens]
                if processor_tokens and processor_tokens[-1] == input_token:
                    processor_tokens.pop()
                item = DecodeWorkItem(
                    prompt_cache=decode.prompt_cache,
                    input_token=input_token,
                    sampler=decode.sampler,
                    detokenizer=decode.detokenizer,
                    stop_token_ids=decode.stop_token_ids,
                    logits_processors=decode.logits_processors,
                    logits_processor_tokens=processor_tokens,
                    completion_tokens=len(output_tokens),
                    max_tokens=request.max_tokens,
                    request_id=request.trace_id,
                    logits_processor_context_size=decode.logits_processor_context_size,
                )
                observer.begin_step()
                step_started = time.perf_counter()
                result = runner.decode_batch_step([item])[0]
                step_elapsed = time.perf_counter() - step_started
                if not isinstance(result, DecodeResult) or result.token_id is None:
                    raise MatrixError(f"Aster decode failed: {result!r}")
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started
                decode.prompt_cache = result.prompt_cache
                input_token = result.token_id
                output_tokens.append(result.token_id)
                text_segments.append(result.text)
                if decode_index > 0:
                    lower_level_steps.append(observer.finish_step(step_elapsed))
                if result.finish_reason is not None:
                    finish_reason = result.finish_reason
                    break
        decode_seconds = time.perf_counter() - decode_started
        text_segments.append(runner.finalize_detokenizer(decode.detokenizer))
        lower_level_payload = _lower_level_decode_trace(
            engine="aster",
            output_token_count=len(output_tokens),
            steps=lower_level_steps,
        )
    else:
        for _ in range(request.max_tokens):
            processor_tokens = [*prompt_tokens, *output_tokens]
            if processor_tokens and processor_tokens[-1] == input_token:
                processor_tokens.pop()
            item = DecodeWorkItem(
                prompt_cache=decode.prompt_cache,
                input_token=input_token,
                sampler=decode.sampler,
                detokenizer=decode.detokenizer,
                stop_token_ids=decode.stop_token_ids,
                logits_processors=decode.logits_processors,
                logits_processor_tokens=processor_tokens,
                completion_tokens=len(output_tokens),
                max_tokens=request.max_tokens,
                request_id=request.trace_id,
                logits_processor_context_size=decode.logits_processor_context_size,
            )
            result = runner.decode_batch_step([item])[0]
            if not isinstance(result, DecodeResult) or result.token_id is None:
                raise MatrixError(f"Aster decode failed: {result!r}")
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            decode.prompt_cache = result.prompt_cache
            input_token = result.token_id
            output_tokens.append(result.token_id)
            text_segments.append(result.text)
            if result.finish_reason is not None:
                finish_reason = result.finish_reason
                break
        decode_seconds = time.perf_counter() - decode_started
        text_segments.append(runner.finalize_detokenizer(decode.detokenizer))
    payload = {
        "output_token_ids": output_tokens,
        "text_sha256": public.sha256_text("".join(text_segments)),
        "finish_reason": finish_reason,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "ttft_seconds": first_token_seconds if first_token_seconds is not None else 0.0,
        "end_to_end_seconds": time.perf_counter() - started,
    }
    if component_payload is not None:
        payload["component_trace"] = component_payload
    if lower_level_payload is not None:
        payload["lower_level_decode_trace"] = lower_level_payload
    return payload


def _mlx_lm_generate(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    *,
    max_tokens: int,
    prefill_step: int,
    component_trace: bool = False,
    lower_level_decode_trace: bool = False,
) -> dict[str, Any]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if component_trace and lower_level_decode_trace:
        raise MatrixError("component trace and lower-level decode trace are mutually exclusive")
    responses: list[Any] = []
    started = time.perf_counter()
    first_token_seconds: float | None = None
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0)
    lower_level_observer = (
        _DirectLowerLevelDecodeObserver(model, sampler) if lower_level_decode_trace else None
    )
    generator = stream_generate(
        lower_level_observer.model if lower_level_observer is not None else model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        sampler=lower_level_observer.sampler if lower_level_observer is not None else sampler,
        prefill_step_size=prefill_step,
    )
    component_payload: dict[str, Any] | None = None
    if component_trace:
        raw_generation_advance_seconds = 0.0
        caller_bookkeeping_seconds = 0.0
        while True:
            advance_started = time.perf_counter()
            try:
                response = next(generator)
            except StopIteration:
                break
            advance_elapsed = time.perf_counter() - advance_started
            raw_generation_advance_seconds += advance_elapsed
            bookkeeping_started = time.perf_counter()
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            responses.append(response)
            caller_bookkeeping_seconds += time.perf_counter() - bookkeeping_started
        end_to_end_seconds = time.perf_counter() - started
        if not responses:
            raise MatrixError("direct MLX-LM produced no generation responses")
        post_decode_started = time.perf_counter()
        final = responses[-1]
        prompt_tps = float(getattr(final, "prompt_tps", 0.0) or 0.0)
        prefill_seconds = len(prompt_tokens) / prompt_tps if prompt_tps > 0 else 0.0
        output_token_ids = [int(response.token) for response in responses]
        text = "".join(str(response.text) for response in responses)
        post_decode_delivery_seconds = time.perf_counter() - post_decode_started
        component_payload = _mlx_lm_component_trace(
            decode_steps=len(output_token_ids),
            driver_seconds=max(0.0, raw_generation_advance_seconds - prefill_seconds),
            caller_bookkeeping_seconds=caller_bookkeeping_seconds,
            post_decode_delivery_seconds=post_decode_delivery_seconds,
        )
        component_payload["seconds"]["raw_generation_advance_seconds"] = (
            raw_generation_advance_seconds
        )
        payload = {
            "output_token_ids": output_token_ids,
            "text_sha256": public.sha256_text(text),
            "finish_reason": str(getattr(final, "finish_reason", "length") or "length"),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": max(0.0, end_to_end_seconds - prefill_seconds),
            "ttft_seconds": first_token_seconds if first_token_seconds is not None else 0.0,
            "end_to_end_seconds": end_to_end_seconds,
            "component_trace": component_payload,
        }
        return payload

    if lower_level_observer is not None:
        lower_level_steps: list[dict[str, float | int]] = []
        while True:
            lower_level_observer.begin_step()
            advance_started = time.perf_counter()
            try:
                response = next(generator)
            except StopIteration:
                break
            advance_elapsed = time.perf_counter() - advance_started
            if responses:
                lower_level_steps.append(lower_level_observer.finish_step(advance_elapsed))
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            responses.append(response)
        end_to_end_seconds = time.perf_counter() - started
        if not responses:
            raise MatrixError("direct MLX-LM produced no generation responses")
        final = responses[-1]
        prompt_tps = float(getattr(final, "prompt_tps", 0.0) or 0.0)
        prefill_seconds = len(prompt_tokens) / prompt_tps if prompt_tps > 0 else 0.0
        output_token_ids = [int(response.token) for response in responses]
        text = "".join(str(response.text) for response in responses)
        return {
            "output_token_ids": output_token_ids,
            "text_sha256": public.sha256_text(text),
            "finish_reason": str(getattr(final, "finish_reason", "length") or "length"),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": max(0.0, end_to_end_seconds - prefill_seconds),
            "ttft_seconds": first_token_seconds if first_token_seconds is not None else 0.0,
            "end_to_end_seconds": end_to_end_seconds,
            "lower_level_decode_trace": _lower_level_decode_trace(
                engine="mlx-lm",
                output_token_count=len(output_token_ids),
                steps=lower_level_steps,
            ),
        }

    for response in generator:
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - started
        responses.append(response)
    end_to_end_seconds = time.perf_counter() - started
    if not responses:
        raise MatrixError("direct MLX-LM produced no generation responses")
    final = responses[-1]
    prompt_tps = float(getattr(final, "prompt_tps", 0.0) or 0.0)
    prefill_seconds = len(prompt_tokens) / prompt_tps if prompt_tps > 0 else 0.0
    text = "".join(str(response.text) for response in responses)
    return {
        "output_token_ids": [int(response.token) for response in responses],
        "text_sha256": public.sha256_text(text),
        "finish_reason": str(getattr(final, "finish_reason", "length") or "length"),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": max(0.0, end_to_end_seconds - prefill_seconds),
        "ttft_seconds": first_token_seconds if first_token_seconds is not None else 0.0,
        "end_to_end_seconds": end_to_end_seconds,
    }


class MemorySampler:
    """Sample process RSS during one timed generation without retaining output text."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise MatrixError("memory sample interval must be positive")
        self._interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_before_bytes = 0
        self.swap_before_bytes = 0
        self._peak_rss_bytes = 0

    def _sample(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(self._process.memory_info().rss))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def start(self) -> None:
        self.rss_before_bytes = int(self._process.memory_info().rss)
        self.swap_before_bytes = int(psutil.swap_memory().used)
        self._peak_rss_bytes = self.rss_before_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, int]:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds * 2 + 1.0)
        rss_after_bytes = int(self._process.memory_info().rss)
        swap_after_bytes = int(psutil.swap_memory().used)
        return {
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": rss_after_bytes,
            "peak_rss_bytes": max(self._peak_rss_bytes, rss_after_bytes),
            "swap_before_bytes": self.swap_before_bytes,
            "swap_after_bytes": swap_after_bytes,
            "swap_delta_bytes": max(0, swap_after_bytes - self.swap_before_bytes),
        }


def _fingerprint_files(model_path: Path, names: tuple[str, ...]) -> list[Path]:
    files = [model_path / name for name in names if (model_path / name).is_file()]
    return sorted(files, key=lambda path: path.name)


def _hash_file_group(model_path: Path, paths: list[Path]) -> str:
    if not paths:
        raise MatrixError(f"model fingerprint has no files under {model_path}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(model_path).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_path: Path) -> dict[str, str]:
    model_files = sorted(model_path.glob("model*.safetensors"), key=lambda path: path.name)
    model_files.extend(
        _fingerprint_files(model_path, ("model.safetensors.index.json", "config.json"))
    )
    tokenizer_files = _fingerprint_files(
        model_path,
        (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "chat_template.jinja",
        ),
    )
    return {
        "model_sha256": _hash_file_group(model_path, model_files),
        "tokenizer_sha256": _hash_file_group(model_path, tokenizer_files),
    }


def _load_fingerprint(path: Path | None, model_path: Path) -> dict[str, str]:
    if path is None:
        return model_fingerprint(model_path)
    payload = _read_json(path)
    if payload.get("model_path") != str(model_path):
        raise MatrixError("model fingerprint file points at a different model")
    fingerprint = payload.get("model_fingerprint")
    if not isinstance(fingerprint, dict):
        raise MatrixError("model fingerprint file has no fingerprint")
    model_sha256 = fingerprint.get("model_sha256")
    tokenizer_sha256 = fingerprint.get("tokenizer_sha256")
    if not isinstance(model_sha256, str) or not isinstance(tokenizer_sha256, str):
        raise MatrixError("model fingerprint file is incomplete")
    return {"model_sha256": model_sha256, "tokenizer_sha256": tokenizer_sha256}


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _engine_version(engine: str) -> str:
    if engine == "aster":
        return f"aster={_distribution_version('aster')};mlx={_distribution_version('mlx')}"
    return f"mlx-lm={_distribution_version('mlx-lm')};mlx={_distribution_version('mlx')}"


def execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = {
        "generation_mode": "greedy",
        "input_mode": "pinned-public-source-rendered-token-ids",
        "input_truncation_policy": TRUNCATION_POLICY,
        "max_input_tokens": args.max_input_tokens,
        "prefill_step_tokens": args.prefill_step,
        "warmup": {
            "per_isolated_process": 1,
            "max_tokens": args.warmup_tokens,
        },
        "process_isolation": "fresh-process-per-engine-task-shard",
        "record_order": "workload-manifest-order-within-shard",
        "decode_cache_counter_scope": "per-public-request",
        "memory_peak": "psutil-rss-sampled",
        "memory_sample_interval_seconds": args.memory_sample_interval,
    }
    component_trace = _component_trace_config(args)
    if component_trace is not None:
        contract["component_trace"] = component_trace
    lower_level_trace = _lower_level_decode_trace_config(args)
    if lower_level_trace is not None:
        contract["lower_level_decode_trace"] = lower_level_trace
    return contract


def _metric_record(
    record: dict[str, Any],
    prompt_tokens: list[int],
    original_token_count: int,
    truncated: bool,
    result: dict[str, Any],
    memory: dict[str, int],
    include_output_token_ids: bool,
) -> dict[str, Any]:
    output_tokens = result["output_token_ids"]
    prefill_seconds = float(result["prefill_seconds"])
    decode_seconds = float(result["decode_seconds"])
    payload = {
        "workload_id": record["workload_id"],
        "prompt_sha256": record["prompt"]["sha256"],
        "prompt_token_ids_sha256": _json_hash(prompt_tokens),
        "prompt_token_count": len(prompt_tokens),
        "input_truncation": {
            "applied": truncated,
            "original_token_count": original_token_count,
            "effective_token_count": len(prompt_tokens),
        },
        "output_token_ids_sha256": _json_hash(output_tokens),
        "output_token_count": len(output_tokens),
        "text_sha256": result["text_sha256"],
        "finish_reason": result["finish_reason"],
        "metrics": {
            "ttft_seconds": float(result["ttft_seconds"]),
            "end_to_end_seconds": float(result["end_to_end_seconds"]),
            "prefill_tokens_per_second": (
                len(prompt_tokens) / prefill_seconds if prefill_seconds > 0 else 0.0
            ),
            "decode_tokens_per_second": (
                len(output_tokens) / decode_seconds if decode_seconds > 0 else 0.0
            ),
            "peak_rss_bytes": memory["peak_rss_bytes"],
            "swap_delta_bytes": memory["swap_delta_bytes"],
        },
        "resources": memory,
    }
    if include_output_token_ids:
        payload["debug_output_token_ids"] = output_tokens
    if "component_trace" in result:
        payload["component_trace"] = result["component_trace"]
    if "lower_level_decode_trace" in result:
        payload["lower_level_decode_trace"] = result["lower_level_decode_trace"]
    return payload


def _resolve_selected_records(workload: dict[str, Any], shard: str) -> list[dict[str, Any]]:
    try:
        records = workload_shards(workload)[shard]
    except KeyError as error:
        raise MatrixError(f"workload has no shard named {shard}") from error
    if not records:
        raise MatrixError(f"workload shard {shard} is empty")
    return records


def _adapter_source_hashes(engine: str) -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "public_benchmark.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
    ]
    if engine == "mlx-lm":
        import mlx_lm

        source_path = Path(getattr(mlx_lm, "__file__", ""))
        if source_path.is_file():
            paths.append(source_path)
    return {
        str(
            path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        ): _sha256_file(path)
        for path in paths
    }


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    state_trace_config = _state_trace_config(args)
    component_trace_config = _component_trace_config(args)
    lower_level_trace_config = _lower_level_decode_trace_config(args)
    process_started_utc = _now()
    state_before_model_load = _host_state_snapshot() if state_trace_config else None
    workload_path = args.workload.resolve()
    workload = _load_workload(workload_path)
    records = _resolve_selected_records(workload, args.shard)
    lock_path = args.lock.resolve()
    if workload.get("lock_sha256") != _sha256_file(lock_path):
        raise MatrixError("workload source lock hash differs from the active source lock")
    lock = public.load_lock(lock_path)
    resolver = public.PublicWorkloadResolver(lock, args.data_root.resolve())
    fingerprint = _load_fingerprint(args.model_fingerprint, args.model)
    contract = execution_contract(args)
    max_output_tokens = max(int(record["max_tokens"]) for record in records)
    results: list[dict[str, Any]] = []

    if args.engine == "aster":
        runner = ModelRunner(
            _aster_settings(
                args.config,
                args.model,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=max_output_tokens,
            )
        )
        runner.warmup()

        def encode(prompt: str) -> list[int]:
            return runner.encode_request(
                InferenceRequest(
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    enable_thinking=False,
                    trace_id="public-tokenize",
                )
            ).prompt_tokens

        def generate(tokens: list[int], max_tokens: int, trace_id: str) -> dict[str, Any]:
            return _aster_generate(
                runner,
                tokens,
                _request(max_tokens, trace_id),
                prefill_step=args.prefill_step,
                component_trace=component_trace_config is not None,
                lower_level_decode_trace=lower_level_trace_config is not None,
            )

    else:
        from mlx_lm import load

        model, tokenizer = load(str(args.model), lazy=False)

        def encode(prompt: str) -> list[int]:
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token or ""
            )
            return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))

        def generate(tokens: list[int], max_tokens: int, trace_id: str) -> dict[str, Any]:
            del trace_id
            return _mlx_lm_generate(
                model,
                tokenizer,
                tokens,
                max_tokens=max_tokens,
                prefill_step=args.prefill_step,
                component_trace=component_trace_config is not None,
                lower_level_decode_trace=lower_level_trace_config is not None,
            )

    first_record = records[0]
    warmup_prompt = resolver.resolve(first_record)
    warmup_tokens, _ = _trim_input_tokens(encode(warmup_prompt), args.max_input_tokens)
    generate(
        warmup_tokens,
        min(args.warmup_tokens, int(first_record["max_tokens"])),
        f"public-warmup-{args.engine}-{args.shard}",
    )

    state_before_timed_records = _host_state_snapshot() if state_trace_config else None

    for index, record in enumerate(records):
        prompt = resolver.resolve(record)
        raw_tokens = encode(prompt)
        prompt_tokens, truncated = _trim_input_tokens(raw_tokens, args.max_input_tokens)
        sampler = MemorySampler(args.memory_sample_interval)
        sampler.start()
        try:
            result = generate(
                prompt_tokens,
                int(record["max_tokens"]),
                f"public-{args.engine}-{args.shard}-{index}",
            )
        finally:
            memory = sampler.finish()
        results.append(
            _metric_record(
                record,
                prompt_tokens,
                len(raw_tokens),
                truncated,
                result,
                memory,
                getattr(args, "include_output_token_ids", False),
            )
        )

    payload = {
        "schema_version": 1,
        "kind": "public-engine-result-shard",
        "created_utc": _now(),
        "engine": args.engine,
        "engine_version": _engine_version(args.engine),
        "workload_sha256": _sha256_file(workload_path),
        "source_lock_sha256": _sha256_file(lock_path),
        "generation": workload["generation"],
        "execution": contract,
        "model_fingerprint": fingerprint,
        "shard": {
            "key": args.shard,
            "record_count": len(records),
            "workload_ids": [record["workload_id"] for record in records],
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "execution_order": {
            "mode": args.engine_order_mode,
            "position": args.engine_position,
        },
        "adapter_source_sha256": _adapter_source_hashes(args.engine),
        "records": results,
    }
    if state_trace_config:
        payload["state_trace"] = {
            **state_trace_config,
            "process": {
                "pid": os.getpid(),
                "started_utc": process_started_utc,
                "finished_utc": _now(),
            },
            "snapshots": {
                "before_model_load": state_before_model_load,
                "before_timed_records": state_before_timed_records,
                "after_timed_records": _host_state_snapshot(),
            },
        }
    return payload


def _validate_state_trace_payload(
    value: Any,
    expected: dict[str, Any] | None,
) -> None:
    if expected is None:
        if value is not None:
            raise MatrixError("unexpected state trace metadata in a non-traced matrix")
        return
    if not isinstance(value, dict):
        raise MatrixError("traced shard result has no state trace metadata")
    if value.get("schema_version") != expected["schema_version"]:
        raise MatrixError("state trace schema version differs")
    if value.get("timing_boundary") != expected["timing_boundary"]:
        raise MatrixError("state trace timing boundary differs")
    if value.get("block") != expected["block"]:
        raise MatrixError("state trace block metadata differs")
    process = value.get("process")
    if not isinstance(process, dict):
        raise MatrixError("state trace has no process metadata")
    if (
        isinstance(process.get("pid"), bool)
        or not isinstance(process.get("pid"), int)
        or process["pid"] < 1
        or not isinstance(process.get("started_utc"), str)
        or not isinstance(process.get("finished_utc"), str)
    ):
        raise MatrixError("state trace process metadata is invalid")
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, dict):
        raise MatrixError("state trace has no host/process snapshots")
    required_snapshots = (
        "before_model_load",
        "before_timed_records",
        "after_timed_records",
    )
    for name in required_snapshots:
        snapshot = snapshots.get(name)
        if not isinstance(snapshot, dict):
            raise MatrixError(f"state trace has no {name} snapshot")
        if not isinstance(snapshot.get("captured_utc"), str):
            raise MatrixError(f"state trace {name} has no capture timestamp")
        for field in (
            "available_memory_bytes",
            "system_swap_used_bytes",
            "process_rss_bytes",
            "process_vms_bytes",
        ):
            field_value = snapshot.get(field)
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
                raise MatrixError(f"state trace {name} has invalid {field}")


def _trace_seconds(value: Any, *, engine: str, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise MatrixError(f"{engine} component trace has invalid {field}")
    return float(value)


def _validate_component_trace_payload(
    value: Any,
    expected: dict[str, Any] | None,
    *,
    engine: str,
    output_token_count: int,
) -> None:
    if expected is None:
        if value is not None:
            raise MatrixError("unexpected component trace metadata in a non-traced matrix")
        return
    if not isinstance(value, dict):
        raise MatrixError(f"{engine} traced record has no component trace metadata")
    if value.get("schema_version") != expected["schema_version"]:
        raise MatrixError(f"{engine} component trace schema version differs")
    if value.get("timing_boundary") != expected["timing_boundary"]:
        raise MatrixError(f"{engine} component trace timing boundary differs")
    if value.get("mode") != expected["mode"]:
        raise MatrixError(f"{engine} component trace mode differs")
    expected_boundary = {
        "aster": "aster-manual-model-runner-single-decode-step",
        "mlx-lm": "mlx-lm-stream-generate-next",
    }[engine]
    if value.get("engine_boundary") != expected_boundary:
        raise MatrixError(f"{engine} component trace engine boundary differs")
    if value.get("cross_engine_comparable_boundary") != "decode_driver_seconds":
        raise MatrixError(f"{engine} component trace has no common decode driver boundary")
    decode = value.get("decode")
    if not isinstance(decode, dict):
        raise MatrixError(f"{engine} component trace has no decode metadata")
    expected_decode = {
        "steps": output_token_count,
        "batch_size_min": 1,
        "batch_size_max": 1,
        "batch_size_total_items": output_token_count,
    }
    if decode != expected_decode:
        raise MatrixError(f"{engine} component trace decode metadata differs")
    seconds = value.get("seconds")
    if not isinstance(seconds, dict):
        raise MatrixError(f"{engine} component trace has no timing values")
    for field in (
        "decode_driver_seconds",
        "caller_bookkeeping_seconds",
        "post_decode_delivery_seconds",
    ):
        _trace_seconds(seconds.get(field), engine=engine, field=field)
    if engine == "mlx-lm":
        raw_seconds = _trace_seconds(
            seconds.get("raw_generation_advance_seconds"),
            engine=engine,
            field="raw_generation_advance_seconds",
        )
        if raw_seconds + 1e-9 < _trace_seconds(
            seconds.get("decode_driver_seconds"),
            engine=engine,
            field="decode_driver_seconds",
        ):
            raise MatrixError("direct MLX-LM component trace decode exceeds raw generator time")
        if "cache" in value:
            raise MatrixError("direct MLX-LM component trace has Aster cache metadata")
        return
    component_seconds = sum(
        _trace_seconds(seconds.get(field), engine=engine, field=field)
        for field in ASTER_INTERNAL_COMPONENT_SECONDS
    )
    driver_seconds = _trace_seconds(
        seconds.get("decode_driver_seconds"),
        engine=engine,
        field="decode_driver_seconds",
    )
    if not math.isclose(component_seconds, driver_seconds, rel_tol=0.01, abs_tol=1e-6):
        raise MatrixError("Aster component trace does not reconcile to its decode driver")
    cache = value.get("cache")
    if not isinstance(cache, dict) or cache.get("decode_mode") != "single-request-no-batch-merge":
        raise MatrixError("Aster component trace has invalid cache mode")
    for field in (
        "batch_cache_reuses",
        "batch_cache_rebuilds",
        "single_steps",
        "cache_clear_attempts",
        "cache_clears",
        "cache_clear_failures",
    ):
        field_value = cache.get(field)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise MatrixError(f"Aster component trace has invalid cache {field}")
    if cache["single_steps"] != output_token_count:
        raise MatrixError("Aster component trace single-step count differs from output tokens")
    if cache["batch_cache_reuses"] != 0 or cache["batch_cache_rebuilds"] != 0:
        raise MatrixError("Aster single-request trace unexpectedly used batch cache merging")


def _validate_lower_level_decode_trace_payload(
    value: Any,
    expected: dict[str, Any] | None,
    *,
    engine: str,
    output_token_count: int,
) -> None:
    if expected is None:
        if value is not None:
            raise MatrixError("unexpected lower-level trace metadata in a non-traced matrix")
        return
    if not isinstance(value, dict):
        raise MatrixError(f"{engine} traced record has no lower-level trace metadata")
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise MatrixError(f"{engine} lower-level trace {field} differs")
    expected_boundary = {
        "aster": "aster-manual-single-decode-step",
        "mlx-lm": "mlx-lm-stream-generate-post-first-next",
    }[engine]
    if value.get("engine_boundary") != expected_boundary:
        raise MatrixError(f"{engine} lower-level trace engine boundary differs")
    if value.get("cross_engine_comparable_components") != [
        "outer_step_seconds",
        *LOWER_LEVEL_COMPONENT_SECONDS,
    ]:
        raise MatrixError(f"{engine} lower-level trace comparable components differ")
    decode = value.get("decode")
    if not isinstance(decode, dict):
        raise MatrixError(f"{engine} lower-level trace has no decode metadata")
    expected_steps = max(output_token_count - 1, 0)
    if decode != {
        "generated_output_steps": output_token_count,
        "traced_post_prefill_steps": expected_steps,
        "excluded_initial_output_steps": 1 if output_token_count else 0,
    }:
        raise MatrixError(f"{engine} lower-level trace decode metadata differs")
    seconds = value.get("seconds")
    if not isinstance(seconds, dict):
        raise MatrixError(f"{engine} lower-level trace has no timing values")
    outer_step_seconds = _trace_seconds(
        seconds.get("outer_step_seconds"), engine=engine, field="outer_step_seconds"
    )
    component_seconds = sum(
        _trace_seconds(seconds.get(field), engine=engine, field=field)
        for field in LOWER_LEVEL_COMPONENT_SECONDS
    )
    if not math.isclose(component_seconds, outer_step_seconds, rel_tol=0.01, abs_tol=1e-6):
        raise MatrixError(f"{engine} lower-level trace does not reconcile to its outer step")
    calls = value.get("calls")
    if not isinstance(calls, dict):
        raise MatrixError(f"{engine} lower-level trace has no submission counters")
    for field in ("model_submit_calls", "sampler_submit_calls"):
        count = calls.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count != expected_steps:
            raise MatrixError(f"{engine} lower-level trace has invalid {field}")


def _validate_shard_payload(
    payload: dict[str, Any],
    *,
    engine: str,
    expected_records: list[dict[str, Any]],
    workload_sha256: str,
    generation: dict[str, Any],
    contract: dict[str, Any],
    fingerprint: dict[str, str],
    source_lock_sha256: str,
    engine_order_mode: str,
    engine_position: int,
    state_trace: dict[str, Any] | None,
    component_trace: dict[str, Any] | None,
    lower_level_decode_trace: dict[str, Any] | None,
) -> None:
    if payload.get("kind") != "public-engine-result-shard":
        raise MatrixError("shard result has an unexpected kind")
    if payload.get("engine") != engine:
        raise MatrixError("shard result has an unexpected engine")
    if payload.get("workload_sha256") != workload_sha256:
        raise MatrixError("shard result workload hash differs")
    if payload.get("source_lock_sha256") != source_lock_sha256:
        raise MatrixError("shard result source lock hash differs")
    if payload.get("generation") != generation:
        raise MatrixError("shard result generation contract differs")
    if payload.get("execution") != contract:
        raise MatrixError("shard result execution contract differs")
    if payload.get("model_fingerprint") != fingerprint:
        raise MatrixError("shard result model fingerprint differs")
    expected_ids = [record["workload_id"] for record in expected_records]
    actual_rows = payload.get("records")
    if not isinstance(actual_rows, list):
        raise MatrixError("shard result has no records")
    actual_ids = [row.get("workload_id") for row in actual_rows if isinstance(row, dict)]
    if actual_ids != expected_ids:
        raise MatrixError("shard result record IDs differ from the workload shard")
    if payload.get("execution_order") != {
        "mode": engine_order_mode,
        "position": engine_position,
    }:
        raise MatrixError("shard result execution order metadata differs")
    _validate_state_trace_payload(payload.get("state_trace"), state_trace)
    for row in actual_rows:
        if not isinstance(row, dict):
            raise MatrixError("shard result record is not an object")
        output_token_count = row.get("output_token_count")
        if isinstance(output_token_count, bool) or not isinstance(output_token_count, int):
            raise MatrixError("shard result record has invalid output token count")
        _validate_component_trace_payload(
            row.get("component_trace"),
            component_trace,
            engine=engine,
            output_token_count=output_token_count,
        )
        _validate_lower_level_decode_trace_payload(
            row.get("lower_level_decode_trace"),
            lower_level_decode_trace,
            engine=engine,
            output_token_count=output_token_count,
        )


def _descriptor(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}


def _aggregate_engine_result(
    workload: dict[str, Any],
    workload_path: Path,
    engine: str,
    shard_payloads: list[dict[str, Any]],
    shard_paths: list[Path],
    run_root: Path,
) -> dict[str, Any]:
    if not shard_payloads:
        raise MatrixError(f"no shard payloads available for {engine}")
    engine_versions = {str(payload.get("engine_version")) for payload in shard_payloads}
    fingerprints = {_json_hash(payload.get("model_fingerprint")) for payload in shard_payloads}
    contracts = {_json_hash(payload.get("execution")) for payload in shard_payloads}
    lock_hashes = {str(payload.get("source_lock_sha256")) for payload in shard_payloads}
    if (
        len(engine_versions) != 1
        or len(fingerprints) != 1
        or len(contracts) != 1
        or len(lock_hashes) != 1
    ):
        raise MatrixError(f"{engine} shard metadata is inconsistent")
    records_by_id: dict[str, dict[str, Any]] = {}
    for payload in shard_payloads:
        for row in payload["records"]:
            record_id = row["workload_id"]
            if record_id in records_by_id:
                raise MatrixError(f"duplicate {engine} result row for {record_id}")
            records_by_id[record_id] = row
    ordered_records = []
    for record in workload["records"]:
        record_id = record["workload_id"]
        try:
            ordered_records.append(records_by_id.pop(record_id))
        except KeyError as error:
            raise MatrixError(f"{engine} result misses workload record {record_id}") from error
    if records_by_id:
        raise MatrixError(f"{engine} result has unexpected workload records")
    return {
        "schema_version": 1,
        "kind": "public-engine-result",
        "created_utc": _now(),
        "engine": engine,
        "engine_version": next(iter(engine_versions)),
        "workload_sha256": _sha256_file(workload_path),
        "source_lock_sha256": next(iter(lock_hashes)),
        "generation": workload["generation"],
        "execution": shard_payloads[0]["execution"],
        "model_fingerprint": shard_payloads[0]["model_fingerprint"],
        "shards": [
            {
                "key": payload["shard"]["key"],
                **_descriptor(path, run_root),
            }
            for payload, path in zip(shard_payloads, shard_paths, strict=True)
        ],
        "records": ordered_records,
    }


def _initial_manifest(
    workload_path: Path,
    workload: dict[str, Any],
    args: argparse.Namespace,
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    shards = workload_shards(workload)
    engine_order_mode = getattr(args, "engine_order_mode", "alternating")
    state_trace = _state_trace_config(args)
    component_trace = _component_trace_config(args)
    lower_level_decode_trace = _lower_level_decode_trace_config(args)
    manifest = {
        "schema_version": 1,
        "kind": "public-engine-matrix",
        "created_utc": _now(),
        "updated_utc": _now(),
        "status": "running",
        "workload": {
            "path": str(workload_path),
            "sha256": _sha256_file(workload_path),
            "profile": workload.get("profile"),
            "record_count": len(workload["records"]),
        },
        "model_path": str(args.model),
        "model_fingerprint": fingerprint,
        "execution": execution_contract(args),
        "engines": list(ENGINE_NAMES),
        "engine_order_mode": engine_order_mode,
        "shards": [
            {
                "key": shard,
                "record_count": len(records),
                "order": list(engine_order_for_shard(index, engine_order_mode)),
                "results": {},
                **({"state_traces": {}} if state_trace else {}),
            }
            for index, (shard, records) in enumerate(shards.items())
        ],
    }
    if state_trace:
        manifest["state_trace"] = state_trace
    if component_trace:
        manifest["component_trace"] = component_trace
    if lower_level_decode_trace:
        manifest["lower_level_decode_trace"] = lower_level_decode_trace
    return manifest


def _check_resume_manifest(
    manifest: dict[str, Any],
    workload_path: Path,
    args: argparse.Namespace,
    fingerprint: dict[str, str],
) -> None:
    expected_contract = execution_contract(args)
    if manifest.get("kind") != "public-engine-matrix":
        raise MatrixError("resume manifest has an unexpected kind")
    workload = manifest.get("workload")
    if not isinstance(workload, dict) or workload.get("sha256") != _sha256_file(workload_path):
        raise MatrixError("resume manifest targets a different workload")
    if manifest.get("model_path") != str(args.model):
        raise MatrixError("resume manifest targets a different model")
    if manifest.get("model_fingerprint") != fingerprint:
        raise MatrixError("resume manifest model fingerprint differs")
    if manifest.get("execution") != expected_contract:
        raise MatrixError("resume manifest execution contract differs")
    engine_order_mode = getattr(args, "engine_order_mode", "alternating")
    if manifest.get("engine_order_mode") != engine_order_mode:
        raise MatrixError("resume manifest engine order mode differs")
    if manifest.get("state_trace") != _state_trace_config(args):
        raise MatrixError("resume manifest state trace configuration differs")
    if manifest.get("component_trace") != _component_trace_config(args):
        raise MatrixError("resume manifest component trace configuration differs")
    if manifest.get("lower_level_decode_trace") != _lower_level_decode_trace_config(args):
        raise MatrixError("resume manifest lower-level decode trace configuration differs")
    workload = _load_workload(workload_path)
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise MatrixError("resume manifest has no shard schedule")
    entries_by_key = {
        entry.get("key"): entry for entry in entries if isinstance(entry, dict)
    }
    for index, shard in enumerate(workload_shards(workload)):
        entry = entries_by_key.get(shard)
        if not isinstance(entry, dict):
            raise MatrixError(f"resume manifest misses shard {shard}")
        if entry.get("order") != list(engine_order_for_shard(index, engine_order_mode)):
            raise MatrixError("resume manifest shard order differs")


def _run_child(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise MatrixError(
            "public engine shard failed: " + " ".join(command) + "\n" + completed.stderr.strip()
        )


def _child_command(
    args: argparse.Namespace,
    *,
    engine: str,
    shard: str,
    output: Path,
    fingerprint_path: Path,
    engine_position: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-shard",
        "--engine",
        engine,
        "--workload",
        str(args.workload),
        "--shard",
        shard,
        "--lock",
        str(args.lock),
        "--data-root",
        str(args.data_root),
        "--config",
        str(args.config),
        "--model",
        str(args.model),
        "--model-fingerprint",
        str(fingerprint_path),
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--prefill-step",
        str(args.prefill_step),
        "--memory-sample-interval",
        str(args.memory_sample_interval),
        "--engine-order-mode",
        args.engine_order_mode,
        "--engine-position",
        str(engine_position),
        "--output",
        str(output),
    ]
    state_trace = _state_trace_config(args)
    if state_trace:
        command.extend(
            [
                "--state-trace",
                "--state-trace-block-id",
                str(state_trace["block"]["id"]),
                "--state-trace-block-index",
                str(state_trace["block"]["index"]),
            ]
        )
    if _component_trace_config(args):
        command.append("--component-trace")
    if _lower_level_decode_trace_config(args):
        command.append("--lower-level-decode-trace")
    return command


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    workload_path = args.workload.resolve()
    workload = _load_workload(workload_path)
    run_root = args.run_dir.resolve()
    manifest_path = run_root / "matrix-manifest.json"
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise MatrixError(f"run directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint_path = run_root / "model-fingerprint.json"
    fingerprint = model_fingerprint(args.model)
    state_trace = _state_trace_config(args)
    component_trace = _component_trace_config(args)
    lower_level_decode_trace = _lower_level_decode_trace_config(args)
    if fingerprint_path.is_file():
        if _load_fingerprint(fingerprint_path, args.model) != fingerprint:
            raise MatrixError("cached model fingerprint differs from the current model files")
    else:
        public.write_json(
            fingerprint_path,
            {
                "schema_version": 1,
                "created_utc": _now(),
                "model_path": str(args.model),
                "model_fingerprint": fingerprint,
            },
        )
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        _check_resume_manifest(manifest, workload_path, args, fingerprint)
    else:
        manifest = _initial_manifest(workload_path, workload, args, fingerprint)
        public.write_json(manifest_path, manifest)

    shards = workload_shards(workload)
    workload_digest = _sha256_file(workload_path)
    source_lock_sha256 = _sha256_file(args.lock)
    contract = execution_contract(args)
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_shard = {entry["key"]: entry for entry in manifest["shards"]}
    paths_by_engine: dict[str, list[Path]] = {engine: [] for engine in ENGINE_NAMES}
    payloads_by_engine: dict[str, list[dict[str, Any]]] = {engine: [] for engine in ENGINE_NAMES}

    for shard_index, (shard, expected_records) in enumerate(shards.items()):
        entry = manifest_by_shard.get(shard)
        if not isinstance(entry, dict):
            raise MatrixError(f"resume manifest misses shard {shard}")
        order = engine_order_for_shard(shard_index, args.engine_order_mode)
        if entry.get("order") != list(order):
            raise MatrixError(f"matrix manifest order differs for shard {shard}")
        for engine_position, engine in enumerate(order):
            output_path = records_dir / f"{shard_index + 1:02d}-{shard}-{engine}.json"
            if output_path.is_file():
                if not args.resume:
                    raise MatrixError(f"existing shard output requires --resume: {output_path}")
                payload = _read_json(output_path)
            else:
                _run_child(
                    _child_command(
                        args,
                        engine=engine,
                        shard=shard,
                        output=output_path,
                        fingerprint_path=fingerprint_path,
                        engine_position=engine_position,
                    )
                )
                payload = _read_json(output_path)
            _validate_shard_payload(
                payload,
                engine=engine,
                expected_records=expected_records,
                workload_sha256=workload_digest,
                generation=workload["generation"],
                contract=contract,
                fingerprint=fingerprint,
                source_lock_sha256=source_lock_sha256,
                engine_order_mode=args.engine_order_mode,
                engine_position=engine_position,
                state_trace=state_trace,
                component_trace=component_trace,
                lower_level_decode_trace=lower_level_decode_trace,
            )
            paths_by_engine[engine].append(output_path)
            payloads_by_engine[engine].append(payload)
            entry["results"][engine] = _descriptor(output_path, run_root)
            if state_trace:
                state_traces = entry.setdefault("state_traces", {})
                state_traces[engine] = payload["state_trace"]
            manifest["updated_utc"] = _now()
            public.write_json(manifest_path, manifest)
            print(
                f"completed shard {shard_index + 1}/{len(shards)} {shard} {engine}",
                flush=True,
            )

    result_paths: list[Path] = []
    for engine in ENGINE_NAMES:
        aggregate = _aggregate_engine_result(
            workload,
            workload_path,
            engine,
            payloads_by_engine[engine],
            paths_by_engine[engine],
            run_root,
        )
        output_path = run_root / f"{engine}.json"
        public.write_json(output_path, aggregate)
        result_paths.append(output_path)
    validation = public.validate_engine_results(
        workload_path,
        result_paths,
        set(ENGINE_NAMES),
    )
    comparison = {
        "schema_version": 1,
        "kind": "public-engine-matrix-comparison",
        "created_utc": _now(),
        "matrix_manifest": {"path": str(manifest_path.relative_to(run_root))},
        "engine_order_mode": args.engine_order_mode,
        **({"state_trace": state_trace} if state_trace else {}),
        **({"component_trace": component_trace} if component_trace else {}),
        **({"lower_level_decode_trace": lower_level_decode_trace} if lower_level_decode_trace else {}),
        "shard_engine_orders": [
            {"key": entry["key"], "order": entry["order"]}
            for entry in manifest["shards"]
        ],
        "engine_results": {
            engine: _descriptor(run_root / f"{engine}.json", run_root) for engine in ENGINE_NAMES
        },
        "validation": validation,
    }
    comparison_path = run_root / "comparison.json"
    public.write_json(comparison_path, comparison)
    manifest["updated_utc"] = _now()
    manifest["status"] = validation["decision"]
    manifest["comparison"] = _descriptor(comparison_path, run_root)
    public.write_json(manifest_path, manifest)
    if validation["decision"] != "comparable":
        raise MatrixError("public engine matrix failed its complete comparison gate")
    return comparison


def _load_completed_matrix(run_dir: Path) -> dict[str, Any]:
    run_root = run_dir.resolve()
    manifest = _read_json(run_root / "matrix-manifest.json")
    if manifest.get("kind") != "public-engine-matrix":
        raise MatrixError(f"{run_root} does not contain a public engine matrix")
    if manifest.get("status") != "comparable":
        raise MatrixError(f"{run_root} matrix is not comparable")
    workload_metadata = manifest.get("workload")
    if not isinstance(workload_metadata, dict):
        raise MatrixError(f"{run_root} matrix has no workload metadata")
    workload_value = workload_metadata.get("path")
    if not isinstance(workload_value, str) or not workload_value:
        raise MatrixError(f"{run_root} matrix has no workload path")
    workload_path = Path(workload_value).resolve()
    workload = _load_workload(workload_path)
    workload_sha256 = _sha256_file(workload_path)
    if workload_metadata.get("sha256") != workload_sha256:
        raise MatrixError(f"{run_root} workload hash differs from its manifest")

    result_paths = [run_root / f"{engine}.json" for engine in ENGINE_NAMES]
    validation = public.validate_engine_results(workload_path, result_paths, set(ENGINE_NAMES))
    if validation["decision"] != "comparable" or not all(validation["gates"].values()):
        raise MatrixError(f"{run_root} matrix fails public comparability validation")
    results = {
        engine: _read_json(path)
        for engine, path in zip(ENGINE_NAMES, result_paths, strict=True)
    }
    source_locks = {result.get("source_lock_sha256") for result in results.values()}
    if len(source_locks) != 1 or not isinstance(next(iter(source_locks)), str):
        raise MatrixError(f"{run_root} matrix has inconsistent source lock hashes")
    execution = results[ENGINE_NAMES[0]].get("execution")
    if manifest.get("execution") != execution:
        raise MatrixError(f"{run_root} execution contract differs from its manifest")
    rows = {
        engine: {row["workload_id"]: row for row in result["records"]}
        for engine, result in results.items()
    }
    return {
        "root": run_root,
        "manifest": manifest,
        "workload_path": workload_path,
        "workload": workload,
        "workload_sha256": workload_sha256,
        "source_lock_sha256": next(iter(source_locks)),
        "execution": execution,
        "model_fingerprint": results[ENGINE_NAMES[0]]["model_fingerprint"],
        "results": results,
        "rows": rows,
        "validation": validation,
    }


def _first_engine_by_record(matrix: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    manifest = matrix["manifest"]
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise MatrixError("matrix manifest has no shard entries")
    entries_by_key = {
        entry.get("key"): entry for entry in entries if isinstance(entry, dict)
    }
    first_engine: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for shard_index, (shard, records) in enumerate(workload_shards(matrix["workload"]).items()):
        entry = entries_by_key.get(shard)
        if not isinstance(entry, dict):
            raise MatrixError(f"matrix manifest misses shard {shard}")
        order = entry.get("order")
        if not isinstance(order, list) or tuple(order) not in {
            ENGINE_NAMES,
            ENGINE_NAMES[::-1],
        }:
            raise MatrixError(f"matrix manifest has invalid order for shard {shard}")
        if entry.get("record_count") != len(records):
            raise MatrixError(f"matrix manifest has invalid record count for shard {shard}")
        for record in records:
            first_engine[record["workload_id"]] = order[0]
        summaries.append(
            {
                "index": shard_index,
                "key": shard,
                "record_count": len(records),
                "order": order,
            }
        )
    return first_engine, summaries


def _require_same_cross_matrix_contract(
    original: dict[str, Any], reversed_matrix: dict[str, Any]
) -> None:
    checks = {
        "workload": original["workload_sha256"] == reversed_matrix["workload_sha256"],
        "source lock": original["source_lock_sha256"] == reversed_matrix["source_lock_sha256"],
        "execution": original["execution"] == reversed_matrix["execution"],
        "model fingerprint": original["model_fingerprint"] == reversed_matrix["model_fingerprint"],
        "generation": original["workload"]["generation"]
        == reversed_matrix["workload"]["generation"],
    }
    mismatch = next((name for name, same in checks.items() if not same), None)
    if mismatch is not None:
        raise MatrixError(f"crossed matrices differ in {mismatch}")

    expected_ids = [record["workload_id"] for record in original["workload"]["records"]]
    if expected_ids != [record["workload_id"] for record in reversed_matrix["workload"]["records"]]:
        raise MatrixError("crossed matrices use different workload record order")
    parity_fields = (
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "prompt_token_count",
        "output_token_ids_sha256",
        "output_token_count",
        "text_sha256",
        "finish_reason",
    )
    for engine in ENGINE_NAMES:
        for workload_id in expected_ids:
            original_row = original["rows"][engine].get(workload_id)
            reversed_row = reversed_matrix["rows"][engine].get(workload_id)
            if original_row is None or reversed_row is None:
                raise MatrixError(f"crossed matrices miss {engine} result for {workload_id}")
            if any(original_row.get(field) != reversed_row.get(field) for field in parity_fields):
                raise MatrixError(
                    f"crossed matrices have deterministic input/output drift for {engine}:{workload_id}"
                )


def _input_length_bin(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MatrixError("matrix has an invalid prompt token count")
    for lower, upper, label in LENGTH_BINS:
        if value >= lower and (upper is None or value < upper):
            return label
    raise MatrixError("matrix prompt token count does not fit a length bin")


def _metric_value(row: dict[str, Any], metric: str, *, engine: str, workload_id: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise MatrixError(f"{engine}:{workload_id} has no metrics")
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise MatrixError(f"{engine}:{workload_id} has invalid {metric}")
    return float(value)


def _paired_percent(aster_value: float, mlx_lm_value: float, metric: str) -> float:
    if mlx_lm_value == 0:
        if aster_value == 0:
            return 0.0
        raise MatrixError(f"cannot compare {metric} against a zero direct-MLX-LM value")
    return (aster_value / mlx_lm_value - 1.0) * 100.0


def _bootstrap_interval(values: list[float], *, seed: str, samples: int) -> dict[str, float]:
    if not values:
        raise MatrixError("cannot bootstrap an empty paired effect")
    random_source = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    sample_size = len(values)
    medians = [
        statistics.median(values[random_source.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(samples)
    ]
    medians.sort()
    return {
        "low": public.percentile(medians, 0.025),
        "high": public.percentile(medians, 0.975),
    }


def _effect_class(interval: dict[str, float]) -> str:
    if interval["low"] > NO_OP_PERCENT:
        return "positive-material"
    if interval["high"] < -NO_OP_PERCENT:
        return "negative-material"
    return "inconclusive"


def _effect_summary(
    values: list[float],
    *,
    seed: str,
    samples: int,
    min_order_stratum_records: int,
) -> dict[str, Any]:
    if len(values) < min_order_stratum_records:
        return {
            "records": len(values),
            "paired_median_aster_vs_mlx_lm_percent": statistics.median(values),
            "bootstrap_95_percent": None,
            "effect_class": "insufficient",
        }
    interval = _bootstrap_interval(values, seed=seed, samples=samples)
    return {
        "records": len(values),
        "paired_median_aster_vs_mlx_lm_percent": statistics.median(values),
        "bootstrap_95_percent": interval,
        "effect_class": _effect_class(interval),
    }


def _order_agreement(aster_first: dict[str, Any], mlx_lm_first: dict[str, Any]) -> str:
    first_class = aster_first["effect_class"]
    second_class = mlx_lm_first["effect_class"]
    if "insufficient" in {first_class, second_class}:
        return "insufficient"
    if first_class == second_class and first_class != "inconclusive":
        return "material-agreement"
    if {
        first_class,
        second_class,
    } == {"positive-material", "negative-material"}:
        return "directional-disagreement"
    return "inconclusive"


def _swap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aster_values = [row["swap"]["aster"] for row in rows]
    mlx_lm_values = [row["swap"]["mlx-lm"] for row in rows]
    return {
        "records": len(rows),
        "max_aster_swap_delta_bytes": max(aster_values),
        "max_mlx_lm_swap_delta_bytes": max(mlx_lm_values),
        "any_swap_growth": any(value > 0 for value in [*aster_values, *mlx_lm_values]),
    }


def _group_summary(
    group_id: str,
    metadata: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    min_order_stratum_records: int,
) -> dict[str, Any]:
    strata = {
        engine: [row for row in rows if row["first_engine"] == engine]
        for engine in ENGINE_NAMES
    }
    if not all(strata.values()):
        raise MatrixError(f"order strata are incomplete for group {group_id}")
    metrics: dict[str, dict[str, Any]] = {}
    for metric in PAIRED_METRICS:
        summaries = {
            "aster_first": _effect_summary(
                [row["effects"][metric] for row in strata["aster"]],
                seed=f"{group_id}:{metric}:aster-first",
                samples=bootstrap_samples,
                min_order_stratum_records=min_order_stratum_records,
            ),
            "mlx_lm_first": _effect_summary(
                [row["effects"][metric] for row in strata["mlx-lm"]],
                seed=f"{group_id}:{metric}:mlx-lm-first",
                samples=bootstrap_samples,
                min_order_stratum_records=min_order_stratum_records,
            ),
        }
        summaries["order_agreement"] = _order_agreement(
            summaries["aster_first"], summaries["mlx_lm_first"]
        )
        metrics[metric] = summaries
    return {
        "id": group_id,
        **metadata,
        "records": len(rows),
        "metrics": metrics,
        "swap": {
            "aster_first": _swap_summary(strata["aster"]),
            "mlx_lm_first": _swap_summary(strata["mlx-lm"]),
        },
    }


def compare_matrices(
    original_run_dir: Path,
    reversed_run_dir: Path,
    *,
    bootstrap_samples: int = 2000,
    min_order_stratum_records: int = MIN_ORDER_STRATUM_RECORDS,
) -> dict[str, Any]:
    """Join I066 and I067 matrices without selecting a runtime candidate.

    Every public record must appear once with Aster first and once with direct
    MLX-LM first. The result reports raw Aster/direct effect signs by order
    stratum, then leaves component attribution to a later measured iteration.
    """

    if bootstrap_samples < 100:
        raise MatrixError("bootstrap samples must be at least 100")
    if min_order_stratum_records < 1:
        raise MatrixError("minimum order-stratum records must be positive")
    original = _load_completed_matrix(original_run_dir)
    reversed_matrix = _load_completed_matrix(reversed_run_dir)
    _require_same_cross_matrix_contract(original, reversed_matrix)
    original_first, original_shards = _first_engine_by_record(original)
    reversed_first, reversed_shards = _first_engine_by_record(reversed_matrix)
    reversed_by_key = {entry["key"]: entry for entry in reversed_shards}
    order_summary: list[dict[str, Any]] = []
    for original_entry in original_shards:
        reversed_entry = reversed_by_key.get(original_entry["key"])
        if reversed_entry is None or original_entry["order"] != list(
            reversed(reversed_entry["order"])
        ):
            raise MatrixError("crossed matrices do not use opposite engine order per shard")
        order_summary.append(
            {
                "key": original_entry["key"],
                "record_count": original_entry["record_count"],
                "original_order": original_entry["order"],
                "reversed_order": reversed_entry["order"],
            }
        )

    rows: list[dict[str, Any]] = []
    for matrix_name, matrix, first_by_record in (
        ("original", original, original_first),
        ("reversed", reversed_matrix, reversed_first),
    ):
        for record in matrix["workload"]["records"]:
            workload_id = record["workload_id"]
            aster_row = matrix["rows"]["aster"][workload_id]
            mlx_lm_row = matrix["rows"]["mlx-lm"][workload_id]
            effects = {
                metric: _paired_percent(
                    _metric_value(aster_row, metric, engine="aster", workload_id=workload_id),
                    _metric_value(mlx_lm_row, metric, engine="mlx-lm", workload_id=workload_id),
                    metric,
                )
                for metric in PAIRED_METRICS
            }
            rows.append(
                {
                    "workload_id": workload_id,
                    "matrix": matrix_name,
                    "first_engine": first_by_record[workload_id],
                    "workload": _workload_shard_key(record),
                    "input_length_bin": _input_length_bin(aster_row["prompt_token_count"]),
                    "effects": effects,
                    "swap": {
                        "aster": _metric_value(
                            aster_row,
                            "swap_delta_bytes",
                            engine="aster",
                            workload_id=workload_id,
                        ),
                        "mlx-lm": _metric_value(
                            mlx_lm_row,
                            "swap_delta_bytes",
                            engine="mlx-lm",
                            workload_id=workload_id,
                        ),
                    },
                }
            )

    expected_count = len(original["workload"]["records"])
    aster_first_count = sum(row["first_engine"] == "aster" for row in rows)
    mlx_lm_first_count = sum(row["first_engine"] == "mlx-lm" for row in rows)
    if aster_first_count != expected_count or mlx_lm_first_count != expected_count:
        raise MatrixError("crossed matrices do not balance first-engine record counts")

    group_rows: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    group_metadata: dict[str, dict[str, str]] = {
        "overall": {"dimension": "overall", "workload": "all", "input_length_bin": "all"}
    }
    for row in rows:
        workload_group = f"workload:{row['workload']}"
        length_group = f"input-length:{row['input_length_bin']}"
        joint_group = f"workload-input-length:{row['workload']}:{row['input_length_bin']}"
        for group_id, metadata in (
            (
                workload_group,
                {
                    "dimension": "workload",
                    "workload": row["workload"],
                    "input_length_bin": "all",
                },
            ),
            (
                length_group,
                {
                    "dimension": "input-length",
                    "workload": "all",
                    "input_length_bin": row["input_length_bin"],
                },
            ),
            (
                joint_group,
                {
                    "dimension": "workload-input-length",
                    "workload": row["workload"],
                    "input_length_bin": row["input_length_bin"],
                },
            ),
        ):
            group_rows.setdefault(group_id, []).append(row)
            group_metadata.setdefault(group_id, metadata)
    groups = [
        _group_summary(
            group_id,
            group_metadata[group_id],
            group_rows[group_id],
            bootstrap_samples=bootstrap_samples,
            min_order_stratum_records=min_order_stratum_records,
        )
        for group_id in ["overall", *sorted(key for key in group_rows if key != "overall")]
    ]
    agreements = [
        metric["order_agreement"]
        for group in groups
        for metric in group["metrics"].values()
    ]
    any_swap_growth = any(
        summary["any_swap_growth"]
        for group in groups
        for summary in group["swap"].values()
    )
    if any_swap_growth:
        decision = "reject-swap-growth"
    elif "directional-disagreement" in agreements:
        decision = "reject-directional-disagreement"
    elif "material-agreement" in agreements:
        decision = "order-confirmed-effects-require-component-attribution"
    else:
        decision = "no-material-order-confirmed-effect"
    return {
        "schema_version": 1,
        "kind": "public-engine-crossed-order-comparison",
        "created_utc": _now(),
        "original_matrix": {
            "run_dir": str(original["root"]),
            "engine_order_mode": original["manifest"].get("engine_order_mode", "legacy"),
        },
        "reversed_matrix": {
            "run_dir": str(reversed_matrix["root"]),
            "engine_order_mode": reversed_matrix["manifest"].get("engine_order_mode", "legacy"),
        },
        "source": {
            "workload_sha256": original["workload_sha256"],
            "source_lock_sha256": original["source_lock_sha256"],
            "model_fingerprint": original["model_fingerprint"],
            "execution": original["execution"],
        },
        "gates": {
            "both_matrices_comparable": True,
            "same_workload": True,
            "same_source_lock": True,
            "same_model_and_tokenizer": True,
            "same_execution_contract": True,
            "deterministic_cross_matrix_parity": True,
            "opposite_engine_order_per_shard": True,
            "balanced_first_engine_records": True,
            "zero_swap_growth": not any_swap_growth,
        },
        "shards": order_summary,
        "order_balance": {
            "records_per_matrix": expected_count,
            "aster_first_records": aster_first_count,
            "mlx_lm_first_records": mlx_lm_first_count,
        },
        "bootstrap_samples": bootstrap_samples,
        "minimum_order_stratum_records": min_order_stratum_records,
        "groups": groups,
        "decision": decision,
        "production_candidate": "none",
    }


def _lower_level_trace_execution_without_observer(
    execution: dict[str, Any],
    *,
    traced: bool,
) -> dict[str, Any]:
    """Validate I070's trace contract and return the shared execution contract."""

    if not isinstance(execution, dict):
        raise MatrixError("trace no-op matrix has no execution contract")
    normalized = dict(execution)
    observer = normalized.pop("lower_level_decode_trace", None)
    if traced:
        if observer != _lower_level_decode_trace_metadata():
            raise MatrixError("traced matrix lacks the lower-level decode trace contract")
    elif observer is not None:
        raise MatrixError("untraced matrix unexpectedly enables lower-level decode tracing")
    return normalized


def _require_lower_level_trace_rows(matrix: dict[str, Any]) -> None:
    expected_metadata = _lower_level_decode_trace_metadata()
    for engine in ENGINE_NAMES:
        for workload_id, row in matrix["rows"][engine].items():
            trace = row.get("lower_level_decode_trace")
            if not isinstance(trace, dict):
                raise MatrixError(f"{engine}:{workload_id} has no lower-level decode trace")
            for field, expected in expected_metadata.items():
                if trace.get(field) != expected:
                    raise MatrixError(
                        f"{engine}:{workload_id} has an incompatible lower-level trace {field}"
                    )


def _trace_no_op_percent(traced_value: float, baseline_value: float, metric: str) -> float:
    if baseline_value <= 0:
        raise MatrixError(f"trace no-op baseline has a non-positive {metric}")
    return (traced_value / baseline_value - 1.0) * 100.0


def _compare_lower_level_trace_noop_views(
    untraced: dict[str, Any],
    traced: dict[str, Any],
) -> dict[str, Any]:
    """Gate I070 observer overhead on an identical locked public workload.

    This is a trace-integrity screen, not a performance comparison between
    engines. It keeps the engine-local traced/untraced deltas separate and
    requires exact deterministic output before admitting the observer for the
    QMSUM ABBA experiment.
    """

    if untraced["workload_sha256"] != traced["workload_sha256"]:
        raise MatrixError("trace no-op matrices use different workloads")
    if untraced["source_lock_sha256"] != traced["source_lock_sha256"]:
        raise MatrixError("trace no-op matrices use different source locks")
    if untraced["model_fingerprint"] != traced["model_fingerprint"]:
        raise MatrixError("trace no-op matrices use different model fingerprints")
    if untraced["generation"] != traced["generation"]:
        raise MatrixError("trace no-op matrices use different generation settings")
    if _lower_level_trace_execution_without_observer(
        untraced["execution"], traced=False
    ) != _lower_level_trace_execution_without_observer(traced["execution"], traced=True):
        raise MatrixError("trace no-op matrices differ outside the observer contract")
    _require_lower_level_trace_rows(traced)
    untraced_adapter_sources = untraced.get("adapter_source_fingerprints")
    traced_adapter_sources = traced.get("adapter_source_fingerprints")
    same_adapter_source = (
        untraced_adapter_sources is None and traced_adapter_sources is None
    ) or (
        isinstance(untraced_adapter_sources, dict)
        and isinstance(traced_adapter_sources, dict)
        and set(untraced_adapter_sources) == set(ENGINE_NAMES)
        and set(traced_adapter_sources) == set(ENGINE_NAMES)
        and all(
            isinstance(untraced_adapter_sources[engine], dict)
            and untraced_adapter_sources[engine]
            and untraced_adapter_sources[engine] == traced_adapter_sources[engine]
            for engine in ENGINE_NAMES
        )
    )

    parity_drift_counts = {field: 0 for field in TRACE_NO_OP_PARITY_FIELDS}
    engine_summaries: dict[str, Any] = {}
    any_swap_growth = False
    all_metrics_within_no_op_band = True
    for engine in ENGINE_NAMES:
        baseline_rows = untraced["rows"][engine]
        traced_rows = traced["rows"][engine]
        if set(baseline_rows) != set(traced_rows):
            raise MatrixError(f"trace no-op {engine} records differ between matrices")
        effects_by_metric = {metric: [] for metric in TRACE_NO_OP_METRICS}
        swap_samples = {"untraced": [], "traced": []}
        for workload_id in sorted(baseline_rows):
            baseline_row = baseline_rows[workload_id]
            traced_row = traced_rows[workload_id]
            for field in TRACE_NO_OP_PARITY_FIELDS:
                if baseline_row.get(field) != traced_row.get(field):
                    parity_drift_counts[field] += 1
            for metric in TRACE_NO_OP_METRICS:
                effects_by_metric[metric].append(
                    _trace_no_op_percent(
                        _metric_value(traced_row, metric, engine=engine, workload_id=workload_id),
                        _metric_value(
                            baseline_row,
                            metric,
                            engine=engine,
                            workload_id=workload_id,
                        ),
                        metric,
                    )
                )
            for label, row in (("untraced", baseline_row), ("traced", traced_row)):
                swap_delta = _metric_value(
                    row,
                    "swap_delta_bytes",
                    engine=engine,
                    workload_id=workload_id,
                )
                swap_samples[label].append(swap_delta)
                any_swap_growth = any_swap_growth or swap_delta != 0
        metric_summaries: dict[str, Any] = {}
        for metric, effects in effects_by_metric.items():
            median_effect = statistics.median(effects)
            within_no_op_band = abs(median_effect) <= NO_OP_PERCENT
            all_metrics_within_no_op_band = (
                all_metrics_within_no_op_band and within_no_op_band
            )
            metric_summaries[metric] = {
                "records": len(effects),
                "median_traced_vs_untraced_percent": median_effect,
                "minimum_traced_vs_untraced_percent": min(effects),
                "maximum_traced_vs_untraced_percent": max(effects),
                "within_3_percent_no_op_band": within_no_op_band,
            }
        engine_summaries[engine] = {
            "records": len(baseline_rows),
            "metrics": metric_summaries,
            "swap_delta_bytes": {
                "untraced_nonzero_records": sum(value != 0 for value in swap_samples["untraced"]),
                "traced_nonzero_records": sum(value != 0 for value in swap_samples["traced"]),
            },
        }

    deterministic_parity = not any(parity_drift_counts.values())
    if not same_adapter_source:
        decision = "trace-no-op-rejected-adapter-source-mismatch"
    elif not deterministic_parity:
        decision = "trace-no-op-rejected-output-drift"
    elif any_swap_growth:
        decision = "trace-no-op-rejected-swap-growth"
    elif not all_metrics_within_no_op_band:
        decision = "trace-no-op-rejected-metric-movement"
    else:
        decision = "trace-no-op-admitted"
    return {
        "schema_version": 1,
        "kind": "public-engine-lower-level-trace-noop-comparison",
        "created_utc": _now(),
        "untraced_matrix": {"run_dir": str(untraced["root"])},
        "traced_matrix": {"run_dir": str(traced["root"])},
        "source": {
            "workload_sha256": untraced["workload_sha256"],
            "source_lock_sha256": untraced["source_lock_sha256"],
            "model_fingerprint": untraced["model_fingerprint"],
            "execution": untraced["execution"],
            "adapter_source_sha256": {
                "untraced": untraced_adapter_sources,
                "traced": traced_adapter_sources,
            },
        },
        "measurement": {
            "trace_contract": _lower_level_decode_trace_metadata(),
            "scope": "engine-local public trace no-op screen",
            "metric_gate": "absolute median traced-versus-untraced movement <= 3 percent",
        },
        "gates": {
            "both_matrices_comparable": True,
            "same_workload": True,
            "same_source_lock": True,
            "same_model_and_tokenizer": True,
            "same_generation_settings": True,
            "same_execution_except_trace": True,
            "same_adapter_source": same_adapter_source,
            "complete_lower_level_trace": True,
            "deterministic_token_text_finish_parity": deterministic_parity,
            "zero_swap_growth": not any_swap_growth,
            "no_material_metric_movement": all_metrics_within_no_op_band,
        },
        "parity_drift_counts": parity_drift_counts,
        "engines": engine_summaries,
        "decision": decision,
        "production_candidate": "none",
    }


def _lower_level_trace_noop_matrix_view(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": matrix["root"],
        "workload_sha256": matrix["workload_sha256"],
        "source_lock_sha256": matrix["source_lock_sha256"],
        "model_fingerprint": matrix["model_fingerprint"],
        "generation": matrix["workload"].get("generation"),
        "execution": matrix["execution"],
        "rows": matrix["rows"],
    }


def compare_lower_level_trace_noop(
    untraced_run_dir: Path,
    traced_run_dir: Path,
) -> dict[str, Any]:
    """Gate I070 observer overhead on two complete public engine matrices."""

    return _compare_lower_level_trace_noop_views(
        _lower_level_trace_noop_matrix_view(_load_completed_matrix(untraced_run_dir)),
        _lower_level_trace_noop_matrix_view(_load_completed_matrix(traced_run_dir)),
    )


def _load_lower_level_trace_noop_shard_view(
    workload_path: Path,
    shard: str,
    result_paths: dict[str, Path],
) -> dict[str, Any]:
    """Load one complete two-engine public shard without rerunning valid work."""

    if set(result_paths) != set(ENGINE_NAMES):
        raise MatrixError("trace no-op shard requires exactly one result per engine")
    workload = _load_workload(workload_path.resolve())
    expected_records = _resolve_selected_records(workload, shard)
    expected_workload_ids = {record["workload_id"] for record in expected_records}
    expected_workload_sha256 = _sha256_file(workload_path.resolve())
    expected_lock = workload.get("lock_sha256")
    expected_generation = workload.get("generation")
    if not isinstance(expected_lock, str) or not isinstance(expected_generation, dict):
        raise MatrixError("trace no-op workload has no source lock or generation settings")

    payloads: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in ENGINE_NAMES:
        payload = _read_json(result_paths[engine].resolve())
        if payload.get("kind") != "public-engine-result-shard":
            raise MatrixError(f"trace no-op {engine} input is not a shard result")
        if payload.get("engine") != engine:
            raise MatrixError(f"trace no-op shard engine does not match {engine}")
        payload_shard = payload.get("shard")
        payload_shard_key = (
            payload_shard.get("key") if isinstance(payload_shard, dict) else payload_shard
        )
        if payload_shard_key != shard:
            raise MatrixError(f"trace no-op {engine} result uses a different shard")
        if payload.get("workload_sha256") != expected_workload_sha256:
            raise MatrixError(f"trace no-op {engine} result uses a different workload")
        if payload.get("source_lock_sha256") != expected_lock:
            raise MatrixError(f"trace no-op {engine} result uses a different source lock")
        if payload.get("generation") != expected_generation:
            raise MatrixError(f"trace no-op {engine} result uses different generation settings")
        adapter_source = payload.get("adapter_source_sha256")
        if not isinstance(adapter_source, dict) or not adapter_source:
            raise MatrixError(f"trace no-op {engine} result has no adapter source fingerprint")
        records = payload.get("records")
        if not isinstance(records, list):
            raise MatrixError(f"trace no-op {engine} result has no records")
        engine_rows: dict[str, dict[str, Any]] = {}
        for row in records:
            if not isinstance(row, dict):
                raise MatrixError(f"trace no-op {engine} result has an invalid record")
            workload_id = row.get("workload_id")
            if not isinstance(workload_id, str) or workload_id in engine_rows:
                raise MatrixError(f"trace no-op {engine} result has duplicate workload IDs")
            engine_rows[workload_id] = row
        if set(engine_rows) != expected_workload_ids:
            raise MatrixError(f"trace no-op {engine} result has incomplete shard coverage")
        if isinstance(payload_shard, dict) and payload_shard.get("record_count") != len(engine_rows):
            raise MatrixError(f"trace no-op {engine} result has an invalid shard record count")
        payloads[engine] = payload
        rows[engine] = engine_rows

    reference = payloads[ENGINE_NAMES[0]]
    for engine in ENGINE_NAMES[1:]:
        payload = payloads[engine]
        for field in ("model_fingerprint", "execution"):
            if payload.get(field) != reference.get(field):
                raise MatrixError(f"trace no-op shard engines differ in {field}")
    for workload_id in expected_workload_ids:
        first = rows[ENGINE_NAMES[0]][workload_id]
        second = rows[ENGINE_NAMES[1]][workload_id]
        for field in TRACE_NO_OP_PARITY_FIELDS:
            if first.get(field) != second.get(field):
                raise MatrixError(
                    f"trace no-op shard engines differ in deterministic {field}"
                )
    return {
        "root": result_paths[ENGINE_NAMES[0]].resolve().parent,
        "workload_sha256": expected_workload_sha256,
        "source_lock_sha256": expected_lock,
        "model_fingerprint": reference.get("model_fingerprint"),
        "generation": expected_generation,
        "execution": reference.get("execution"),
        "adapter_source_fingerprints": {
            engine: payloads[engine]["adapter_source_sha256"] for engine in ENGINE_NAMES
        },
        "rows": rows,
    }


def compare_lower_level_trace_noop_shards(
    workload_path: Path,
    shard: str,
    *,
    untraced_results: dict[str, Path],
    traced_results: dict[str, Path],
) -> dict[str, Any]:
    """Gate I070 trace overhead from complete per-engine shard results."""

    return _compare_lower_level_trace_noop_views(
        _load_lower_level_trace_noop_shard_view(workload_path, shard, untraced_results),
        _load_lower_level_trace_noop_shard_view(workload_path, shard, traced_results),
    )


def derive_public_shard_workload(
    parent_workload_path: Path,
    shard: str,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize a reproducible, prompt-free all-record public shard manifest."""

    parent_path = parent_workload_path.resolve()
    parent = _load_workload(parent_path)
    lock_sha256 = parent.get("lock_sha256")
    if not isinstance(lock_sha256, str) or len(lock_sha256) != 64:
        raise MatrixError("parent public workload has no valid source lock hash")
    profile = parent.get("profile")
    if not isinstance(profile, str) or not profile:
        raise MatrixError("parent public workload has no profile")
    records = _resolve_selected_records(parent, shard)
    selected_records = json.loads(json.dumps(records, ensure_ascii=False))
    derived = {
        "schema_version": 1,
        "kind": "public-cross-engine-workload",
        "lock_sha256": lock_sha256,
        "data_root": parent.get("data_root"),
        "profile": f"{profile}:{shard}:order-state-trace",
        "selection": {
            "origin": "public-dataset-only-derived-shard",
            "parent_workload_path": str(parent_path),
            "parent_workload_sha256": _sha256_file(parent_path),
            "parent_profile": profile,
            "shard": shard,
            "all_parent_shard_records": True,
            "global_cross_engine_claim_eligible": False,
        },
        "generation": parent["generation"],
        "records": selected_records,
    }
    existing = _read_json(output_path) if output_path.is_file() else None
    if existing is not None:
        if existing != derived:
            raise MatrixError("derived public shard workload differs from its parent selection")
        return existing
    public.write_json(output_path, derived)
    return derived


def _state_trace_block_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, mode in enumerate(STATE_TRACE_ABBA_MODES, start=1):
        order = list(engine_order_for_shard(0, mode))
        first_engine = order[0]
        specs.append(
            {
                "index": index,
                "id": f"{index:02d}-{first_engine}-first",
                "engine_order_mode": mode,
                "order": order,
                "run_dir": f"blocks/{index:02d}-{first_engine}-first",
            }
        )
    return specs


def _state_trace_manifest(
    args: argparse.Namespace,
    parent_workload_path: Path,
    derived_workload_path: Path,
    record_count: int,
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "kind": "public-engine-order-state-trace",
        "created_utc": _now(),
        "updated_utc": _now(),
        "status": "running",
        "parent_workload": {
            "path": str(parent_workload_path),
            "sha256": _sha256_file(parent_workload_path),
        },
        "selected_workload": {
            "path": str(derived_workload_path),
            "sha256": _sha256_file(derived_workload_path),
            "shard": args.shard,
            "record_count": record_count,
        },
        "model_path": str(args.model),
        "model_fingerprint": fingerprint,
        "execution": execution_contract(args),
        "schedule": _state_trace_block_specs(),
        "blocks": {},
    }
    component_trace = _component_trace_config(args)
    if component_trace is not None:
        manifest["component_trace"] = component_trace
    lower_level_decode_trace = _lower_level_decode_trace_config(args)
    if lower_level_decode_trace is not None:
        manifest["lower_level_decode_trace"] = lower_level_decode_trace
    return manifest


def _check_state_trace_resume_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    parent_workload_path: Path,
    derived_workload_path: Path,
    record_count: int,
    fingerprint: dict[str, str],
) -> None:
    if manifest.get("kind") != "public-engine-order-state-trace":
        raise MatrixError("resume manifest has an unexpected order-state trace kind")
    expected = _state_trace_manifest(
        args,
        parent_workload_path,
        derived_workload_path,
        record_count,
        fingerprint,
    )
    for field in (
        "parent_workload",
        "selected_workload",
        "model_path",
        "model_fingerprint",
        "execution",
        "schedule",
        "component_trace",
        "lower_level_decode_trace",
    ):
        if manifest.get(field) != expected.get(field):
            raise MatrixError(f"resume order-state trace manifest differs in {field}")


def _matrix_state_traces(
    matrix: dict[str, Any],
    spec: dict[str, Any],
    shard: str,
) -> dict[str, dict[str, Any]]:
    expected_trace = {
        "schema_version": 1,
        "timing_boundary": STATE_TRACE_TIMING_BOUNDARY,
        "block": {"id": spec["id"], "index": spec["index"]},
    }
    manifest = matrix["manifest"]
    if manifest.get("state_trace") != expected_trace:
        raise MatrixError(f"state trace matrix {spec['id']} has unexpected trace metadata")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or len(entries) != 1:
        raise MatrixError(f"state trace matrix {spec['id']} must contain one public shard")
    entry = entries[0]
    if not isinstance(entry, dict) or entry.get("key") != shard:
        raise MatrixError(f"state trace matrix {spec['id']} selects an unexpected shard")
    if entry.get("order") != spec["order"]:
        raise MatrixError(f"state trace matrix {spec['id']} has an unexpected ABBA order")
    traces = entry.get("state_traces")
    if not isinstance(traces, dict) or set(traces) != set(ENGINE_NAMES):
        raise MatrixError(f"state trace matrix {spec['id']} has incomplete engine state traces")
    for engine in ENGINE_NAMES:
        _validate_state_trace_payload(traces[engine], expected_trace)
    return {engine: traces[engine] for engine in ENGINE_NAMES}


def _block_reproducibility(block_summaries: list[dict[str, Any]]) -> str:
    classes = [str(summary["effect_class"]) for summary in block_summaries]
    if len(classes) != 2:
        raise MatrixError("ABBA order stratum must contain exactly two blocks")
    if classes[0] == classes[1] and classes[0] in {
        "positive-material",
        "negative-material",
    }:
        return classes[0]
    if set(classes) == {"positive-material", "negative-material"}:
        return "directional-disagreement"
    return "inconclusive"


def _order_state_class(aster_first: str, mlx_lm_first: str) -> str:
    material = {"positive-material", "negative-material"}
    if aster_first in material and mlx_lm_first in material:
        if aster_first == mlx_lm_first:
            return "reproduced-order-stable-effect"
        return "reproduced-directional-disagreement"
    return "inconclusive"


def _state_trace_group_summary(
    group_id: str,
    metadata: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    specs: list[dict[str, Any]],
    bootstrap_samples: int,
    min_order_stratum_records: int,
) -> dict[str, Any]:
    rows_by_block = {
        spec["id"]: [row for row in rows if row["block_id"] == spec["id"]]
        for spec in specs
    }
    if not all(rows_by_block.values()):
        raise MatrixError(f"state trace group {group_id} misses an ABBA block")
    metrics: dict[str, dict[str, Any]] = {}
    for metric in PAIRED_METRICS:
        block_summaries = [
            {
                "block": spec["id"],
                "index": spec["index"],
                "first_engine": spec["order"][0],
                **_effect_summary(
                    [row["effects"][metric] for row in rows_by_block[spec["id"]]],
                    seed=f"{group_id}:{metric}:{spec['id']}",
                    samples=bootstrap_samples,
                    min_order_stratum_records=min_order_stratum_records,
                ),
            }
            for spec in specs
        ]
        first_engine_summaries: dict[str, dict[str, Any]] = {}
        for engine in ENGINE_NAMES:
            selected_blocks = [
                summary
                for summary in block_summaries
                if summary["first_engine"] == engine
            ]
            selected_rows = [row for row in rows if row["first_engine"] == engine]
            first_engine_summaries[engine] = {
                "blocks": selected_blocks,
                "combined": _effect_summary(
                    [row["effects"][metric] for row in selected_rows],
                    seed=f"{group_id}:{metric}:{engine}-first-combined",
                    samples=bootstrap_samples,
                    min_order_stratum_records=min_order_stratum_records,
                ),
                "reproducibility": _block_reproducibility(selected_blocks),
            }
        metrics[metric] = {
            "blocks": block_summaries,
            "aster_first": first_engine_summaries["aster"],
            "mlx_lm_first": first_engine_summaries["mlx-lm"],
            "order_state": _order_state_class(
                first_engine_summaries["aster"]["reproducibility"],
                first_engine_summaries["mlx-lm"]["reproducibility"],
            ),
        }
    swap_by_block = {
        spec["id"]: _swap_summary(rows_by_block[spec["id"]]) for spec in specs
    }
    return {
        "id": group_id,
        **metadata,
        "records": len(rows),
        "metrics": metrics,
        "swap_by_block": swap_by_block,
    }


def analyze_order_state_trace(
    block_run_dirs: list[Path],
    *,
    shard: str = STATE_TRACE_QMSUM_SHARD,
    bootstrap_samples: int = 2000,
    min_order_stratum_records: int = MIN_ORDER_STRATUM_RECORDS,
) -> dict[str, Any]:
    """Validate and analyze a four-block public ABBA state-trace run."""

    specs = _state_trace_block_specs()
    if len(block_run_dirs) != len(specs):
        raise MatrixError("order-state trace requires exactly four ABBA block directories")
    if bootstrap_samples < 100:
        raise MatrixError("bootstrap samples must be at least 100")
    if min_order_stratum_records < 1:
        raise MatrixError("minimum order-stratum records must be positive")
    matrices = [_load_completed_matrix(path) for path in block_run_dirs]
    reference = matrices[0]
    for matrix in matrices[1:]:
        _require_same_cross_matrix_contract(reference, matrix)
    traces_by_block = [
        _matrix_state_traces(matrix, spec, shard)
        for matrix, spec in zip(matrices, specs, strict=True)
    ]
    expected_records = _resolve_selected_records(reference["workload"], shard)
    if len(reference["workload"]["records"]) != len(expected_records):
        raise MatrixError("order-state trace workload includes records outside its selected shard")
    if shard == STATE_TRACE_QMSUM_SHARD and len(expected_records) != 200:
        raise MatrixError("public QMSUM order-state trace must retain all 200 source records")
    rows: list[dict[str, Any]] = []
    for matrix, spec in zip(matrices, specs, strict=True):
        first_by_record, _ = _first_engine_by_record(matrix)
        for record in matrix["workload"]["records"]:
            workload_id = record["workload_id"]
            if first_by_record[workload_id] != spec["order"][0]:
                raise MatrixError(f"state trace block {spec['id']} has inconsistent first-engine rows")
            aster_row = matrix["rows"]["aster"][workload_id]
            mlx_lm_row = matrix["rows"]["mlx-lm"][workload_id]
            rows.append(
                {
                    "workload_id": workload_id,
                    "block_id": spec["id"],
                    "block_index": spec["index"],
                    "first_engine": spec["order"][0],
                    "input_length_bin": _input_length_bin(aster_row["prompt_token_count"]),
                    "effects": {
                        metric: _paired_percent(
                            _metric_value(
                                aster_row,
                                metric,
                                engine="aster",
                                workload_id=workload_id,
                            ),
                            _metric_value(
                                mlx_lm_row,
                                metric,
                                engine="mlx-lm",
                                workload_id=workload_id,
                            ),
                            metric,
                        )
                        for metric in PAIRED_METRICS
                    },
                    "swap": {
                        "aster": _metric_value(
                            aster_row,
                            "swap_delta_bytes",
                            engine="aster",
                            workload_id=workload_id,
                        ),
                        "mlx-lm": _metric_value(
                            mlx_lm_row,
                            "swap_delta_bytes",
                            engine="mlx-lm",
                            workload_id=workload_id,
                        ),
                    },
                }
            )
    expected_total = len(expected_records) * len(specs)
    if len(rows) != expected_total:
        raise MatrixError("order-state trace record count is incomplete")
    if sum(row["first_engine"] == "aster" for row in rows) != expected_total // 2:
        raise MatrixError("order-state trace does not balance Aster-first records")
    if sum(row["first_engine"] == "mlx-lm" for row in rows) != expected_total // 2:
        raise MatrixError("order-state trace does not balance MLX-LM-first records")
    group_rows: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    group_metadata: dict[str, dict[str, str]] = {
        "overall": {
            "dimension": "overall",
            "workload": shard,
            "input_length_bin": "all",
        }
    }
    for row in rows:
        group_id = f"input-length:{row['input_length_bin']}"
        group_rows.setdefault(group_id, []).append(row)
        group_metadata.setdefault(
            group_id,
            {
                "dimension": "input-length",
                "workload": shard,
                "input_length_bin": row["input_length_bin"],
            },
        )
    groups = [
        _state_trace_group_summary(
            group_id,
            group_metadata[group_id],
            group_rows[group_id],
            specs=specs,
            bootstrap_samples=bootstrap_samples,
            min_order_stratum_records=min_order_stratum_records,
        )
        for group_id in ["overall", *sorted(key for key in group_rows if key != "overall")]
    ]
    overall = groups[0]
    order_states = [metric["order_state"] for metric in overall["metrics"].values()]
    any_swap_growth = any(
        summary["any_swap_growth"]
        for group in groups
        for summary in group["swap_by_block"].values()
    )
    if any_swap_growth:
        decision = "reject-swap-growth"
    elif "reproduced-directional-disagreement" in order_states:
        decision = "order-interaction-reproduced"
    elif "reproduced-order-stable-effect" in order_states:
        decision = "order-stable-effect-requires-component-trace"
    else:
        decision = "inconclusive-order-state"
    return {
        "schema_version": 1,
        "kind": "public-engine-order-state-analysis",
        "created_utc": _now(),
        "source": {
            "workload_sha256": reference["workload_sha256"],
            "source_lock_sha256": reference["source_lock_sha256"],
            "model_fingerprint": reference["model_fingerprint"],
            "execution": reference["execution"],
            "shard": shard,
            "records_per_block": len(expected_records),
        },
        "gates": {
            "all_blocks_comparable": True,
            "same_workload": True,
            "same_source_lock": True,
            "same_model_and_tokenizer": True,
            "same_execution_contract": True,
            "deterministic_cross_block_parity": True,
            "abba_order": True,
            "balanced_first_engine_blocks": True,
            "state_trace_complete": True,
            "zero_swap_growth": not any_swap_growth,
        },
        "schedule": specs,
        "block_state_traces": [
            {
                "id": spec["id"],
                "index": spec["index"],
                "order": spec["order"],
                "engines": traces,
            }
            for spec, traces in zip(specs, traces_by_block, strict=True)
        ],
        "bootstrap_samples": bootstrap_samples,
        "minimum_order_stratum_records": min_order_stratum_records,
        "groups": groups,
        "decision": decision,
        "production_candidate": "none",
    }


def _matrix_component_traces(matrix: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate and return per-record decode traces from one completed matrix."""

    expected = _component_trace_metadata()
    if matrix["manifest"].get("component_trace") != expected:
        raise MatrixError("component trace matrix has unexpected trace metadata")
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in ENGINE_NAMES:
        engine_traces: dict[str, dict[str, Any]] = {}
        for row in matrix["results"][engine]["records"]:
            workload_id = row.get("workload_id")
            output_token_count = row.get("output_token_count")
            if not isinstance(workload_id, str) or not workload_id:
                raise MatrixError(f"{engine} component trace has invalid workload ID")
            if isinstance(output_token_count, bool) or not isinstance(output_token_count, int):
                raise MatrixError(f"{engine} component trace has invalid output token count")
            trace = row.get("component_trace")
            _validate_component_trace_payload(
                trace,
                expected,
                engine=engine,
                output_token_count=output_token_count,
            )
            if not isinstance(trace, dict):
                raise MatrixError(f"{engine} component trace is not an object")
            engine_traces[workload_id] = trace
        traces[engine] = engine_traces
    return traces


def _matrix_lower_level_decode_traces(
    matrix: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate I070's source-aligned traces from one completed matrix."""

    expected = _lower_level_decode_trace_metadata()
    if matrix["manifest"].get("lower_level_decode_trace") != expected:
        raise MatrixError("lower-level decode trace matrix has unexpected trace metadata")
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for engine in ENGINE_NAMES:
        engine_traces: dict[str, dict[str, Any]] = {}
        for row in matrix["results"][engine]["records"]:
            workload_id = row.get("workload_id")
            output_token_count = row.get("output_token_count")
            if not isinstance(workload_id, str) or not workload_id:
                raise MatrixError(f"{engine} lower-level trace has invalid workload ID")
            if isinstance(output_token_count, bool) or not isinstance(output_token_count, int):
                raise MatrixError(f"{engine} lower-level trace has invalid output token count")
            trace = row.get("lower_level_decode_trace")
            _validate_lower_level_decode_trace_payload(
                trace,
                expected,
                engine=engine,
                output_token_count=output_token_count,
            )
            if not isinstance(trace, dict):
                raise MatrixError(f"{engine} lower-level trace is not an object")
            engine_traces[workload_id] = trace
        traces[engine] = engine_traces
    return traces


def _aster_component_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise MatrixError("cannot summarize an empty Aster component group")
    components: dict[str, dict[str, float]] = {}
    for field in ASTER_INTERNAL_COMPONENT_SECONDS:
        milliseconds_per_token: list[float] = []
        shares: list[float] = []
        for row in rows:
            trace = row["aster_component_trace"]
            seconds = trace["seconds"]
            output_token_count = row["output_token_count"]
            driver_seconds = _trace_seconds(
                seconds.get("decode_driver_seconds"),
                engine="aster",
                field="decode_driver_seconds",
            )
            if driver_seconds <= 0:
                raise MatrixError("Aster component trace has zero decode driver time")
            component_seconds = _trace_seconds(seconds.get(field), engine="aster", field=field)
            milliseconds_per_token.append(component_seconds * 1000.0 / output_token_count)
            shares.append(component_seconds * 100.0 / driver_seconds)
        components[field] = {
            "median_milliseconds_per_output_token": statistics.median(milliseconds_per_token),
            "median_decode_driver_share_percent": statistics.median(shares),
        }
    cache_fields = (
        "batch_cache_reuses",
        "batch_cache_rebuilds",
        "single_steps",
        "cache_clear_attempts",
        "cache_clears",
        "cache_clear_failures",
    )
    cache_totals = {
        field: sum(int(row["aster_component_trace"]["cache"][field]) for row in rows)
        for field in cache_fields
    }
    return {
        "records": len(rows),
        "components": components,
        "cache_totals": cache_totals,
    }


def _component_trace_group_summary(
    group_id: str,
    metadata: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    specs: list[dict[str, Any]],
    bootstrap_samples: int,
    min_order_stratum_records: int,
) -> dict[str, Any]:
    rows_by_block = {
        spec["id"]: [row for row in rows if row["block_id"] == spec["id"]]
        for spec in specs
    }
    if not all(rows_by_block.values()):
        raise MatrixError(f"component trace group {group_id} misses an ABBA block")
    block_summaries = [
        {
            "block": spec["id"],
            "index": spec["index"],
            "first_engine": spec["order"][0],
            **_effect_summary(
                [row["driver_effect_percent"] for row in rows_by_block[spec["id"]]],
                seed=f"{group_id}:decode-driver:{spec['id']}",
                samples=bootstrap_samples,
                min_order_stratum_records=min_order_stratum_records,
            ),
        }
        for spec in specs
    ]
    by_first_engine: dict[str, dict[str, Any]] = {}
    for engine in ENGINE_NAMES:
        selected_blocks = [
            summary for summary in block_summaries if summary["first_engine"] == engine
        ]
        selected_rows = [row for row in rows if row["first_engine"] == engine]
        by_first_engine[engine] = {
            "blocks": selected_blocks,
            "combined": _effect_summary(
                [row["driver_effect_percent"] for row in selected_rows],
                seed=f"{group_id}:decode-driver:{engine}-first-combined",
                samples=bootstrap_samples,
                min_order_stratum_records=min_order_stratum_records,
            ),
            "reproducibility": _block_reproducibility(selected_blocks),
            "aster_internal_accounting": _aster_component_accounting(selected_rows),
        }
    return {
        "id": group_id,
        **metadata,
        "records": len(rows),
        "decode_driver_seconds_per_output_token": {
            "effect_direction": "positive means Aster uses more seconds per output token",
            "blocks": block_summaries,
            "aster_first": by_first_engine["aster"],
            "mlx_lm_first": by_first_engine["mlx-lm"],
            "order_state": _order_state_class(
                by_first_engine["aster"]["reproducibility"],
                by_first_engine["mlx-lm"]["reproducibility"],
            ),
        },
        "swap_by_block": {
            spec["id"]: _swap_summary(rows_by_block[spec["id"]]) for spec in specs
        },
    }


def analyze_order_component_trace(
    block_run_dirs: list[Path],
    *,
    shard: str = STATE_TRACE_QMSUM_SHARD,
    bootstrap_samples: int = 2000,
    min_order_stratum_records: int = MIN_ORDER_STRATUM_RECORDS,
) -> dict[str, Any]:
    """Analyze I069's common decode-driver boundary and Aster-only accounting."""

    state_analysis = analyze_order_state_trace(
        block_run_dirs,
        shard=shard,
        bootstrap_samples=bootstrap_samples,
        min_order_stratum_records=min_order_stratum_records,
    )
    specs = _state_trace_block_specs()
    matrices = [_load_completed_matrix(path) for path in block_run_dirs]
    traces_by_block = [_matrix_component_traces(matrix) for matrix in matrices]
    reference = matrices[0]
    expected_records = _resolve_selected_records(reference["workload"], shard)
    rows: list[dict[str, Any]] = []
    for matrix, spec, traces in zip(matrices, specs, traces_by_block, strict=True):
        first_by_record, _ = _first_engine_by_record(matrix)
        for record in matrix["workload"]["records"]:
            workload_id = record["workload_id"]
            aster_row = matrix["rows"]["aster"][workload_id]
            mlx_lm_row = matrix["rows"]["mlx-lm"][workload_id]
            if aster_row["output_token_count"] != mlx_lm_row["output_token_count"]:
                raise MatrixError("component trace output token counts differ across engines")
            output_token_count = aster_row["output_token_count"]
            if isinstance(output_token_count, bool) or not isinstance(output_token_count, int):
                raise MatrixError("component trace has invalid output token count")
            aster_trace = traces["aster"].get(workload_id)
            mlx_lm_trace = traces["mlx-lm"].get(workload_id)
            if aster_trace is None or mlx_lm_trace is None:
                raise MatrixError("component trace misses an engine record")
            aster_driver = _trace_seconds(
                aster_trace["seconds"].get("decode_driver_seconds"),
                engine="aster",
                field="decode_driver_seconds",
            )
            mlx_lm_driver = _trace_seconds(
                mlx_lm_trace["seconds"].get("decode_driver_seconds"),
                engine="mlx-lm",
                field="decode_driver_seconds",
            )
            if aster_driver <= 0 or mlx_lm_driver <= 0:
                raise MatrixError("component trace has a zero common decode driver duration")
            rows.append(
                {
                    "workload_id": workload_id,
                    "block_id": spec["id"],
                    "block_index": spec["index"],
                    "first_engine": first_by_record[workload_id],
                    "input_length_bin": _input_length_bin(aster_row["prompt_token_count"]),
                    "output_token_count": output_token_count,
                    "driver_effect_percent": _paired_percent(
                        aster_driver / output_token_count,
                        mlx_lm_driver / output_token_count,
                        "decode_driver_seconds_per_output_token",
                    ),
                    "aster_component_trace": aster_trace,
                    "swap": {
                        "aster": _metric_value(
                            aster_row,
                            "swap_delta_bytes",
                            engine="aster",
                            workload_id=workload_id,
                        ),
                        "mlx-lm": _metric_value(
                            mlx_lm_row,
                            "swap_delta_bytes",
                            engine="mlx-lm",
                            workload_id=workload_id,
                        ),
                    },
                }
            )
    expected_total = len(expected_records) * len(specs)
    if len(rows) != expected_total:
        raise MatrixError("component trace record count is incomplete")
    group_rows: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    group_metadata: dict[str, dict[str, str]] = {
        "overall": {
            "dimension": "overall",
            "workload": shard,
            "input_length_bin": "all",
        }
    }
    for row in rows:
        group_id = f"input-length:{row['input_length_bin']}"
        group_rows.setdefault(group_id, []).append(row)
        group_metadata.setdefault(
            group_id,
            {
                "dimension": "input-length",
                "workload": shard,
                "input_length_bin": row["input_length_bin"],
            },
        )
    groups = [
        _component_trace_group_summary(
            group_id,
            group_metadata[group_id],
            group_rows[group_id],
            specs=specs,
            bootstrap_samples=bootstrap_samples,
            min_order_stratum_records=min_order_stratum_records,
        )
        for group_id in ["overall", *sorted(key for key in group_rows if key != "overall")]
    ]
    overall_state = groups[0]["decode_driver_seconds_per_output_token"]["order_state"]
    any_swap_growth = any(
        summary["any_swap_growth"]
        for group in groups
        for summary in group["swap_by_block"].values()
    )
    if any_swap_growth:
        decision = "reject-swap-growth"
    elif overall_state == "reproduced-directional-disagreement":
        decision = "reject-driver-directional-disagreement"
    elif overall_state == "reproduced-order-stable-effect":
        decision = "stable-decode-driver-gap-requires-lower-level-boundary"
    else:
        decision = "inconclusive-decode-component-trace"
    return {
        "schema_version": 1,
        "kind": "public-engine-order-component-analysis",
        "created_utc": _now(),
        "source": {
            "workload_sha256": reference["workload_sha256"],
            "source_lock_sha256": reference["source_lock_sha256"],
            "model_fingerprint": reference["model_fingerprint"],
            "execution": reference["execution"],
            "shard": shard,
            "records_per_block": len(expected_records),
        },
        "measurement": {
            "cross_engine_boundary": "decode_driver_seconds",
            "direct_prefill_exclusion": "reported-prompt-tps-derived-prefill-seconds",
            "aster_internal_boundary": (
                "sampling_completion_seconds includes the lazy MLX completion barrier; "
                "it is not directly comparable to a direct-MLX-LM private sub-step"
            ),
        },
        "gates": {
            "state_trace_gates_pass": all(state_analysis["gates"].values()),
            "component_trace_complete": True,
            "same_component_trace_contract": True,
            "single_request_batch_shape": True,
            "aster_single_request_has_no_batch_cache_merge": True,
            "common_decode_driver_boundary": True,
            "zero_swap_growth": not any_swap_growth,
        },
        "schedule": specs,
        "state_trace_decision": state_analysis["decision"],
        "bootstrap_samples": bootstrap_samples,
        "minimum_order_stratum_records": min_order_stratum_records,
        "groups": groups,
        "decision": decision,
        "production_candidate": "none",
    }


def _lower_level_engine_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source-aligned timing shares without treating them as a speed claim."""

    if not rows:
        raise MatrixError("cannot summarize an empty lower-level trace group")
    engines: dict[str, dict[str, Any]] = {}
    for engine in ENGINE_NAMES:
        components: dict[str, dict[str, float]] = {}
        for field in ("outer_step_seconds", *LOWER_LEVEL_COMPONENT_SECONDS):
            milliseconds_per_step: list[float] = []
            outer_shares: list[float] = []
            for row in rows:
                trace = row["lower_level_traces"][engine]
                seconds = trace["seconds"]
                traced_steps = row["traced_post_prefill_steps"]
                outer_step_seconds = _trace_seconds(
                    seconds.get("outer_step_seconds"),
                    engine=engine,
                    field="outer_step_seconds",
                )
                if outer_step_seconds <= 0:
                    raise MatrixError(f"{engine} lower-level trace has a zero outer step duration")
                component_seconds = _trace_seconds(seconds.get(field), engine=engine, field=field)
                milliseconds_per_step.append(component_seconds * 1000.0 / traced_steps)
                outer_shares.append(component_seconds * 100.0 / outer_step_seconds)
            components[field] = {
                "median_milliseconds_per_traced_post_prefill_step": statistics.median(
                    milliseconds_per_step
                ),
                "median_outer_step_share_percent": statistics.median(outer_shares),
            }
        engines[engine] = {"components": components}
    return {"records": len(rows), "engines": engines}


def _lower_level_component_group_summary(
    group_id: str,
    metadata: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    specs: list[dict[str, Any]],
    bootstrap_samples: int,
    min_order_stratum_records: int,
) -> dict[str, Any]:
    rows_by_block = {
        spec["id"]: [row for row in rows if row["block_id"] == spec["id"]]
        for spec in specs
    }
    if not all(rows_by_block.values()):
        raise MatrixError(f"lower-level trace group {group_id} misses an ABBA block")
    components: dict[str, dict[str, Any]] = {}
    for field in ("outer_step_seconds", *LOWER_LEVEL_COMPONENT_SECONDS):
        block_summaries = [
            {
                "block": spec["id"],
                "index": spec["index"],
                "first_engine": spec["order"][0],
                **_effect_summary(
                    [row["component_effects"][field] for row in rows_by_block[spec["id"]]],
                    seed=f"{group_id}:lower-level:{field}:{spec['id']}",
                    samples=bootstrap_samples,
                    min_order_stratum_records=min_order_stratum_records,
                ),
            }
            for spec in specs
        ]
        by_first_engine: dict[str, dict[str, Any]] = {}
        for engine in ENGINE_NAMES:
            selected_blocks = [
                summary for summary in block_summaries if summary["first_engine"] == engine
            ]
            selected_rows = [row for row in rows if row["first_engine"] == engine]
            by_first_engine[engine] = {
                "blocks": selected_blocks,
                "combined": _effect_summary(
                    [row["component_effects"][field] for row in selected_rows],
                    seed=f"{group_id}:lower-level:{field}:{engine}-first-combined",
                    samples=bootstrap_samples,
                    min_order_stratum_records=min_order_stratum_records,
                ),
                "reproducibility": _block_reproducibility(selected_blocks),
            }
        components[field] = {
            "effect_direction": (
                "positive means Aster uses more seconds per traced post-prefill decode step"
            ),
            "blocks": block_summaries,
            "aster_first": by_first_engine["aster"],
            "mlx_lm_first": by_first_engine["mlx-lm"],
            "order_state": _order_state_class(
                by_first_engine["aster"]["reproducibility"],
                by_first_engine["mlx-lm"]["reproducibility"],
            ),
        }
    return {
        "id": group_id,
        **metadata,
        "records": len(rows),
        "post_prefill_decode_step_seconds": {
            "components": components,
            "engine_median_accounting": _lower_level_engine_accounting(rows),
        },
        "swap_by_block": {
            spec["id"]: _swap_summary(rows_by_block[spec["id"]]) for spec in specs
        },
    }


def analyze_order_lower_level_decode_trace(
    block_run_dirs: list[Path],
    *,
    shard: str = STATE_TRACE_QMSUM_SHARD,
    bootstrap_samples: int = 2000,
    min_order_stratum_records: int = MIN_ORDER_STRATUM_RECORDS,
) -> dict[str, Any]:
    """Analyze I070's common steady-state submit/materialization boundary."""

    state_analysis = analyze_order_state_trace(
        block_run_dirs,
        shard=shard,
        bootstrap_samples=bootstrap_samples,
        min_order_stratum_records=min_order_stratum_records,
    )
    specs = _state_trace_block_specs()
    matrices = [_load_completed_matrix(path) for path in block_run_dirs]
    traces_by_block = [_matrix_lower_level_decode_traces(matrix) for matrix in matrices]
    reference = matrices[0]
    expected_records = _resolve_selected_records(reference["workload"], shard)
    rows: list[dict[str, Any]] = []
    total_output_records = 0
    initial_only_output_records = 0
    for matrix, spec, traces in zip(matrices, specs, traces_by_block, strict=True):
        first_by_record, _ = _first_engine_by_record(matrix)
        for record in matrix["workload"]["records"]:
            total_output_records += 1
            workload_id = record["workload_id"]
            aster_row = matrix["rows"]["aster"][workload_id]
            mlx_lm_row = matrix["rows"]["mlx-lm"][workload_id]
            if aster_row["output_token_count"] != mlx_lm_row["output_token_count"]:
                raise MatrixError("lower-level trace output token counts differ across engines")
            output_token_count = aster_row["output_token_count"]
            if isinstance(output_token_count, bool) or not isinstance(output_token_count, int):
                raise MatrixError("lower-level trace has invalid output token count")
            aster_trace = traces["aster"].get(workload_id)
            mlx_lm_trace = traces["mlx-lm"].get(workload_id)
            if aster_trace is None or mlx_lm_trace is None:
                raise MatrixError("lower-level trace misses an engine record")
            aster_decode = aster_trace["decode"]
            mlx_lm_decode = mlx_lm_trace["decode"]
            if aster_decode != mlx_lm_decode:
                raise MatrixError("lower-level trace decode coverage differs across engines")
            traced_steps = aster_decode["traced_post_prefill_steps"]
            if isinstance(traced_steps, bool) or not isinstance(traced_steps, int) or traced_steps < 0:
                raise MatrixError("lower-level trace has invalid post-prefill step count")
            if traced_steps == 0:
                initial_only_output_records += 1
                continue
            component_effects: dict[str, float] = {}
            for field in ("outer_step_seconds", *LOWER_LEVEL_COMPONENT_SECONDS):
                aster_seconds = _trace_seconds(
                    aster_trace["seconds"].get(field), engine="aster", field=field
                )
                mlx_lm_seconds = _trace_seconds(
                    mlx_lm_trace["seconds"].get(field), engine="mlx-lm", field=field
                )
                if aster_seconds <= 0 or mlx_lm_seconds <= 0:
                    raise MatrixError(f"lower-level trace has a zero {field} duration")
                component_effects[field] = _paired_percent(
                    aster_seconds / traced_steps,
                    mlx_lm_seconds / traced_steps,
                    f"{field}_per_traced_post_prefill_step",
                )
            rows.append(
                {
                    "workload_id": workload_id,
                    "block_id": spec["id"],
                    "block_index": spec["index"],
                    "first_engine": first_by_record[workload_id],
                    "input_length_bin": _input_length_bin(aster_row["prompt_token_count"]),
                    "traced_post_prefill_steps": traced_steps,
                    "component_effects": component_effects,
                    "lower_level_traces": {"aster": aster_trace, "mlx-lm": mlx_lm_trace},
                    "swap": {
                        "aster": _metric_value(
                            aster_row,
                            "swap_delta_bytes",
                            engine="aster",
                            workload_id=workload_id,
                        ),
                        "mlx-lm": _metric_value(
                            mlx_lm_row,
                            "swap_delta_bytes",
                            engine="mlx-lm",
                            workload_id=workload_id,
                        ),
                    },
                }
            )
    expected_total = len(expected_records) * len(specs)
    if total_output_records != expected_total:
        raise MatrixError("lower-level trace record count is incomplete")
    if not rows:
        raise MatrixError("lower-level trace has no post-prefill decode steps to compare")
    if sum(row["first_engine"] == "aster" for row in rows) != len(rows) // 2:
        raise MatrixError("lower-level trace does not balance Aster-first records")
    if sum(row["first_engine"] == "mlx-lm" for row in rows) != len(rows) // 2:
        raise MatrixError("lower-level trace does not balance MLX-LM-first records")
    group_rows: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    group_metadata: dict[str, dict[str, str]] = {
        "overall": {
            "dimension": "overall",
            "workload": shard,
            "input_length_bin": "all",
        }
    }
    for row in rows:
        group_id = f"input-length:{row['input_length_bin']}"
        group_rows.setdefault(group_id, []).append(row)
        group_metadata.setdefault(
            group_id,
            {
                "dimension": "input-length",
                "workload": shard,
                "input_length_bin": row["input_length_bin"],
            },
        )
    groups = [
        _lower_level_component_group_summary(
            group_id,
            group_metadata[group_id],
            group_rows[group_id],
            specs=specs,
            bootstrap_samples=bootstrap_samples,
            min_order_stratum_records=min_order_stratum_records,
        )
        for group_id in ["overall", *sorted(key for key in group_rows if key != "overall")]
    ]
    overall_components = groups[0]["post_prefill_decode_step_seconds"]["components"]
    component_order_states = {
        field: component["order_state"]
        for field, component in overall_components.items()
        if field != "outer_step_seconds"
    }
    stable_components = sorted(
        field
        for field, order_state in component_order_states.items()
        if order_state == "reproduced-order-stable-effect"
    )
    any_directional_disagreement = any(
        order_state == "reproduced-directional-disagreement"
        for order_state in component_order_states.values()
    )
    any_swap_growth = any(
        summary["any_swap_growth"]
        for group in groups
        for summary in group["swap_by_block"].values()
    )
    if any_swap_growth:
        decision = "reject-swap-growth"
    elif any_directional_disagreement:
        decision = "reject-lower-level-component-directional-disagreement"
    elif stable_components:
        decision = "stable-lower-level-component-gap-requires-source-candidate"
    else:
        decision = "inconclusive-lower-level-decode-trace"
    return {
        "schema_version": 1,
        "kind": "public-engine-order-lower-level-decode-analysis",
        "created_utc": _now(),
        "source": {
            "workload_sha256": reference["workload_sha256"],
            "source_lock_sha256": reference["source_lock_sha256"],
            "model_fingerprint": reference["model_fingerprint"],
            "execution": reference["execution"],
            "shard": shard,
            "records_per_block": len(expected_records),
        },
        "measurement": {
            "cross_engine_boundary": LOWER_LEVEL_DECODE_TRACE_TIMING_BOUNDARY,
            "direct_steady_state_pipeline": (
                "each traced direct-MLX-LM generator advance submits the next token and "
                "materializes the previously submitted token; the prefill-bearing first "
                "advance is excluded"
            ),
            "aster_boundary": (
                "each traced Aster single-request decode step submits model and sampler work "
                "and materializes its sampled token through the existing runner path"
            ),
            "initial_output_policy": LOWER_LEVEL_INITIAL_OUTPUT_POLICY,
        },
        "gates": {
            "state_trace_gates_pass": all(state_analysis["gates"].values()),
            "lower_level_trace_complete": True,
            "same_lower_level_trace_contract": True,
            "source_aligned_post_prefill_boundary": True,
            "direct_prefill_bearing_first_advance_excluded": True,
            "balanced_first_engine_blocks": True,
            "zero_swap_growth": not any_swap_growth,
        },
        "schedule": specs,
        "state_trace_decision": state_analysis["decision"],
        "bootstrap_samples": bootstrap_samples,
        "minimum_order_stratum_records": min_order_stratum_records,
        "coverage": {
            "output_records": total_output_records,
            "traced_post_prefill_records": len(rows),
            "initial_only_output_records": initial_only_output_records,
            "traced_post_prefill_steps": sum(
                row["traced_post_prefill_steps"] for row in rows
            ),
        },
        "groups": groups,
        "stable_common_components": stable_components,
        "decision": decision,
        "production_candidate": "none",
    }


def run_order_state_trace(args: argparse.Namespace) -> dict[str, Any]:
    """Run the public QMSUM ABBA state trace with one opt-in decode observer."""

    if args.shard != STATE_TRACE_QMSUM_SHARD:
        raise MatrixError(f"I068 order-state trace only supports {STATE_TRACE_QMSUM_SHARD}")
    parent_workload_path = args.workload.resolve()
    parent_workload = _load_workload(parent_workload_path)
    selected_records = _resolve_selected_records(parent_workload, args.shard)
    if len(selected_records) != 200:
        raise MatrixError("I068 requires all 200 public QMSUM records from cross-engine-core")
    run_root = args.run_dir.resolve()
    manifest_path = run_root / "state-trace-manifest.json"
    if run_root.exists() and any(run_root.iterdir()) and not args.resume:
        raise MatrixError(f"run directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    derived_workload_path = run_root / "qmsum-workload.json"
    derived_workload = derive_public_shard_workload(
        parent_workload_path,
        args.shard,
        derived_workload_path,
    )
    fingerprint = model_fingerprint(args.model)
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        _check_state_trace_resume_manifest(
            manifest,
            args,
            parent_workload_path,
            derived_workload_path,
            len(derived_workload["records"]),
            fingerprint,
        )
    else:
        manifest = _state_trace_manifest(
            args,
            parent_workload_path,
            derived_workload_path,
            len(derived_workload["records"]),
            fingerprint,
        )
        public.write_json(manifest_path, manifest)
    block_roots: list[Path] = []
    for spec in _state_trace_block_specs():
        block_root = run_root / spec["run_dir"]
        block_args = argparse.Namespace(**vars(args))
        block_args.workload = derived_workload_path
        block_args.run_dir = block_root
        block_args.engine_order_mode = spec["engine_order_mode"]
        block_args.state_trace = True
        block_args.state_trace_block_id = spec["id"]
        block_args.state_trace_block_index = spec["index"]
        run_matrix(block_args)
        block_manifest_path = block_root / "matrix-manifest.json"
        manifest["blocks"][spec["id"]] = {
            "index": spec["index"],
            "order": spec["order"],
            "matrix_manifest": _descriptor(block_manifest_path, run_root),
            "comparison": _descriptor(block_root / "comparison.json", run_root),
        }
        manifest["updated_utc"] = _now()
        public.write_json(manifest_path, manifest)
        block_roots.append(block_root)
    analysis = analyze_order_state_trace(
        block_roots,
        shard=args.shard,
        bootstrap_samples=args.bootstrap_samples,
        min_order_stratum_records=args.min_order_stratum_records,
    )
    analysis_path = run_root / "state-trace-comparison.json"
    public.write_json(analysis_path, analysis)
    component_analysis: dict[str, Any] | None = None
    component_analysis_path: Path | None = None
    if _component_trace_config(args) is not None:
        component_analysis = analyze_order_component_trace(
            block_roots,
            shard=args.shard,
            bootstrap_samples=args.bootstrap_samples,
            min_order_stratum_records=args.min_order_stratum_records,
        )
        component_analysis_path = run_root / "decode-component-comparison.json"
        public.write_json(component_analysis_path, component_analysis)
    lower_level_analysis: dict[str, Any] | None = None
    lower_level_analysis_path: Path | None = None
    if _lower_level_decode_trace_config(args) is not None:
        lower_level_analysis = analyze_order_lower_level_decode_trace(
            block_roots,
            shard=args.shard,
            bootstrap_samples=args.bootstrap_samples,
            min_order_stratum_records=args.min_order_stratum_records,
        )
        lower_level_analysis_path = run_root / "lower-level-decode-comparison.json"
        public.write_json(lower_level_analysis_path, lower_level_analysis)
    manifest["updated_utc"] = _now()
    final_analysis = lower_level_analysis or component_analysis or analysis
    manifest["status"] = final_analysis["decision"]
    manifest["comparison"] = _descriptor(analysis_path, run_root)
    if component_analysis_path is not None:
        manifest["component_trace_comparison"] = _descriptor(component_analysis_path, run_root)
    if lower_level_analysis_path is not None:
        manifest["lower_level_decode_trace_comparison"] = _descriptor(
            lower_level_analysis_path,
            run_root,
        )
    public.write_json(manifest_path, manifest)
    return final_analysis


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock", type=Path, default=public.DEFAULT_LOCK_PATH)
    parser.add_argument("--data-root", type=Path, default=public.DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)


def _add_state_trace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-trace", action="store_true")
    parser.add_argument("--state-trace-block-id")
    parser.add_argument("--state-trace-block-index", type=int)


def _add_component_trace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--component-trace", action="store_true")
    parser.add_argument("--lower-level-decode-trace", action="store_true")


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--engine", choices=ENGINE_NAMES, required=True)
    shard.add_argument("--workload", type=Path, required=True)
    shard.add_argument("--shard", required=True)
    shard.add_argument("--model-fingerprint", type=Path)
    shard.add_argument("--include-output-token-ids", action="store_true")
    shard.add_argument("--engine-order-mode", choices=ENGINE_ORDER_MODES, default="alternating")
    shard.add_argument("--engine-position", type=int, choices=(0, 1))
    shard.add_argument("--output", type=Path, required=True)
    _add_common_arguments(shard)
    _add_state_trace_arguments(shard)
    _add_component_trace_arguments(shard)
    matrix = subparsers.add_parser("run-matrix")
    matrix.add_argument("--workload", type=Path, required=True)
    matrix.add_argument("--run-dir", type=Path, required=True)
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--engine-order-mode", choices=ENGINE_ORDER_MODES, default="alternating")
    _add_common_arguments(matrix)
    _add_state_trace_arguments(matrix)
    _add_component_trace_arguments(matrix)
    state_trace = subparsers.add_parser("run-order-state-trace")
    state_trace.add_argument("--workload", type=Path, required=True)
    state_trace.add_argument("--run-dir", type=Path, required=True)
    state_trace.add_argument("--resume", action="store_true")
    state_trace.add_argument("--shard", default=STATE_TRACE_QMSUM_SHARD)
    state_trace.add_argument("--bootstrap-samples", type=int, default=2000)
    state_trace.add_argument(
        "--min-order-stratum-records",
        type=int,
        default=MIN_ORDER_STRATUM_RECORDS,
    )
    _add_common_arguments(state_trace)
    _add_component_trace_arguments(state_trace)
    compare = subparsers.add_parser("compare-matrices")
    compare.add_argument("--original-run-dir", type=Path, required=True)
    compare.add_argument("--reversed-run-dir", type=Path, required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    compare.add_argument(
        "--min-order-stratum-records",
        type=int,
        default=MIN_ORDER_STRATUM_RECORDS,
    )
    compare.add_argument("--output", type=Path, required=True)
    trace_noop = subparsers.add_parser("compare-lower-level-trace-noop")
    trace_noop.add_argument("--untraced-run-dir", type=Path, required=True)
    trace_noop.add_argument("--traced-run-dir", type=Path, required=True)
    trace_noop.add_argument("--output", type=Path, required=True)
    trace_noop_shards = subparsers.add_parser("compare-lower-level-trace-noop-shards")
    trace_noop_shards.add_argument("--workload", type=Path, required=True)
    trace_noop_shards.add_argument("--shard", required=True)
    trace_noop_shards.add_argument("--untraced-aster", type=Path, required=True)
    trace_noop_shards.add_argument("--untraced-mlx-lm", type=Path, required=True)
    trace_noop_shards.add_argument("--traced-aster", type=Path, required=True)
    trace_noop_shards.add_argument("--traced-mlx-lm", type=Path, required=True)
    trace_noop_shards.add_argument("--output", type=Path, required=True)
    return command_parser


def main() -> None:
    args = parser().parse_args()
    if args.command == "compare-lower-level-trace-noop-shards":
        payload = compare_lower_level_trace_noop_shards(
            args.workload.resolve(),
            args.shard,
            untraced_results={
                "aster": args.untraced_aster.resolve(),
                "mlx-lm": args.untraced_mlx_lm.resolve(),
            },
            traced_results={
                "aster": args.traced_aster.resolve(),
                "mlx-lm": args.traced_mlx_lm.resolve(),
            },
        )
        public.write_json(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "compare-lower-level-trace-noop":
        payload = compare_lower_level_trace_noop(
            args.untraced_run_dir.resolve(),
            args.traced_run_dir.resolve(),
        )
        public.write_json(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "compare-matrices":
        if args.bootstrap_samples < 100:
            raise MatrixError("bootstrap samples must be at least 100")
        if args.min_order_stratum_records < 1:
            raise MatrixError("minimum order-stratum records must be positive")
        payload = compare_matrices(
            args.original_run_dir.resolve(),
            args.reversed_run_dir.resolve(),
            bootstrap_samples=args.bootstrap_samples,
            min_order_stratum_records=args.min_order_stratum_records,
        )
        public.write_json(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    args.workload = args.workload.resolve()
    args.lock = args.lock.resolve()
    args.data_root = args.data_root.resolve()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.max_input_tokens = _require_even_positive(args.max_input_tokens, "--max-input-tokens")
    if args.warmup_tokens < 1 or args.prefill_step < 1:
        raise MatrixError("warmup tokens and prefill step must be positive")
    if args.memory_sample_interval <= 0:
        raise MatrixError("memory sample interval must be positive")
    if not args.model.is_dir():
        raise MatrixError(f"model directory does not exist: {args.model}")
    if args.command in {"run-shard", "run-matrix"}:
        has_state_trace_values = (
            args.state_trace_block_id is not None or args.state_trace_block_index is not None
        )
        if not args.state_trace and has_state_trace_values:
            raise MatrixError("state trace block metadata requires --state-trace")
        _state_trace_config(args)
    if args.command == "run-order-state-trace":
        if args.bootstrap_samples < 100:
            raise MatrixError("bootstrap samples must be at least 100")
        if args.min_order_stratum_records < 1:
            raise MatrixError("minimum order-stratum records must be positive")
        payload = run_order_state_trace(args)
    elif args.command == "run-shard":
        payload = run_shard(args)
        public.write_json(args.output.resolve(), payload)
    else:
        payload = run_matrix(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (MatrixError, public.PublicBenchmarkError) as error:
        print(f"public engine matrix error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
