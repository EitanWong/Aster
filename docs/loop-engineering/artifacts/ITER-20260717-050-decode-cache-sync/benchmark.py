#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.model_runner import (  # noqa: E402
    DecodeResult,
    DecodeWorkItem,
    ModelRunner,
)

POLICIES = (
    "baseline",
    "skip-eval-clear-each",
    "periodic-256",
    "periodic-512",
    "periodic-2048",
    "skip-eval-no-clear",
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _thermal_state() -> str:
    try:
        return subprocess.check_output(
            ["pmset", "-g", "therm"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def _prompt(context_words: int) -> str:
    prefix = (
        "System: You are a deterministic local inference benchmark. "
        "Preserve every stated fact and answer with a concise technical summary. "
    )
    words = " ".join(
        f"fact{index % 97:02d}" for index in range(max(context_words, 1))
    )
    return f"{prefix}{words}\nAssistant:"


def _settings(config_path: Path, model_path: Path, *, cache_kind: str, batch_size: int):
    base = load_settings(str(config_path))
    model = base.model.model_copy(
        update={
            "name": "Qwen3.5-0.8B-4bit",
            "path": str(model_path),
            "context_length": max(base.model.context_length, 32_768),
            "enable_thinking": False,
        }
    )
    direct = cache_kind == "direct"
    engine = base.engine.model_copy(
        update={
            "engine_type": "manual",
            "runtime_kernel": "manual",
            "max_active_requests": max(batch_size, 1),
            "max_decode_batch": batch_size,
            "prefill_token_budget": 1024,
            "idle_prefill_token_limit": 1024,
            "pressure_prefill_token_budget": 1024,
            "prefix_cache_enabled": False,
            "prefix_cache_load_on_warmup": False,
            "prefix_cache_save_on_shutdown": False,
            "warm_prompts_path": None,
            "paged_cache_enabled": direct,
            "paged_cache_direct_attention_enabled": direct,
        }
    )
    return base.model_copy(
        update={
            "model": model,
            "engine": engine,
            "speculative": base.speculative.model_copy(
                update={"enabled": False, "max_draft_tokens": 0}
            ),
            "embeddings": base.embeddings.model_copy(update={"enabled": False}),
        }
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


class DecodePolicyMX:
    def __init__(self, base: Any, policy: str) -> None:
        self._base = base
        self.policy = policy
        self._active_depth = 0
        self.cache_eval_requests = 0
        self.cache_eval_executed = 0
        self.cache_eval_skipped = 0
        self.cache_eval_seconds: list[float] = []
        self.clear_requests = 0
        self.clear_executed = 0
        self.clear_skipped = 0
        self.clear_seconds: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @contextlib.contextmanager
    def decode_scope(self) -> Iterator[None]:
        self._active_depth += 1
        try:
            yield
        finally:
            self._active_depth -= 1

    def eval(self, *args: Any) -> None:
        is_cache_tree = (
            self._active_depth > 0
            and len(args) == 1
            and isinstance(args[0], list)
        )
        if not is_cache_tree:
            self._base.eval(*args)
            return

        self.cache_eval_requests += 1
        started = time.perf_counter()
        try:
            if self.policy == "baseline":
                self.cache_eval_executed += 1
                self._base.eval(*args)
            else:
                self.cache_eval_skipped += 1
        finally:
            self.cache_eval_seconds.append(time.perf_counter() - started)

    def clear_cache(self) -> None:
        if self._active_depth <= 0:
            self._base.clear_cache()
            return

        self.clear_requests += 1
        execute = self.policy in ("baseline", "skip-eval-clear-each")
        if self.policy.startswith("periodic-"):
            interval = int(self.policy.rsplit("-", 1)[1])
            execute = self.clear_requests % interval == 0

        started = time.perf_counter()
        try:
            if execute:
                self.clear_executed += 1
                self._base.clear_cache()
            else:
                self.clear_skipped += 1
        finally:
            self.clear_seconds.append(time.perf_counter() - started)

    def metrics(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "cache_eval_requests": self.cache_eval_requests,
            "cache_eval_executed": self.cache_eval_executed,
            "cache_eval_skipped": self.cache_eval_skipped,
            "cache_eval_seconds": _summary(self.cache_eval_seconds),
            "clear_requests": self.clear_requests,
            "clear_executed": self.clear_executed,
            "clear_skipped": self.clear_skipped,
            "clear_seconds": _summary(self.clear_seconds),
        }


@dataclass
class Lane:
    request_id: str
    prompt_tokens: list[int]
    prompt_cache: Any
    input_token: int
    sampler: Any
    detokenizer: Any
    logits_processors: tuple[Any, ...]
    output_tokens: list[int] = field(default_factory=list)
    text_segments: list[str] = field(default_factory=list)


def _prepare_lane(
    runner: ModelRunner,
    *,
    request_id: str,
    prompt: str,
    max_tokens: int,
    prefill_step: int,
) -> tuple[Lane, float]:
    request = InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        trace_id=request_id,
    )
    prepared = runner.encode_request(request)
    target = len(prepared.prompt_tokens) - 1
    prompt_cache = None
    cache_token_count = 0
    started = time.perf_counter()
    while cache_token_count < target:
        result = runner.prefill_to(
            prompt_tokens=prepared.prompt_tokens,
            prompt_cache=prompt_cache,
            cache_token_count=cache_token_count,
            target_cache_token_count=min(cache_token_count + prefill_step, target),
        )
        prompt_cache = result.prompt_cache
        cache_token_count = result.cache_token_count
    decode = runner.initialize_decode(
        prompt_tokens=prepared.prompt_tokens,
        cache_token_count=cache_token_count,
        prompt_cache=prompt_cache,
        request=request,
    )
    return (
        Lane(
            request_id=request_id,
            prompt_tokens=prepared.prompt_tokens,
            prompt_cache=decode.prompt_cache,
            input_token=decode.next_input_token,
            sampler=decode.sampler,
            detokenizer=decode.detokenizer,
            logits_processors=decode.logits_processors,
        ),
        time.perf_counter() - started,
    )


def _work_item(lane: Lane, max_tokens: int) -> DecodeWorkItem:
    processor_tokens = lane.prompt_tokens + lane.output_tokens
    if processor_tokens and processor_tokens[-1] == lane.input_token:
        processor_tokens = processor_tokens[:-1]
    return DecodeWorkItem(
        prompt_cache=lane.prompt_cache,
        input_token=lane.input_token,
        sampler=lane.sampler,
        detokenizer=lane.detokenizer,
        stop_token_ids=frozenset(),
        logits_processors=lane.logits_processors,
        logits_processor_tokens=processor_tokens,
        completion_tokens=len(lane.output_tokens),
        max_tokens=max_tokens,
        request_id=lane.request_id,
    )


def _install_policy(runner: ModelRunner, policy: str) -> tuple[DecodePolicyMX, Any]:
    base_mx = runner._mx
    if base_mx is None:
        raise RuntimeError("MLX is not loaded")
    proxy = DecodePolicyMX(base_mx, policy)
    original_single = runner._decode_single
    original_batch = runner._decode_batch

    def decode_single(item: DecodeWorkItem) -> DecodeResult:
        with proxy.decode_scope():
            return original_single(item)

    def decode_batch(items: list[DecodeWorkItem]) -> list[DecodeResult]:
        with proxy.decode_scope():
            return original_batch(items)

    runner._mx = proxy
    runner._decode_single = decode_single  # type: ignore[method-assign]
    runner._decode_batch = decode_batch  # type: ignore[method-assign]

    def restore() -> None:
        runner._mx = base_mx
        runner._decode_single = original_single  # type: ignore[method-assign]
        runner._decode_batch = original_batch  # type: ignore[method-assign]

    return proxy, restore


def _release_lane(runner: ModelRunner, lane: Lane) -> None:
    cache = runner._resolve_decode_cache(lane.prompt_cache)
    release = getattr(cache, "release", None)
    if callable(release):
        release()


def _warmup(runner: ModelRunner, *, tokens: int, prefill_step: int) -> None:
    lane, _ = _prepare_lane(
        runner,
        request_id="iter050-warmup",
        prompt="Warm the deterministic hybrid decode graph.",
        max_tokens=tokens,
        prefill_step=prefill_step,
    )
    for _ in range(tokens):
        result = runner.decode_batch_step([_work_item(lane, tokens)])[0]
        if not isinstance(result, DecodeResult) or result.token_id is None:
            raise RuntimeError(f"warmup decode failed: {result!r}")
        lane.prompt_cache = result.prompt_cache
        lane.input_token = result.token_id
        lane.output_tokens.append(result.token_id)
    _release_lane(runner, lane)
    mx = runner._mx
    if mx is not None:
        mx.clear_cache()


def _rss_sampler(stop: threading.Event, samples: list[int]) -> None:
    process = psutil.Process()
    while not stop.wait(0.01):
        samples.append(int(process.memory_info().rss))


def _mlx_memory(mx: Any) -> dict[str, int]:
    values: dict[str, int] = {}
    for key, name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        function = getattr(mx, name, None)
        values[key] = int(function()) if callable(function) else 0
    return values


def _flatten_arrays(value: Any, output: list[Any]) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _flatten_arrays(value[key], output)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_arrays(item, output)
        return
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        output.append(value)


def _logical_cache_arrays(runner: ModelRunner, lane: Lane) -> list[Any]:
    cache = runner._resolve_decode_cache(lane.prompt_cache)
    arrays: list[Any] = []
    for layer in cache:
        _flatten_arrays(layer.state, arrays)
    return arrays


def _cache_digest(mx: Any, runner: ModelRunner, lanes: list[Lane]) -> str:
    digest = hashlib.sha256()
    for lane_index, lane in enumerate(lanes):
        arrays = _logical_cache_arrays(runner, lane)
        mx.eval(arrays)
        digest.update(f"lane:{lane_index};arrays:{len(arrays)};".encode())
        for array_index, array in enumerate(arrays):
            metadata = f"{array_index}:{tuple(array.shape)}:{array.dtype};"
            digest.update(metadata.encode())
            raw = np.asarray(array.view(mx.uint8))
            digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


def _model_inputs(model_path: Path) -> tuple[Path, ...]:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    return tuple(path for name in names if (path := model_path / name).is_file())


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.cache_kind == "direct" and args.batch_size != 1:
        raise ValueError("direct paged cache currently supports batch size 1")
    settings = _settings(
        args.config,
        args.model,
        cache_kind=args.cache_kind,
        batch_size=args.batch_size,
    )
    runner = ModelRunner(settings)
    runner.warmup()
    base_mx = runner._mx
    if base_mx is None:
        raise RuntimeError("MLX failed to load")
    _warmup(runner, tokens=args.warmup_tokens, prefill_step=args.prefill_step)

    lanes: list[Lane] = []
    prefill_seconds: list[float] = []
    for lane_index in range(args.batch_size):
        lane, elapsed = _prepare_lane(
            runner,
            request_id=f"iter050-lane-{lane_index}",
            prompt=_prompt(args.context_words),
            max_tokens=args.max_tokens,
            prefill_step=args.prefill_step,
        )
        lanes.append(lane)
        prefill_seconds.append(elapsed)

    proxy, restore = _install_policy(runner, args.policy)
    reset_peak = getattr(base_mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    memory_before = _mlx_memory(base_mx)
    swap_before = int(psutil.swap_memory().used)
    rss_samples = [int(psutil.Process().memory_info().rss)]
    rss_stop = threading.Event()
    rss_thread = threading.Thread(target=_rss_sampler, args=(rss_stop, rss_samples), daemon=True)
    rss_thread.start()
    thermal_before = _thermal_state()
    step_seconds: list[float] = []
    memory_curve: list[dict[str, int]] = []

    decode_started = time.perf_counter()
    try:
        for step in range(args.max_tokens):
            started = time.perf_counter()
            results = runner.decode_batch_step(
                [_work_item(lane, args.max_tokens) for lane in lanes]
            )
            step_seconds.append(time.perf_counter() - started)
            if len(results) != len(lanes):
                raise RuntimeError("decode result count mismatch")
            for lane, result in zip(lanes, results, strict=True):
                if not isinstance(result, DecodeResult) or result.token_id is None:
                    raise RuntimeError(f"decode failed: {result!r}")
                lane.prompt_cache = result.prompt_cache
                lane.input_token = result.token_id
                lane.output_tokens.append(result.token_id)
                lane.text_segments.append(result.text)
            if step == 0 or (step + 1) % args.memory_sample_interval == 0:
                memory_curve.append({"step": step + 1, **_mlx_memory(base_mx)})
    finally:
        decode_elapsed = time.perf_counter() - decode_started
        rss_stop.set()
        rss_thread.join(timeout=5.0)
        restore()

    memory_after_decode = _mlx_memory(base_mx)
    swap_after = int(psutil.swap_memory().used)
    cache_digest = _cache_digest(base_mx, runner, lanes)
    texts: list[str] = []
    for lane in lanes:
        lane.text_segments.append(runner.finalize_detokenizer(lane.detokenizer))
        texts.append("".join(lane.text_segments))

    payload: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "run_id": args.run_id,
        "policy": args.policy,
        "cache_kind": args.cache_kind,
        "batch_size": args.batch_size,
        "context_words": args.context_words,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "numpy": _version("numpy"),
            "psutil": _version("psutil"),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "thermal_before": thermal_before,
            "thermal_after": _thermal_state(),
        },
        "settings": {
            "model_path": str(args.model.relative_to(PROJECT_ROOT)),
            "config_path": str(args.config.relative_to(PROJECT_ROOT)),
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "prefill_step": args.prefill_step,
            "kv_cache_step_tokens": settings.engine.kv_cache_step_tokens,
            "paged_cache_block_size": settings.engine.paged_cache_block_size,
            "temperature": 0.0,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                PROJECT_ROOT / "aster/core/config.py",
                PROJECT_ROOT / "aster/inference/model_runner.py",
                PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
                PROJECT_ROOT / "aster/inference/paged_attention_bridge.py",
                PROJECT_ROOT / "aster/inference/metal_paged_attention.py",
                args.config,
                Path(__file__).resolve(),
            )
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in _model_inputs(args.model)
        },
        "prefill": {
            "seconds": prefill_seconds,
            "prompt_tokens": [len(lane.prompt_tokens) for lane in lanes],
        },
        "decode": {
            "elapsed_seconds": decode_elapsed,
            "tokens": args.max_tokens * args.batch_size,
            "tokens_per_second": (args.max_tokens * args.batch_size) / decode_elapsed,
            "step_seconds": _summary(step_seconds),
            "raw_step_seconds": step_seconds,
            "token_ids": [lane.output_tokens for lane in lanes],
            "text_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
            "cache_digest": cache_digest,
        },
        "policy_metrics": proxy.metrics(),
        "memory": {
            "mlx_before_decode": memory_before,
            "mlx_after_decode": memory_after_decode,
            "curve": memory_curve,
            "rss_peak_bytes": max(rss_samples),
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
        },
    }

    for lane in lanes:
        _release_lane(runner, lane)
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--cache-kind", choices=("native", "direct"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-words", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--memory-sample-interval", type=int, default=32)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.model = args.model.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.batch_size < 1 or args.context_words < 1:
        raise ValueError("batch size and context words must be positive")
    if args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("token counts must be positive")
    if args.memory_sample_interval < 1:
        raise ValueError("memory sample interval must be positive")

    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
