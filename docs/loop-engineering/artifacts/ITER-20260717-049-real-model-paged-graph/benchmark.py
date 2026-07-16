#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_DIR))

from profile_lib import (  # noqa: E402
    TimingCollector,
    patch_method,
    phase_for_query,
    query_tokens,
    summarize_samples,
)

from aster.core.config import load_settings  # noqa: E402
from aster.inference.contracts import InferenceRequest  # noqa: E402
from aster.inference.engine import InferenceEngine  # noqa: E402
from aster.inference.model_runner import ModelRunner  # noqa: E402
from aster.inference.paged_kv_adapter import (  # noqa: E402
    PagedAttentionView,
    PagedKVCacheLayer,
    _PagedKVBlockPool,
)
from aster.telemetry.metrics import MetricsRegistry  # noqa: E402


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
        "System: You are a local Apple Silicon assistant. "
        "Keep answers precise. Reuse prior context when possible. "
        + " ".join(
            "Preserve this stable operating policy across every agent turn."
            for _ in range(16)
        )
        + " "
    )
    return prefix + " ".join(["section"] * context_words)


def _settings(config_path: Path, model_path: Path, *, direct: bool):
    base = load_settings(str(config_path))
    model = base.model.model_copy(
        update={
            "name": "Qwen3.5-0.8B-4bit",
            "path": str(model_path),
            "context_length": max(base.model.context_length, 16_384),
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


def _method_metadata(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
    item = args[1]
    return {
        "phase": "decode",
        "kv_tokens": int(getattr(item, "completion_tokens", 0)),
    }


def _prefill_metadata(
    _args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    start = int(kwargs.get("cache_token_count", 0))
    end = int(kwargs.get("target_cache_token_count", start))
    return {"phase": "prefill", "query_tokens": end - start, "kv_tokens": end}


def _update_metadata(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
    cache = args[0]
    tokens = query_tokens(args[1])
    return {
        "phase": phase_for_query(tokens),
        "query_tokens": tokens,
        "kv_tokens": int(cache.offset) + int(tokens or 0),
    }


def _attention_metadata(
    args: tuple[Any, ...], _kwargs: dict[str, Any]
) -> dict[str, Any]:
    view = args[0]
    tokens = query_tokens(args[1])
    return {
        "phase": phase_for_query(tokens),
        "query_tokens": tokens,
        "kv_tokens": int(view.sequence_length),
    }


def _native_attention_metadata(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    tokens = query_tokens(args[0])
    cache = kwargs.get("cache")
    return {
        "phase": phase_for_query(tokens),
        "query_tokens": tokens,
        "kv_tokens": int(getattr(cache, "offset", 0)),
    }


def _pool_write_metadata(
    args: tuple[Any, ...], _kwargs: dict[str, Any]
) -> dict[str, Any]:
    tokens = query_tokens(args[3])
    return {
        "phase": "decode" if tokens is not None and tokens <= 8 else "promotion",
        "query_tokens": tokens,
    }


class _TimedModel:
    def __init__(self, model: Any, collector: TimingCollector) -> None:
        self._model = model
        self._collector = collector

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tokens = _model_query_tokens(args[0]) if args else None
        started = time.perf_counter_ns()
        try:
            return self._model(*args, **kwargs)
        finally:
            self._collector.record(
                "model_forward_enqueue",
                time.perf_counter_ns() - started,
                phase=phase_for_query(tokens),
                query_tokens=tokens,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _model_query_tokens(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[-1])


class _Instrumentation:
    def __init__(self, collector: TimingCollector, *, profile: bool) -> None:
        self.collector = collector
        self.profile = profile
        self._restores: list[Any] = []

    def install(self) -> None:
        if not self.profile:
            self._install_token_capture()
            return

        import mlx_lm.models.qwen3_next as qwen3_next

        self._restores.extend(
            [
                patch_method(
                    qwen3_next,
                    "scaled_dot_product_attention",
                    self.collector,
                    "native_attention_enqueue",
                    metadata=_native_attention_metadata,
                ),
                patch_method(
                    PagedKVCacheLayer,
                    "update_and_fetch",
                    self.collector,
                    "kv_update_enqueue",
                    metadata=_update_metadata,
                ),
                patch_method(
                    PagedAttentionView,
                    "attention",
                    self.collector,
                    "paged_attention_enqueue",
                    metadata=_attention_metadata,
                ),
                patch_method(
                    PagedAttentionView,
                    "block_pool",
                    self.collector,
                    "paged_block_pool",
                    metadata=lambda args, _kwargs: {
                        "phase": "metadata",
                        "kv_tokens": int(args[0].sequence_length),
                    },
                ),
                patch_method(
                    PagedKVCacheLayer,
                    "prepare_direct_attention",
                    self.collector,
                    "paged_promote_enqueue",
                    metadata=lambda args, _kwargs: {
                        "phase": "promotion",
                        "kv_tokens": int(args[0].offset),
                    },
                ),
                patch_method(
                    _PagedKVBlockPool,
                    "write",
                    self.collector,
                    "kv_pool_write_enqueue",
                    metadata=_pool_write_metadata,
                ),
                patch_method(
                    ModelRunner,
                    "prefill_to",
                    self.collector,
                    "prefill_chunk_complete",
                    metadata=_prefill_metadata,
                ),
                patch_method(
                    ModelRunner,
                    "initialize_decode",
                    self.collector,
                    "initialize_decode_complete",
                    metadata=lambda _args, kwargs: {
                        "phase": "promotion",
                        "kv_tokens": int(kwargs.get("cache_token_count", 0)),
                    },
                ),
                patch_method(
                    ModelRunner,
                    "_decode_single",
                    self.collector,
                    "decode_step_complete",
                    metadata=_method_metadata,
                ),
                patch_method(
                    ModelRunner,
                    "_sample_token",
                    self.collector,
                    "sample_token_sync",
                    metadata=lambda _args, _kwargs: {"phase": "decode"},
                    capture_result=self.collector.record_token,
                ),
                patch_method(
                    ModelRunner,
                    "_eval_cache",
                    self.collector,
                    "cache_eval_complete",
                    metadata=lambda _args, _kwargs: {"phase": "cache_sync"},
                ),
            ]
        )

    def _install_token_capture(self) -> None:
        original = ModelRunner._sample_token
        collector = self.collector

        def capture(runner: ModelRunner, logprobs: Any, sampler: Any) -> int:
            token = original(runner, logprobs, sampler)
            collector.record_token(token)
            return token

        ModelRunner._sample_token = capture

        def restore() -> None:
            if ModelRunner._sample_token is capture:
                ModelRunner._sample_token = original

        self._restores.append(restore)

    def wrap_model(self, runner: ModelRunner) -> None:
        if self.profile:
            runner._model = _TimedModel(runner._model, self.collector)

    def restore(self) -> None:
        for restore in reversed(self._restores):
            restore()


async def _sample_rss(stop: asyncio.Event, values: list[int]) -> None:
    process = psutil.Process()
    while not stop.is_set():
        values.append(int(process.memory_info().rss))
        await asyncio.sleep(0.01)


def _mlx_memory(mx: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, function_name in (
        ("active_bytes", "get_active_memory"),
        ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"),
    ):
        function = getattr(mx, function_name, None)
        result[key] = int(function()) if callable(function) else 0
    return result


def _resolve_runner(engine: InferenceEngine) -> ModelRunner:
    runner = getattr(engine, "model_runner", None)
    if not isinstance(runner, ModelRunner):
        raise TypeError("manual benchmark requires InferenceEngine.model_runner")
    return runner


async def run(args: argparse.Namespace) -> dict[str, object]:
    direct = args.variant == "direct"
    settings = _settings(args.config, args.model, direct=direct)
    collector = TimingCollector()
    instrumentation = _Instrumentation(collector, profile=args.mode == "profile")
    instrumentation.install()
    metrics = MetricsRegistry(f"aster_iter049_{os.getpid()}")
    engine = InferenceEngine(settings, metrics)

    try:
        await engine.start()
        with collector.paused():
            await engine.warmup()
            runner = _resolve_runner(engine)
            instrumentation.wrap_model(runner)
            await engine.submit(
                InferenceRequest(
                    prompt="Warm the deterministic decode graph.",
                    max_tokens=args.warmup_tokens,
                    temperature=0.0,
                    trace_id=f"iter049-warmup-{args.run_id}",
                )
            )

        collector.clear()
        mx = runner._mx
        assert mx is not None
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        before_memory = _mlx_memory(mx)
        before_swap = int(psutil.swap_memory().used)
        thermal_before = _thermal_state()
        rss_values = [int(psutil.Process().memory_info().rss)]
        rss_stop = asyncio.Event()
        rss_task = asyncio.create_task(_sample_rss(rss_stop, rss_values))

        request = InferenceRequest(
            prompt=_prompt(args.context_words),
            max_tokens=args.max_tokens,
            temperature=0.0,
            trace_id=f"iter049-{args.mode}-{args.variant}-{args.context_words}-{args.run_id}",
        )
        started = time.perf_counter()
        try:
            response = await engine.submit(request)
        finally:
            elapsed = time.perf_counter() - started
            rss_stop.set()
            await rss_task

        after_memory = _mlx_memory(mx)
        after_swap = int(psutil.swap_memory().used)
        status = engine.status()
        token_ids = list(collector.tokens)
        if len(token_ids) != response.completion_tokens:
            raise RuntimeError(
                f"captured {len(token_ids)} tokens for {response.completion_tokens} completions"
            )

        return {
            "schema_version": 1,
            "mode": args.mode,
            "variant": args.variant,
            "run_id": args.run_id,
            "context_words": args.context_words,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
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
            "source_sha256": {
                (
                    str(path.relative_to(PROJECT_ROOT))
                    if path.is_relative_to(PROJECT_ROOT)
                    else str(path)
                ): _sha256(path)
                for path in (
                    PROJECT_ROOT / "aster/core/config.py",
                    PROJECT_ROOT / "aster/inference/contracts.py",
                    PROJECT_ROOT / "aster/inference/engine.py",
                    PROJECT_ROOT / "aster/inference/model_runner.py",
                    PROJECT_ROOT / "aster/inference/paged_kv_adapter.py",
                    PROJECT_ROOT / "aster/inference/paged_attention_bridge.py",
                    PROJECT_ROOT / "aster/inference/metal_paged_attention.py",
                    PROJECT_ROOT / "aster/inference/runtime_kernel.py",
                    PROJECT_ROOT / "aster/telemetry/metrics.py",
                    args.config.resolve(),
                    Path(__file__).resolve(),
                    ARTIFACT_DIR / "profile_lib.py",
                )
            },
            "settings": {
                "model_path": str(args.model.relative_to(PROJECT_ROOT)),
                "max_tokens": args.max_tokens,
                "prefill_token_budget": settings.engine.prefill_token_budget,
                "paged_cache_block_size": settings.engine.paged_cache_block_size,
                "paged_cache_enabled": settings.engine.paged_cache_enabled,
                "paged_cache_direct_attention_enabled": (
                    settings.engine.paged_cache_direct_attention_enabled
                ),
                "prefix_cache_enabled": settings.engine.prefix_cache_enabled,
                "max_decode_batch": settings.engine.max_decode_batch,
                "temperature": 0.0,
            },
            "response": {
                "elapsed_seconds": elapsed,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "generation_tps": response.generation_tps,
                "prompt_tps": response.prompt_tps,
                "finish_reason": response.finish_reason,
                "token_ids": token_ids,
                "text_sha256": hashlib.sha256(response.text.encode()).hexdigest(),
                "mlx_peak_memory_bytes": after_memory["peak_bytes"],
                "process_rss_peak_bytes": max(rss_values),
                "swap_before_bytes": before_swap,
                "swap_after_bytes": after_swap,
            },
            "mlx_memory_before": before_memory,
            "mlx_memory_after": after_memory,
            "timings": summarize_samples(collector.samples),
            "raw_timings": [asdict(sample) for sample in collector.samples],
            "request_timeline": status.get("recent_request_timelines", [])[-2:],
            "decode_diagnostics": status.get("decode_batch_diagnostics", {}),
        }
    finally:
        await engine.aclose()
        instrumentation.restore()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--variant", choices=("native", "direct"), required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--context-words", type=int, choices=(2048, 8192), required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_absolute():
        args.model = (PROJECT_ROOT / args.model).resolve()
    if not args.config.is_absolute():
        args.config = (PROJECT_ROOT / args.config).resolve()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.max_tokens < 1 or args.warmup_tokens < 1:
        raise ValueError("token counts must be positive")

    payload = asyncio.run(run(args))
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
