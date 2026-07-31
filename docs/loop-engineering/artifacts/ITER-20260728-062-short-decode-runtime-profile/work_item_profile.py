#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
I061_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260728-061-local-cross-engine-baseline"
)
if str(I061_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(I061_ARTIFACT_DIR))

import preflight as i061  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_processor_tokens(
    *,
    prompt_tokens: list[int],
    output_tokens: list[int],
    input_token: int,
    context_size: int | None,
) -> list[int]:
    """Mirror InferenceEngine._logits_processor_tokens for a manual work item."""
    if context_size == 0:
        return []
    if context_size is None:
        tokens = prompt_tokens + output_tokens
        if tokens and tokens[-1] == input_token:
            return tokens[:-1]
        return tokens

    prompt_end = len(prompt_tokens)
    output_end = len(output_tokens)
    if output_end and output_tokens[-1] == input_token:
        output_end -= 1
    elif not output_end and prompt_end and prompt_tokens[-1] == input_token:
        prompt_end -= 1
    preceding = max(context_size - 1, 0)
    output_start = max(output_end - preceding, 0)
    recent_output = output_tokens[output_start:output_end]
    remaining = preceding - len(recent_output)
    recent_prompt = prompt_tokens[max(prompt_end - remaining, 0) : prompt_end]
    return [*recent_prompt, *recent_output]


def _legacy_processor_tokens(
    *,
    prompt_tokens: list[int],
    output_tokens: list[int],
    input_token: int,
) -> list[int]:
    tokens = [*prompt_tokens, *output_tokens]
    if tokens and tokens[-1] == input_token:
        tokens.pop()
    return tokens


def _processor_tokens(
    variant: str,
    *,
    prompt_tokens: list[int],
    output_tokens: list[int],
    input_token: int,
    context_size: int | None,
) -> list[int]:
    if variant == "legacy-full-history":
        return _legacy_processor_tokens(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            input_token=input_token,
        )
    return _runtime_processor_tokens(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        input_token=input_token,
        context_size=context_size,
    )


def _generate(
    runner: ModelRunner,
    request: Any,
    *,
    prefill_step: int,
    variant: str,
) -> dict[str, Any]:
    prepared = runner.encode_request(request)
    prompt_tokens = prepared.prompt_tokens
    if len(prompt_tokens) < 2:
        raise ValueError("profile prompt must contain at least two tokens")
    prompt_cache = None
    cache_token_count = 0
    target = len(prompt_tokens) - 1
    while cache_token_count < target:
        prefill = runner.prefill_to(
            prompt_tokens=prompt_tokens,
            prompt_cache=prompt_cache,
            cache_token_count=cache_token_count,
            target_cache_token_count=min(cache_token_count + prefill_step, target),
        )
        prompt_cache = prefill.prompt_cache
        cache_token_count = prefill.cache_token_count
    decode = runner.initialize_decode(
        prompt_tokens=prompt_tokens,
        cache_token_count=cache_token_count,
        prompt_cache=prompt_cache,
        request=request,
    )
    if variant == "runtime-context" and decode.logits_processor_context_size != 0:
        raise RuntimeError("runtime-context screen requires a processor-free request")

    input_token = decode.next_input_token
    output_tokens: list[int] = []
    text_segments: list[str] = []
    processor_tokens_seconds = 0.0
    work_item_seconds = 0.0
    runner_decode_seconds = 0.0
    loop_started = time.perf_counter()
    finish_reason = "length"
    for _ in range(request.max_tokens):
        item_started = time.perf_counter()
        processor_started = time.perf_counter()
        processor_tokens = _processor_tokens(
            variant,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            input_token=input_token,
            context_size=decode.logits_processor_context_size,
        )
        processor_tokens_seconds += time.perf_counter() - processor_started
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
        work_item_seconds += time.perf_counter() - item_started
        decode_started = time.perf_counter()
        result = runner.decode_batch_step([item])[0]
        runner_decode_seconds += time.perf_counter() - decode_started
        if not isinstance(result, DecodeResult) or result.token_id is None:
            raise RuntimeError(f"Aster decode failed: {result!r}")
        decode.prompt_cache = result.prompt_cache
        input_token = result.token_id
        output_tokens.append(result.token_id)
        text_segments.append(result.text)
        if result.finish_reason is not None:
            finish_reason = result.finish_reason
            break
    decode_loop_seconds = time.perf_counter() - loop_started
    text_segments.append(runner.finalize_detokenizer(decode.detokenizer))
    return {
        "prompt_token_ids": prompt_tokens,
        "output_token_ids": output_tokens,
        "text_sha256": hashlib.sha256("".join(text_segments).encode()).hexdigest(),
        "finish_reason": finish_reason,
        "decode_loop_seconds": decode_loop_seconds,
        "runner_decode_seconds": runner_decode_seconds,
        "work_item_seconds": work_item_seconds,
        "processor_tokens_seconds": processor_tokens_seconds,
        "logits_processor_context_size": decode.logits_processor_context_size,
        "logits_processor_count": len(decode.logits_processors),
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        I061_ARTIFACT_DIR / "preflight.py",
        PROJECT_ROOT / "aster/inference/engine.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
    )


def run_variant(args: argparse.Namespace) -> dict[str, Any]:
    prompt = i061._prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    runner = ModelRunner(i061._settings(args.config, args.model))
    runner.warmup()
    _generate(
        runner,
        i061._request(prompt, args.warmup_tokens),
        prefill_step=args.prefill_step,
        variant=args.variant,
    )
    result = _generate(
        runner,
        i061._request(prompt, args.max_tokens),
        prefill_step=args.prefill_step,
        variant=args.variant,
    )
    return {
        "schema_version": 1,
        "variant": args.variant,
        "pid": os.getpid(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "settings": {
            "model_path": str(args.model.relative_to(PROJECT_ROOT)),
            "config_path": str(args.config.relative_to(PROJECT_ROOT)),
            "context_words": args.context_words,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "structured_output": False,
            "prefix_cache_enabled": False,
            "speculative_enabled": False,
        },
        "result": {
            **result,
            "output_tokens_per_second": (
                len(result["output_token_ids"]) / result["decode_loop_seconds"]
                if result["decode_loop_seconds"]
                else 0.0
            ),
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": int(psutil.swap_memory().used),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in _source_paths()
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in i061._model_inputs(args.model)
        },
    }


def compare(legacy_path: Path, runtime_path: Path) -> dict[str, Any]:
    legacy = json.loads(legacy_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    legacy_result = legacy["result"]
    runtime_result = runtime["result"]
    gates = {
        "model_inputs_equal": legacy["model_input_sha256"] == runtime["model_input_sha256"],
        "settings_equal": legacy["settings"] == runtime["settings"],
        "prompt_tokens_equal": (
            legacy_result["prompt_token_ids"] == runtime_result["prompt_token_ids"]
        ),
        "output_tokens_equal": (
            legacy_result["output_token_ids"] == runtime_result["output_token_ids"]
        ),
        "text_equal": legacy_result["text_sha256"] == runtime_result["text_sha256"],
        "finish_reason_equal": (
            legacy_result["finish_reason"] == runtime_result["finish_reason"]
        ),
        "processor_context_expected": (
            runtime_result["logits_processor_context_size"] == 0
            and runtime_result["logits_processor_count"] == 0
        ),
        "swap_non_growth": (
            legacy["memory"]["swap_after_bytes"] <= legacy["memory"]["swap_before_bytes"]
            and runtime["memory"]["swap_after_bytes"] <= runtime["memory"]["swap_before_bytes"]
        ),
    }
    legacy_loop = float(legacy_result["decode_loop_seconds"])
    runtime_loop = float(runtime_result["decode_loop_seconds"])
    legacy_runner = float(legacy_result["runner_decode_seconds"])
    runtime_runner = float(runtime_result["runner_decode_seconds"])
    return {
        "schema_version": 1,
        "legacy_record": str(legacy_path.relative_to(PROJECT_ROOT)),
        "runtime_record": str(runtime_path.relative_to(PROJECT_ROOT)),
        "comparable": all(gates.values()),
        "gates": gates,
        "output_tokens": len(legacy_result["output_token_ids"]),
        "legacy_output_tokens_per_second": legacy_result["output_tokens_per_second"],
        "runtime_output_tokens_per_second": runtime_result["output_tokens_per_second"],
        "decode_loop_change_percent": (runtime_loop / legacy_loop - 1.0) * 100.0,
        "runner_decode_change_percent": (runtime_runner / legacy_runner - 1.0) * 100.0,
        "legacy_processor_tokens_percent": (
            legacy_result["processor_tokens_seconds"] / legacy_loop * 100.0
        ),
        "runtime_processor_tokens_percent": (
            runtime_result["processor_tokens_seconds"] / runtime_loop * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile I061 harness work-item construction against engine semantics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--variant", choices=("legacy-full-history", "runtime-context"), required=True
    )
    run_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    run_parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    run_parser.add_argument("--context-words", type=int, default=128)
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.add_argument("--warmup-tokens", type=int, default=32)
    run_parser.add_argument("--prefill-step", type=int, default=1024)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--legacy", type=Path, required=True)
    compare_parser.add_argument("--runtime", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "compare":
        payload = compare(args.legacy.resolve(), args.runtime.resolve())
    else:
        args.config = args.config.resolve()
        args.model = args.model.resolve()
        if min(args.context_words, args.max_tokens, args.warmup_tokens, args.prefill_step) < 1:
            raise ValueError("context, token, warmup, and prefill values must be positive")
        payload = run_variant(args)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
