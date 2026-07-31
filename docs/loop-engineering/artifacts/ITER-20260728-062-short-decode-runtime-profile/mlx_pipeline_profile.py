#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import psutil

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


def _prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token or ""
    )
    return list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))


def _prefill_to_last_token(
    mx: Any,
    model: Any,
    prompt_tokens: list[int],
    prompt_cache: list[Any],
    *,
    prefill_step: int,
    generation_stream: Any,
) -> Any:
    prompt = mx.array(prompt_tokens)
    with mx.stream(generation_stream):
        while len(prompt) > 1:
            n_to_process = min(prefill_step, len(prompt) - 1)
            model(prompt[:n_to_process][None], cache=prompt_cache)
            mx.eval([cache_item.state for cache_item in prompt_cache])
            prompt = prompt[n_to_process:]
            mx.clear_cache()
    return prompt


def _serial_decode(
    mx: Any,
    model: Any,
    input_tokens: Any,
    prompt_cache: list[Any],
    sampler: Any,
    *,
    max_tokens: int,
    generation_stream: Any,
) -> list[int]:
    output_tokens: list[int] = []
    with mx.stream(generation_stream):
        current = input_tokens
        for index in range(max_tokens):
            logits = model(current[None], cache=prompt_cache)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            mx.eval(sampled)
            output_tokens.append(int(sampled.item()))
            if index % 256 == 0:
                mx.clear_cache()
            current = sampled
    return output_tokens


def _pipeline_decode(
    mx: Any,
    model: Any,
    input_tokens: Any,
    prompt_cache: list[Any],
    sampler: Any,
    *,
    max_tokens: int,
) -> list[int]:
    generate = importlib.import_module("mlx_lm.generate")
    output_tokens: list[int] = []
    for token, _logprobs in generate.generate_step(
        input_tokens,
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        output_tokens.append(int(token))
    return output_tokens


def _generate(
    variant: str,
    *,
    mx: Any,
    model: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    prefill_step: int,
    sampler: Any,
    make_prompt_cache: Any,
    generation_stream: Any,
) -> dict[str, Any]:
    prompt_cache = make_prompt_cache(model)
    input_tokens = _prefill_to_last_token(
        mx,
        model,
        prompt_tokens,
        prompt_cache,
        prefill_step=prefill_step,
        generation_stream=generation_stream,
    )
    started = time.perf_counter()
    if variant == "serial":
        output_tokens = _serial_decode(
            mx,
            model,
            input_tokens,
            prompt_cache,
            sampler,
            max_tokens=max_tokens,
            generation_stream=generation_stream,
        )
    else:
        output_tokens = _pipeline_decode(
            mx,
            model,
            input_tokens,
            prompt_cache,
            sampler,
            max_tokens=max_tokens,
        )
    decode_seconds = time.perf_counter() - started
    return {
        "prompt_token_ids": prompt_tokens,
        "output_token_ids": output_tokens,
        "finish_reason": "length",
        "decode_seconds": decode_seconds,
        "output_tokens_per_second": len(output_tokens) / decode_seconds,
    }


def _source_paths() -> tuple[Path, ...]:
    generate = importlib.import_module("mlx_lm.generate")
    generate_path = Path(inspect.getsourcefile(generate.generate_step) or "")
    return tuple(
        path
        for path in (
            Path(__file__).resolve(),
            I061_ARTIFACT_DIR / "preflight.py",
            generate_path,
        )
        if path.is_file()
    )


def run_variant(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler

    generate = importlib.import_module("mlx_lm.generate")
    prompt = i061._prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    model, tokenizer = load(str(args.model), lazy=False)
    prompt_tokens = _prompt_tokens(tokenizer, prompt)
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0)
    _generate(
        args.variant,
        mx=mx,
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=args.warmup_tokens,
        prefill_step=args.prefill_step,
        sampler=sampler,
        make_prompt_cache=cache.make_prompt_cache,
        generation_stream=generate.generation_stream,
    )
    result = _generate(
        args.variant,
        mx=mx,
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        prefill_step=args.prefill_step,
        sampler=sampler,
        make_prompt_cache=cache.make_prompt_cache,
        generation_stream=generate.generation_stream,
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
            "context_words": args.context_words,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "prefill_step": args.prefill_step,
        },
        "result": result,
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": int(psutil.swap_memory().used),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): _sha256(path)
            for path in _source_paths()
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in i061._model_inputs(args.model)
        },
    }


def _comparison_gates(serial_result: dict[str, Any], pipeline_result: dict[str, Any]) -> dict[str, bool]:
    return {
        "prompt_tokens_equal": (
            serial_result["prompt_token_ids"] == pipeline_result["prompt_token_ids"]
        ),
        "output_tokens_equal": (
            serial_result["output_token_ids"] == pipeline_result["output_token_ids"]
        ),
        "finish_reason_equal": (
            serial_result["finish_reason"] == pipeline_result["finish_reason"]
        ),
    }


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler

    generate = importlib.import_module("mlx_lm.generate")
    prompt = i061._prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    model, tokenizer = load(str(args.model), lazy=False)
    prompt_tokens = _prompt_tokens(tokenizer, prompt)
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0)
    common = {
        "mx": mx,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "prefill_step": args.prefill_step,
        "sampler": sampler,
        "make_prompt_cache": cache.make_prompt_cache,
        "generation_stream": generate.generation_stream,
    }
    # Warm both graphs before timing so only the declared measurement order varies.
    for variant in ("serial", "pipeline"):
        _generate(variant, max_tokens=args.warmup_tokens, **common)
    results = {
        variant: _generate(variant, max_tokens=args.max_tokens, **common)
        for variant in args.order
    }
    serial_result = results["serial"]
    pipeline_result = results["pipeline"]
    gates = _comparison_gates(serial_result, pipeline_result)
    swap_after = int(psutil.swap_memory().used)
    gates["swap_non_growth"] = swap_after <= swap_before
    serial_seconds = float(serial_result["decode_seconds"])
    pipeline_seconds = float(pipeline_result["decode_seconds"])
    return {
        "schema_version": 1,
        "order": list(args.order),
        "pid": os.getpid(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        },
        "settings": {
            "model_path": str(args.model.relative_to(PROJECT_ROOT)),
            "context_words": args.context_words,
            "max_tokens": args.max_tokens,
            "warmup_tokens": args.warmup_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "prefill_step": args.prefill_step,
        },
        "comparable": all(gates.values()),
        "gates": gates,
        "results": results,
        "pipeline_elapsed_gain_percent": (serial_seconds / pipeline_seconds - 1.0)
        * 100.0,
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): _sha256(path)
            for path in _source_paths()
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in i061._model_inputs(args.model)
        },
    }


def compare(serial_path: Path, pipeline_path: Path) -> dict[str, Any]:
    serial = json.loads(serial_path.read_text())
    pipeline = json.loads(pipeline_path.read_text())
    serial_result = serial["result"]
    pipeline_result = pipeline["result"]
    gates = {
        "model_inputs_equal": serial["model_input_sha256"] == pipeline["model_input_sha256"],
        "settings_equal": serial["settings"] == pipeline["settings"],
        **_comparison_gates(serial_result, pipeline_result),
        "swap_non_growth": (
            serial["memory"]["swap_after_bytes"] <= serial["memory"]["swap_before_bytes"]
            and pipeline["memory"]["swap_after_bytes"] <= pipeline["memory"]["swap_before_bytes"]
        ),
    }
    serial_seconds = float(serial_result["decode_seconds"])
    pipeline_seconds = float(pipeline_result["decode_seconds"])
    return {
        "schema_version": 1,
        "serial_record": str(serial_path.relative_to(PROJECT_ROOT)),
        "pipeline_record": str(pipeline_path.relative_to(PROJECT_ROOT)),
        "comparable": all(gates.values()),
        "gates": gates,
        "output_tokens": len(serial_result["output_token_ids"]),
        "serial_output_tokens_per_second": serial_result["output_tokens_per_second"],
        "pipeline_output_tokens_per_second": pipeline_result["output_tokens_per_second"],
        "pipeline_decode_change_percent": (
            pipeline_seconds / serial_seconds - 1.0
        )
        * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile MLX-LM serial versus lookahead-pipelined raw decode."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--variant", choices=("serial", "pipeline"), required=True)
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
    pair_parser = subparsers.add_parser("pair")
    pair_parser.add_argument(
        "--order",
        nargs=2,
        choices=("serial", "pipeline"),
        required=True,
    )
    pair_parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    pair_parser.add_argument("--context-words", type=int, default=128)
    pair_parser.add_argument("--max-tokens", type=int, default=256)
    pair_parser.add_argument("--warmup-tokens", type=int, default=32)
    pair_parser.add_argument("--prefill-step", type=int, default=1024)
    pair_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--serial", type=Path, required=True)
    compare_parser.add_argument("--pipeline", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "compare":
        payload = compare(args.serial.resolve(), args.pipeline.resolve())
    else:
        args.model = args.model.resolve()
        if min(args.context_words, args.max_tokens, args.warmup_tokens, args.prefill_step) < 1:
            raise ValueError("context, token, warmup, and prefill values must be positive")
        if args.command == "pair":
            if len(set(args.order)) != 2:
                raise ValueError("pair order must contain serial and pipeline exactly once")
            payload = run_pair(args)
        else:
            payload = run_variant(args)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
