#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import psutil

from aster.core.config import load_settings
from aster.inference.contracts import InferenceRequest
from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _settings(config_path: Path, model_path: Path):
    base = load_settings(str(config_path))
    model = base.model.model_copy(
        update={
            "name": "Qwen3.5-0.8B-4bit",
            "path": str(model_path),
            "context_length": max(base.model.context_length, 32_768),
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
            "speculative": base.speculative.model_copy(
                update={"enabled": False, "max_draft_tokens": 0}
            ),
            "embeddings": base.embeddings.model_copy(update={"enabled": False}),
        }
    )


def _prompt(context_words: int) -> str:
    facts = " ".join(f"fact{index % 97:02d}" for index in range(max(context_words, 1)))
    return (
        "System: You are a deterministic local inference benchmark. "
        "Preserve every stated fact and answer with a concise technical summary.\n"
        f"{facts}\nAssistant:"
    )


def _request(prompt: str, max_tokens: int) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        enable_thinking=False,
        trace_id="iter061-preflight",
    )


def _aster_generate(
    runner: ModelRunner,
    request: InferenceRequest,
    *,
    prefill_step: int,
) -> dict[str, Any]:
    prepared = runner.encode_request(request)
    prompt_tokens = prepared.prompt_tokens
    if len(prompt_tokens) < 2:
        raise ValueError("preflight prompt must contain at least two tokens")
    prompt_cache = None
    cache_token_count = 0
    target = len(prompt_tokens) - 1
    prefill_started = time.perf_counter()
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
    prefill_seconds = time.perf_counter() - prefill_started
    input_token = decode.next_input_token
    output_tokens: list[int] = []
    text_segments: list[str] = []
    decode_started = time.perf_counter()
    finish_reason = "length"
    for _step in range(request.max_tokens):
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
            raise RuntimeError(f"Aster decode failed: {result!r}")
        decode.prompt_cache = result.prompt_cache
        input_token = result.token_id
        output_tokens.append(result.token_id)
        text_segments.append(result.text)
        if result.finish_reason is not None:
            finish_reason = result.finish_reason
            break
    decode_seconds = time.perf_counter() - decode_started
    text_segments.append(runner.finalize_detokenizer(decode.detokenizer))
    return {
        "prompt_token_ids": prompt_tokens,
        "output_token_ids": output_tokens,
        "text_sha256": hashlib.sha256("".join(text_segments).encode()).hexdigest(),
        "finish_reason": finish_reason,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
    }


def _mlx_lm_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token or ""
    )
    prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    responses: list[Any] = []
    started = time.perf_counter()
    first_response_seconds: float | None = None
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0)
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if first_response_seconds is None:
            first_response_seconds = time.perf_counter() - started
        responses.append(response)
    elapsed = time.perf_counter() - started
    if not responses:
        raise RuntimeError("MLX-LM produced no generation responses")
    final = responses[-1]
    text = "".join(str(response.text) for response in responses)
    prefill_seconds = 0.0
    if getattr(final, "prompt_tps", 0.0):
        prefill_seconds = len(prompt_tokens) / float(final.prompt_tps)
    return {
        "prompt_token_ids": prompt_tokens,
        "output_token_ids": [int(response.token) for response in responses],
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": str(final.finish_reason),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": max(0.0, elapsed - prefill_seconds),
        "first_response_seconds": first_response_seconds,
    }


def _source_paths(engine: str) -> tuple[Path, ...]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
    ]
    if engine == "mlx-lm":
        import mlx_lm

        path = Path(inspect.getsourcefile(mlx_lm.stream_generate) or "")
        if path.is_file():
            paths.append(path)
    return tuple(paths)


def run_engine(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    if args.engine == "aster":
        runner = ModelRunner(_settings(args.config, args.model))
        runner.warmup()
        _aster_generate(runner, _request(prompt, args.warmup_tokens), prefill_step=args.prefill_step)
        result = _aster_generate(runner, _request(prompt, args.max_tokens), prefill_step=args.prefill_step)
    else:
        from mlx_lm import load

        model, tokenizer = load(str(args.model), lazy=False)
        _mlx_lm_generate(model, tokenizer, prompt, max_tokens=args.warmup_tokens)
        result = _mlx_lm_generate(model, tokenizer, prompt, max_tokens=args.max_tokens)
    decode_seconds = float(result["decode_seconds"])
    output_tokens = result["output_token_ids"]
    return {
        "schema_version": 1,
        "engine": args.engine,
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
            "output_tokens_per_second": len(output_tokens) / decode_seconds if decode_seconds else 0.0,
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": int(psutil.swap_memory().used),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): _sha256(path)
            for path in _source_paths(args.engine)
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in _model_inputs(args.model)
        },
    }


def compare(aster_path: Path, reference_path: Path) -> dict[str, Any]:
    aster = json.loads(aster_path.read_text())
    reference = json.loads(reference_path.read_text())
    aster_result = aster["result"]
    reference_result = reference["result"]
    gates = {
        "model_inputs_equal": aster["model_input_sha256"] == reference["model_input_sha256"],
        "settings_equal": aster["settings"] == reference["settings"],
        "prompt_tokens_equal": aster_result["prompt_token_ids"] == reference_result["prompt_token_ids"],
        "output_tokens_equal": aster_result["output_token_ids"] == reference_result["output_token_ids"],
        "text_equal": aster_result["text_sha256"] == reference_result["text_sha256"],
        "finish_reason_equal": aster_result["finish_reason"] == reference_result["finish_reason"],
        "swap_non_growth": (
            aster["memory"]["swap_after_bytes"] <= aster["memory"]["swap_before_bytes"]
            and reference["memory"]["swap_after_bytes"] <= reference["memory"]["swap_before_bytes"]
        ),
    }
    return {
        "schema_version": 1,
        "aster_record": str(aster_path.relative_to(PROJECT_ROOT)),
        "reference_record": str(reference_path.relative_to(PROJECT_ROOT)),
        "comparable": all(gates.values()),
        "gates": gates,
        "output_tokens": len(aster_result["output_token_ids"]),
        "aster_output_tokens_per_second": aster_result["output_tokens_per_second"],
        "reference_output_tokens_per_second": reference_result["output_tokens_per_second"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--engine", choices=("aster", "mlx-lm"), required=True)
    run_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    run_parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    run_parser.add_argument("--context-words", type=int, default=128)
    run_parser.add_argument("--max-tokens", type=int, default=64)
    run_parser.add_argument("--warmup-tokens", type=int, default=8)
    run_parser.add_argument("--prefill-step", type=int, default=1024)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--aster", type=Path, required=True)
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "compare":
        payload = compare(args.aster.resolve(), args.reference.resolve())
    else:
        args.config = args.config.resolve()
        args.model = args.model.resolve()
        if min(args.context_words, args.max_tokens, args.warmup_tokens, args.prefill_step) < 1:
            raise ValueError("context, token, warmup, and prefill values must be positive")
        payload = run_engine(args)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
