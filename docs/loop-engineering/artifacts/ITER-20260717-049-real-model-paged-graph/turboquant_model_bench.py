#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _command(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _swap_used_bytes() -> int | None:
    output = _command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _model_file_hashes(model_path: Path) -> dict[str, str]:
    runtime_suffixes = {".json", ".jinja", ".model", ".safetensors", ".txt"}
    return {
        path.relative_to(model_path).as_posix(): _sha256(path)
        for path in sorted(model_path.rglob("*"))
        if path.is_file()
        and path.suffix in runtime_suffixes
        and not any(part.startswith(".") for part in path.relative_to(model_path).parts)
    }


def _cache_bytes(cache: list[Any]) -> int:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return sum(
        int(value.nbytes)
        for _, value in tree_flatten([item.state for item in cache])
        if isinstance(value, mx.array)
    )


def _eval_cache(mx: Any, cache: list[Any]) -> None:
    mx.eval([item.state for item in cache])


def _encode_required_prefix(tokenizer: Any, corpus: str, required: int) -> list[int]:
    character_limit = min(len(corpus), max(4096, required * 8))
    while True:
        token_ids = tokenizer.encode(corpus[:character_limit])
        if len(token_ids) >= required or character_limit == len(corpus):
            return token_ids
        character_limit = min(len(corpus), character_limit * 2)


def _prefill(
    mx: Any,
    model: Any,
    token_ids: list[int],
    cache: list[Any],
    *,
    step: int,
) -> tuple[Any, float]:
    logits = None
    started = time.perf_counter()
    for start in range(0, len(token_ids), step):
        chunk = mx.array(token_ids[start : start + step], dtype=mx.int32)[None]
        logits = model(chunk, cache=cache)[:, -1, :]
        mx.eval(logits)
        _eval_cache(mx, cache)
        mx.clear_cache()
    if logits is None:
        raise ValueError("prefill token list is empty")
    return logits, time.perf_counter() - started


def _convert_cache(cache: list[Any], *, bits: float, seed: int) -> tuple[list[Any], int]:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.turboquant import TurboQuantKVCache

    converted: list[Any] = []
    converted_layers = 0
    for item in cache:
        if isinstance(item, KVCache):
            converted.append(TurboQuantKVCache.from_cache(item, bits=bits, seed=seed))
            converted_layers += 1
        else:
            converted.append(item)
    _eval_cache(mx, converted)
    return converted, converted_layers


def _prepare(
    mx: Any,
    model: Any,
    make_prompt_cache: Any,
    token_ids: list[int],
    *,
    variant: str,
    prefill_step: int,
    bits: float,
    seed: int,
) -> tuple[Any, list[Any], dict[str, float | int]]:
    cache = make_prompt_cache(model)
    logits, prefill_seconds = _prefill(
        mx, model, token_ids, cache, step=prefill_step
    )
    fp16_cache_bytes = _cache_bytes(cache)
    conversion_seconds = 0.0
    converted_layers = 0
    if variant == "turboquant":
        started = time.perf_counter()
        cache, converted_layers = _convert_cache(cache, bits=bits, seed=seed)
        conversion_seconds = time.perf_counter() - started
    cache_bytes = _cache_bytes(cache)
    return logits, cache, {
        "prefill_seconds": prefill_seconds,
        "conversion_seconds": conversion_seconds,
        "fp16_cache_bytes": fp16_cache_bytes,
        "active_cache_bytes": cache_bytes,
        "compression_ratio": fp16_cache_bytes / cache_bytes,
        "converted_full_attention_layers": converted_layers,
    }


def _greedy(
    mx: Any,
    model: Any,
    logits: Any,
    cache: list[Any],
    *,
    tokens: int,
) -> dict[str, object]:
    token_ids: list[int] = []
    decode_step_ms: list[float] = []
    started = time.perf_counter()
    first_started = time.perf_counter_ns()
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    token_id = int(token.item())
    token_ids.append(token_id)
    first_token_ms = (time.perf_counter_ns() - first_started) / 1_000_000

    for _ in range(1, tokens):
        step_started = time.perf_counter_ns()
        logits = model(mx.array([[token_id]], dtype=mx.int32), cache=cache)[
            :, -1, :
        ]
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        token_id = int(token.item())
        token_ids.append(token_id)
        decode_step_ms.append((time.perf_counter_ns() - step_started) / 1_000_000)
    elapsed = time.perf_counter() - started
    return {
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "elapsed_seconds": elapsed,
        "generation_tps": tokens / elapsed,
        "first_token_ms": first_token_ms,
        "decode_median_ms": statistics.median(decode_step_ms)
        if decode_step_ms
        else 0.0,
        "decode_step_ms": decode_step_ms,
        "cache_bytes_after": _cache_bytes(cache),
    }


def _teacher_forced(
    mx: Any,
    model: Any,
    logits: Any,
    cache: list[Any],
    target_ids: list[int],
) -> dict[str, object]:
    top1_ids: list[int] = []
    target_logprobs: list[float] = []
    started = time.perf_counter()
    for index, target_id in enumerate(target_ids):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top1 = mx.argmax(logits, axis=-1)
        target_logprob = logprobs[0, target_id]
        mx.eval(top1, target_logprob)
        top1_ids.append(int(top1.item()))
        target_logprobs.append(float(target_logprob.item()))
        if index + 1 < len(target_ids):
            logits = model(mx.array([[target_id]], dtype=mx.int32), cache=cache)[
                :, -1, :
            ]
    elapsed = time.perf_counter() - started
    negative_log_likelihood = -sum(target_logprobs) / len(target_logprobs)
    return {
        "target_ids": target_ids,
        "top1_ids": top1_ids,
        "target_logprobs": target_logprobs,
        "negative_log_likelihood": negative_log_likelihood,
        "perplexity": math.exp(negative_log_likelihood),
        "elapsed_seconds": elapsed,
        "tokens_per_second": len(target_ids) / elapsed,
        "cache_bytes_after": _cache_bytes(cache),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if str(PROJECT_ROOT / "examples/omlx") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "examples/omlx"))

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_vlm.turboquant import TurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    corpus = args.corpus.read_text()
    corpus_sha256 = _sha256(args.corpus)
    model, tokenizer = load(str(args.model))
    apply_turboquant_attention_patch()
    required = args.offset + args.context_tokens + args.teacher_tokens
    all_token_ids = _encode_required_prefix(tokenizer, corpus, required)
    if len(all_token_ids) < required:
        raise ValueError(f"corpus has {len(all_token_ids)} tokens, needs {required}")
    prompt_ids = all_token_ids[args.offset : args.offset + args.context_tokens]
    target_ids = all_token_ids[
        args.offset + args.context_tokens : required
    ]
    mx.random.seed(args.seed)

    warm_logits, warm_cache, _ = _prepare(
        mx,
        model,
        make_prompt_cache,
        prompt_ids,
        variant=args.variant,
        prefill_step=args.prefill_step,
        bits=args.bits,
        seed=args.seed,
    )
    _greedy(mx, model, warm_logits, warm_cache, tokens=2)
    del warm_logits, warm_cache
    gc.collect()
    mx.clear_cache()
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    swap_before = _swap_used_bytes()
    thermal_before = _command("pmset", "-g", "therm")

    greedy_logits, greedy_cache, greedy_prepare = _prepare(
        mx,
        model,
        make_prompt_cache,
        prompt_ids,
        variant=args.variant,
        prefill_step=args.prefill_step,
        bits=args.bits,
        seed=args.seed,
    )
    greedy = _greedy(
        mx,
        model,
        greedy_logits,
        greedy_cache,
        tokens=args.generation_tokens,
    )
    del greedy_logits, greedy_cache
    gc.collect()
    mx.clear_cache()

    teacher_logits, teacher_cache, teacher_prepare = _prepare(
        mx,
        model,
        make_prompt_cache,
        prompt_ids,
        variant=args.variant,
        prefill_step=args.prefill_step,
        bits=args.bits,
        seed=args.seed,
    )
    teacher = _teacher_forced(
        mx, model, teacher_logits, teacher_cache, target_ids
    )
    thermal_after = _command("pmset", "-g", "therm")
    swap_after = _swap_used_bytes()
    runtime_source = Path(inspect.getsourcefile(TurboQuantKVCache) or "")
    model_files = _model_file_hashes(args.model)
    return {
        "schema_version": 1,
        "benchmark": "turboquant_qwen35_model",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "pid": os.getpid(),
        "variant": args.variant,
        "context_tokens": args.context_tokens,
        "teacher_tokens": args.teacher_tokens,
        "generation_tokens": args.generation_tokens,
        "bits": args.bits,
        "seed": args.seed,
        "dataset": {
            "path": str(args.corpus.resolve()),
            "sha256": corpus_sha256,
            "source": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
            "offset": args.offset,
            "prompt_ids_sha256": hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "model": {
            "path": str(args.model.resolve()),
            "files": model_files,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "mlx_vlm": _version("mlx-vlm"),
            "cpu": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        },
        "provenance": {
            "aster_commit": _command("git", "rev-parse", "HEAD", cwd=PROJECT_ROOT),
            "omlx_commit": _command(
                "git", "rev-parse", "HEAD", cwd=PROJECT_ROOT / "examples/omlx"
            ),
            "benchmark_sha256": _sha256(Path(__file__)),
            "omlx_patch_sha256": _sha256(
                PROJECT_ROOT / "examples/omlx/omlx/patches/turboquant_attention.py"
            ),
            "mlx_vlm_turboquant_sha256": _sha256(runtime_source),
        },
        "greedy_prepare": greedy_prepare,
        "greedy": greedy,
        "teacher_prepare": teacher_prepare,
        "teacher_forced": teacher,
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "swap_used_before_bytes": swap_before,
        "swap_used_after_bytes": swap_after,
        "swap_delta_bytes": (
            None
            if swap_before is None or swap_after is None
            else swap_after - swap_before
        ),
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--variant", choices=("fp16", "turboquant"), required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--teacher-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=1024)
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=49_217)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.context_tokens,
        args.teacher_tokens,
        args.generation_tokens,
        args.prefill_step,
        args.run_id,
    ) < 1:
        raise ValueError("token counts, prefill step, and run id must be positive")
    if not args.model.is_dir() or not args.corpus.is_file():
        raise FileNotFoundError("model or corpus is missing")
    _write_json(args.output.resolve(), run(args))


if __name__ == "__main__":
    main()
