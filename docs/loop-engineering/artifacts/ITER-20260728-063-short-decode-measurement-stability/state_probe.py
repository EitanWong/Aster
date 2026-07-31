#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
I062_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260728-062-short-decode-runtime-profile"
)
if str(I062_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(I062_ARTIFACT_DIR))

import mlx_pipeline_profile as pipeline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mlx_memory(mx: Any) -> dict[str, int | None]:
    def read(name: str) -> int | None:
        getter = getattr(mx, name, None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except Exception:
            return None

    return {
        "active_bytes": read("get_active_memory"),
        "peak_bytes": read("get_peak_memory"),
    }


def _run_variant(
    variant: str,
    *,
    mx: Any,
    common: dict[str, Any],
    max_tokens: int,
    clear_before: bool,
    collect_between: bool,
) -> dict[str, Any]:
    if clear_before:
        mx.clear_cache()
    if collect_between:
        gc.collect()
    before = _mlx_memory(mx)
    result = pipeline._generate(variant, max_tokens=max_tokens, **common)
    after = _mlx_memory(mx)
    mx.clear_cache()
    collected_objects = gc.collect() if collect_between else 0
    after_clear = _mlx_memory(mx)
    return {
        "variant": variant,
        "before": before,
        "after": after,
        "after_clear": after_clear,
        "collected_objects": collected_objects,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify MLX state around an exact serial/pipeline decode pair."
    )
    parser.add_argument("--order", nargs=2, choices=("serial", "pipeline"), required=True)
    parser.add_argument("--clear-before", action="store_true")
    parser.add_argument("--collect-between", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--context-words", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.order)) != 2:
        raise SystemExit("--order must contain serial and pipeline exactly once")
    if min(args.context_words, args.max_tokens, args.warmup_tokens, args.prefill_step) < 1:
        raise SystemExit("context, token, warmup, and prefill values must be positive")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler

    generate = __import__("importlib").import_module("mlx_lm.generate")
    args.model = args.model.resolve()
    prompt = pipeline.i061._prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    model, tokenizer = load(str(args.model), lazy=False)
    prompt_tokens = pipeline._prompt_tokens(tokenizer, prompt)
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
    # Prewarm both graph shapes. The declared order applies only to measurement.
    for variant in ("serial", "pipeline"):
        pipeline._generate(variant, max_tokens=args.warmup_tokens, **common)
    observations = [
        _run_variant(
            variant,
            mx=mx,
            common=common,
            max_tokens=args.max_tokens,
            clear_before=args.clear_before,
            collect_between=args.collect_between,
        )
        for variant in args.order
    ]
    serial = next(item["result"] for item in observations if item["variant"] == "serial")
    pipelined = next(item["result"] for item in observations if item["variant"] == "pipeline")
    gates = pipeline._comparison_gates(serial, pipelined)
    swap_after = int(psutil.swap_memory().used)
    gates["swap_non_growth"] = swap_after <= swap_before
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "order": args.order,
        "clear_before": args.clear_before,
        "collect_between": args.collect_between,
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
            "prefill_step": args.prefill_step,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
        },
        "comparable": all(gates.values()),
        "gates": gates,
        "observations": observations,
        "pipeline_elapsed_gain_percent": (
            float(serial["decode_seconds"]) / float(pipelined["decode_seconds"]) - 1.0
        )
        * 100.0,
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                I062_ARTIFACT_DIR / "mlx_pipeline_profile.py",
            )
        },
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
