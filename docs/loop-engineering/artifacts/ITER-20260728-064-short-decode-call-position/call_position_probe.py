#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
I063_ARTIFACT_DIR = (
    ARTIFACT_DIR.parent / "ITER-20260728-063-short-decode-measurement-stability"
)
if str(I063_ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(I063_ARTIFACT_DIR))

import state_probe as state  # noqa: E402
import warmup_order_probe as warmup  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distinct_order(order: list[str]) -> None:
    if len(order) != 2 or set(order) != {"serial", "pipeline"}:
        raise SystemExit("--warmup-order must contain serial and pipeline exactly once")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure same-variant short-decode call position with fresh prompt caches."
    )
    parser.add_argument("--variant", choices=("serial", "pipeline"), required=True)
    parser.add_argument("--warmup-order", nargs=2, choices=("serial", "pipeline"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
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
    _distinct_order(args.warmup_order)
    if args.replicate < 1:
        raise SystemExit("--replicate must be positive")
    if min(args.context_words, args.max_tokens, args.warmup_tokens, args.prefill_step) < 1:
        raise SystemExit("context, token, warmup, and prefill values must be positive")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.sample_utils import make_sampler

    generate = __import__("importlib").import_module("mlx_lm.generate")
    args.model = args.model.resolve()
    prompt = state.pipeline.i061._prompt(args.context_words)
    swap_before = int(psutil.swap_memory().used)
    rss_before = int(psutil.Process().memory_info().rss)
    model, tokenizer = load(str(args.model), lazy=False)
    prompt_tokens = state.pipeline._prompt_tokens(tokenizer, prompt)
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

    host_before_warmup = warmup._host_state(mx)
    warmups = [
        {
            "variant": variant,
            "before": warmup._host_state(mx),
            "result": state.pipeline._generate(
                variant,
                max_tokens=args.warmup_tokens,
                **common,
            ),
            "after": warmup._host_state(mx),
        }
        for variant in args.warmup_order
    ]
    host_after_warmup = warmup._host_state(mx)
    observations = []
    for position in (1, 2):
        observations.append(
            {
                "call_position": position,
                "host_before": warmup._host_state(mx),
                "run": state._run_variant(
                    args.variant,
                    mx=mx,
                    common=common,
                    max_tokens=args.max_tokens,
                    clear_before=False,
                    collect_between=False,
                ),
                "host_after": warmup._host_state(mx),
            }
        )
    host_after_measurement = warmup._host_state(mx)
    first = observations[0]["run"]["result"]
    second = observations[1]["run"]["result"]
    warmup_serial = next(item["result"] for item in warmups if item["variant"] == "serial")
    warmup_pipelined = next(item["result"] for item in warmups if item["variant"] == "pipeline")
    gates = state.pipeline._comparison_gates(first, second)
    gates["warmup_output_tokens_equal"] = (
        warmup_serial["output_token_ids"] == warmup_pipelined["output_token_ids"]
    )
    gates["warmup_finish_reason_equal"] = (
        warmup_serial["finish_reason"] == warmup_pipelined["finish_reason"]
    )
    swap_after = int(psutil.swap_memory().used)
    gates["swap_non_growth"] = swap_after <= swap_before
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "replicate": args.replicate,
        "variant": args.variant,
        "warmup_order": args.warmup_order,
        "warmup_terminal_variant": args.warmup_order[-1],
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
        "host_before_warmup": host_before_warmup,
        "warmups": warmups,
        "host_after_warmup": host_after_warmup,
        "observations": observations,
        "host_after_measurement": host_after_measurement,
        "second_vs_first_elapsed_gain_percent": (
            float(first["decode_seconds"]) / float(second["decode_seconds"]) - 1.0
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
                I063_ARTIFACT_DIR / "state_probe.py",
                I063_ARTIFACT_DIR / "warmup_order_probe.py",
                state.I062_ARTIFACT_DIR / "mlx_pipeline_profile.py",
            )
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in state.pipeline.i061._model_inputs(args.model)
        },
    }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
