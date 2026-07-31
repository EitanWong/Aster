#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
BASE_DIR = ARTIFACT_DIR.parent / "ITER-20260717-051-batch-sampler-sync"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import sampling_benchmark as sampling  # noqa: E402

from aster.inference.model_runner import DecodeResult, DecodeWorkItem, ModelRunner  # noqa: E402

base = sampling.base


def _source_paths(config: Path) -> tuple[Path, ...]:
    return (
        PROJECT_ROOT / "aster/core/config.py",
        PROJECT_ROOT / "aster/inference/model_runner.py",
        PROJECT_ROOT / "aster/inference/constrained/json_schema_processor.py",
        config,
        Path(__file__).resolve(),
        BASE_DIR / "sampling_benchmark.py",
        ARTIFACT_DIR.parent / "ITER-20260717-050-decode-cache-sync/benchmark.py",
    )


def _schema_valid(text: str) -> bool:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"answer", "score"}
        and isinstance(value["answer"], str)
        and isinstance(value["score"], int)
        and not isinstance(value["score"], bool)
    )


@dataclass
class _Lane:
    request_id: str
    prompt_tokens: list[int]
    prompt_cache: Any
    input_token: int
    sampler: Any
    detokenizer: Any
    stop_token_ids: frozenset[int]
    logits_processors: tuple[Any, ...]
    max_tokens: int
    output_tokens: list[int] = field(default_factory=list)
    text_segments: list[str] = field(default_factory=list)
    finish_reason: str | None = None


def _prepare_lane(
    runner: ModelRunner,
    *,
    lane_index: int,
    max_tokens: int,
    prefill_step: int,
) -> _Lane:
    request = sampling._request_for_lane(
        workload="structured",
        lane_index=lane_index,
        prompt=(
            "Return a compact object with an answer string and integer score. "
            f"Use score {lane_index + 1}."
        ),
        max_tokens=max_tokens,
    )
    prepared = runner.encode_request(request)
    target = len(prepared.prompt_tokens) - 1
    prompt_cache = None
    cache_token_count = 0
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
    return _Lane(
        request_id=f"structured-stop-lane-{lane_index}",
        prompt_tokens=prepared.prompt_tokens,
        prompt_cache=decode.prompt_cache,
        input_token=decode.next_input_token,
        sampler=decode.sampler,
        detokenizer=decode.detokenizer,
        stop_token_ids=decode.stop_token_ids,
        logits_processors=decode.logits_processors,
        max_tokens=max_tokens,
    )


def _work_item(lane: _Lane) -> DecodeWorkItem:
    processor_tokens = lane.prompt_tokens + lane.output_tokens
    if processor_tokens and processor_tokens[-1] == lane.input_token:
        processor_tokens = processor_tokens[:-1]
    return DecodeWorkItem(
        prompt_cache=lane.prompt_cache,
        input_token=lane.input_token,
        sampler=lane.sampler,
        detokenizer=lane.detokenizer,
        stop_token_ids=lane.stop_token_ids,
        logits_processors=lane.logits_processors,
        logits_processor_tokens=processor_tokens,
        completion_tokens=len(lane.output_tokens),
        max_tokens=lane.max_tokens,
        request_id=lane.request_id,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    settings = base._settings(
        args.config,
        args.model,
        cache_kind="native",
        batch_size=args.batch_size,
    )
    runner = ModelRunner(settings)
    runner.warmup()
    mx = runner._mx
    if mx is None:
        raise RuntimeError("MLX failed to load")
    base._warmup(runner, tokens=args.warmup_tokens, prefill_step=args.prefill_step)
    lanes = [
        _prepare_lane(
            runner,
            lane_index=index,
            max_tokens=args.max_tokens,
            prefill_step=args.prefill_step,
        )
        for index in range(args.batch_size)
    ]
    swap_before = int(psutil.swap_memory().used)
    membership_sizes: list[int] = []
    for _step in range(args.max_tokens):
        active = [lane for lane in lanes if lane.finish_reason is None]
        if not active:
            break
        membership_sizes.append(len(active))
        results = runner.decode_batch_step([_work_item(lane) for lane in active])
        for lane, result in zip(active, results, strict=True):
            if not isinstance(result, DecodeResult) or result.token_id is None:
                raise RuntimeError(f"structured decode failed: {result!r}")
            lane.prompt_cache = result.prompt_cache
            lane.input_token = result.token_id
            lane.output_tokens.append(result.token_id)
            lane.text_segments.append(result.text)
            if result.finish_reason is not None:
                lane.finish_reason = result.finish_reason

    texts: list[str] = []
    for lane in lanes:
        lane.text_segments.append(runner.finalize_detokenizer(lane.detokenizer))
        texts.append("".join(lane.text_segments))
    payload: dict[str, object] = {
        "schema_version": 1,
        "pid": os.getpid(),
        "run_id": args.run_id,
        "batch_size": args.batch_size,
        "membership_sizes": membership_sizes,
        "all_schema_valid": all(_schema_valid(text) for text in texts),
        "all_stopped_before_limit": all(
            lane.finish_reason == "stop" and len(lane.output_tokens) < args.max_tokens
            for lane in lanes
        ),
        "lanes": [
            {
                "request_id": lane.request_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "completion_tokens": len(lane.output_tokens),
                "finish_reason": lane.finish_reason,
                "schema_valid": _schema_valid(text),
            }
            for lane, text in zip(lanes, texts, strict=True)
        ],
        "memory": {
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "rss_bytes": int(psutil.Process().memory_info().rss),
            "swap_before_bytes": swap_before,
            "swap_after_bytes": int(psutil.swap_memory().used),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): base._sha256(path)
            for path in _source_paths(args.config)
        },
        "model_input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): base._sha256(path)
            for path in base._model_inputs(args.model)
        },
    }
    for lane in lanes:
        base._release_lane(runner, lane)  # type: ignore[arg-type]
    mx.clear_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit",
    )
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--prefill-step", type=int, default=1024)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    payload = run(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
