#!/usr/bin/env python3
"""Trace a near-tie greedy token across active-cap decode cohort shapes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aster.inference.model_runner import ModelRunner  # noqa: E402
from scripts.dev import benchmark_active_cap_frontier as frontier  # noqa: E402
from scripts.dev import public_arrival_load as arrival  # noqa: E402

TARGET_REQUEST_ID = "public-arrival:mixed-short-3"
CANDIDATE_TOKEN_IDS = (364, 421, 8574)


async def run_diagnostic(cap: int) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    original_processors = ModelRunner._apply_logits_processors
    original_batch = ModelRunner._decode_batch
    original_single = ModelRunner._decode_single
    original_result = ModelRunner._decode_result

    def traced_processors(self: ModelRunner, logits: Any, *, item: Any) -> Any:
        processed = original_processors(self, logits, item=item)
        if item.request_id != TARGET_REQUEST_ID:
            return processed
        mx = self._mx
        assert mx is not None
        top_indices = mx.argpartition(processed[0], kth=-2)[-2:]
        top_values = processed[0][top_indices]
        candidate_indices = mx.array(CANDIDATE_TOKEN_IDS, dtype=mx.uint32)
        candidate_values = processed[0][candidate_indices]
        mx.eval(top_indices, top_values, candidate_values)
        top = sorted(
            zip(top_indices.tolist(), top_values.tolist(), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        trace.append(
            {
                "completion_tokens": item.completion_tokens,
                "input_token": item.input_token,
                "top": top,
                "top_margin": float(top[0][1] - top[1][1]),
                "candidate_logits": dict(
                    zip(
                        (str(token_id) for token_id in CANDIDATE_TOKEN_IDS),
                        candidate_values.tolist(),
                        strict=True,
                    )
                ),
                "selected_token": None,
            }
        )
        return processed

    def record_cohort(mode: str, items: list[Any]) -> None:
        target = next((item for item in items if item.request_id == TARGET_REQUEST_ID), None)
        if target is not None:
            cohorts.append(
                {
                    "mode": mode,
                    "completion_tokens": target.completion_tokens,
                    "request_ids": [item.request_id for item in items],
                }
            )

    def traced_batch(self: ModelRunner, items: list[Any]) -> Any:
        record_cohort("batch", items)
        return original_batch(self, items)

    def traced_single(self: ModelRunner, item: Any) -> Any:
        record_cohort("single", [item])
        return original_single(self, item)

    def traced_result(
        self: ModelRunner,
        *,
        item: Any,
        token: int,
        prompt_cache: Any,
        peak_memory_gb: float,
    ) -> Any:
        if item.request_id == TARGET_REQUEST_ID:
            matching = [
                entry for entry in trace if entry["completion_tokens"] == item.completion_tokens
            ]
            if len(matching) != 1:
                raise RuntimeError("target token trace is missing or duplicated")
            matching[0]["selected_token"] = token
        return original_result(
            self,
            item=item,
            token=token,
            prompt_cache=prompt_cache,
            peak_memory_gb=peak_memory_gb,
        )

    ModelRunner._apply_logits_processors = traced_processors
    ModelRunner._decode_batch = traced_batch
    ModelRunner._decode_single = traced_single
    ModelRunner._decode_result = traced_result
    try:
        args = argparse.Namespace(
            source_workload=frontier.DEFAULT_WORKLOAD,
            workload="mixed",
            config=PROJECT_ROOT / "configs/config.yaml",
            lock=arrival.public.DEFAULT_LOCK_PATH,
            data_root=arrival.public.DEFAULT_DATA_ROOT,
            timeout_seconds=180.0,
            cap=cap,
            sequence=100 + cap,
        )
        payload = await frontier._run_cell(args)
    finally:
        ModelRunner._apply_logits_processors = original_processors
        ModelRunner._decode_batch = original_batch
        ModelRunner._decode_single = original_single
        ModelRunner._decode_result = original_result

    event = next(event for event in payload["result"]["events"] if event["key"] == "mixed-short-3")
    return {
        "schema_version": 1,
        "kind": "active-cap-greedy-logit-diagnostic",
        "performance_measurement_valid": False,
        "cap": cap,
        "model": payload["experiment"]["compact"]["model"],
        "target_request_id": TARGET_REQUEST_ID,
        "candidate_token_ids": list(CANDIDATE_TOKEN_IDS),
        "output_token_ids_sha256": event["timeline"]["output_token_ids_sha256"],
        "text_sha256": event["response"]["text_sha256"],
        "request_contract_passed": payload["experiment"]["compact"]["contract_passed"],
        "cohorts": cohorts,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=frontier.CAPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(run_diagnostic(args.cap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
