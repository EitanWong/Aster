# Iteration 071: Public Arrival/Load Baseline

- **Date:** 2026-07-29
- **Phase:** completed (baseline-only)
- **Scope:** establish a public-source, Aster scheduler baseline before a
  scheduler, prefill, or cache production candidate. No runtime behavior change
  is in scope.

## Objective

I070 established that another private lazy-decode split would perturb the timed
path. Measure the observable request lifecycle instead: controlled B1/B4/B8
arrivals, one staggered long-prefill cohort, one shared-prefix cohort, and one
cancellation cohort, all reconstructed from the pinned MT-Bench/LongBench
public sources.

## Hypothesis

The next actionable bottleneck, if any, is visible in request-level queue wait,
TTFT, p50/p95 end-to-end latency, throughput, cache reuse, and memory behavior
under controlled arrivals. A single-request private decode label is insufficient
to select a scheduler or cache change.

## Reference Design

- vLLM's `Scheduler.schedule` uses one token budget across running and waiting
  work, schedules active requests first, and admits waiting requests only within
  request/KV limits (`examples/vllm/vllm/v1/core/sched/scheduler.py:424`).
- SGLang's `get_new_batch_prefill` checks waiting-queue and pool capacity before
  creating a prefill batch (`examples/sglang/python/sglang/srt/managers/scheduler.py:2847`).

These are design references for workload dimensions and telemetry, not code to
copy or an argument for a policy change.

## Predeclared Gates

1. Public prompts are reconstructed by the existing locked-source resolver;
   manifests retain IDs and hashes, not copied prompt text.
2. Each arrival schedule is deterministic, monotonic, and records submission,
   admission, first-token, completion, cancellation, cache, RSS, and swap
   observations.
3. Each baseline case validates deterministic tokens/text/finish where its
   request completes, and verifies cancellation cleanup where it does not.
4. No scheduler, prefill, prefix, compression, SIMD/Metal, tokenizer, or
   speculative candidate is admitted by the baseline itself.

## First Actions

1. Audit the existing engine benchmark and public-workload resolver for a
   minimal source-bound arrival manifest and request telemetry seam.
2. Add focused unit tests for schedule validation and source reconstruction.
3. Collect a fixed Aster-only baseline before choosing one production candidate.

## Implementation

- Added `scripts/dev/public_arrival_load.py`. It creates deterministic arrival
  plans from public workload IDs only, resolves prompt text through the pinned
  public resolver at execution, and records request lifecycle, cache, RSS, and
  swap observations without placing prompt text in a manifest.
- Added terminal-only token-ID and response-text hashes plus finish reason to
  `InferenceEngine.recent_request_timelines`. They are calculated only after a
  request is terminal, keeping the decode step free of this evidence work.
- Added focused coverage for simultaneous, staggered prefill, shared-prefix,
  and cancellation dependency release paths.

## Results

All cases used Qwen3.5-9B, greedy 8-token completion, the locked
`cross-engine-core` source, a cold in-memory prefix store, and the manual
runtime. The compact, hash-bound result is
`artifacts/ITER-20260729-071-public-arrival-load-baseline/public-arrival-load-baseline.json`.

- B1 completed `mt-bench:81:turn-1` in `1.284s`: TTFT `0.955s`, aggregate
  generation `16.985 tok/s`, and zero swap growth.
- B4 completed 32 tokens in `2.028s` at `51.247 tok/s`; all 9 decode batches
  succeeded, average decode batch size was `2.909`, and swap was unchanged.
- B8 completed 64 tokens in `3.539s` at `53.410 tok/s`, with no decode
  fallback. Its tail TTFT reached `2.856s` and swap grew `430,440,448` bytes,
  so it is a pressure observation rather than a throughput admission.
- A 10,334-token QMSUM primary request had TTFT `14.597s`. Its exact-prefix
  replay reused 10,333 tokens, retained exact token/text/finish parity, and
  reduced TTFT to `0.169s`. The case grew swap by `933,429,248` bytes.
- The long-request cancellation was accepted at the first 1,024-token
  checkpoint, left zero active estimated bytes/pending requests, retained one
  85,065,728-byte checkpoint, completed the follow-up, and had zero swap
  growth.
- Two staggered QMSUM-plus-MT-Bench runs each recorded one prefill-yield
  rotation. The short request's queue wait was `1.255s`/`1.190s`; its
  decode duration was `9.346s`/`9.614s` and end-to-end latency
  `10.829s`/`11.034s`. A 1,024-token long-prefill block is therefore a
  repeatable source of interactive decode delay.

## Decision

I071 admits the public-source arrival/load harness and its baseline evidence.
It enables no runtime policy. I072 evaluates one opt-in, reversible candidate:
cap a continuing prefill chunk at the existing 512-token pressure budget when
decode work is active, then compare it against the unchanged default on the
same staggered source workload.

## Verification

- `uv run pytest -q tests/test_public_arrival_load.py` -> `5 passed`
- `uv run pytest -q tests/test_engine_runtime.py -k batches_decode_steps_for_concurrent_requests` -> `1 passed, 53 deselected`
- `uv run ruff check scripts/dev/public_arrival_load.py aster/inference/engine.py aster/inference/request_state.py tests/test_public_arrival_load.py tests/test_engine_runtime.py` -> passed
- `uv run python scripts/dev/check_loop_workspace.py --strict` -> warning-only;
  inherited foreign artifacts and seven generated caches remain explicit.
