# Per-Profile BatchGenerator Lanes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in bounded multi-lane BatchGenerator scheduler that restores heterogeneous progress without mixing incompatible cache profiles.

**Architecture:** Keep one engine loop and shared model/tokenizer, but move BatchGenerator state, UID mappings, response polling, and cache ownership into `_BatchLane`. Use a profile-keyed admission policy and `engine.batch_generator_max_lanes`, defaulting to one lane. Validate the candidate against the existing Iteration 028 workload harness before considering any default change.

**Tech Stack:** Python 3.14, asyncio, Pydantic settings, `mlx_lm.BatchGenerator`, pytest, Ruff, existing Aster benchmark harness.

## Global Constraints

- Preserve exact greedy token/text parity; hybrid cache profiles must never be merged.
- Keep all MLX calls on the existing single engine-loop owner.
- Keep `engine_type=manual` as the production default and do not enable `BatchGeneratorRuntimeKernel`.
- Do not modify or stage the user-owned `.codegraph/.gitignore` change.
- No new external dependency.

### Task 1: Add the lane configuration and pure profile model

**Files:**
- Modify: `aster/core/config.py`
- Test: `tests/test_config.py`
- Test: `tests/test_batched_engine.py`

- [ ] Write a test that `EngineSettings()` defaults `batch_generator_max_lanes` to `1`, accepts `2`, and rejects `0`.
- [ ] Run the focused tests and observe failure because the field is absent.
- [ ] Add a bounded integer field with default `1` and a profile/lane dataclass in `aster/inference/batched_engine.py`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Make request ownership lane-local

**Files:**
- Modify: `aster/inference/batched_engine.py`
- Test: `tests/test_batched_engine.py`

- [ ] Add failing tests proving two profiles can select two lanes, while a third request is deferred at the configured lane limit.
- [ ] Run the tests and verify the failure is caused by the current single-generator admission path.
- [ ] Add `_BatchLane` with a generator, profile, request set, and per-lane UID maps; add a deterministic lane-selection helper.
- [ ] Run the focused tests and confirm admission and deferral pass.

### Task 3: Route polling, finish, prefix extraction, and aborts by lane

**Files:**
- Modify: `aster/inference/batched_engine.py`
- Test: `tests/test_batched_engine.py`

- [ ] Add failing tests for lane-local `remove`, prompt-boundary `extract_cache`, finish response routing, and cancellation cleanup.
- [ ] Run the tests and confirm they fail against global `_batch_generator` ownership.
- [ ] Thread the owning lane through `_process_prompt_responses`, `_process_responses`, `_finish_request`, and `_process_aborts`; close all lanes during stop/finally.
- [ ] Run focused tests, then the full suite.

### Task 4: Extend the benchmark for lane A/B comparison

**Files:**
- Modify: `scripts/dev/benchmark_batched_engine.py`
- Test: `tests/test_benchmark_batched_engine.py`

- [ ] Add a failing helper test for setting the lane limit independently of cache on/off.
- [ ] Add `--max-lanes` and record it in benchmark metadata without changing existing defaults.
- [ ] Run harness helper tests and a small 0.8B smoke with `--max-lanes 1` and `2`.

### Task 5: Verify, record, and decide

**Files:**
- Create: `docs/loop-engineering/iterations/ITER-20260714-029-per-profile-batch-generator-lanes.md`
- Create: `docs/loop-engineering/iterations/artifacts/ITER-20260714-029-per-profile-batch-generator-lanes/summary.json`
- Modify: `docs/loop-engineering/STATUS.md`
- Modify: `docs/loop-engineering/DECISIONS.md`
- Modify: `docs/reference/ENGINE_SCHEDULER.md`

- [ ] Run the matched 0.8B matrix with lane limits `1` and `2`, including structured output and cancellation.
- [ ] Compare hashes, errors, swap, p50/p95, warm elapsed, and peak memory.
- [ ] Keep default `1` unless the stated gates pass; record failures and rollback if they do not.
- [ ] Run full pytest, Ruff, compileall, lock, package, and diff checks.
- [ ] Commit implementation and records as separate conventional commits.
