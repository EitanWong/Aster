# LOOP ITERATION: ITER-20260713-001-admission-before-prefill

STATUS: SUPERSEDED (initial scenario result; later randomized A/B caused rollback)

START COMMIT: `25067b8`

END COMMIT: `32addf1`

## Focus

Reduce short-request queue delay when decode is idle but a long prompt is
currently being prefetched.

## Root Cause And Hypothesis

`InferenceEngine._scheduler_step()` executed one prefill step before draining
waiting submissions. A newly submitted short request therefore waited for an
extra prefill turn. Hypothesis: draining admissions immediately after decode,
then placing those new requests before existing prefill continuations, reduces
mixed-workload latency without changing decode correctness.

## References

- Aster: `aster/inference/engine.py` (`_scheduler_step`, `_step_prefill`, `_drain_submissions`).
- Aster tests: `tests/test_engine_runtime.py` scheduler and prefill fairness cases.
- Local reference: `examples/vllm-mlx/vllm_mlx/scheduler.py` and `mllm_scheduler.py` waiting-to-running admission paths.

## Changes

- Move admission before prefill while preserving cancellation and decode priority.
- Record the pre-admission prefill queue only when submissions are waiting, avoiding an unconditional hot-path set allocation.
- Prioritize requests admitted in the current scheduler turn ahead of existing prefill work.
- Add tests for scheduler ordering and short-request prefill preemption.

## Correctness Validation

```text
.venv/bin/pytest tests/test_engine_runtime.py tests/test_model_runner.py tests/test_prefix_cache.py tests/test_scheduler.py -q
52 passed, 2 warnings

.venv/bin/pytest -q
376 passed, 9 skipped, 3 warnings

.venv/bin/python -m compileall -q aster tests
PASS
```

The full suite initially failed on an unrelated pre-existing CLI warning
expectation. `25067b8` fixed that separately; the full suite then passed.

## Performance Validation

Command, with the same command run from the detached baseline worktree and the
current worktree:

```bash
ASTER_CONFIG_OVERRIDE=$'model:\n  name: Qwen3.5-0.8B-4bit\n  path: /Users/eitan/Documents/Projects/Python/Aster/models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit\nengine:\n  max_active_requests: 8\n  max_decode_batch: 2\n  prefill_token_budget: 2\n  idle_prefill_token_limit: 4096\n  pressure_prefill_token_budget: 1\n' \
  /Users/eitan/Documents/Projects/Python/Aster/.venv/bin/python \
  scripts/dev/benchmark_live.py \
  --config configs/config.yaml --workload mixed --concurrency-levels 4 \
  --runtime-kernel manual
```

Seven trials were run on each side after engine warmup. Equal-token baseline
samples (288 completion tokens) were compared with the seven current samples:

| Metric | Baseline median | Current median | Change |
| --- | ---: | ---: | ---: |
| Elapsed seconds | 3.872710 | 3.346841 | -13.58% |
| Average latency | 3.040467 | 2.559043 | -15.83% |
| p95 latency | 3.872539 | 3.346681 | -13.58% |
| Completion tok/s | 74.366980 | 86.051294 | +15.71% |
| Average generation tok/s | 54.853256 | 61.005396 | +11.22% |

A 20,000-resample median bootstrap over these samples gave a 95% interval for
the elapsed-time delta of `[-0.612703s, -0.392361s]`. This interval does not
remove the sequential-order and stochastic-sampling limitations, so the result
is treated as scenario evidence rather than a default-profile gate.

## Memory And Power

- `memory_pressure`: 58% system-wide free memory.
- `pmset -g therm`: no thermal or performance warning recorded.
- `powermetrics`: unavailable without superuser privileges.
- Per-process RSS and MLX peak memory: unavailable in this benchmark artifact.

## Regression And Decision

- All tests passed; all benchmark trials completed 4/4 requests with zero
  failures and zero cancellations.
- The initial grouped result provisionally kept `32addf1`; iterations 004 and
  005 superseded that conclusion with randomized mixed/staggered A/B evidence.
- Do not claim Aster has closed the vllm-mlx continuous-batching gap. The next
  gate is a deterministic, randomized benchmark with memory fields and a 9B
  long-context/mixed workload.

## Next Priority

Upgrade the benchmark runner to support deterministic sampling, randomized A/B
ordering, and per-trial resource metadata before accepting scheduler policy as
the default across the workload matrix.
