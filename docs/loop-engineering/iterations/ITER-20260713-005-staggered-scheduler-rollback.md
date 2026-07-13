# LOOP ITERATION: ITER-20260713-005-staggered-scheduler-rollback

STATUS: ROLLED_BACK

START COMMIT: `0c07d13`

END COMMIT: `5f2b952`

## Focus

Evaluate the scheduler candidate on the workload that motivated it: a long
prefill starts first, then short requests arrive at fixed intervals.

## Workload

- Model: Qwen3.5-0.8B-4bit.
- Runtime: manual Aster engine.
- `prefill_token_budget=64`, `idle_prefill_token_limit=64`, pressure budget 32.
- One 4096-word long request starts first; three short requests are submitted
  after active prefill is observed, with 50ms spacing.
- Seven baseline/current trials were interleaved with `temperature=0.0`.

## Results

| Metric | Baseline median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Short request p95 latency | 2.444696s | 2.745431s | +12.30% |
| Long request latency | 4.229011s | 4.150410s | -1.86% |
| Aggregate elapsed | 4.235210s | 4.188500s | -1.10% |
| Completion throughput | 64.223494 tok/s | 64.939713 tok/s | +1.12% |
| RSS peak | 1,349,500,928 | 1,292,599,296 | -4.22% |

All 14 trials completed 4/4 requests, failed zero requests, cancelled zero
requests, and produced exactly 272 completion tokens. Bootstrap 95% intervals
were `[-1.582823s, 1.587049s]` for short p95 delta,
`[-0.167598s, 0.091877s]` for aggregate elapsed delta, and
`[-1.456436, 2.504113]` for throughput delta.

## Failed Experiments

- A 2-token prefill budget caused a 38.5s trial and about 4.44GB swap growth;
  it was excluded as an intentionally over-aggressive pressure configuration.
- Two smoke commands used malformed model paths and were aborted during
  warmup; their failed records were excluded from the artifact.

## Decision

Rollback `32addf1` via `5f2b952`. The candidate did not meet the 3% performance
gate on the protected short-request metric and its aggregate result was within
noise. Keep the benchmark harness enhancements for future candidates.

## Regression Verification

```text
.venv/bin/pytest -q
379 passed, 9 skipped, 3 warnings

.venv/bin/python -m compileall -q aster scripts/dev/benchmark_live.py tests
PASS
```

## Next Priority

Run the resource-aware benchmark on the 9B model for long-context, prefix reuse,
and memory-pressure baselines before choosing the next runtime change.
