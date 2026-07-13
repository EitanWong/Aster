# LOOP ITERATION: ITER-20260713-004-randomized-scheduler-ab

STATUS: INVESTIGATING

START COMMIT: `740f4a6`

END COMMIT: `740f4a6` (measurement-only iteration)

## Focus

Remove execution-order bias from the iteration 001 scheduler A/B result.

## Method

- Baseline source: `25067b8`, with the iteration 003 benchmark harness copied into the temporary worktree.
- Current source: `740f4a6`.
- Model: Qwen3.5-0.8B-4bit.
- Runtime: manual; `max_active_requests=8`, `max_decode_batch=2`, prefill budget 2, pressure budget 1.
- Workload: mixed, concurrency level 4, `temperature=0.0`.
- Seven baseline and seven current trials were interleaved using a fixed randomized sequence.

## Results

| Metric | Baseline median | Current median | Change |
| --- | ---: | ---: | ---: |
| Elapsed seconds | 3.260047 | 3.353254 | +2.86% |
| Average request latency | 2.470631 | 2.574088 | +4.19% |
| Completion throughput | 88.342288 | 85.886713 | -2.78% |
| RSS peak bytes | 1,327,939,584 | 1,258,504,192 | -5.23% |

All trials produced 288 completion tokens and 4/4 successful requests. Bootstrap
95% intervals were `[-0.046119s, 0.320934s]` for elapsed delta,
`[-0.051882s, 0.277328s]` for average latency delta, and
`[-8.146609, 0.786641]` for completion throughput delta.

## Analysis

The earlier grouped result is not robust enough to claim a performance win. The
mixed workload submits all requests together, while the original root-cause
evidence concerned short requests arriving while a long prefill is already in
progress. This A/B run therefore does not prove the scheduler change is wrong,
but it does reject the prior broad performance conclusion.

## Failed Experiment

The first randomized command used a malformed local model path and was aborted
during warmup. It produced no valid artifact and is excluded from the result.

## Decision

Keep the implementation temporarily because it has a clear fairness behavior
covered by unit tests, but do not promote it as a default performance profile.
The next experiment must model staggered arrivals. Roll back `32addf1` if that
workload also fails to show a meaningful benefit.

## Next Priority

Add or run a direct staggered workload: submit one long prompt, wait until its
prefill has started, then submit short requests at fixed intervals. Compare
short-request queue wait, p95 latency, completion tokens, RSS, and failures.
