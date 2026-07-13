# LOOP ITERATION: ITER-20260713-002-deterministic-benchmark

STATUS: SUCCESS (measurement infrastructure improvement)

START COMMIT: `9595842`

END COMMIT: `d521c22`

## Focus

Make direct benchmark workload sampling explicit and reproducible enough for
token-count and latency A/B comparisons.

## Root Cause And Hypothesis

`scripts/dev/benchmark_live.py` constructed requests without specifying
temperature, so `InferenceRequest` supplied its service default `0.7`. The
resulting completion lengths varied between repeated trials. Hypothesis: an
explicit greedy benchmark default will keep workload outputs stable without
changing service defaults.

## Changes

- Add `--temperature` to `benchmark_live.py`, defaulting to `0.0`.
- Propagate the value to every request in the single, reuse, mixed, and long workloads.
- Include the chosen temperature in each `BenchmarkRecord`.
- Add a unit test that verifies explicit propagation.

## Validation

```text
.venv/bin/pytest tests/test_benchmark_live.py -q
1 passed

.venv/bin/pytest -q
377 passed, 9 skipped, 3 warnings

.venv/bin/python -m compileall -q scripts/dev/benchmark_live.py
PASS
```

The validation run used the same Qwen3.5-0.8B mixed workload and runtime
configuration as iteration 001, with:

```bash
... scripts/dev/benchmark_live.py \
  --workload mixed --concurrency-levels 4 --runtime-kernel manual \
  --temperature 0.0
```

Seven trials all recorded `temperature=0.0`, `total_completion_tokens=288`,
`completed_requests=4`, `failed_requests=0`, and `cancelled_requests=0`.
Elapsed-time median was `3.225234s`; mean `3.274447s`; min/max
`3.140911s/3.559012s`; standard deviation `0.143175s`.

The previous seven current-side trials from iteration 001 used implicit
temperature `0.7` and produced completion token totals
`[242, 288, 271, 288, 288, 288, 288]`. This confirms the measurement problem;
it is not a performance comparison.

## Memory And Power

No new resource collection was added in this iteration. Per-process RSS, MLX
peak memory, swap delta, and power remain unavailable from this artifact.

## Decision

Keep `d521c22`. Greedy sampling is now the benchmark default, while callers can
opt into stochastic sampling explicitly. Do not use this iteration as evidence
of a runtime speedup.

## Next Priority

Add randomized/interleaved A/B execution and per-trial machine, process RSS,
MLX memory, and swap metadata, then rerun iteration 001's scheduler candidate.
