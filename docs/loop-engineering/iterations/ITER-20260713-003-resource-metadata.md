# LOOP ITERATION: ITER-20260713-003-resource-metadata

STATUS: SUCCESS (observability improvement)

START COMMIT: `0469f6c`

END COMMIT: `96ff8a6`

## Focus

Add machine and process resource evidence to each direct benchmark workload
record.

## Root Cause And Hypothesis

The deterministic benchmark from iteration 002 made token counts stable, but
its JSON still lacked environment and memory evidence. Hypothesis: collecting
lightweight process RSS samples and system snapshots around each workload will
make performance results auditable without changing request execution.

## Changes

- Record platform, Python, MLX-LM version, and total system memory.
- Sample process RSS every 50ms and retain the observed peak.
- Record swap used before and after each workload.
- Keep unavailable package/resource values explicit rather than inventing data.
- Add unit coverage for metadata fields.

## Validation

```text
.venv/bin/pytest tests/test_benchmark_live.py -q
2 passed

.venv/bin/pytest -q
378 passed, 9 skipped, 3 warnings

.venv/bin/python -m compileall -q scripts/dev/benchmark_live.py
PASS
```

Seven live MLX trials used Qwen3.5-0.8B-4bit, manual runtime, mixed workload,
concurrency level 4, and `temperature=0.0`:

- platform: `macOS-27.0-arm64-arm-64bit-Mach-O`
- Python: `3.14.5`
- MLX-LM: `0.31.3`
- system memory: `25,769,803,776` bytes
- completion tokens: 288 in every trial
- outcome: 4/4 successful requests in every trial
- RSS peak: median `1,328,529,408` bytes; min/max `1,255,702,528` / `1,330,987,008`
- swap delta: 0 bytes in every trial
- elapsed time: median `3.259s`, min/max `3.079s` / `3.409s`

Raw baseline and current records are under
`iterations/artifacts/ITER-20260713-003-resource-metadata/`.

## Memory And Power

RSS and swap are now available per workload. MLX allocator-level memory and
energy remain unavailable; `powermetrics` still requires superuser privileges.

## Decision

Keep `96ff8a6`. The benchmark now has enough environment and process-memory
metadata to support a stronger scheduler A/B run. This iteration itself is an
observability change and has no runtime performance claim.

## Next Priority

Randomize or interleave baseline/current trial order and rerun the scheduler
candidate, then move to the 9B long-context and mixed-agent workload matrix.
