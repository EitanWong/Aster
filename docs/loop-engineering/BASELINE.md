# Baseline

## Machine

- macOS `27.0`, Apple M5, 10 cores (4 performance, 6 efficiency), 24 GB unified memory.
- Python `3.14.5` from `.venv`.
- MLX version is not exposed by the installed module; `mlx-lm` is `0.31.3`.
- NumPy `2.4.6`, psutil `7.2.2`.
- Benchmark baseline source: detached `HEAD` `d6f1ca2`; current scheduler source: `32addf1`.

## Test Baseline

```text
.venv/bin/pytest -q
376 passed, 9 skipped, 3 warnings
```

The initial baseline exposed one unrelated pre-existing CLI failure. It was fixed separately in `25067b8` and is not part of the scheduler change.

## Benchmark Configuration

- Model: `models/qwen3.5-0.8b-mlx/Qwen3.5-0.8B-4bit`.
- Runtime: manual Aster engine.
- `max_active_requests=8`, `max_decode_batch=2`, `prefill_token_budget=2`, `pressure_prefill_token_budget=1`.
- Workload: `scripts/dev/benchmark_live.py --workload mixed --concurrency-levels 4`.
- Each trial includes engine start and warmup, then four mixed requests. Seven trials per side.
- The workload uses the existing request default `temperature=0.7`; completion token counts are therefore not fully deterministic.

## Baseline Measurements

| Metric | HEAD median | Current median | Change |
| --- | ---: | ---: | ---: |
| Elapsed seconds, equal-token samples | 3.872710 | 3.346841 | -13.58% |
| Average request latency | 3.040467 | 2.559043 | -15.83% |
| p95 latency | 3.872539 | 3.346681 | -13.58% |
| Completion throughput | 74.366980 | 86.051294 | +15.71% |
| Average generation throughput | 54.853256 | 61.005396 | +11.22% |

The raw seven-trial JSON files are under `iterations/artifacts/ITER-20260713-001-admission-before-prefill/`.

## Unavailable Measurements

- Per-process RSS and MLX peak memory were not emitted by this direct benchmark.
- Power and joules/token were not collected because `powermetrics` requires superuser privileges.
