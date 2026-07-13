# Iteration 006: Package Refresh and Prefix/Memory Baseline

- Iteration ID: `ITER-20260713-006`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `4ae300e`
- Package refresh commit: `1a0b993`
- Benchmark instrumentation commit: `af65098`
- Runtime: Python 3.14.5, manual engine, Qwen3.5-9B-4bit

## Problem and Hypothesis

The dependency environment was behind the latest compatible MLX release, and
the repeated-prefix benchmark could not distinguish a real cache hit from a
divergent LCP that MLX-LM's non-rewindable `ArraysCache` must safely skip. The
hypothesis was that a refreshed MLX stack plus explicit cache diagnostics would
make the 9B baseline reproducible without weakening cache safety.

## Package Research and Resolution

The package metadata was checked on 2026-07-13 using PyPI and the upstream MLX
release metadata:

- `mlx==0.32.0` and `mlx-metal==0.32.0`
- `mlx-lm==0.31.3` (latest available)
- `mlx-audio==0.4.5`
- `fastapi==0.139.0`
- `uvicorn==0.51.0`
- `pydantic==2.13.4`
- `numpy==2.5.1`
- `transformers==5.12.1`
- `sse-starlette==3.4.5`
- `sentencepiece==0.2.2`
- `huggingface-hub==1.23.0`
- `python-multipart==0.0.32`

`transformers==5.13.1` was not selected because `mlx-audio==0.4.5` declares
`transformers<5.13.0`. The project declarations therefore use
`transformers>=5.12.1,<5.13.0`, the newest resolver-compatible range tested on
this machine. `pip check` reported no broken requirements, and MLX selected the
GPU device successfully.

Sources:

- https://pypi.org/project/mlx/
- https://pypi.org/project/mlx-lm/
- https://pypi.org/project/mlx-audio/
- https://github.com/ml-explore/mlx/releases/tag/v0.32.0

## Changes

- Updated `pyproject.toml` and `requirements.txt` to the validated compatible
  dependency floor.
- Exposed nested prefix-store counters in `InferenceEngine.status()`.
- Split the benchmark into exact `reuse` and divergent `reuse-divergent`
  workloads.
- Added exact/LCP/unsafe-LCP/store counters, prompt token counts, and admission
  rejection counts to benchmark records.
- Added `--long-prompt-words` for reproducible long-context probes.

## Verification

Commands:

```text
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q aster scripts/dev/benchmark_live.py tests
.venv/bin/pytest -q
```

The full suite passed with `384 passed, 9 skipped, 1 warning`; the affected
benchmark and engine suites passed with `43 passed` before the final additive
record field, and the final full run covered the complete resulting suite.

## 9B Benchmark Evidence

The all-workload run used greedy sampling, one loaded model process, manual
runtime, concurrency level 2, and the resource-aware harness. All requests
completed successfully:

| Workload | Completion tokens | Elapsed | Completion tok/s | Prefix result | Swap delta |
| --- | ---: | ---: | ---: | --- | ---: |
| `single` | 128 | 15.072s | 8.49 | none | +32 MiB |
| `reuse` | 192 | 17.261s | 11.12 | 1 exact hit, 188 tokens | 0 |
| `reuse-divergent` | 192 | 18.312s | 10.48 | 2 unsafe LCP skips | 0 |
| `mixed` | 288 | 22.612s | 12.74 | 2 unsafe LCP skips | 0 |
| `staggered` | 272 | 36.101s | 7.53 | 1 unsafe LCP skip | +1.67 GiB |
| `long` | 256 | 21.685s | 11.81 | 2 exact hits, 8,552 tokens | -8 MiB |

The exact reuse result proves that the cache path is active. The divergent
result proves that the current runtime reports and safely skips non-rewindable
LCP reuse rather than claiming a false hit.

Long-context probes were run in fresh processes with one active request:

| Prompt tokens | Result | Elapsed | RSS peak | Swap delta |
| ---: | --- | ---: | ---: | ---: |
| 8,181 | completed, 128 output tokens | 44.970s | 988 MiB | +1.61 GiB |
| 12,181 | admission failure | 0.028s | 2.50 GiB | 0 |
| 16,181 | admission failure | 0.040s | 2.21 GiB | 0 |
| 30,181 | admission failure | 0.070s | 1.00 GiB | 0 |

The direct diagnostic for the 30,005-token request identified
`OverloadedError(code=memory_pressure)`, `admission_retries=16`, and an
estimated allocation of `15,798,370,304` bytes. The 8K request completed but
caused substantial swap growth, so it is not evidence of a healthy long-context
profile. The current successful single-request boundary is therefore between
8K and 12K prompt tokens under the measured machine state.

## Conclusion and Rollback

Keep both commits. The package refresh is compatible and fully tested, while
the benchmark changes improve evidence quality without changing scheduling or
cache correctness policy. No performance win is claimed from the package
upgrade because this iteration did not run a randomized pre/post A/B against
the old environment.

Rollback is `git revert af65098` followed by `git revert 1a0b993`; do not revert
the user's unrelated `.codegraph/.gitignore` change.

## Next Priority

Investigate the memory-pressure boundary and MLX allocator visibility before
attempting 32K agent workloads. The next experiment should compare the current
admission estimate, actual unified-memory/swap behavior, and cache budget under
8K, 12K, and 16K prompts, then evaluate a memory-aware prefix eviction or KV
representation change with correctness and rollback gates.
