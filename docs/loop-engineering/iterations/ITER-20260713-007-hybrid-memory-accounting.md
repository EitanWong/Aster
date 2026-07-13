# Iteration 007: Hybrid Attention Memory Accounting

- Iteration ID: `ITER-20260713-007`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `1dc643b`
- End commit: `4de13e1`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`, manual engine
- Model: Qwen3.5-9B-4bit

## Problem

The 9B long-context baseline rejected 12K, 16K, and 30K prompts before
prefill. `ModelRunner.estimate_request_bytes()` read only top-level config
fields. MLX-LM returned Qwen3.5 metadata under `text_config`, so the estimator
fell back to 32 layers, 32 KV heads, and a full KV cache for every layer.

Qwen3.5-9B actually has 32 layers, 8 `full_attention` layers, and 24
`linear_attention` layers. Linear attention keeps a fixed recurrent state; it
does not allocate sequence-growing KV for every layer.

## Hypothesis and Change

Read nested `text_config` metadata and estimate:

- sequence-growing KV bytes only for `full_attention` layers;
- fixed float32 recurrent state for linear-attention layers;
- fixed convolution state for linear-attention layers;
- dtype-aware bytes for sequence and convolution buffers.

The estimator retains the generic all-KV fallback for models without a
`layer_types` description. A unit test locks the nested hybrid calculation.
The benchmark also now records `mlx_peak_memory_gb` from the response's MLX
allocator measurement.

## Estimate Evidence

| Prompt + 128 output | Before | After |
| ---: | ---: | ---: |
| 8,181 tokens | 4,356,308,992 B | 323,780,608 B |
| 12,181 tokens | 6,453,460,992 B | 454,852,608 B |
| 16,181 tokens | 8,550,612,992 B | 585,924,608 B |
| 30,181 tokens | 15,890,644,992 B | 1,044,676,608 B |

## 9B Runtime Evidence

Each request used a fresh process, one active request, greedy sampling, and
128 output tokens. The previous 12K/16K/30K probes were rejected by admission;
all three completed after this change.

| Prompt tokens | Result | MLX peak | Elapsed | Completion tok/s | Swap delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12,181 | completed | 12.19 GB | 48.580s | 2.63 | +1.73 GiB |
| 16,181 | completed | unavailable in this artifact | 59.790s | 2.14 | +2.16 GiB |
| 30,181 | completed | unavailable in this artifact | 160.188s | 0.80 | +3.56 GiB |

A 9B single-request smoke recorded `mlx_peak_memory_gb=5.169 GB`, completed 128
tokens in 11.400s, and showed 11.23 completion tok/s. The long-context runs
prove that the prior failures were false admission rejects, not context-limit
failures. They do not prove healthy long-context performance: swap growth and
end-to-end throughput degradation remain severe.

## Verification

```text
.venv/bin/pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q aster scripts/dev/benchmark_live.py tests
.venv/bin/python -m pytest -q tests/test_model_runner.py tests/test_benchmark_live.py
```

Results: `386 passed, 9 skipped, 1 warning`; `pip check` and compileall passed.
`ruff` was unavailable in the active environment and was not used as evidence.

## Conclusion

Keep `4de13e1`. This is a correctness and availability improvement in memory
admission accounting, not a global performance optimization. The next change
must address actual unified-memory and swap behavior rather than lowering the
estimate further. Candidate directions are allocator-aware admission,
prefill-time memory sampling, prefix eviction under pressure, and a paged or
quantized KV representation, each with output parity and rollback gates.
