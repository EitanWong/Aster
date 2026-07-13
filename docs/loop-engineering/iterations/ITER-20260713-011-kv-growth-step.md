# Iteration 011: KV Growth Step Tuning

- Iteration ID: `ITER-20260713-011`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Start commit: `85657b0`
- End commit: `d525ef6`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`, manual engine
- Model: Qwen3.5-9B-4bit

## Problem and Hypothesis

MLX-LM `KVCache` grows with a 256-token step. During 512-token prefill,
capacity expansion repeatedly allocates and concatenates the existing keys and
values. Increasing the growth step should reduce repeated copies without
changing the logical cache or attention calculation.

## Change

Add `engine.kv_cache_step_tokens`, default `2048`. After MLX-LM creates its
native prompt cache, Aster applies this step only to `KVCache` instances;
Qwen3.5 linear-attention `ArraysCache` instances are unchanged. The setting is
configurable and the cache factory behavior is unit tested.

## 9B Evidence

All runs used cap-0 prefix snapshots, one active request, greedy sampling, and
512-token prefill chunks. Results are single trials; machine swap state varied.

| Prompt | Previous step 256 | Prototype step 2048 | Production default 2048 |
| ---: | --- | --- | --- |
| 12,181 | 35.665s / 3.59 tok/s | 32.321s / 3.96 tok/s | 33.221s / 3.85 tok/s |
| 30,181 | 88.132s / 1.45 tok/s | 83.247s / 1.54 tok/s | 79.810s / 1.60 tok/s |

Peak memory remained approximately `9.121 GB` at 12K and `12.126 GB` at 30K;
the gain is from reducing allocation/copy work, not reducing retained KV
bytes. All runs completed 128 tokens with zero admission or request failures.

## Correctness Gate

A fixed short prompt with `temperature=0.0`, 32 output tokens, and the default
2048 growth step produced the same greedy text as the previous normal KV path:

`Unified memory is a critical architectural feature in modern GPUs (like NVIDIA's CUDA cores) ...`

The cache step changes capacity growth only; existing full test coverage and
the real smoke parity check passed.

## Scope and Next Priority

Keep `d525ef6`. This is a bounded improvement to dynamic KV allocation, not a
paged KV implementation. A true paged adapter still requires model attention
to consume block tables without rebuilding contiguous key/value tensors. The
next loop should specify that ownership/kernel contract and compare it against
this tuned contiguous baseline at 12K, 16K, and 30K.
