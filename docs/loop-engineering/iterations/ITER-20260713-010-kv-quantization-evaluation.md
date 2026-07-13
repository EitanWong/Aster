# Iteration 010: KV Quantization Evaluation

- Iteration ID: `ITER-20260713-010`
- Date: 2026-07-13
- Machine: macOS 27.0, Apple M5, 10 cores, 24 GiB unified memory
- Code baseline: `85657b0`
- Runtime: Python 3.14.5, MLX `0.32.0`, MLX-LM `0.31.3`, manual engine
- Model: Qwen3.5-9B-4bit

## Candidate

The installed MLX-LM runtime provides `QuantizedKVCache`. A prototype replaced
only full-attention `KVCache` layers with 4-bit or 8-bit quantized caches; the
Qwen3.5 linear-attention `ArraysCache` layers were left unchanged. This was an
isolated experiment and did not change Aster source or defaults.

## 12K Evidence

Each run used one request, 512-token prefill chunks, greedy sampling, and 128
output tokens. Results are single trials under changing system swap state.

| Cache | MLX peak | Prefill peak | Elapsed | Completion tok/s | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal KV | 9.121 GB | 9.121 GB | 35.665s | 3.589 | 0 |
| 8-bit KV | 8.940 GB | 8.940 GB | 36.284s | 3.528 | 0 |
| 4-bit KV | 8.843 GB | 8.843 GB | 35.587s | 3.597 | 0 |

The memory reduction is modest and does not clear the prefill working-set
bottleneck. The 8-bit run was slower than normal in this trial; the 4-bit run
does not establish a stable performance gain from one sample.

## Correctness Gate

Fixed short-prompt greedy output differed:

- Normal KV: `Unified memory is a critical architectural feature in modern GPUs (like NVIDIA's CUDA cores)...`
- 4-bit KV: `Unified memory is a critical architectural feature in modern GPUs (like those from NVIDIA) ...`

Both generated 32 tokens, but the token sequence was not identical. Since the
loop goal prohibits trading correctness for a small memory result, 4-bit KV is
rejected for the default path. 8-bit KV was not promoted because it lacks a
measured material gain and still requires a broader parity/stability matrix.

## Conclusion

Do not merge the prototype. The existing standalone `PagedCacheManager` is only
a Python block index and does not own or expose MLX attention tensors; wiring it
into the current model would require a new cache representation and attention
kernel contract. The next candidate is therefore a deliberate lossless paged
KV adapter design, with block ownership, COW, MLX kernel access, and exact
greedy parity specified before implementation.

Artifacts contain the normal, 8-bit, and 4-bit raw benchmark records. No Aster
source change was made in this iteration.
