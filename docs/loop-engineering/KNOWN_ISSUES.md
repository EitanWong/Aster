# Known Issues

- Benchmark sampling is now explicit and defaults to greedy `temperature=0.0`; non-greedy runs remain intentionally stochastic.
- Randomized A/B is available in the recorded harness runs; future candidates must use it rather than grouped execution.
- Direct benchmark records include RSS, swap deltas, prompt token counts, prefix-store counters, admission rejections, overall MLX peak, and prefill active/peak memory for completed responses, but not energy or allocator peaks for failed requests. The paged-attention probe separately records randomized A/B timing and allocator peaks.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- 9B hybrid memory accounting and bounded prefix snapshots now allow 30K prompts to complete at about 1.38 tok/s with 1.10 GiB swap growth in the best measured run; 32K mixed-agent, 35B, cancellation pressure, and 30-minute stability evidence remain incomplete.
- The admission-before-prefill scheduler experiment was rolled back: randomized mixed and staggered A/B did not meet a reliable performance gate.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
- Experimental KV quantization is not enabled: 4-bit KV failed fixed greedy token parity and 8-bit has no demonstrated material gain.
- `kv_cache_step_tokens` reduces native KV growth copies but does not reduce retained KV memory; a true paged attention path remains unimplemented.
- The experimental `PagedKVCacheLayer` is lossless and COW-capable, and its block pool no longer repacks with `mx.stack` on every view. `PagedKVCacheBundle` reclaims full-attention pools after the last fork releases, but mixed recurrent/full-attention bundles are rejected and the MLX integration still materializes contiguous K/V on every update; batch merge falls back to native contiguous caches. It remains disabled in production paths.
- The opt-in hybrid list boundary is parity-clean on the Qwen3.5-0.8B greedy smoke. Contiguous-buffer reuse brings 8.4K randomized A/B to within `0.4%` median of native, but peak memory remains about `7.6%` higher (`2.471 GB` vs `2.297 GB`); the 2.2K single-run path remains about `10.6%` slower. Prefix snapshots are disabled and decode batch size is restricted to one; it is not a default path.
- The experimental block-indexed Metal kernel is numerically correct on Qwen3.5-shaped FP16 input after the threadgroup-grid fix, but the corrected median benchmark is `1.56x/3.42x/7.44x` slower than native at 512/2K/8K in the recorded run. Pool reclamation and hybrid bundle lifecycle are incomplete, so it is not a serving path.
