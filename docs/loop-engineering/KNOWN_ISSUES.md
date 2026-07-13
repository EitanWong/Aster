# Known Issues

- Benchmark sampling is now explicit and defaults to greedy `temperature=0.0`; non-greedy runs remain intentionally stochastic.
- Randomized A/B is available in the recorded harness runs; future candidates must use it rather than grouped execution.
- Direct benchmark records include RSS, swap deltas, prompt token counts, prefix-store counters, admission rejections, overall MLX peak, and prefill active/peak memory for completed responses, but not energy or allocator peaks for failed requests.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- 9B hybrid memory accounting and bounded prefix snapshots now allow 30K prompts to complete at about 1.38 tok/s with 1.10 GiB swap growth in the best measured run; 32K mixed-agent, 35B, cancellation pressure, and 30-minute stability evidence remain incomplete.
- The admission-before-prefill scheduler experiment was rolled back: randomized mixed and staggered A/B did not meet a reliable performance gate.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
- Experimental KV quantization is not enabled: 4-bit KV failed fixed greedy token parity and 8-bit has no demonstrated material gain.
