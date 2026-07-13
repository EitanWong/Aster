# Known Issues

- Benchmark sampling is now explicit and defaults to greedy `temperature=0.0`; non-greedy runs remain intentionally stochastic.
- Randomized A/B is available in the recorded harness runs; future candidates must use it rather than grouped execution.
- Direct benchmark records include RSS, swap deltas, prompt token counts, prefix-store counters, admission rejections, and MLX allocator peak memory for completed responses, but not energy or allocator peaks for failed requests.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- 9B hybrid memory accounting now allows 30K prompts to complete, but 12K/16K/30K runs added about 1.73/2.16/3.56 GiB swap and 30K completion throughput fell to 0.80 tok/s; 32K mixed-agent, 35B, cancellation pressure, and 30-minute stability evidence remain incomplete.
- The admission-before-prefill scheduler experiment was rolled back: randomized mixed and staggered A/B did not meet a reliable performance gate.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
