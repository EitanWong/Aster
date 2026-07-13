# Known Issues

- Benchmark sampling is now explicit and defaults to greedy `temperature=0.0`; non-greedy runs remain intentionally stochastic.
- Randomized A/B is available in the recorded harness runs; future candidates must use it rather than grouped execution.
- Direct benchmark records include RSS and swap deltas, but not MLX allocator-level peak memory or energy.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- Long-context 9B/35B, 32K mixed-agent, cancellation pressure, and 30-minute stability evidence remain incomplete.
- The admission-before-prefill scheduler experiment was rolled back: randomized mixed and staggered A/B did not meet a reliable performance gate.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
