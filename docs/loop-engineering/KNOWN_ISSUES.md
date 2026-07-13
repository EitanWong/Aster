# Known Issues

- Benchmark sampling is now explicit and defaults to greedy `temperature=0.0`; non-greedy runs remain intentionally stochastic.
- Benchmark A/B execution must be randomized or interleaved to reduce thermal and process-order bias.
- Direct benchmark records include RSS and swap deltas, but not MLX allocator-level peak memory or energy.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- Long-context 9B/35B, 32K mixed-agent, cancellation pressure, and 30-minute stability evidence remain incomplete.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
