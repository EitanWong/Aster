# Known Issues

- Benchmark sampling is stochastic because the direct workload uses `temperature=0.7`; fixed-seed greedy comparisons are still needed for token-level A/B evidence.
- Benchmark A/B execution must be randomized or interleaved to reduce thermal and process-order bias.
- Direct benchmark records do not include per-trial RSS, MLX peak memory, swap delta, or energy.
- `BatchGeneratorRuntimeKernel.available` is false; continuous batching is implemented by the manual scheduler and decode batch runner only.
- Long-context 9B/35B, 32K mixed-agent, cancellation pressure, and 30-minute stability evidence remain incomplete.
- Paged KV, SSD cache, KV-cache quantization, structured-output black-box parity, and full tool parser parity remain open.
