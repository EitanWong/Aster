# Aster Architecture

## Design goal

Aster is an Apple Silicon-first local inference runtime for long-context, tool-heavy, OpenClaw-style workloads. It is designed around measured performance, stable background serving, and graceful fallback when an optimization regresses.

## High-level flow

API Gateway
→ Request Queue
→ Scheduler
→ Prefill Engine
→ Decode Engine
→ Streaming Layer
→ Metrics / Telemetry
→ Worker Supervisor

## Key modules

### api/
OpenAI-compatible surface:
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`
- `GET /ready`
- `GET /metrics`

### scheduler/
Queue-aware adaptive scheduling:
- separates intake from execution
- chooses batching windows dynamically
- guards queue depth and overload behavior
- leaves room for future prefill/decode lane separation

### inference/
Execution path:
- `prefill_engine.py` handles prompt ingestion and prefix reuse lookup
- `decode_engine.py` handles token streaming cadence
- `speculative.py` controls speculative enable/disable decisions
- `speculative_pipeline.py` models draft/verify flow and rollback accounting
- `engine.py` coordinates all inference stages

### cache/
Long-context reuse:
- `prefix_cache.py` hashes repeated prefixes deterministically
- `paged_kv_cache.py` models page-based KV allocation rather than monolithic request allocation
- designed so shared-prefix reuse and eviction policy can become more sophisticated without rewriting the server surface

### autotune/
Measured policy selection:
- benchmarks multiple candidate runtime policies
- scores latency, throughput, and stability
- persists selected profile
- re-applies the fastest stable profile at startup

### telemetry/
Observability:
- Prometheus-compatible metrics
- JSON logging
- tracks request latency, prefill, decode, batching, cache hits, speculative acceptance, KV page use, and worker restarts

### workers/
Long-lived service stability:
- heartbeat-based worker supervision
- restart accounting
- degraded-state hooks for crash recovery and future isolation improvements

## Why hybrid rather than full multi-process today

The current implementation uses a robust hybrid architecture:
- separate lifecycle domains for API, scheduler, and worker supervision
- one Python process for lower coordination complexity initially
- clear interfaces so the inference worker can be promoted to a separate process later

This is the right tradeoff for a serious first production-oriented codebase because it preserves modularity without prematurely adding IPC overhead and operational complexity.

## Dynamic optimization policy

Aster does not assume an optimization is good.

Candidate features are benchmarked and policy-selected:
- speculative decoding
- draft token count
- batch window size
- max batch size
- streaming flush cadence

Future benchmark dimensions fit naturally into the same system:
- prefix cache modes
- page sizes
- prefill/decode scheduling modes
- concurrency classes

## Fallback behavior

If an optimization underperforms or destabilizes service behavior, Aster falls back by policy:
- speculative can be disabled
- aggressive draft lengths can be reduced
- batch windows can shrink
- max batch size can be lowered
- previously selected stable tuning profile can be reused

## OpenClaw-oriented advantages

Aster is designed for workloads with:
- repeated system prompts
- repeated tool definitions
- long prompt scaffolding
- multi-turn agent sessions
- 4k–16k prompt lengths

That makes prefix caching and cache-aware prefill especially important. The architecture keeps correctness first: prefix reuse is explicit and deterministic, not approximate.

## Current implementation status

Implemented now:
- full scaffold
- config system
- API routes
- scheduler
- policy engine
- prefix cache
- paged KV abstraction
- speculative controller and pipeline abstraction
- telemetry and logs
- supervisor
- autotune scaffolding and persisted tuning profile selection
- startup scripts and tests

Planned next hardening pass:
- real MLX model/tokenizer binding
- true paged KV residency tied to MLX tensors
- multiprocess inference worker isolation
- realistic benchmark harness against Qwen3.5-9B + Qwen3.5-0.8B
- deeper backpressure and memory-pressure adaptation
