# Aster Roadmap

This document records the long-term architectural direction for Aster.

It is intended to be a stable reference for future development so the project can evolve deliberately over time rather than drifting through one-off optimizations.

---

## Status and intent

Aster's **current codebase remains the baseline**.

The project already has important working foundations:

- MLX-based local inference
- OpenAI-compatible API endpoints
- streaming token output
- request metrics and structured logs
- health / readiness checks
- request tracing
- basic queueing and scheduling
- prefix-cache-related instrumentation
- an operational shell for local service use

The purpose of this roadmap is **not** to trigger a large rewrite now.

Instead, it defines how Aster should gradually evolve from:

> a local model-serving application

into:

> a scheduler-driven, MLX-native inference engine for Apple Silicon

The roadmap should be followed **incrementally**.

### Ground rules

- keep the current runtime working while improving it
- prefer gradual refactors over disruptive rewrites
- preserve correctness and stability at each step
- avoid large architectural churn unless there is a clear payoff and migration path
- use measurements to guide optimization work
- treat production quality, debuggability, and maintainability as first-class goals

In short:

> **Do not throw away the current system. Evolve it carefully.**

---

## Long-term architectural direction

Aster should move toward the design of a real inference engine rather than remaining just an API wrapper around a decode loop.

### Core shift

The major architectural transition is:

### Current mental model

```text
request -> tokenize -> prefill -> decode loop -> respond
```

### Target mental model

```text
many requests -> admission -> scheduler -> shared execution runtime -> stream multiplexer -> responses
```

This means the runtime should increasingly be organized around:

- a scheduler-owned execution loop
- explicit request lifecycle state
- token-level scheduling decisions
- structured KV memory management
- prompt/prefix reuse
- chunked prefill
- shared decode execution across multiple requests
- observability built into the runtime itself

---

## Architectural principles

These principles should guide future work.

### 1. Treat Aster as an inference engine, not just an API app

The API surface is important, but it should sit on top of a serving runtime.

Future design should emphasize:

- request admission
- request state tracking
- batch formation
- runtime scheduling
- memory management
- stream delivery
- metrics and tracing

The serving core should be the heart of the system.

### 2. Move toward a scheduler-driven runtime

The decode loop should eventually be owned by the scheduler, not by individual requests.

Requests should become state objects managed by a shared runtime that can:

- admit new work dynamically
- interleave prefill and decode
- join requests into active batches
- remove finished requests immediately
- preserve fairness and latency goals

### 3. Prefer token-level continuous batching

Decode on modern LLMs is largely memory-bandwidth-bound.

Single-request decode underutilizes available hardware resources.

Aster should evolve toward:

- continuous batching
- batch resizing
- mid-generation request joins
- early completion removal
- token-step scheduling

This is one of the most important long-term performance multipliers.

### 4. Introduce structured KV cache memory management

KV state should become a first-class managed resource.

Rather than treating cache state as an opaque side effect, Aster should move toward:

- paged or block-structured KV allocation
- explicit request-to-KV mappings
- clearer residency and reuse behavior
- cleaner handling of long contexts and mixed workloads

Over time this should reduce fragmentation and improve concurrency.

### 5. Make prompt/prefix reuse a core serving primitive

Agent-style workloads often repeat large prompt prefixes.

Aster should increasingly optimize for:

- long system prompts
- shared instruction scaffolding
- repeated tool definitions
- multi-turn context reuse

Prefix and prompt-state reuse should become a central part of lowering TTFT and increasing throughput.

### 6. Prefer chunked prefill over monolithic prompt ingestion

Large prefills can monopolize the machine and harm latency.

Aster should move toward chunked prompt ingestion so that:

- latency spikes are reduced
- prefill and decode can interleave more gracefully
- large prompts do not block all active work
- scheduler flexibility improves

### 7. Separate runtime responsibilities more clearly

Over time the system should distinguish these layers more explicitly:

- API / transport layer
- request admission and lifecycle layer
- scheduler
- executor / MLX runtime layer
- KV memory manager
- prefix reuse system
- stream multiplexer
- telemetry / tracing layer

This separation will make optimization work safer and more maintainable.

### 8. Build observability into the runtime

As Aster becomes more complex, debugging must become easier, not harder.

The engine should continue improving in:

- request-level metrics
- system-level metrics
- scheduler visibility
- request tracing
- cache hit/miss visibility
- latency decomposition
- runtime health reporting

The more advanced the runtime becomes, the more important this is.

---

## Phased roadmap

The phases below are intended to be followed progressively. They do not require a single large rewrite.

---

## Phase 0 — Baseline stabilization and operator tooling

### Goal
Strengthen the current system so it is easier to operate, debug, and trust.

### Affects
- service scripts
- local chat tooling
- logging
- health / readiness
- metrics and request tracing
- documentation

### Why it matters
A stronger operational foundation makes future architectural work safer.

### Expected benefits
- easier debugging
- faster iteration
- clearer regressions
- more confidence in performance experiments

### Notes
This phase keeps the current architecture intact while making it more usable.

---

## Phase 1 — Runtime structure refactor

### Goal
Refactor the codebase toward clearer runtime boundaries without changing the external product shape.

### Affects
- runtime control flow
- request lifecycle modeling
- module structure
- separation of scheduling, execution, and streaming responsibilities

### Direction
Move gradually toward more explicit subsystems such as:

- request admission
- request state objects
- scheduler loop management
- stream multiplexing
- executor abstraction
- KV management abstraction

### Expected benefits
- cleaner code ownership
- easier future optimization work
- safer changes to scheduling and memory logic
- reduced coupling between API and runtime internals

### Guidance
This phase should be mostly refactoring and boundary clarification, not a full rewrite.

---

## Phase 2 — Scheduler-owned execution loop

### Goal
Introduce a stronger scheduler model so the runtime, not each request, owns execution flow.

### Affects
- request queueing
- scheduling policy
- active request tracking
- decode orchestration
- stream dispatch timing

### Direction
Evolve from a simple request submission path into a scheduler-driven loop that can:

- track active requests
- decide what work runs next
- manage fairness
- prepare future continuous batching support

### Expected benefits
- better control over latency and concurrency
- foundation for token-level scheduling
- more predictable runtime behavior under load

### Guidance
Do this in stages. It is better to first centralize control flow than to jump straight to advanced batching.

---

## Phase 3 — Token-level continuous batching

### Goal
Support batching at decode-step granularity rather than treating each request as an isolated decode stream.

### Affects
- scheduler
- active batch formation
- decode step execution
- request state transitions
- streaming behavior

### Direction
Introduce capabilities such as:

- dynamic request admission
- active decode batch management
- mid-generation batch joins
- early completion removal
- batch resizing

### Expected benefits
- much better hardware utilization under concurrency
- improved aggregate throughput
- better scaling for multiple simultaneous requests

### Guidance
This should be built on top of a scheduler that already has clear control over request state.

---

## Phase 4 — Structured KV cache management

### Goal
Turn KV cache into an explicitly managed memory system rather than a loosely tracked runtime artifact.

### Affects
- cache allocation strategy
- request state
- memory accounting
- long-context behavior
- concurrency efficiency

### Direction
Move toward:

- paged or block-based KV allocation
- explicit request-to-KV mapping
- structured append / release behavior
- clearer ownership and reuse rules

### Expected benefits
- less memory fragmentation
- better long-context support
- more scalable batching behavior
- stronger foundations for reuse and tiering

### Guidance
Introduce structure gradually. Avoid destabilizing the current runtime all at once.

---

## Phase 5 — Prompt and prefix reuse

### Goal
Make prompt-state reuse a central optimization path for agent-like workloads.

### Affects
- prompt hashing
- cache lookup policy
- prefill path
- request initialization
- metrics and observability

### Direction
Support:

- exact prefix reuse
- increasingly better shared-prefix detection
- prompt-state checkpoints where useful
- safer reuse of stable prompt scaffolding

### Expected benefits
- lower TTFT
- lower prefill cost
- better throughput on repeated workloads
- much better performance on agent sessions with long stable prefixes

### Guidance
Correctness matters more than aggressiveness here. Reuse bugs are subtle and can be expensive.

---

## Phase 6 — Chunked prefill and prefill/decode interleaving

### Goal
Reduce latency spikes and make prefill coexist more gracefully with active decode work.

### Affects
- prompt ingestion
- scheduler policy
- runtime fairness
- latency behavior under mixed workloads

### Direction
Process large prompts in chunks rather than in one monolithic prefill step, and let the scheduler decide how to interleave prefill with decode.

### Expected benefits
- lower TTFT variance
- fewer large-latency stalls
- better scheduler flexibility
- improved responsiveness for streaming users while large prompts are entering the system

### Guidance
This phase becomes much easier once the scheduler and request lifecycle are already explicit.

---

## Phase 7 — Multi-tier memory and advanced KV residency

### Goal
Improve memory efficiency and support larger contexts by introducing a clearer residency model.

### Affects
- KV manager
- memory accounting
- eviction policy
- prefix reuse retention
- long-context support

### Direction
Move toward logical tiers such as:

- hot active KV state
- warm retained KV state
- optional cold spill mechanisms

Even on unified memory systems, locality and access policy still matter.

### Expected benefits
- more resilient memory behavior
- better support for many active or very large contexts
- improved reuse retention under pressure

### Guidance
This should come after structured KV management exists. Do not jump to tiering before the base cache model is clean.

---

## Phase 8 — Speculative decoding as a mature subsystem

### Goal
Improve decode throughput by introducing a correct, observable, maintainable speculative decoding path.

### Affects
- executor
- cache interaction model
- decode policy
- runtime metrics
- fallback behavior

### Direction
Speculative decoding should be treated as an explicit subsystem with:

- clear operating modes
- correct draft/target cache handling
- verification metrics
- safe fallback logic

### Expected benefits
- higher decode throughput
- better small-batch responsiveness
- potential 1.3×–2× speedup depending on workload and implementation quality

### Guidance
Do not rush this phase. It should only be pursued after scheduler and cache foundations are trustworthy.

---

## Phase 9 — Runtime and MLX optimization

### Goal
Optimize the execution layer once higher-level scheduling and memory architecture are in place.

### Affects
- MLX execution strategy
- graph reuse
- shape bucketing
- host/runtime synchronization
- decode pipeline efficiency

### Direction
Pursue optimizations such as:

- lower synchronization overhead
- more stable execution shapes
- better graph or plan reuse
- tighter decode hot path implementation
- improved memory locality and runtime efficiency

### Expected benefits
- lower kernel overhead
- better sustained throughput
- more consistent latency
- better hardware utilization on Apple Silicon

### Guidance
Kernel-level or runtime-level tuning is most effective after the scheduler and memory model are strong.

---

## Phase 10 — Workload-specialized optimization for agents

### Goal
Make Aster unusually strong for the workloads it is meant to serve.

### Affects
- prefix reuse policy
- cache retention policy
- session-local reuse
- request classification
- scheduler prioritization

### Direction
Optimize explicitly for patterns like:

- repeated long system prompts
- repeated tool schema blocks
- multi-turn agent loops
- stable scaffolding with small deltas

### Expected benefits
- lower latency on real agent workloads
- more effective reuse
- differentiated performance beyond generic local serving

### Guidance
This phase should build on the general-purpose engine improvements from earlier phases.

---

## Cross-cutting concerns

These should be improved throughout all phases.

### Correctness first

Any optimization that risks silent corruption, invalid cache reuse, or broken streaming behavior should be treated conservatively.

### Incremental migration

When possible:

- keep old paths available while introducing new ones
- gate new runtime behavior behind clear switches or rollout boundaries
- validate each phase under real workloads before moving deeper

### Observability-first engineering

Every major runtime change should come with:

- request-level visibility
- system-level metrics
- clear debug hooks
- useful logs for failure diagnosis

### Performance must be measured, not assumed

Each phase should define measurable outcomes such as:

- TTFT changes
- decode throughput changes
- aggregate throughput under concurrency
- cache hit rate changes
- memory behavior under pressure

---

## Practical interpretation for future work

When future tasks touch architecture, they should ask:

1. Which roadmap phase does this belong to?
2. Does it improve the current baseline incrementally?
3. Does it preserve correctness and operational clarity?
4. Is it building toward a scheduler-driven, memory-aware runtime?
5. Is it improving Aster as an inference engine rather than only adding API-layer behavior?

If a proposed change does not fit these principles, it should be reconsidered.

---

## Long-term goal

The long-term objective is for Aster to become:

- MLX-native
- Apple Silicon optimized
- scheduler-driven
- memory-aware
- concurrency-capable
- operationally observable
- well suited for long-context and agent workloads

In other words:

> **Aster should evolve toward a real inference engine for Apple Silicon, not remain just a thin local serving wrapper.**

That evolution should happen carefully, one stable phase at a time.
