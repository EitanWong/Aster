# Implementation Roadmap

## Phase 1: Canonical Core And OpenAI

- finalize canonical types, capability model, adapter contracts, and runtime registry/dispatch
- complete OpenAI Responses and Chat Completions request/response/stream/error coverage
- add fixtures for response payloads and stream events
- lock in unsupported-feature behavior with contract tests

## Phase 2: Anthropic

- deepen Anthropic content-block coverage
- harden streaming for `text_delta`, `input_json_delta`, pings, stop events, and malformed frames
- expand beta-header and version-header contract coverage
- add regression tests for tool result mapping and system prompt extraction

## Phase 3: Gemini

- expand native part coverage for multimodal and file-backed inputs
- harden `responseSchema` and thinking controls
- cover Gemini OpenAI compatibility endpoint separately from native semantics
- add drift fixtures around candidate/finish-reason behavior

## Phase 4: Bedrock

- enrich Converse support for model-specific capability matrices
- formalize AWS OpenAI-compatible endpoint request/response fixtures
- add more metadata and additional-model-field preservation tests
- introduce model-capability lookup hooks keyed by region/model family

## Phase 5: Mistral, Cohere, xAI

- flesh out Mistral conversation/agent lifecycle semantics from current docs
- expand Cohere stream-event coverage for tool-plan and tool-call deltas
- harden xAI modern inference semantics and native tool preservation
- add provider-specific regression fixtures for citation, reasoning, and tool metadata

## Phase 6: Conformance Hardening

- broaden cross-provider conformance matrix
- add malformed payload and truncated-stream robustness cases
- add compatibility-drift fixtures for docs that change frequently
- document canonical promotion criteria for new shared features

## Phase 7: Runtime Integration

- integrate dispatcher into higher-level gateway surfaces
- add tracing, retry hooks, cancellation, and timeout orchestration
- wire provider adapters into a transport client layer
- add integration tests with mock HTTP transports

## Phase 8: Production Hardening

- CI matrix for adapter tests
- richer observability around raw envelopes and provider error taxonomy
- backwards-safe canonical versioning guidance
- release process for provider drift review and fixture refresh
