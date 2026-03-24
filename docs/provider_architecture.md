# Provider Architecture

## Goal

Aster uses a dual-layer provider design:

- Canonical abstraction: a stable internal domain model for Aster runtime logic, tests, routing, tracing, retries, and upper-layer orchestration.
- Provider-native adapters: dedicated implementations that preserve each provider's official request, response, streaming, tool, versioning, and error semantics as faithfully as possible.

This avoids the common failure mode of "OpenAI-shaped everything" where materially different APIs get flattened into a lossy pseudo-standard.

## Why A Single Lossy Layer Is Unacceptable

A shallow single abstraction causes correctness bugs:

- Conversation state is not the same across OpenAI Responses, Anthropic Messages, Gemini contents, Bedrock Converse, and agent/conversation APIs.
- Tool use is structurally similar across vendors but not semantically identical. Anthropic tool results are user content blocks, Gemini uses `functionCall` / `functionResponse`, Bedrock uses `toolUse` / `toolResult`, and OpenAI Chat uses assistant `tool_calls` plus tool-role messages.
- Streaming event taxonomies differ sharply. OpenAI Responses emits named lifecycle events, Anthropic emits structured SSE events such as `content_block_delta`, Bedrock uses event-union frames, Cohere uses stream event names like `tool-call-start`, and Gemini streams candidate fragments.
- Structured output support varies from first-class JSON-schema support to provider-specific or model-specific controls.
- Authentication, versioning, beta headers, and error envelopes are all provider-specific.

If Aster flattened these differences into one shape, it would either silently drop data or lie about provider behavior. Both are unacceptable in long-lived infrastructure.

## Canonical Layer

The canonical layer lives under `aster/core/` and defines:

- provider identity: `ProviderRef`, `ModelRef`
- request model: `CanonicalRequest`, `CanonicalMessage`, `CanonicalContentPart`
- tool model: `CanonicalToolDefinition`, `CanonicalToolCall`, `CanonicalToolResult`
- response model: `CanonicalResponseChunk`, `CanonicalFinalResponse`, `CanonicalStreamEvent`
- accounting and errors: `CanonicalUsage`, `CanonicalError`
- capability model: `ProviderFeatureFlags`, `ProviderCapabilities`
- traceability and escape hatches: `ProviderExtensionData`, `RawProviderEnvelope`

Design rules:

- Canonical fields only exist when they represent real cross-provider semantics.
- Provider-specific data that cannot be normalized losslessly is stored in `ProviderExtensionData`.
- Unknown or future provider fields are preserved rather than discarded whenever practical.
- Unsupported combinations fail with typed errors instead of silent coercion.

## Provider-Native Layer

Each provider family has its own package under `aster/providers/<provider>/` with explicit boundaries:

- `models.py`: provider-specific constants and known schema markers
- `request_builder.py`: canonical request to provider-native outbound shape
- `response_parser.py`: provider-native payload to canonical final response
- `stream_decoder.py`: provider-native streaming frames to canonical chunks
- `tool_mapper.py`: tool definition and tool call mapping
- `capabilities.py`: provider capability descriptor
- `errors.py`: provider-native error mapping
- `extensions.py`: provider-specific extension preservation
- `adapter.py`: composition root implementing the adapter contract

## Mapping Strategy

Mapping is intentionally asymmetric:

- Canonical to native: convert only fields with true semantic equivalents.
- Native to canonical: normalize shared semantics and preserve the rest in extensions.
- Native-only features: expose via `provider_options` on requests and `provider_extensions` on parsed outputs.
- Unsupported features: raise a typed `UnsupportedProviderFeatureError`.

Examples:

- OpenAI Responses `previous_response_id` maps to canonical conversation linkage. OpenAI Chat does not, so the Chat adapter rejects it explicitly.
- Anthropic system prompts are lifted out of canonical system/developer messages into top-level `system`.
- Gemini structured output maps to `generationConfig.responseMimeType` plus `responseSchema`.
- Bedrock native advanced fields remain in `additionalModelRequestFields` / `additionalModelResponseFields` extensions.
- xAI built-in tools such as web search remain provider-specific tools instead of being mislabeled as generic OpenAI tools.

## Extension Strategy

Aster uses two explicit escape hatches:

- `provider_options` on `CanonicalRequest`: provider-native outbound controls that should not pollute canonical fields.
- `ProviderExtensionData` on canonical entities: preserved inbound or outbound native metadata that has no safe canonical home.

This lets Aster stay internally stable while still tracking provider evolution without emergency type churn.

## Runtime Boundaries

The runtime layer in `aster/runtime/` is intentionally thin:

- `ProviderRegistry`: adapter registration and lookup
- `ProviderRouter`: model-to-adapter resolution
- `ProviderDispatcher`: request preparation, response parsing, and stream decoding

This keeps transport logic out of upper layers and prevents provider-specific hacks from leaking into application code.

## Testing Philosophy

The adapter layer is treated like infrastructure:

- unit tests validate canonical models and isolated mappers
- contract tests validate outbound request shapes and explicit rejection paths
- conformance tests compare the same canonical request across adapters and assert intentional differences
- streaming tests validate provider event decoding and malformed-event handling
- regression tests lock in extension preservation and previously fragile edge cases

## Evolution Policy

When providers drift:

1. update the provider-native adapter first
2. preserve new native fields in extensions immediately
3. only promote fields into the canonical layer if they represent durable multi-provider semantics
4. add fixtures and regression tests before broadening normalization

That policy keeps the canonical model stable without freezing Aster's ability to adopt provider-native features.
