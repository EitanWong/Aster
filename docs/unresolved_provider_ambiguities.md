# Unresolved Provider Ambiguities

These are explicit notes for follow-up rather than silent assumptions:

- Mistral conversations/agents: the repository now models this surface separately, but the exact native lifecycle and path variants need a deeper pass against the latest agents docs before claiming full conformance.
- Bedrock OpenAI-compatible endpoints: the compatibility adapters are scaffolded separately, but capability support remains model-dependent and should be verified against target regions/models before broad production enablement.
- Cohere structured outputs: Cohere response-format support is kept provider-specific for now rather than promoted into broader canonical guarantees.
- Anthropic structured outputs: no direct canonical JSON-schema mapping is treated as lossless today; tool-mediated or provider-specific handling remains explicit.
- Gemini multimodal external-URI handling: native Gemini parts support multiple media patterns, but arbitrary canonical image URLs are not coerced silently without explicit native extension data.
- xAI modern inference semantics: xAI’s preferred modern path is modeled separately from legacy chat semantics, but additional request/response controls should be widened as the official docs stabilize.
