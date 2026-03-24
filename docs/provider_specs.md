# Provider Specs

## OpenAI

- Auth: `Authorization: Bearer <api_key>`
- Required headers: `Authorization`, `Content-Type: application/json`
- Versioning: endpoint-family evolution; Responses is the primary modern path, Chat Completions is compatibility-oriented
- Beta/preview: model and endpoint rollout rather than a required version header
- Endpoint families: `/v1/responses`, `/v1/chat/completions`
- Request style: Responses uses response-oriented `input` plus tools and lifecycle fields; Chat uses message arrays
- Response style: Responses returns response lifecycle objects with `output`; Chat returns `choices`
- Streaming: Responses emits named lifecycle events like text deltas and completion events; Chat streams `chat.completion.chunk`
- Tool calling: Responses supports function tools and built-in tools; Chat supports function tools via `tool_calls`
- Structured output: first-class JSON-schema support on both paths, but not with identical request fields
- Multimodal: supported, with path-specific part shapes
- Conversation state: Responses supports response chaining with `previous_response_id`; Chat does not
- Background/deferred: Responses supports background/deferred semantics; Chat does not
- Usage: `input_tokens`, `output_tokens`, `total_tokens` on Responses; prompt/completion token accounting on Chat
- Stop reasons: Responses lifecycle/incomplete details; Chat uses `stop`, `length`, `tool_calls`, `content_filter`
- Error model: structured `error` envelope with request ID headers
- Canonical mismatch notes: Chat cannot represent response lifecycle losslessly; Responses built-in tools remain partly provider-specific
- OpenAI Responses mismatch notes: n/a for native Responses; Chat is intentionally a secondary compatibility surface

## Anthropic

- Auth: `x-api-key`
- Required headers: `x-api-key`, `anthropic-version`
- Versioning: mandatory version header
- Beta/preview: `anthropic-beta` header
- Endpoint families: `/v1/messages`
- Request style: top-level `system` plus `messages`; tool definitions use `input_schema`
- Response style: message object with ordered content blocks
- Streaming: SSE event taxonomy including `message_start`, `content_block_delta`, `message_delta`, `message_stop`, `ping`, `error`
- Tool calling: assistant emits `tool_use`; tool results are sent back as user content blocks of type `tool_result`
- Structured output: no direct cross-provider JSON-schema equivalent; must stay provider-specific or tool-mediated
- Multimodal: content blocks support text and supported media formats
- Conversation state: primarily stateless message replay; no OpenAI-style response lifecycle
- Background/deferred: no direct equivalent
- Usage: `usage.input_tokens`, `usage.output_tokens`
- Stop reasons: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal`
- Error model: typed `error` envelope with request ID header
- Canonical mismatch notes: system prompt is not a normal message; tool results are not a native tool role
- OpenAI Responses mismatch notes: no `previous_response_id`, no equivalent background response lifecycle, different stream event taxonomy

## Google Gemini

- Auth: `x-goog-api-key` on native API; OpenAI compatibility uses the compatibility endpoint contract
- Required headers: native API key header for the native adapter
- Versioning: endpoint version in path, typically `v1beta` for documented Gemini API surface
- Beta/preview: model and endpoint labels; changelog-driven
- Endpoint families: `:generateContent`, `:streamGenerateContent`, secondary OpenAI-compatible chat path
- Request style: `contents` with `parts`, `systemInstruction`, `generationConfig`, `toolConfig`
- Response style: `candidates`, `content.parts`, `usageMetadata`, `promptFeedback`
- Streaming: candidate-fragment streaming rather than OpenAI/Anthropic-style event taxonomies
- Tool calling: `functionCall` and `functionResponse`
- Structured output: native `responseMimeType` + `responseSchema`
- Multimodal: native parts model for text and supported media
- Conversation state: request-local contents history; not response-lifecycle state
- Background/deferred: no direct equivalent documented here
- Usage: `promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`
- Stop reasons: candidate `finishReason`
- Error model: Google-style error payloads
- Canonical mismatch notes: native Gemini parts and thinking config cannot be flattened into chat messages without losing detail
- OpenAI Responses mismatch notes: different request envelope, candidate model, function-call representation, and thinking controls

## Amazon Bedrock

- Auth: AWS SigV4 / IAM credentials
- Required headers: SigV4-signed AWS request headers
- Versioning: API versioning via AWS service API plus model-specific capability matrix
- Beta/preview: model availability and AWS docs/changelog rather than a universal beta header
- Endpoint families: native `Converse`, `ConverseStream`, secondary OpenAI-compatible chat/responses paths
- Request style: `messages`, `system`, `inferenceConfig`, `toolConfig`, `additionalModelRequestFields`
- Response style: `output.message`, `usage`, `stopReason`, `additionalModelResponseFields`
- Streaming: event-union stream frames such as content block deltas, metadata, and stop events
- Tool calling: `toolUse` and `toolResult`
- Structured output: model-specific and commonly routed through provider-specific additional request fields
- Multimodal: model-dependent
- Conversation state: request-local message replay in Converse; no universal provider-managed response state
- Background/deferred: no general Bedrock-native equivalent exposed here
- Usage: `inputTokens`, `outputTokens`, `totalTokens`
- Stop reasons: model- and Bedrock-specific, surfaced as `stopReason`
- Error model: AWS service errors with request IDs
- Canonical mismatch notes: Converse capabilities depend on hosted model family; additional native fields must remain in extensions
- OpenAI Responses mismatch notes: even the compatible endpoints run inside AWS auth, model routing, and feature-gating semantics

## Mistral

- Auth: `Authorization: Bearer <api_key>`
- Required headers: `Authorization`
- Versioning: model/version naming and endpoint evolution
- Beta/preview: agents/conversations surfaces may evolve independently from chat/completions
- Endpoint families: native `/v1/chat/completions`, agent/conversation-oriented surface
- Request style: chat-completions is OpenAI-like; conversations/agents add conversation lifecycle semantics
- Response style: chat-completions uses `choices`; conversation surfaces may attach additional lifecycle data
- Streaming: chat-completions is SSE chunk oriented
- Tool calling: function-style tools
- Structured output: available through response-format style controls
- Multimodal: model-dependent
- Conversation state: separate conversations/agents surface instead of OpenAI Responses lifecycle
- Background/deferred: not modeled as OpenAI-style background execution
- Usage: OpenAI-like chat accounting
- Stop reasons: chat-completion finish reasons
- Error model: provider JSON errors
- Canonical mismatch notes: agent/conversation APIs should not be collapsed into chat-completions
- OpenAI Responses mismatch notes: no direct `previous_response_id` lifecycle equivalent on chat-completions

## Cohere

- Auth: `Authorization: Bearer <api_key>`
- Required headers: `Authorization`
- Versioning: v2 Chat API
- Beta/preview: release-note driven
- Endpoint families: `/v2/chat`, `/v2/chat` streaming
- Request style: message list plus Cohere tool and response-format controls
- Response style: `message` object plus usage information
- Streaming: named event stream with message and tool events such as `content-delta` and `tool-call-start`
- Tool calling: provider-native tool events and tool result messages
- Structured output: provider-specific response-format controls
- Multimodal: evolving/model-specific; not treated as universally equivalent here
- Conversation state: request-local replay
- Background/deferred: no direct equivalent
- Usage: usage token accounting nested under usage/tokens
- Stop reasons: provider finish reason on message end
- Error model: provider JSON errors
- Canonical mismatch notes: event taxonomy and tool streaming are distinct from OpenAI/Anthropic
- OpenAI Responses mismatch notes: no response-lifecycle object model, different stream event taxonomy, different tool protocol

## xAI

- Auth: `Authorization: Bearer <api_key>`
- Required headers: `Authorization`
- Versioning: modern inference surface plus legacy chat-completions surface
- Beta/preview: docs-driven model capability evolution
- Endpoint families: preferred modern response-style inference path, legacy chat-completions path
- Request style: OpenAI-like at first glance, but includes xAI-native fields such as citations, truncation, reasoning, and provider-native tools
- Response style: response-style or chat-completion-style depending on endpoint family
- Streaming: OpenAI-like chunking, but semantics are not assumed identical
- Tool calling: user-defined functions plus xAI-native tools such as web search
- Structured output: supported on modern and compatibility paths where documented
- Multimodal: model-dependent
- Conversation state: modern path may support stored or linked responses; legacy chat is request-local
- Background/deferred: not assumed equivalent to OpenAI unless documented on the specific surface
- Usage: provider usage accounting plus provider-native extensions such as citations/reasoning metadata
- Stop reasons: endpoint-family specific
- Error model: provider JSON error envelopes
- Canonical mismatch notes: xAI built-in tools and citation metadata stay provider-specific
- OpenAI Responses mismatch notes: similar shapes do not imply identical semantics; xAI-native tools and reasoning metadata must remain explicit
