from __future__ import annotations

RESPONSES_DOCS = (
    "https://platform.openai.com/docs/api-reference/responses",
    "https://platform.openai.com/docs/guides/structured-outputs",
    "https://platform.openai.com/docs/guides/function-calling",
    "https://platform.openai.com/docs/guides/streaming-responses",
)

CHAT_DOCS = (
    "https://platform.openai.com/docs/api-reference/chat",
    "https://platform.openai.com/docs/guides/structured-outputs",
    "https://platform.openai.com/docs/guides/function-calling",
)

RESPONSES_KNOWN_KEYS = {
    "id",
    "object",
    "status",
    "model",
    "output",
    "usage",
    "metadata",
    "previous_response_id",
    "store",
    "parallel_tool_calls",
    "incomplete_details",
}

CHAT_KNOWN_KEYS = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "usage",
    "system_fingerprint",
    "service_tier",
}
