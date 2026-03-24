from __future__ import annotations

MESSAGES_DOCS = (
    "https://docs.anthropic.com/en/api/messages",
    "https://docs.anthropic.com/en/api/messages-streaming",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview",
    "https://docs.anthropic.com/en/release-notes/api",
)

KNOWN_KEYS = {
    "id",
    "type",
    "role",
    "content",
    "model",
    "stop_reason",
    "stop_sequence",
    "usage",
}
