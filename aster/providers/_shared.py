from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalToolCall,
    CanonicalToolDefinition,
    CanonicalUsage,
    ContentPartType,
    MessageRole,
    ProviderExtensionData,
)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_json_or_none(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_extensions(payload: Mapping[str, Any], known_keys: Iterable[str]) -> ProviderExtensionData:
    known = set(known_keys)
    return ProviderExtensionData(values={key: value for key, value in payload.items() if key not in known})


def text_part(text: str, *, provider_extensions: dict[str, Any] | None = None) -> CanonicalContentPart:
    return CanonicalContentPart(
        type=ContentPartType.TEXT,
        text=text,
        provider_extensions=ProviderExtensionData(values=provider_extensions or {}),
    )


def json_part(value: Any, *, provider_extensions: dict[str, Any] | None = None) -> CanonicalContentPart:
    return CanonicalContentPart(
        type=ContentPartType.JSON,
        json_value=value,
        provider_extensions=ProviderExtensionData(values=provider_extensions or {}),
    )


def provider_native_part(
    value: Any,
    *,
    provider_extensions: dict[str, Any] | None = None,
) -> CanonicalContentPart:
    return CanonicalContentPart(
        type=ContentPartType.PROVIDER_NATIVE,
        json_value=value,
        provider_extensions=ProviderExtensionData(values=provider_extensions or {}),
    )


def join_text_parts(parts: Iterable[CanonicalContentPart]) -> str:
    fragments = [part.text for part in parts if part.text]
    return "\n".join(fragment for fragment in fragments if fragment).strip()


def system_message_text(messages: Iterable[CanonicalMessage]) -> str | None:
    fragments = [
        join_text_parts(message.content)
        for message in messages
        if message.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}
    ]
    joined = "\n".join(fragment for fragment in fragments if fragment).strip()
    return joined or None


def non_system_messages(messages: Iterable[CanonicalMessage]) -> list[CanonicalMessage]:
    return [
        message
        for message in messages
        if message.role not in {MessageRole.SYSTEM, MessageRole.DEVELOPER}
    ]


def tool_definition_to_function(tool: CanonicalToolDefinition) -> dict[str, Any]:
    payload = {
        "name": tool.name,
        "parameters": tool.input_schema,
    }
    if tool.description:
        payload["description"] = tool.description
    if tool.strict is not None:
        payload["strict"] = tool.strict
    payload.update(tool.provider_extensions.values)
    return payload


def function_call_to_canonical(
    *,
    call_id: str,
    name: str,
    arguments_json: str | None,
    provider_extensions: dict[str, Any] | None = None,
) -> CanonicalToolCall:
    return CanonicalToolCall(
        call_id=call_id,
        name=name,
        arguments_json=arguments_json,
        arguments=parse_json_or_none(arguments_json),
        provider_extensions=ProviderExtensionData(values=provider_extensions or {}),
    )


def usage_from_token_payload(
    payload: Mapping[str, Any] | None,
    *,
    input_key: str,
    output_key: str,
    total_key: str,
    reasoning_key: str | None = None,
    extra_keys: Iterable[str] = (),
) -> CanonicalUsage:
    data = payload or {}
    extensions = {
        key: data[key]
        for key in extra_keys
        if key in data
    }
    return CanonicalUsage(
        input_tokens=_int_or_none(data.get(input_key)),
        output_tokens=_int_or_none(data.get(output_key)),
        total_tokens=_int_or_none(data.get(total_key)),
        reasoning_tokens=_int_or_none(data.get(reasoning_key)) if reasoning_key else None,
        provider_extensions=ProviderExtensionData(values=extensions),
    )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
