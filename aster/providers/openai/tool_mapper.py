from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import CanonicalToolCall, CanonicalToolDefinition
from aster.providers._shared import function_call_to_canonical, tool_definition_to_function


def build_responses_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    native_tools: list[dict[str, Any]] = []
    for tool in tools:
        tool_type = tool.provider_extensions.values.get("type", "function")
        if tool_type == "function":
            payload = {"type": "function"}
            payload.update(tool_definition_to_function(tool))
            native_tools.append(payload)
            continue
        payload = dict(tool.provider_extensions.values)
        payload.setdefault("type", tool_type)
        if tool.description:
            payload.setdefault("description", tool.description)
        native_tools.append(payload)
    return native_tools


def build_chat_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool_definition_to_function(tool)} for tool in tools]


def parse_response_output_tool_call(item: Mapping[str, Any]) -> CanonicalToolCall:
    return function_call_to_canonical(
        call_id=str(item.get("call_id") or item.get("id") or "tool_call"),
        name=str(item.get("name") or "function"),
        arguments_json=_arguments_json(item),
        provider_extensions={key: value for key, value in item.items() if key not in {"call_id", "id", "name", "arguments", "status"}},
    )


def parse_chat_tool_call(item: Mapping[str, Any]) -> CanonicalToolCall:
    function_data = item.get("function", {})
    if not isinstance(function_data, Mapping):
        function_data = {}
    return function_call_to_canonical(
        call_id=str(item.get("id") or "tool_call"),
        name=str(function_data.get("name") or "function"),
        arguments_json=function_data.get("arguments") if isinstance(function_data.get("arguments"), str) else None,
        provider_extensions={key: value for key, value in item.items() if key not in {"id", "function", "type"}},
    )


def _arguments_json(item: Mapping[str, Any]) -> str | None:
    arguments = item.get("arguments")
    return arguments if isinstance(arguments, str) else None
