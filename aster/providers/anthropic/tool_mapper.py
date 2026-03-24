from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import CanonicalToolCall, CanonicalToolDefinition, CanonicalToolResult
from aster.providers._shared import function_call_to_canonical


def build_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    payloads = []
    for tool in tools:
        payload = {
            "name": tool.name,
            "input_schema": tool.input_schema,
        }
        if tool.description:
            payload["description"] = tool.description
        payload.update(tool.provider_extensions.values)
        payloads.append(payload)
    return payloads


def parse_tool_use(block: Mapping[str, Any]) -> CanonicalToolCall:
    return function_call_to_canonical(
        call_id=str(block.get("id") or "tool_use"),
        name=str(block.get("name") or "tool"),
        arguments_json=_input_json(block.get("input")),
        provider_extensions={key: value for key, value in block.items() if key not in {"type", "id", "name", "input"}},
    )


def tool_result_block(result: CanonicalToolResult) -> dict[str, Any]:
    content = result.output if isinstance(result.output, str) else str(result.output)
    return {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": content,
        "is_error": result.is_error,
    }


def _input_json(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    from aster.providers._shared import stable_json_dumps

    return stable_json_dumps(value)
