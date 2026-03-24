from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import CanonicalToolCall, CanonicalToolDefinition, CanonicalToolResult
from aster.providers._shared import function_call_to_canonical


def build_tool_config(tools: Iterable[CanonicalToolDefinition], tool_choice: str | dict[str, Any] | None) -> dict[str, Any]:
    tool_specs = []
    for tool in tools:
        tool_specs.append(
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": {"json": tool.input_schema},
                }
            }
        )
    payload: dict[str, Any] = {"tools": tool_specs}
    if tool_choice is not None:
        payload["toolChoice"] = tool_choice if isinstance(tool_choice, dict) else {"auto": {} if tool_choice == "auto" else {"name": tool_choice}}
    return payload


def parse_tool_use(block: Mapping[str, Any]) -> CanonicalToolCall:
    tool_use = block.get("toolUse", {})
    if not isinstance(tool_use, Mapping):
        tool_use = {}
    from aster.providers._shared import stable_json_dumps

    arguments = tool_use.get("input") if isinstance(tool_use.get("input"), Mapping) else {}
    return function_call_to_canonical(
        call_id=str(tool_use.get("toolUseId") or "tool_use"),
        name=str(tool_use.get("name") or "tool"),
        arguments_json=stable_json_dumps(arguments),
        provider_extensions={key: value for key, value in block.items() if key != "toolUse"},
    )


def tool_result_block(result: CanonicalToolResult) -> dict[str, Any]:
    content = result.output if isinstance(result.output, dict) else {"text": str(result.output)}
    inner_content = [{"json": content}] if isinstance(result.output, dict) else [{"text": str(result.output)}]
    payload = {
        "toolResult": {
            "toolUseId": result.call_id,
            "content": inner_content,
        }
    }
    if result.is_error:
        payload["toolResult"]["status"] = "error"
    return payload
