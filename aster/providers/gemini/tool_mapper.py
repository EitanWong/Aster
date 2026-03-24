from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import CanonicalToolCall, CanonicalToolDefinition
from aster.providers._shared import function_call_to_canonical


def build_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    declarations = []
    for tool in tools:
        declaration = {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }
        declaration.update(tool.provider_extensions.values)
        declarations.append({"functionDeclarations": [declaration]})
    return declarations


def parse_function_call(part: Mapping[str, Any]) -> CanonicalToolCall:
    function_call = part.get("functionCall", {})
    if not isinstance(function_call, Mapping):
        function_call = {}
    arguments = function_call.get("args") if isinstance(function_call.get("args"), Mapping) else {}
    from aster.providers._shared import stable_json_dumps

    return function_call_to_canonical(
        call_id=str(function_call.get("id") or function_call.get("name") or "function_call"),
        name=str(function_call.get("name") or "function"),
        arguments_json=stable_json_dumps(arguments),
        provider_extensions={key: value for key, value in part.items() if key != "functionCall"},
    )
