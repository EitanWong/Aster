from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aster.core.canonical import CanonicalToolDefinition
from aster.providers._shared import tool_definition_to_function


def build_responses_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    payloads = []
    for tool in tools:
        if tool.provider_extensions.values.get("type") == "web_search":
            payloads.append({"type": "web_search", **tool.provider_extensions.values})
            continue
        payloads.append({"type": "function", **tool_definition_to_function(tool)})
    return payloads


def build_chat_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool_definition_to_function(tool)} for tool in tools]
