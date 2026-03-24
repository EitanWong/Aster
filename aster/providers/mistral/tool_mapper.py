from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aster.core.canonical import CanonicalToolDefinition
from aster.providers._shared import tool_definition_to_function


def build_tools(tools: Iterable[CanonicalToolDefinition]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": tool_definition_to_function(tool)} for tool in tools]


def normalize_tool_choice(choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    return choice
