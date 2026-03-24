from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ToolExecutionContext:
    request_id: str
    provider: str
    api_family: str
    model: str


ToolHandler = Callable[[dict[str, Any], ToolExecutionContext], Any | Awaitable[Any]]


@dataclass(slots=True)
class RegisteredTool:
    name: str
    description: str | None
    handler: ToolHandler


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, RegisteredTool] = field(default_factory=dict)

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str | None = None,
    ) -> None:
        self._tools[name] = RegisteredTool(name=name, description=description, handler=handler)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        result = tool.handler(arguments, context)
        if inspect.isawaitable(result):
            return await result
        return result


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", _tool_echo, description="Return the provided text.")
    registry.register("add_numbers", _tool_add_numbers, description="Add two numbers and return their sum.")
    registry.register("get_current_time", _tool_get_current_time, description="Return the current UTC time in ISO-8601 format.")
    return registry


def _tool_echo(arguments: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
    return {"text": arguments.get("text")}


def _tool_add_numbers(arguments: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
    a = arguments.get("a", 0)
    b = arguments.get("b", 0)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("a and b must both be numeric.")
    return {"sum": a + b}


def _tool_get_current_time(_arguments: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
    return {"utc": datetime.now(timezone.utc).isoformat()}
