from __future__ import annotations

import pytest

from aster.api.feature_emulation import (
    FeaturePlan,
    ToolCallResult,
    ToolSpec,
    validate_tool_call_arguments,
)
from aster.core.errors import AsterError


def _plan(parameters: dict[str, object]) -> FeaturePlan:
    return FeaturePlan(
        mode="tools",
        tools=[
            ToolSpec(
                name="bounded_tool",
                description=None,
                parameters=parameters,
            )
        ],
    )


def test_validate_tool_call_arguments_accepts_common_bounds() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 3,
                "maxLength": 6,
                "pattern": "^[a-z]+$",
            },
            "score": {"type": "number", "minimum": 1, "maximum": 10},
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string"},
            },
        },
        "required": ["query", "score", "tags"],
        "additionalProperties": False,
    }

    validate_tool_call_arguments(
        ToolCallResult(
            call_id="call_1",
            name="bounded_tool",
            arguments={"query": "alpha", "score": 7, "tags": ["one", "two"]},
        ),
        _plan(parameters),
    )


@pytest.mark.parametrize(
    ("property_schema", "arguments", "expected_message"),
    [
        (
            {"query": {"type": "string", "minLength": 3}},
            {"query": "ab"},
            "$.query must contain at least 3 characters.",
        ),
        (
            {"query": {"type": "string", "maxLength": 3}},
            {"query": "abcd"},
            "$.query must contain at most 3 characters.",
        ),
        (
            {"query": {"type": "string", "pattern": "^[0-9]+$"}},
            {"query": "abc"},
            "$.query does not match required pattern.",
        ),
        (
            {"score": {"type": "number", "minimum": 1}},
            {"score": 0},
            "$.score must be greater than or equal to 1.",
        ),
        (
            {"score": {"type": "number", "maximum": 10}},
            {"score": 11},
            "$.score must be less than or equal to 10.",
        ),
        (
            {"score": {"type": "number", "exclusiveMinimum": 0}},
            {"score": 0},
            "$.score must be greater than 0.",
        ),
        (
            {"score": {"type": "number", "exclusiveMaximum": 10}},
            {"score": 10},
            "$.score must be less than 10.",
        ),
        (
            {"tags": {"type": "array", "minItems": 2}},
            {"tags": ["one"]},
            "$.tags must contain at least 2 items.",
        ),
        (
            {"tags": {"type": "array", "maxItems": 1}},
            {"tags": ["one", "two"]},
            "$.tags must contain at most 1 items.",
        ),
    ],
)
def test_validate_tool_call_arguments_rejects_common_bounds(
    property_schema: dict[str, object],
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    parameters = {
        "type": "object",
        "properties": property_schema,
        "additionalProperties": False,
    }

    with pytest.raises(AsterError) as exc_info:
        validate_tool_call_arguments(
            ToolCallResult(
                call_id="call_1",
                name="bounded_tool",
                arguments=arguments,
            ),
            _plan(parameters),
        )

    assert exc_info.value.code == "tool_arguments_invalid"
    assert expected_message in exc_info.value.message
