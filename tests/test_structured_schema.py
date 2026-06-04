from __future__ import annotations

import pytest

from aster.api.feature_emulation import (
    build_structured_plan,
    decode_local_output,
    validate_structured_output_text,
)
from aster.core.errors import AsterError
from aster.inference.structured_schema import normalize_json_schema


def test_normalize_json_schema_resolves_refs_and_type_arrays() -> None:
    schema = {
        "$defs": {
            "answer": {
                "type": ["string", "null"],
                "description": "metadata should not be passed to enforcers",
            }
        },
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "required": ["answer"],
    }

    normalized = normalize_json_schema(schema)

    assert "description" not in normalized["properties"]["answer"]
    assert normalized["properties"]["answer"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert normalized["additionalProperties"] is False


def test_structured_plan_validation_supports_normalized_anyof() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {"answer": {"type": ["string", "null"]}},
            "required": ["answer"],
        }
    )

    decoded = decode_local_output('{"answer": null}', plan)

    assert decoded.structured_data == {"answer": None}


@pytest.mark.parametrize(
    ("property_schema", "output_text", "expected_message"),
    [
        (
            {"answer": {"type": "string", "minLength": 3}},
            '{"answer":"ok"}',
            "$.answer must contain at least 3 characters.",
        ),
        (
            {"confidence": {"type": "number", "maximum": 1}},
            '{"confidence":1.5}',
            "$.confidence must be less than or equal to 1.",
        ),
        (
            {"tags": {"type": "array", "minItems": 2}},
            '{"tags":["sunny"]}',
            "$.tags must contain at least 2 items.",
        ),
    ],
)
def test_structured_plan_validation_rejects_common_bounds(
    property_schema: dict[str, object],
    output_text: str,
    expected_message: str,
) -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": property_schema,
            "additionalProperties": False,
        }
    )

    with pytest.raises(AsterError) as exc_info:
        decode_local_output(output_text, plan)

    assert exc_info.value.code == "structured_output_invalid"
    assert expected_message in exc_info.value.message


def test_structured_output_extracts_later_balanced_json_candidate() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    decoded = decode_local_output('draft {"answer": } final {"answer": "ok"} done', plan)

    assert decoded.structured_data == {"answer": "ok"}


def test_structured_output_balanced_json_respects_strings_and_nested_values() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["answer", "items"],
            "additionalProperties": False,
        }
    )

    decoded = decode_local_output('prefix {"answer": "literal } brace", "items": [1, 2]} suffix', plan)

    assert decoded.structured_data == {"answer": "literal } brace", "items": [1, 2]}


def test_structured_output_repairs_truncated_json_object() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["answer", "count"],
            "additionalProperties": False,
        }
    )

    decoded = decode_local_output('```json\n{"answer": "ok", "count": 3', plan)

    assert decoded.structured_data == {"answer": "ok", "count": 3}


def test_structured_output_repairs_truncated_nested_json() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["answer", "items"],
            "additionalProperties": False,
        }
    )

    decoded = decode_local_output('{"answer": "ok", "items": [1, 2, 3', plan)

    assert decoded.structured_data == {"answer": "ok", "items": [1, 2, 3]}


def test_structured_output_drops_dangling_key_when_repairing() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    decoded = decode_local_output('{"answer": "ok", "extra":', plan)

    assert decoded.structured_data == {"answer": "ok"}


def test_structured_stream_validation_can_disable_truncated_json_repair() -> None:
    plan = build_structured_plan(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    with pytest.raises(AsterError) as exc_info:
        validate_structured_output_text('{"answer": "ok"', plan, allow_repair=False)
    assert exc_info.value.code == "invalid_json_output"
