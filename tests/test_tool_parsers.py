from __future__ import annotations

import json

from aster.api.feature_emulation import (
    StreamingJsonFenceStripper,
    ToolChoice,
    ToolSpec,
    build_tool_plan,
    decode_local_output,
)
from aster.inference.parser_pipeline import ParserPipeline
from aster.inference.tool_parsers import AutoToolParser, ToolParserManager


def test_auto_tool_parser_extracts_qwen_xml_tool_call() -> None:
    parser = AutoToolParser()

    parsed = parser.extract_tool_calls(
        '<tool_call>{"name":"add_numbers","arguments":{"a":2,"b":3}}</tool_call>'
    )

    assert parsed.tools_called is True
    assert parsed.tool_calls[0]["name"] == "add_numbers"
    assert parsed.tool_calls[0]["arguments"] == {"a": 2, "b": 3}


def test_auto_tool_parser_extracts_minimax_tool_call() -> None:
    parser = AutoToolParser()

    parsed = parser.extract_tool_calls(
        '<minimax:tool_call><invoke name="lookup_weather">'
        '<parameter name="city">"Shanghai"</parameter>'
        '<parameter name="days">3</parameter>'
        "</invoke></minimax:tool_call>"
    )

    assert parsed.tools_called is True
    assert parsed.tool_calls[0]["name"] == "lookup_weather"
    assert parsed.tool_calls[0]["arguments"] == {"city": "Shanghai", "days": 3}


def test_auto_tool_parser_does_not_hijack_name_only_json() -> None:
    parser = AutoToolParser()

    parsed = parser.extract_tool_calls('{"name":"John","age":25}')

    assert parsed.tools_called is False
    assert parsed.content == '{"name":"John","age":25}'


def test_tool_parser_manager_registers_auto_parser() -> None:
    parser_cls = ToolParserManager.get_tool_parser("auto")

    assert parser_cls is AutoToolParser


def test_auto_tool_parser_exposes_single_token_extra_stop_ids() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            if text == "<|tool_response>":
                return [777]
            return [ord(char) for char in text]

    parser = AutoToolParser()

    assert parser.stop_token_ids(FakeTokenizer()) == frozenset({777})


def test_auto_tool_parser_skips_multi_token_extra_stop_ids() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(char) for char in text]

    parser = AutoToolParser()

    assert parser.stop_token_ids(FakeTokenizer()) == frozenset()


def test_feature_emulation_decodes_auto_tool_formats() -> None:
    plan = build_tool_plan(
        tools=[
            ToolSpec(
                name="add_numbers",
                description=None,
                parameters={"type": "object"},
            )
        ],
        tool_choice=ToolChoice(mode="required"),
    )

    decoded = decode_local_output(
        "[Calling tool: add_numbers({\"a\":2,\"b\":3})]",
        plan,
    )

    assert decoded.tool_calls[0].name == "add_numbers"
    assert decoded.tool_calls[0].arguments == {"a": 2, "b": 3}


def test_streaming_json_fence_stripper_strips_split_fences() -> None:
    stripper = StreamingJsonFenceStripper()
    chunks = ["```", 'json\n{"answer":', '"Sunny"}', "\n`", "``\n"]
    emitted = "".join(stripper.feed(chunk) for chunk in chunks) + stripper.finalize()

    assert emitted == '{"answer":"Sunny"}'


def test_streaming_json_fence_stripper_preserves_unfenced_backticks() -> None:
    stripper = StreamingJsonFenceStripper()
    emitted = stripper.feed('{"note":"use `code`"}') + stripper.finalize()

    assert emitted == '{"note":"use `code`"}'


def test_auto_tool_parser_streaming_suppresses_split_bracket_call() -> None:
    parser = AutoToolParser()
    current = ""
    content_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []

    for chunk in ["Before ", '[Calling tool: add_numbers({"a":', '2,"b":3})]', " After"]:
        previous = current
        current += chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk)
        if delta is None:
            continue
        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)
        tool_calls.extend(item for item in delta.get("tool_calls", []) if isinstance(item, dict))

    assert "".join(content_parts) == "Before  After"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"] == {"name": "add_numbers", "arguments": ""}
    assert "".join(tool_call["function"]["arguments"] for tool_call in tool_calls) == '{"a":2,"b":3}'
    assert tool_calls[-1]["finished"] is True


def test_auto_tool_parser_streaming_holds_split_xml_marker() -> None:
    parser = AutoToolParser()
    current = ""
    outputs: list[dict[str, object]] = []

    for chunk in [
        "Hello <too",
        'l_call>{"name":"lookup_weather","arguments":{"city":"Shanghai"}}</tool_call>',
        " done",
    ]:
        previous = current
        current += chunk
        delta = parser.extract_tool_calls_streaming(previous, current, chunk)
        if delta is not None:
            outputs.append(delta)

    assert outputs[0] == {"content": "Hello "}
    tool_call_deltas = outputs[1]["tool_calls"]
    assert tool_call_deltas[0]["function"] == {"name": "lookup_weather", "arguments": ""}
    assert "".join(delta["function"]["arguments"] for delta in tool_call_deltas) == '{"city":"Shanghai"}'
    assert tool_call_deltas[-1]["finished"] is True
    assert outputs[2] == {"content": " done"}


def test_auto_tool_parser_streaming_emits_json_tool_call_argument_deltas() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        "Before ",
        '<tool_call>{"name":"lookup_weather","arguments":{"city":',
        '"Shanghai"',
        ',"unit":"c"}}',
        "</tool_call> After",
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert tool_deltas[0].name == "lookup_weather"
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"city":"Shanghai","unit":"c"}'
    assert [delta.finished for delta in tool_deltas] == [False, False, False, False, True]


def test_auto_tool_parser_streaming_json_arguments_respects_nested_values() -> None:
    parser = AutoToolParser()
    tool_deltas = []

    for chunk in [
        '<tool_call>{"name":"write_record","arguments":{"items":[{"id":',
        '1,"tags":["a","b"]}],"note":"brace } in string"}}',
        "</tool_call>",
    ]:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    arguments = "".join(delta.arguments_delta for delta in tool_deltas)
    assert json.loads(arguments) == {
        "items": [{"id": 1, "tags": ["a", "b"]}],
        "note": "brace } in string",
    }
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_bare_function_body_finishes_call() -> None:
    parser = AutoToolParser()
    tool_deltas = []

    for chunk in [
        '<function=lookup_weather>{"city":',
        '"Shanghai"}</function>',
    ]:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    assert tool_deltas[0].name == "lookup_weather"
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"city":"Shanghai"}'
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_raw_json_tool_protocol_finishes_call() -> None:
    parser = AutoToolParser()
    parser.configure_request(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    )
    content = ""
    tool_deltas = []

    for chunk in [
        '{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":',
        '2,"b":3}}]}',
    ]:
        delta = parser.parse_delta(chunk)
        content += delta.content_delta
        tool_deltas.extend(delta.tool_call_deltas)

    assert content == ""
    assert tool_deltas[0].name == "add_numbers"
    assert json.loads("".join(delta.arguments_delta for delta in tool_deltas)) == {"a": 2, "b": 3}
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_raw_json_without_tools_remains_content() -> None:
    parser = AutoToolParser()
    parser.configure_request(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    )
    content = ""
    tool_deltas = []

    for chunk in ['{"answer":', '"No tool needed."}']:
        delta = parser.parse_delta(chunk)
        content += delta.content_delta
        tool_deltas.extend(delta.tool_call_deltas)

    assert content == '{"answer":"No tool needed."}'
    assert tool_deltas == []


def test_auto_tool_parser_flush_suppresses_partial_raw_json_tool_protocol() -> None:
    parser = AutoToolParser()
    parser.configure_request(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    )

    delta = parser.parse_delta(
        '{"assistant_text":null,"tool_calls":[{"name":"add_numbers","arguments":{"a":'
    )
    flush = parser.flush_delta()

    assert delta.content_delta == ""
    assert delta.tool_call_deltas == ()
    assert flush.content_delta == ""
    assert flush.tool_call_deltas == ()
    assert parser.suppressed_tool_protocol is True


def test_auto_tool_parser_flush_preserves_partial_regular_raw_json_content() -> None:
    parser = AutoToolParser()
    parser.configure_request(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    )

    delta = parser.parse_delta('{"answer":')
    flush = parser.flush_delta()

    assert delta.content_delta == ""
    assert flush.content_delta == '{"answer":'
    assert flush.tool_call_deltas == ()
    assert parser.suppressed_tool_protocol is False


def test_parser_pipeline_uses_real_auto_tool_parser_for_streaming_deltas() -> None:
    parser = AutoToolParser()
    pipeline = ParserPipeline(tool_parser=parser)
    content = ""
    tool_deltas = []

    for chunk in ["Answer ", "[lookup_weather({\"city\":", '"Shanghai"})]', " done"]:
        delta = pipeline.parse_delta(chunk)
        content += delta.content_delta
        tool_deltas.extend(delta.tool_call_deltas)

    assert content == "Answer  done"
    assert tool_deltas[0].name == "lookup_weather"
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"city":"Shanghai"}'
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_bracket_arguments_respects_nested_values() -> None:
    parser = AutoToolParser()
    tool_deltas = []

    for chunk in [
        "[write_record({\"items\":[{\"id\":",
        "1,\"tags\":[\"a\",\"b\"]}],\"note\":\"brace } in string\"}",
        ")]",
    ]:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    assert tool_deltas[0].name == "write_record"
    arguments = "".join(delta.arguments_delta for delta in tool_deltas)
    assert json.loads(arguments) == {
        "items": [{"id": 1, "tags": ["a", "b"]}],
        "note": "brace } in string",
    }
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_bracket_emits_trailing_content_in_same_delta() -> None:
    parser = AutoToolParser()

    delta = parser.parse_delta('[lookup_weather({"city":"Shanghai"})] done')

    assert delta.content_delta == " done"
    assert delta.tool_call_deltas[0].name == "lookup_weather"
    assert "".join(tool_delta.arguments_delta for tool_delta in delta.tool_call_deltas) == '{"city":"Shanghai"}'
    assert delta.tool_call_deltas[-1].finished is True


def test_auto_tool_parser_streaming_mistral_waits_for_complete_arguments() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in ["Before ", '[TOOL_CALLS] lookup_weather{"city":', '"Shanghai"}', " After"]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert tool_deltas[0].name == "lookup_weather"
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"city":"Shanghai"}'
    assert [delta.finished for delta in tool_deltas] == [False, False, False, True]


def test_auto_tool_parser_streaming_mistral_arguments_respects_nested_values() -> None:
    parser = AutoToolParser()
    tool_deltas = []

    for chunk in [
        "[TOOL_CALLS] write_record{\"items\":[{\"id\":",
        "1,\"tags\":[\"a\",\"b\"]}],\"note\":\"brace } in string\"}",
    ]:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    assert tool_deltas[0].name == "write_record"
    arguments = "".join(delta.arguments_delta for delta in tool_deltas)
    assert json.loads(arguments) == {
        "items": [{"id": 1, "tags": ["a", "b"]}],
        "note": "brace } in string",
    }
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_mistral_old_array_format_waits_for_complete_json() -> None:
    parser = AutoToolParser()
    tool_deltas = []

    for chunk in ['[TOOL_CALLS] [{"name":"lookup_weather","arguments":{"city":', '"Shanghai"}}]']:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    assert len(tool_deltas) == 1
    assert tool_deltas[0].name == "lookup_weather"
    assert json.loads(tool_deltas[0].arguments_delta) == {"city": "Shanghai"}
    assert tool_deltas[0].finished is True


def test_auto_tool_parser_streaming_mistral_old_array_emits_each_complete_call() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        '[TOOL_CALLS] [{"name":"lookup_weather","arguments":{"city":"Shanghai"}},',
        '{"name":"remote_add_numbers","arguments":{"a":2,"b":3}}] done',
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == " done"
    assert [delta.name for delta in tool_deltas] == ["lookup_weather", "remote_add_numbers"]
    assert [json.loads(delta.arguments_delta) for delta in tool_deltas] == [{"city": "Shanghai"}, {"a": 2, "b": 3}]
    assert [delta.finished for delta in tool_deltas] == [True, True]


def test_auto_tool_parser_extracts_deepseek_tool_call() -> None:
    parser = AutoToolParser()

    parsed = parser.extract_tool_calls(
        "Before "
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup_weather\n"
        "```json\n"
        '{"city":"Shanghai"}'
        "\n```<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert parsed.content == "Before"
    assert parsed.tools_called is True
    assert parsed.tool_calls[0]["name"] == "lookup_weather"
    assert json.loads(parsed.tool_calls[0]["arguments"]) == {"city": "Shanghai"}


def test_auto_tool_parser_streaming_deepseek_emits_argument_deltas() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        "Before ",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup_weather\n```json\n",
        '{"city":',
        '"Shanghai"}',
        "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> After",
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert tool_deltas[0].name == "lookup_weather"
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"city":"Shanghai"}'
    assert [delta.finished for delta in tool_deltas] == [False, False, False, True]


def test_auto_tool_parser_extracts_minimax_bare_invoke() -> None:
    parser = AutoToolParser()

    parsed = parser.extract_tool_calls(
        'Before <invoke name="lookup_weather">'
        '<parameter name="city">"Shanghai"</parameter>'
        '<parameter name="days">3</parameter>'
        "</invoke>"
    )

    assert parsed.content == "Before"
    assert parsed.tools_called is True
    assert parsed.tool_calls[0]["name"] == "lookup_weather"
    assert parsed.tool_calls[0]["arguments"] == {"city": "Shanghai", "days": 3}


def test_auto_tool_parser_streaming_minimax_bare_invoke_emits_parameter_deltas() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        "Before ",
        '<invoke name="lookup_weather">',
        '<parameter name="city">"Shanghai"</parameter>',
        '<parameter name="days">3</parameter></invoke>',
        " After",
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert [delta.name for delta in tool_deltas] == ["lookup_weather", None, None, None, None]
    assert json.loads("".join(delta.arguments_delta for delta in tool_deltas)) == {"city": "Shanghai", "days": 3}
    assert [delta.finished for delta in tool_deltas] == [False, False, False, False, True]


def test_auto_tool_parser_streaming_minimax_wrapped_invoke_emits_trailing_content() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        "Before ",
        '<minimax:tool_call><invoke name="lookup_weather">',
        '<parameter name="city">"Shanghai"</parameter></invoke></minimax:tool_call> After',
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert tool_deltas[0].name == "lookup_weather"
    assert json.loads("".join(delta.arguments_delta for delta in tool_deltas)) == {"city": "Shanghai"}
    assert tool_deltas[-1].finished is True


def test_auto_tool_parser_streaming_emits_xml_parameter_argument_deltas() -> None:
    parser = AutoToolParser()
    content_parts: list[str] = []
    tool_deltas = []

    for chunk in [
        "Before ",
        "<tool_call><function=add_numbers>",
        "<parameter=a>2</parameter>",
        "<parameter=b>3</parameter></function></tool_call>",
        " After",
    ]:
        delta = parser.parse_delta(chunk)
        content_parts.append(delta.content_delta)
        tool_deltas.extend(delta.tool_call_deltas)

    assert "".join(content_parts) == "Before  After"
    assert [delta.name for delta in tool_deltas] == ["add_numbers", None, None, None, None]
    assert "".join(delta.arguments_delta for delta in tool_deltas) == '{"a": 2, "b": 3}'
    assert [delta.finished for delta in tool_deltas] == [False, False, False, False, True]


def test_auto_tool_parser_streaming_coerces_xml_parameters_from_tool_schema() -> None:
    parser = AutoToolParser()
    parser.configure_request(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_record",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count": {"type": "integer"},
                                "ratio": {"type": "number"},
                                "enabled": {"type": "boolean"},
                                "metadata": {"type": "object"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                }
            ]
        }
    )
    tool_deltas = []

    for chunk in [
        "<tool_call><function=write_record>",
        "<parameter=count>2</parameter>",
        "<parameter=ratio>2.5</parameter>",
        "<parameter=enabled>true</parameter>",
        '<parameter=metadata>{"nested":1}</parameter>',
        '<parameter=content>{"keep":"as text"}</parameter>',
        "</function></tool_call>",
    ]:
        tool_deltas.extend(parser.parse_delta(chunk).tool_call_deltas)

    arguments = "".join(delta.arguments_delta for delta in tool_deltas)
    assert json.loads(arguments) == {
        "count": 2,
        "ratio": 2.5,
        "enabled": True,
        "metadata": {"nested": 1},
        "content": '{"keep":"as text"}',
    }
