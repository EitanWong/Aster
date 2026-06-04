from __future__ import annotations

from aster.inference.parser_pipeline import ParserPipeline
from aster.inference.reasoning_parsers import (
    AutoReasoningParser,
    DeepSeekR1ReasoningParser,
    Gemma4ReasoningParser,
    Glm4ReasoningParser,
    GptOssReasoningParser,
    HarmonyReasoningParser,
    Qwen3ReasoningParser,
    get_reasoning_parser,
    list_reasoning_parsers,
    parse_reasoning_output,
)


def test_qwen3_reasoning_parser_extracts_think_block() -> None:
    parser = Qwen3ReasoningParser()

    parsed = parser.parse_full("<think>step 1</think>The answer is 42.")

    assert parsed.reasoning_content == "step 1"
    assert parsed.content == "The answer is 42."


def test_qwen3_reasoning_parser_treats_unmarked_text_as_content() -> None:
    parser = Qwen3ReasoningParser()

    parsed = parser.parse_full("plain answer")

    assert parsed.reasoning_content == ""
    assert parsed.content == "plain answer"


def test_deepseek_reasoning_parser_supports_implicit_start_tag() -> None:
    parser = DeepSeekR1ReasoningParser()

    parsed = parser.parse_full("reasoning only</think>final")

    assert parsed.reasoning_content == "reasoning only"
    assert parsed.content == "final"


def test_reasoning_parser_works_in_parser_pipeline() -> None:
    pipeline = ParserPipeline(reasoning_parser=Qwen3ReasoningParser())

    parsed = pipeline.parse_full("<think>scratch</think>final")

    assert parsed.reasoning_content == "scratch"
    assert parsed.content == "final"


def test_reasoning_parser_registry_lists_parsers() -> None:
    assert get_reasoning_parser("qwen3") is Qwen3ReasoningParser
    assert "deepseek_r1" in list_reasoning_parsers()
    assert "gpt_oss" in list_reasoning_parsers()
    assert "gemma4" in list_reasoning_parsers()


def test_auto_reasoning_parser_buffers_split_think_tags() -> None:
    parser = AutoReasoningParser()

    first = parser.parse_delta("<thi")
    second = parser.parse_delta("nk>scratch</thi")
    third = parser.parse_delta("nk>final")

    assert first.content_delta == ""
    assert second.reasoning_delta == "scratch"
    assert third.content_delta == "final"


def test_gpt_oss_reasoning_parser_extracts_analysis_and_final_channels() -> None:
    parser = GptOssReasoningParser()

    parsed = parser.parse_full(
        "<|channel|>analysis<|message|>Need JSON."
        "<|start|>assistant<|channel|>final <|constrain|>JSON<|message|>{\"ok\": true}<|return|>"
    )

    assert parsed.reasoning_content == "Need JSON."
    assert parsed.content == '{"ok": true}'


def test_harmony_reasoning_parser_extracts_completed_blocks() -> None:
    parser = HarmonyReasoningParser()

    parsed = parser.parse_full(
        "<|channel|>analysis<|message|>Think A<|end|>"
        "<|channel|>analysis<|message|>Think B<|end|>"
        "<|channel|>final<|message|>Done<|return|>"
    )

    assert parsed.reasoning_content == "Think A\nThink B"
    assert parsed.content == "Done"


def test_gemma4_reasoning_parser_uses_last_channel_boundary() -> None:
    parser = Gemma4ReasoningParser()

    parsed = parser.parse_full(
        "<|channel>thought\nbad cycle<channel|>response\npartial"
        "<|channel>thought\nfinal thought<channel|>final answer"
    )

    assert parsed.reasoning_content == "bad cycle\npartial\nfinal thought"
    assert parsed.content == "final answer"


def test_glm4_reasoning_parser_strips_box_container_tokens() -> None:
    parser = Glm4ReasoningParser()

    parsed = parser.parse_full("<|begin_of_box|><think>work</think>answer<|end_of_box|>")

    assert parsed.reasoning_content == "work"
    assert parsed.content == "answer"


def test_parse_reasoning_output_auto_selects_channel_parser() -> None:
    parsed = parse_reasoning_output("<|channel|>analysis<|message|>think<|channel|>final<|message|>answer")

    assert parsed.reasoning_content == "think"
    assert parsed.content == "answer"
