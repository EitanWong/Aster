from __future__ import annotations

from typing import Any

from aster.inference.parser_pipeline import (
    ParsedGeneration,
    ParsedGenerationDelta,
    ParserPipeline,
    ToolCallDelta,
)


class FakeToolParser:
    name = "fake-tool"

    def parse_full(self, text: str) -> ParsedGeneration:
        return ParsedGeneration(
            content="",
            tool_calls=({"name": "search", "arguments": {"query": text}},),
            raw_text=text,
        )

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        return ParsedGenerationDelta(
            tool_call_deltas=(
                ToolCallDelta(index=0, name="search", arguments_delta=text_delta),
            ),
            raw_delta=text_delta,
        )

    def stop_token_ids(self, tokenizer: Any) -> frozenset[int]:
        del tokenizer
        return frozenset({42})


def test_parser_pipeline_collects_tool_stop_tokens() -> None:
    pipeline = ParserPipeline(
        tool_parser=FakeToolParser(),
        extra_stop_token_ids={7},
    )

    assert pipeline.stop_token_ids(tokenizer=None) == frozenset({7, 42})


def test_parser_pipeline_exposes_streaming_tool_deltas() -> None:
    pipeline = ParserPipeline(tool_parser=FakeToolParser())

    delta = pipeline.parse_delta('{"query":"mlx"}')

    assert delta.tool_call_deltas
    assert delta.tool_call_deltas[0].name == "search"
    assert delta.tool_call_deltas[0].arguments_delta == '{"query":"mlx"}'
