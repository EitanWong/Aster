from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    name: str | None = None
    arguments_delta: str = ""
    call_id: str | None = None
    finished: bool = False


@dataclass(frozen=True, slots=True)
class ParsedGeneration:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw_text: str = ""


@dataclass(frozen=True, slots=True)
class ParsedGenerationDelta:
    content_delta: str = ""
    reasoning_delta: str = ""
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()
    raw_delta: str = ""


class ToolParser(Protocol):
    name: str

    def parse_full(self, text: str) -> ParsedGeneration:
        ...

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        ...

    def stop_token_ids(self, tokenizer: Any) -> frozenset[int]:
        ...


class ReasoningParser(Protocol):
    name: str

    def parse_full(self, text: str) -> ParsedGeneration:
        ...

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        ...

    def reset(self) -> None:
        ...


class StructuredLogitsProcessor(Protocol):
    name: str

    def build(self, *, tokenizer: Any, schema: dict[str, Any] | None) -> Any:
        ...


@dataclass(slots=True)
class ParserPipeline:
    tool_parser: ToolParser | None = None
    reasoning_parser: ReasoningParser | None = None
    structured_processor: StructuredLogitsProcessor | None = None
    extra_stop_token_ids: set[int] = field(default_factory=set)

    def stop_token_ids(self, tokenizer: Any) -> frozenset[int]:
        token_ids = set(self.extra_stop_token_ids)
        if self.tool_parser is not None:
            token_ids.update(self.tool_parser.stop_token_ids(tokenizer))
        return frozenset(token_ids)

    def parse_full(self, text: str) -> ParsedGeneration:
        parsed = ParsedGeneration(content=text, raw_text=text)
        if self.reasoning_parser is not None:
            parsed = self.reasoning_parser.parse_full(parsed.content)
        if self.tool_parser is not None:
            parsed = self.tool_parser.parse_full(parsed.content)
        return parsed

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        if self.reasoning_parser is not None:
            parsed = self.reasoning_parser.parse_delta(text_delta)
        else:
            parsed = ParsedGenerationDelta(content_delta=text_delta, raw_delta=text_delta)
        if self.tool_parser is not None:
            tool_delta = self.tool_parser.parse_delta(parsed.content_delta)
            return ParsedGenerationDelta(
                content_delta=tool_delta.content_delta,
                reasoning_delta=parsed.reasoning_delta,
                tool_call_deltas=tool_delta.tool_call_deltas,
                raw_delta=parsed.raw_delta,
            )
        return parsed


def merge_deltas(deltas: Iterable[ParsedGenerationDelta]) -> ParsedGenerationDelta:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[ToolCallDelta] = []
    raw_parts: list[str] = []
    for delta in deltas:
        content_parts.append(delta.content_delta)
        reasoning_parts.append(delta.reasoning_delta)
        tool_deltas.extend(delta.tool_call_deltas)
        raw_parts.append(delta.raw_delta)
    return ParsedGenerationDelta(
        content_delta="".join(content_parts),
        reasoning_delta="".join(reasoning_parts),
        tool_call_deltas=tuple(tool_deltas),
        raw_delta="".join(raw_parts),
    )
