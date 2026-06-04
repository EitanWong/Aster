# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from dataclasses import dataclass

from aster.inference.parser_pipeline import ParsedGeneration, ParsedGenerationDelta


@dataclass(slots=True)
class ReasoningDelta:
    content: str | None = None
    reasoning: str | None = None


class ThinkingTagReasoningParser:
    name = "think"
    start_token = "<think>"
    end_token = "</think>"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase = "pre_think"
        self._pending = ""

    def parse_full(self, text: str) -> ParsedGeneration:
        reasoning, content = self.extract_reasoning(text)
        return ParsedGeneration(
            content=content or "",
            reasoning_content=reasoning or "",
            raw_text=text,
        )

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        delta = self.extract_reasoning_streaming(text_delta)
        if delta is None:
            return ParsedGenerationDelta(raw_delta=text_delta)
        return ParsedGenerationDelta(
            content_delta=delta.content or "",
            reasoning_delta=delta.reasoning or "",
            raw_delta=text_delta,
        )

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        text = model_output
        if self.end_token in text:
            return self._extract_complete_reasoning(text)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None
        return None, model_output

    def extract_reasoning_streaming(self, delta_text: str) -> ReasoningDelta | None:
        if not delta_text:
            return None
        combined = self._pending + delta_text
        self._pending = ""
        trailing = _trailing_partial_marker_len(combined, (self.start_token, self.end_token))
        if trailing:
            self._pending = combined[-trailing:]
            combined = combined[:-trailing]
        if not combined:
            return None
        if self._phase == "pre_think":
            if self.start_token in combined:
                before, _, after = combined.partition(self.start_token)
                self._phase = "thinking"
                if self.end_token in after:
                    reasoning, _, content = after.partition(self.end_token)
                    self._phase = "content"
                    return ReasoningDelta(
                        content=(before + content) or None,
                        reasoning=reasoning or None,
                    )
                return ReasoningDelta(content=before or None, reasoning=after or None)
            if self.end_token in combined:
                reasoning, _, content = combined.partition(self.end_token)
                self._phase = "content"
                return ReasoningDelta(content=content or None, reasoning=reasoning or None)
            return ReasoningDelta(content=combined)
        if self._phase == "thinking":
            if self.end_token in combined:
                reasoning, _, content = combined.partition(self.end_token)
                self._phase = "content"
                return ReasoningDelta(content=content or None, reasoning=reasoning or None)
            return ReasoningDelta(reasoning=combined)
        return ReasoningDelta(content=combined)

    def _extract_complete_reasoning(self, text: str) -> tuple[str | None, str | None]:
        reasoning_parts: list[str] = []
        remainder = text
        while remainder:
            stripped = remainder.lstrip()
            if stripped.startswith(self.start_token):
                after_start = stripped[len(self.start_token) :]
                reasoning, found, after_end = after_start.partition(self.end_token)
                if not found:
                    reasoning_parts.append(reasoning)
                    remainder = ""
                    break
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = after_end
                continue
            start_idx = stripped.find(self.start_token)
            end_idx = stripped.find(self.end_token)
            if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
                reasoning = stripped[:end_idx]
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = stripped[end_idx + len(self.end_token) :]
                continue
            remainder = stripped
            break
        return "\n".join(reasoning_parts).strip() or None, remainder.strip() or None


class Qwen3ReasoningParser(ThinkingTagReasoningParser):
    name = "qwen3"

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output)


class DeepSeekR1ReasoningParser(ThinkingTagReasoningParser):
    name = "deepseek_r1"


class Glm4ReasoningParser(ThinkingTagReasoningParser):
    name = "glm4"
    box_start = "<|begin_of_box|>"
    box_end = "<|end_of_box|>"

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        cleaned = model_output.replace(self.box_start, "").replace(self.box_end, "")
        return super().extract_reasoning(cleaned)

    def extract_reasoning_streaming(self, delta_text: str) -> ReasoningDelta | None:
        cleaned = delta_text.replace(self.box_start, "").replace(self.box_end, "")
        return super().extract_reasoning_streaming(cleaned)


_GPT_OSS_STRUCTURAL_TOKENS = re.compile(
    r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|constrain\|>"
)
_GPT_OSS_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|final)(?:[^<]*(?:<\|constrain\|>[^<]*)?)?<\|message\|>"
)
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


class GptOssReasoningParser:
    name = "gpt_oss"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._accumulated = ""
        self._emitted_reasoning_len = 0
        self._emitted_content_len = 0

    def parse_full(self, text: str) -> ParsedGeneration:
        reasoning, content = self.extract_reasoning(text)
        return ParsedGeneration(
            content=content or "",
            reasoning_content=reasoning or "",
            raw_text=text,
        )

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        self._accumulated += text_delta
        reasoning, content = self.extract_reasoning(self._accumulated)
        reasoning = reasoning or ""
        content = content or ""
        reasoning_delta = reasoning[self._emitted_reasoning_len :]
        content_delta = content[self._emitted_content_len :]
        self._emitted_reasoning_len = len(reasoning)
        self._emitted_content_len = len(content)
        return ParsedGenerationDelta(
            content_delta=content_delta,
            reasoning_delta=reasoning_delta,
            raw_delta=text_delta,
        )

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if not model_output or "<|channel|>" not in model_output:
            return None, model_output if model_output else None
        reasoning = _extract_gpt_oss_channel(model_output, "analysis")
        content = _extract_gpt_oss_channel(model_output, "final")
        if reasoning:
            reasoning = _GPT_OSS_STRUCTURAL_TOKENS.sub("", reasoning).strip() or None
        if content:
            content = content.replace("<|return|>", "")
            content = _GPT_OSS_STRUCTURAL_TOKENS.sub("", content).strip() or None
        if reasoning is None and content is None:
            return None, model_output
        return reasoning, content


class HarmonyReasoningParser(GptOssReasoningParser):
    name = "harmony"

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        if "<|channel|>" not in model_output:
            return None, model_output if model_output else None
        analysis_blocks = _HARMONY_ANALYSIS_RE.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks).strip() or None
        final_match = _HARMONY_FINAL_RE.search(model_output)
        content = final_match.group(1).strip() if final_match else None
        if reasoning is None and content is None:
            return super().extract_reasoning(model_output)
        return reasoning, content


_GEMMA_THOUGHT_PREFIX = "thought"
_GEMMA_RESPONSE_MARKER = "<|channel>response"
_GEMMA_START = "<|channel>"
_GEMMA_END = "<channel|>"


class Gemma4ReasoningParser:
    name = "gemma4"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._accumulated = ""
        self._emitted_reasoning_len = 0
        self._emitted_content_len = 0

    def parse_full(self, text: str) -> ParsedGeneration:
        reasoning, content = self.extract_reasoning(text)
        return ParsedGeneration(content=content or "", reasoning_content=reasoning or "", raw_text=text)

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        self._accumulated += text_delta
        reasoning, content = self.extract_reasoning(self._accumulated)
        reasoning = reasoning or ""
        content = content or ""
        reasoning_delta = reasoning[self._emitted_reasoning_len :]
        content_delta = content[self._emitted_content_len :]
        self._emitted_reasoning_len = len(reasoning)
        self._emitted_content_len = len(content)
        return ParsedGenerationDelta(
            content_delta=content_delta,
            reasoning_delta=reasoning_delta,
            raw_delta=text_delta,
        )

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        text = model_output
        if _GEMMA_START in text and _GEMMA_END in text:
            _, _, after_start = text.partition(_GEMMA_START)
            reasoning, _, content = after_start.rpartition(_GEMMA_END)
            return _strip_gemma_channel_tokens(reasoning) or None, _strip_gemma_channel_tokens(content) or None
        if text.count(_GEMMA_START) >= 2 and _GEMMA_RESPONSE_MARKER in text:
            _, _, after_start = text.partition(_GEMMA_START)
            last_response = after_start.rfind(_GEMMA_RESPONSE_MARKER)
            reasoning = after_start[:last_response]
            content = after_start[last_response + len(_GEMMA_RESPONSE_MARKER) :]
            return _strip_gemma_channel_tokens(reasoning) or None, _strip_gemma_channel_tokens(content) or None
        if _GEMMA_END in text:
            reasoning, _, content = text.rpartition(_GEMMA_END)
            return _strip_gemma_channel_tokens(reasoning) or None, _strip_gemma_channel_tokens(content) or None
        if _GEMMA_START in text:
            _, _, reasoning = text.partition(_GEMMA_START)
            return _strip_gemma_channel_tokens(reasoning) or None, None
        return None, model_output


class AutoReasoningParser:
    name = "auto"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._selected: object | None = None
        self._plain_pending = ""

    def parse_full(self, text: str) -> ParsedGeneration:
        parser = _select_reasoning_parser(text)
        if parser is None:
            return ParsedGeneration(content=text, raw_text=text)
        return parser.parse_full(text)

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        if self._selected is None:
            probe = self._plain_pending + text_delta
            self._selected = _select_reasoning_parser(probe)
            if self._selected is None:
                trailing = _trailing_partial_marker_len(probe, _AUTO_MARKERS)
                safe = probe[:-trailing] if trailing else probe
                self._plain_pending = probe[-trailing:] if trailing else ""
                return ParsedGenerationDelta(content_delta=safe, raw_delta=text_delta)
            if hasattr(self._selected, "reset"):
                self._selected.reset()
            pending = self._plain_pending
            self._plain_pending = ""
            delta = self._selected.parse_delta(pending + text_delta)
            return ParsedGenerationDelta(
                content_delta=delta.content_delta,
                reasoning_delta=delta.reasoning_delta,
                raw_delta=text_delta,
            )
        return self._selected.parse_delta(text_delta)  # type: ignore[no-any-return]


_PARSERS: dict[str, type] = {
    "auto": AutoReasoningParser,
    "think": ThinkingTagReasoningParser,
    "qwen3": Qwen3ReasoningParser,
    "deepseek_r1": DeepSeekR1ReasoningParser,
    "glm4": Glm4ReasoningParser,
    "gpt_oss": GptOssReasoningParser,
    "harmony": HarmonyReasoningParser,
    "gemma4": Gemma4ReasoningParser,
}

_AUTO_MARKERS = (
    "<think>",
    "</think>",
    "<|begin_of_box|>",
    "<|channel|>",
    "<|channel>",
    "<channel|>",
)


def parse_reasoning_output(text: str) -> ParsedGeneration:
    return AutoReasoningParser().parse_full(text)


def _select_reasoning_parser(text: str) -> object | None:
    if not text:
        return None
    if "<|channel|>" in text:
        if "<|end|>" in text or "<|return|>" in text:
            return HarmonyReasoningParser()
        return GptOssReasoningParser()
    if "<|channel>" in text or "<channel|>" in text:
        return Gemma4ReasoningParser()
    if "<|begin_of_box|>" in text:
        return Glm4ReasoningParser()
    if "<think>" in text or "</think>" in text:
        return ThinkingTagReasoningParser()
    return None


def _extract_gpt_oss_channel(text: str, channel_name: str) -> str | None:
    for match in _GPT_OSS_CHANNEL_RE.finditer(text):
        if match.group(1) != channel_name:
            continue
        start = match.end()
        end_match = _GPT_OSS_STRUCTURAL_TOKENS.search(text, start)
        content = text[start : end_match.start()] if end_match else text[start:]
        return content.strip() or None
    return None


def _strip_gemma_channel_tokens(text: str) -> str:
    text = text.replace(_GEMMA_END, "\n").replace(_GEMMA_START, "\n")
    cleaned_lines = []
    for line in text.split("\n"):
        if line.strip() in {"thought", "response"}:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()
    for name in ("thought", "response"):
        if text.startswith(f"{name}\n"):
            text = text[len(name) + 1 :]
            break
        if text == name:
            return ""
        if text.startswith(name) and not text[len(name) : len(name) + 1].isalpha():
            text = text[len(name) :]
            break
    return text.strip()


def _trailing_partial_marker_len(text: str, markers: tuple[str, ...]) -> int:
    max_len = 0
    for marker in markers:
        max_prefix = min(len(marker) - 1, len(text))
        for length in range(max_prefix, 0, -1):
            if text.endswith(marker[:length]):
                max_len = max(max_len, length)
                break
    return max_len


def get_reasoning_parser(name: str) -> type:
    return _PARSERS[name]


def list_reasoning_parsers() -> list[str]:
    return sorted(_PARSERS)
