# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from aster.inference.parser_pipeline import ParsedGeneration, ParsedGenerationDelta, ToolCallDelta


@dataclass(slots=True)
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: list[dict[str, Any]]
    content: str | None = None


class ToolParserManager:
    tool_parsers: dict[str, type[AutoToolParserProtocol]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}

    @classmethod
    def get_tool_parser(cls, name: str) -> type[AutoToolParserProtocol]:
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]
        if name in cls.lazy_parsers:
            module_path, class_name = cls.lazy_parsers[name]
            module = importlib.import_module(module_path)
            parser_cls = getattr(module, class_name)
            cls.tool_parsers[name] = parser_cls
            return parser_cls
        raise KeyError(f"Tool parser {name!r} not found. Available parsers: {cls.list_registered()}")

    @classmethod
    def register_module(
        cls,
        name: str | list[str],
        module: type[AutoToolParserProtocol] | None = None,
        *,
        force: bool = True,
    ):
        names = [name] if isinstance(name, str) else name

        def decorator(parser_cls: type[AutoToolParserProtocol]) -> type[AutoToolParserProtocol]:
            for parser_name in names:
                if not force and parser_name in cls.tool_parsers:
                    raise KeyError(f"Parser {parser_name!r} is already registered")
                cls.tool_parsers[parser_name] = parser_cls
            return parser_cls

        if module is not None:
            return decorator(module)
        return decorator

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(set(cls.tool_parsers) | set(cls.lazy_parsers))


class AutoToolParserProtocol:
    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError


def generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _arguments_as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=True)


def _streaming_tool_name(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool_call.get("name")
    return name if isinstance(name, str) else None


def _streaming_tool_arguments(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        return _arguments_as_string(function.get("arguments"))
    return _arguments_as_string(tool_call.get("arguments"))


@ToolParserManager.register_module(["auto", "generic"])
class AutoToolParser(AutoToolParserProtocol):
    extra_stop_tokens = ("<|tool_response>",)
    MISTRAL_TOKEN = "[TOOL_CALLS]"
    DEEPSEEK_CALLS_START = "<｜tool▁calls▁begin｜>"
    DEEPSEEK_CALLS_END = "<｜tool▁calls▁end｜>"
    DEEPSEEK_CALL_START = "<｜tool▁call▁begin｜>"
    DEEPSEEK_CALL_END = "<｜tool▁call▁end｜>"
    DEEPSEEK_TOOL_SEP = "<｜tool▁sep｜>"
    QWEN_BRACKET_PATTERN = re.compile(
        r"\[Calling tool:\s*([\w.-]+)\((\{.*?\})\)\]",
        re.DOTALL,
    )
    QWEN_XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    LLAMA_PATTERN = re.compile(r"<function=([^>]+)>(\{.*?\})</function>", re.DOTALL)
    NEMOTRON_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    NEMOTRON_PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    MINIMAX_PATTERN = re.compile(r"<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>", re.DOTALL)
    MINIMAX_INVOKE_PATTERN = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
    MINIMAX_INVOKE_START_PATTERN = re.compile(r'<invoke\s+name="([^"]+)">', re.DOTALL)
    MINIMAX_PARAM_PATTERN = re.compile(r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>', re.DOTALL)
    MINIMAX_PARAM_START_PATTERN = re.compile(r'<parameter\s+name="([^"]+)">', re.DOTALL)
    GLM_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    GLM_ARG_PATTERN = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
    GLM_ARG_KEY_START = "<arg_key>"
    GLM_ARG_KEY_END = "</arg_key>"
    GLM_ARG_VALUE_START = "<arg_value>"
    GLM_ARG_VALUE_END = "</arg_value>"
    BARE_BRACKET_PATTERN = re.compile(r"\[([\w.-]+)\((\{.*?\})\)\]", re.DOTALL)
    BARE_BRACKET_MARKER_PATTERN = re.compile(r"\[[\w.-]+\(\{", re.DOTALL)
    BARE_BRACKET_PARTIAL_PATTERN = re.compile(r"\[[\w.-]*\(?\{?$", re.DOTALL)
    QWEN_BRACKET_STREAM_START_PATTERN = re.compile(r"\[Calling tool:\s*([\w.-]+)\(\{", re.DOTALL)
    BARE_BRACKET_STREAM_START_PATTERN = re.compile(r"\[([\w.-]+)\(\{", re.DOTALL)
    XML_STREAM_START_PATTERN = re.compile(r"<tool_call>\s*(?=<function=)|(?=<function=)", re.DOTALL)
    STREAMING_MARKERS = (
        DEEPSEEK_CALLS_START,
        DEEPSEEK_CALL_START,
        "<minimax:tool_call>",
        '<invoke name="',
        "<tool_call>",
        "<arg_key>",
        "<function=",
        "[Calling tool:",
        "[TOOL_CALLS]",
        MISTRAL_TOKEN,
    )

    def __init__(self) -> None:
        self._stream_request: dict[str, Any] | None = None
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self.reset()

    def configure_request(self, request: dict[str, Any] | None) -> None:
        self._stream_request = request
        self._tool_schemas = self._tool_schemas_from_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        del request
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        if self.MISTRAL_TOKEN in model_output:
            content, *raw_calls = model_output.split(self.MISTRAL_TOKEN)
            cleaned_text = content.strip()
            for raw in raw_calls:
                self._extend_from_mistral(raw.strip(), tool_calls)

        deepseek_content, deepseek_calls = self._extract_deepseek_tool_calls(cleaned_text)
        if deepseek_calls:
            cleaned_text = deepseek_content or ""
            tool_calls.extend(deepseek_calls)

        cleaned_text, glm_calls = self._extract_glm_tool_calls(cleaned_text)
        tool_calls.extend(glm_calls)

        minimax_calls: list[dict[str, Any]] = []
        for invoke_block in self.MINIMAX_PATTERN.findall(cleaned_text):
            minimax_calls.extend(self._extract_minimax_invokes(invoke_block))
        cleaned_text = self.MINIMAX_PATTERN.sub("", cleaned_text).strip()
        if not minimax_calls:
            minimax_calls = self._extract_minimax_invokes(cleaned_text)
            if minimax_calls:
                cleaned_text = self.MINIMAX_INVOKE_PATTERN.sub("", cleaned_text).strip()
        tool_calls.extend(minimax_calls)

        for pattern in (
            self.QWEN_BRACKET_PATTERN,
            self.BARE_BRACKET_PATTERN,
            self.LLAMA_PATTERN,
        ):
            for name, args in pattern.findall(cleaned_text):
                tool_calls.append(self._tool_call(name, args))
            cleaned_text = pattern.sub("", cleaned_text).strip()

        for name, params_block in self.NEMOTRON_PATTERN.findall(cleaned_text):
            params = {
                param_name.strip(): value.strip()
                for param_name, value in self.NEMOTRON_PARAM_PATTERN.findall(params_block)
            }
            tool_calls.append(self._tool_call(name, params))
        cleaned_text = self.NEMOTRON_PATTERN.sub("", cleaned_text).strip()

        for raw_json in self.QWEN_XML_PATTERN.findall(cleaned_text):
            parsed = self._parse_json(raw_json)
            if self._looks_like_tool_call(parsed):
                tool_calls.append(self._tool_call(parsed["name"], parsed["arguments"]))
        cleaned_text = self.QWEN_XML_PATTERN.sub("", cleaned_text).strip()

        if not tool_calls:
            tool_calls.extend(self._parse_raw_json_tool_calls(cleaned_text))
            if tool_calls:
                cleaned_text = ""

        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=cleaned_text.strip() or None,
        )

    def parse_full(self, text: str) -> ParsedGeneration:
        parsed = self.extract_tool_calls(text)
        return ParsedGeneration(
            content=parsed.content or "",
            tool_calls=tuple(parsed.tool_calls),
            raw_text=text,
        )

    def parse_delta(self, text_delta: str) -> ParsedGenerationDelta:
        previous = self._stream_text
        current = previous + text_delta
        self._stream_text = current
        parsed = self.extract_tool_calls_streaming(previous, current, text_delta, request=self._stream_request)
        if parsed is None:
            return ParsedGenerationDelta(raw_delta=text_delta)
        tool_call_deltas = tuple(
            ToolCallDelta(
                index=int(tool_call.get("index", index)),
                name=_streaming_tool_name(tool_call),
                arguments_delta=_streaming_tool_arguments(tool_call),
                call_id=tool_call.get("id") if isinstance(tool_call.get("id"), str) else None,
                finished=bool(tool_call.get("finished", True)),
            )
            for index, tool_call in enumerate(parsed.get("tool_calls", []))
            if isinstance(tool_call, dict)
        )
        content = parsed.get("content")
        return ParsedGenerationDelta(
            content_delta=content if isinstance(content, str) else "",
            tool_call_deltas=tool_call_deltas,
            raw_delta=text_delta,
        )

    def flush_delta(self) -> ParsedGenerationDelta:
        remaining = self._stream_text[self._stream_cursor :]
        if not remaining:
            return ParsedGenerationDelta()
        self._stream_cursor = len(self._stream_text)
        if self._stream_raw_json_active:
            self._stream_raw_json_active = False
            self._stream_raw_json_content_start = 0
            self._stream_raw_json_start = 0
            if self._looks_like_raw_json_tool_protocol(remaining):
                self._stream_suppressed_tool_protocol = True
                return ParsedGenerationDelta()
        if self._looks_like_unclosed_tool_block(remaining):
            return ParsedGenerationDelta()
        return ParsedGenerationDelta(content_delta=remaining, raw_delta=remaining)

    def stop_token_ids(self, tokenizer: Any) -> frozenset[int]:
        token_ids: set[int] = set()
        for stop_token in self.extra_stop_tokens:
            try:
                encoded = tokenizer.encode(stop_token, add_special_tokens=False)
            except Exception:
                continue
            if len(encoded) == 1:
                token_ids.add(int(encoded[0]))
        return frozenset(token_ids)

    def reset(self) -> None:
        self._stream_text = ""
        self._stream_cursor = 0
        self._stream_emitted_call_count = 0
        self._stream_xml_active = False
        self._stream_xml_call_id: str | None = None
        self._stream_xml_function_name: str | None = None
        self._stream_xml_arguments_started = False
        self._stream_xml_wrapped = False
        self._stream_json_active = False
        self._stream_json_call_id: str | None = None
        self._stream_json_function_name: str | None = None
        self._stream_json_payload_start = 0
        self._stream_json_arguments_start: int | None = None
        self._stream_json_arguments_cursor: int | None = None
        self._stream_json_name_emitted = False
        self._stream_bracket_active = False
        self._stream_bracket_call_id: str | None = None
        self._stream_bracket_function_name: str | None = None
        self._stream_bracket_arguments_start: int | None = None
        self._stream_bracket_arguments_cursor: int | None = None
        self._stream_mistral_active = False
        self._stream_mistral_call_id: str | None = None
        self._stream_mistral_function_name: str | None = None
        self._stream_mistral_payload_start = 0
        self._stream_mistral_arguments_start: int | None = None
        self._stream_mistral_arguments_cursor: int | None = None
        self._stream_mistral_name_emitted = False
        self._stream_mistral_array_mode = False
        self._stream_mistral_array_cursor: int | None = None
        self._stream_deepseek_active = False
        self._stream_deepseek_call_active = False
        self._stream_deepseek_call_id: str | None = None
        self._stream_deepseek_payload_start = 0
        self._stream_deepseek_function_name: str | None = None
        self._stream_deepseek_arguments_start: int | None = None
        self._stream_deepseek_arguments_cursor: int | None = None
        self._stream_deepseek_name_emitted = False
        self._stream_minimax_active = False
        self._stream_minimax_wrapper_active = False
        self._stream_minimax_call_active = False
        self._stream_minimax_call_id: str | None = None
        self._stream_minimax_function_name: str | None = None
        self._stream_minimax_arguments_started = False
        self._stream_glm_active = False
        self._stream_glm_call_id: str | None = None
        self._stream_glm_payload_start = 0
        self._stream_glm_function_name: str | None = None
        self._stream_glm_arguments_started = False
        self._stream_raw_json_active = False
        self._stream_raw_json_content_start = 0
        self._stream_raw_json_start = 0
        self._stream_suppressed_tool_protocol = False

    @property
    def suppressed_tool_protocol(self) -> bool:
        return self._stream_suppressed_tool_protocol

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del previous_text, delta_text
        if request is not None and request is not self._stream_request:
            self.configure_request(request)
        if len(current_text) < self._stream_cursor:
            self.reset()

        deepseek_handled, deepseek_result = self._extract_deepseek_tool_call_streaming(current_text)
        if deepseek_handled:
            return deepseek_result

        minimax_handled, minimax_result = self._extract_minimax_tool_call_streaming(current_text)
        if minimax_handled:
            return minimax_result

        glm_handled, glm_result = self._extract_glm_tool_call_streaming(current_text)
        if glm_handled:
            return glm_result

        xml_handled, xml_result = self._extract_xml_parameter_streaming(current_text)
        if xml_handled:
            return xml_result

        json_handled, json_result = self._extract_json_tool_call_streaming(current_text)
        if json_handled:
            return json_result

        bracket_handled, bracket_result = self._extract_bracket_tool_call_streaming(current_text)
        if bracket_handled:
            return bracket_result

        mistral_handled, mistral_result = self._extract_mistral_tool_call_streaming(current_text)
        if mistral_handled:
            return mistral_result

        raw_json_handled, raw_json_result = self._extract_raw_json_tool_call_streaming(current_text)
        if raw_json_handled:
            return raw_json_result

        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while self._stream_cursor < len(current_text):
            complete_span = self._next_complete_tool_span(current_text, self._stream_cursor)
            incomplete_start = self._next_incomplete_tool_start(current_text, self._stream_cursor)

            if complete_span is None:
                safe_end = len(current_text)
                partial_len = self._partial_marker_suffix_len(current_text[self._stream_cursor :])
                if partial_len:
                    safe_end -= partial_len
                if incomplete_start is not None:
                    safe_end = min(safe_end, incomplete_start)
                if safe_end > self._stream_cursor:
                    content_parts.append(current_text[self._stream_cursor : safe_end])
                    self._stream_cursor = safe_end
                break

            start, end = complete_span
            if incomplete_start is not None and incomplete_start < start:
                if incomplete_start > self._stream_cursor:
                    content_parts.append(current_text[self._stream_cursor : incomplete_start])
                    self._stream_cursor = incomplete_start
                break

            if start > self._stream_cursor:
                content_parts.append(current_text[self._stream_cursor : start])
            self._stream_cursor = end
            tool_call_payloads.extend(self._new_streaming_tool_calls(current_text[:end]))

        if not content_parts and not tool_call_payloads:
            return None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return result

    def _extract_deepseek_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_deepseek_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_deepseek_active:
                calls_start = current_text.find(self.DEEPSEEK_CALLS_START, cursor)
                call_start = current_text.find(self.DEEPSEEK_CALL_START, cursor)
                start = self._earliest_index(calls_start, call_start)
                if start is None:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                self._stream_deepseek_active = True
                handled = True
                if current_text.startswith(self.DEEPSEEK_CALLS_START, start):
                    cursor = start + len(self.DEEPSEEK_CALLS_START)
                    continue
                cursor = start

            if not self._stream_deepseek_call_active:
                while cursor < len(current_text) and current_text[cursor].isspace():
                    cursor += 1
                if cursor >= len(current_text):
                    break
                if current_text.startswith(self.DEEPSEEK_CALLS_END, cursor):
                    cursor += len(self.DEEPSEEK_CALLS_END)
                    self._reset_deepseek_stream_state()
                    continue
                call_start = current_text.find(self.DEEPSEEK_CALL_START, cursor)
                calls_end = current_text.find(self.DEEPSEEK_CALLS_END, cursor)
                if call_start == -1 or (calls_end != -1 and calls_end < call_start):
                    if calls_end != -1:
                        cursor = calls_end + len(self.DEEPSEEK_CALLS_END)
                        self._reset_deepseek_stream_state()
                        continue
                    break
                cursor = call_start + len(self.DEEPSEEK_CALL_START)
                self._stream_deepseek_call_active = True
                self._stream_deepseek_call_id = generate_tool_id()
                self._stream_deepseek_payload_start = cursor
                self._stream_deepseek_function_name = None
                self._stream_deepseek_arguments_start = None
                self._stream_deepseek_arguments_cursor = None
                self._stream_deepseek_name_emitted = False

            call_end = current_text.find(self.DEEPSEEK_CALL_END, self._stream_deepseek_payload_start)
            payload_end = call_end if call_end != -1 else len(current_text)

            if self._stream_deepseek_arguments_start is None:
                name, argument_start = self._deepseek_name_and_argument_start(
                    current_text,
                    self._stream_deepseek_payload_start,
                    payload_end,
                )
                if name and not self._stream_deepseek_name_emitted:
                    self._stream_deepseek_function_name = name
                    self._stream_deepseek_name_emitted = True
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_deepseek_call_id or f"call_{self._stream_emitted_call_count}",
                            name=name,
                            arguments="",
                            finished=False,
                        )
                    )
                if argument_start is not None:
                    self._stream_deepseek_arguments_start = argument_start
                    self._stream_deepseek_arguments_cursor = argument_start

            if self._stream_deepseek_arguments_start is None:
                cursor = payload_end
                break

            argument_end = self._json_value_end(current_text, self._stream_deepseek_arguments_start, payload_end)
            emit_end = argument_end if argument_end is not None else payload_end
            argument_cursor = self._stream_deepseek_arguments_cursor or self._stream_deepseek_arguments_start
            if emit_end > argument_cursor:
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_deepseek_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=current_text[argument_cursor:emit_end],
                        finished=False,
                    )
                )
                self._stream_deepseek_arguments_cursor = emit_end

            if argument_end is None or call_end == -1:
                cursor = emit_end
                break

            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_deepseek_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments="",
                    finished=True,
                )
            )
            self._stream_emitted_call_count += 1
            self._reset_deepseek_call_state()
            cursor = call_end + len(self.DEEPSEEK_CALL_END)

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_minimax_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_minimax_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_minimax_active:
                wrapper_start = current_text.find("<minimax:tool_call>", cursor)
                invoke_match = self.MINIMAX_INVOKE_START_PATTERN.search(current_text, cursor)
                invoke_start = invoke_match.start() if invoke_match is not None else -1
                start = self._earliest_index(wrapper_start, invoke_start)
                if start is None:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                self._stream_minimax_active = True
                handled = True
                if wrapper_start == start:
                    self._stream_minimax_wrapper_active = True
                    cursor = start + len("<minimax:tool_call>")
                    continue
                self._start_minimax_stream_call(invoke_match)
                tool_call_payloads.append(self._minimax_name_payload())
                cursor = invoke_match.end() if invoke_match is not None else start

            if not self._stream_minimax_call_active:
                wrapper_end = current_text.find("</minimax:tool_call>", cursor)
                invoke_match = self.MINIMAX_INVOKE_START_PATTERN.search(current_text, cursor)
                if wrapper_end != -1 and (invoke_match is None or wrapper_end < invoke_match.start()):
                    cursor = wrapper_end + len("</minimax:tool_call>")
                    self._reset_minimax_stream_state()
                    continue
                if invoke_match is None:
                    partial_invoke = current_text.find("<invoke", cursor)
                    cursor = partial_invoke if partial_invoke != -1 else len(current_text)
                    break
                self._start_minimax_stream_call(invoke_match)
                tool_call_payloads.append(self._minimax_name_payload())
                cursor = invoke_match.end()

            next_parameter = self.MINIMAX_PARAM_START_PATTERN.search(current_text, cursor)
            invoke_end = current_text.find("</invoke>", cursor)
            if invoke_end != -1 and (next_parameter is None or invoke_end < next_parameter.start()):
                fragment = "}" if self._stream_minimax_arguments_started else "{}"
                self._stream_minimax_arguments_started = False
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_minimax_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=fragment,
                        finished=False,
                    )
                )
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_minimax_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments="",
                        finished=True,
                    )
                )
                self._stream_emitted_call_count += 1
                cursor = invoke_end + len("</invoke>")
                self._reset_minimax_call_state()
                if not self._stream_minimax_wrapper_active:
                    self._reset_minimax_stream_state()
                continue

            if next_parameter is None:
                partial_parameter = current_text.find("<parameter", cursor)
                cursor = partial_parameter if partial_parameter != -1 else len(current_text)
                break

            close_start = current_text.find("</parameter>", next_parameter.end())
            if close_start == -1:
                cursor = next_parameter.start()
                break
            parameter_name = next_parameter.group(1).strip()
            raw_value = current_text[next_parameter.end() : close_start]
            fragment = self._minimax_argument_fragment(parameter_name, raw_value)
            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_minimax_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments=fragment,
                    finished=False,
                )
            )
            cursor = close_start + len("</parameter>")

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_glm_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_glm_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_glm_active:
                start = current_text.find("<tool_call>", cursor)
                if start == -1:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                payload_start = start + len("<tool_call>")
                while payload_start < len(current_text) and current_text[payload_start].isspace():
                    payload_start += 1
                if payload_start >= len(current_text):
                    return False, None
                if current_text[payload_start] == "{" or current_text.startswith("<function=", payload_start):
                    return False, None
                self._stream_glm_active = True
                self._stream_glm_call_id = generate_tool_id()
                self._stream_glm_payload_start = payload_start
                self._stream_glm_function_name = None
                self._stream_glm_arguments_started = False
                cursor = payload_start
                handled = True

            if self._stream_glm_function_name is None:
                call_end = current_text.find("</tool_call>", self._stream_glm_payload_start)
                body_end = call_end if call_end != -1 else len(current_text)
                name_end = self._glm_name_end(current_text, self._stream_glm_payload_start, body_end)
                if name_end is None:
                    cursor = body_end
                    break
                name = current_text[self._stream_glm_payload_start:name_end].strip()
                if not name:
                    cursor = body_end
                    break
                self._stream_glm_function_name = name
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_glm_call_id or f"call_{self._stream_emitted_call_count}",
                        name=name,
                        arguments="",
                        finished=False,
                    )
                )
                cursor = name_end

            next_arg = current_text.find(self.GLM_ARG_KEY_START, cursor)
            call_end = current_text.find("</tool_call>", cursor)
            if call_end != -1 and (next_arg == -1 or call_end < next_arg):
                fragment = "}" if self._stream_glm_arguments_started else "{}"
                self._stream_glm_arguments_started = False
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_glm_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=fragment,
                        finished=False,
                    )
                )
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_glm_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments="",
                        finished=True,
                    )
                )
                self._stream_emitted_call_count += 1
                cursor = call_end + len("</tool_call>")
                self._reset_glm_stream_state()
                continue

            if next_arg == -1:
                partial_arg = current_text.find("<arg_", cursor)
                cursor = partial_arg if partial_arg != -1 else len(current_text)
                break

            parsed_arg = self._parse_glm_argument(current_text, next_arg)
            if parsed_arg is None:
                cursor = next_arg
                break
            parameter_name, raw_value, arg_end = parsed_arg
            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_glm_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments=self._glm_argument_fragment(parameter_name, raw_value),
                    finished=False,
                )
            )
            cursor = arg_end

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_xml_parameter_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_xml_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_xml_active:
                match = self.XML_STREAM_START_PATTERN.search(current_text, cursor)
                if match is None:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    incomplete_start = self._next_incomplete_tool_start(current_text, cursor)
                    if incomplete_start is not None:
                        safe_end = min(safe_end, incomplete_start)
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                start = match.start()
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                cursor = start
                handled = True
                wrapped = False
                if current_text.startswith("<tool_call>", cursor):
                    wrapped = True
                    cursor += len("<tool_call>")
                    while cursor < len(current_text) and current_text[cursor].isspace():
                        cursor += 1
                if not current_text.startswith("<function=", cursor):
                    break
                function_end = current_text.find(">", cursor)
                if function_end == -1:
                    break
                name = current_text[cursor + len("<function=") : function_end].strip()
                if not name:
                    break
                self._stream_xml_active = True
                self._stream_xml_call_id = generate_tool_id()
                self._stream_xml_function_name = name
                self._stream_xml_arguments_started = False
                self._stream_xml_wrapped = wrapped
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_xml_call_id,
                        name=name,
                        arguments="",
                        finished=False,
                    )
                )
                cursor = function_end + 1
                continue

            next_tag = self._next_xml_stream_tag(current_text, cursor)
            if next_tag is None:
                break
            body_start = cursor
            tag, tag_start = next_tag
            cursor = tag_start
            if tag == "parameter":
                tag_end = current_text.find(">", cursor)
                if tag_end == -1:
                    break
                parameter_name = current_text[cursor + len("<parameter=") : tag_end].strip()
                close_start = current_text.find("</parameter>", tag_end + 1)
                if close_start == -1:
                    break
                raw_value = current_text[tag_end + 1 : close_start]
                fragment = self._xml_argument_fragment(parameter_name, raw_value)
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=fragment,
                        finished=False,
                    )
                )
                cursor = close_start + len("</parameter>")
                continue
            if tag == "function_end":
                raw_body = current_text[body_start:tag_start].strip()
                if raw_body and not self._stream_xml_arguments_started:
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                            name=None,
                            arguments=raw_body,
                            finished=False,
                        )
                    )
                elif self._stream_xml_arguments_started:
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                            name=None,
                            arguments="}",
                            finished=False,
                        )
                    )
                    self._stream_xml_arguments_started = False
                elif not self._stream_xml_wrapped:
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                            name=None,
                            arguments="{}",
                            finished=False,
                        )
                    )
                cursor += len("</function>")
                if not self._stream_xml_wrapped:
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                            name=None,
                            arguments="",
                            finished=True,
                        )
                    )
                    self._stream_emitted_call_count += 1
                    self._reset_xml_stream_state()
                continue
            if tag == "tool_call_end":
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_xml_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments="",
                        finished=True,
                    )
                )
                self._stream_emitted_call_count += 1
                self._reset_xml_stream_state()
                cursor += len("</tool_call>")

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_bracket_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_bracket_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_bracket_active:
                match = self._next_bracket_stream_start(current_text, cursor)
                if match is None:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                start, name, arguments_start = match
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                self._stream_bracket_active = True
                self._stream_bracket_call_id = generate_tool_id()
                self._stream_bracket_function_name = name
                self._stream_bracket_arguments_start = arguments_start
                self._stream_bracket_arguments_cursor = arguments_start
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_bracket_call_id,
                        name=name,
                        arguments="",
                        finished=False,
                    )
                )
                cursor = arguments_start
                handled = True

            if self._stream_bracket_arguments_start is None:
                break
            argument_end = self._json_value_end(current_text, self._stream_bracket_arguments_start, len(current_text))
            emit_end = argument_end if argument_end is not None else len(current_text)
            argument_cursor = self._stream_bracket_arguments_cursor or self._stream_bracket_arguments_start
            if emit_end > argument_cursor:
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_bracket_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=current_text[argument_cursor:emit_end],
                        finished=False,
                    )
                )
                self._stream_bracket_arguments_cursor = emit_end

            if argument_end is None:
                cursor = emit_end
                break

            close_cursor = argument_end
            while close_cursor < len(current_text) and current_text[close_cursor].isspace():
                close_cursor += 1
            if not current_text.startswith(")]", close_cursor):
                cursor = argument_end
                break
            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_bracket_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments="",
                    finished=True,
                )
            )
            self._stream_emitted_call_count += 1
            self._reset_bracket_stream_state()
            cursor = close_cursor + len(")]")

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_mistral_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_mistral_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_mistral_active:
                start = current_text.find(self.MISTRAL_TOKEN, cursor)
                if start == -1:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                payload_start = start + len(self.MISTRAL_TOKEN)
                while payload_start < len(current_text) and current_text[payload_start].isspace():
                    payload_start += 1
                if payload_start >= len(current_text):
                    if start > cursor:
                        content_parts.append(current_text[cursor:start])
                        cursor = start
                    break
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                self._stream_mistral_active = True
                self._stream_mistral_payload_start = payload_start
                if current_text[payload_start] == "[":
                    self._stream_mistral_array_mode = True
                    self._stream_mistral_array_cursor = payload_start + 1
                    cursor = payload_start + 1
                else:
                    self._stream_mistral_array_mode = False
                    self._stream_mistral_array_cursor = None
                    self._stream_mistral_call_id = generate_tool_id()
                    self._stream_mistral_arguments_start = None
                    self._stream_mistral_arguments_cursor = None
                    self._stream_mistral_function_name = None
                    self._stream_mistral_name_emitted = False
                    cursor = payload_start
                handled = True

            if self._stream_mistral_array_mode:
                array_cursor = self._stream_mistral_array_cursor
                if array_cursor is None:
                    cursor = len(current_text)
                    break
                while array_cursor < len(current_text):
                    while array_cursor < len(current_text) and current_text[array_cursor].isspace():
                        array_cursor += 1
                    if array_cursor >= len(current_text):
                        cursor = len(current_text)
                        break
                    if current_text[array_cursor] == ",":
                        array_cursor += 1
                        continue
                    if current_text[array_cursor] == "]":
                        cursor = array_cursor + 1
                        self._reset_mistral_stream_state()
                        break
                    if current_text[array_cursor] != "{":
                        cursor = len(current_text)
                        break
                    item_end = self._json_value_end(current_text, array_cursor, len(current_text))
                    if item_end is None:
                        self._stream_mistral_array_cursor = array_cursor
                        cursor = len(current_text)
                        break
                    item = self._parse_json(current_text[array_cursor:item_end])
                    if self._looks_like_tool_call(item):
                        tool_call_payloads.append(
                            self._streaming_tool_payload(
                                index=self._stream_emitted_call_count,
                                call_id=generate_tool_id(),
                                name=item["name"],
                                arguments=_arguments_as_string(item.get("arguments")),
                                finished=True,
                            )
                        )
                        self._stream_emitted_call_count += 1
                    array_cursor = item_end
                    self._stream_mistral_array_cursor = array_cursor
                if self._stream_mistral_active:
                    break
                continue

            argument_start = self._stream_mistral_arguments_start
            if argument_start is None:
                argument_start = current_text.find("{", self._stream_mistral_payload_start)
                if argument_start == -1:
                    cursor = len(current_text)
                    break
                name = current_text[self._stream_mistral_payload_start:argument_start].strip()
                if not name:
                    cursor = len(current_text)
                    break
                self._stream_mistral_function_name = name
                self._stream_mistral_arguments_start = argument_start
                self._stream_mistral_arguments_cursor = argument_start

            if not self._stream_mistral_name_emitted and self._stream_mistral_function_name:
                self._stream_mistral_name_emitted = True
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_mistral_call_id or f"call_{self._stream_emitted_call_count}",
                        name=self._stream_mistral_function_name,
                        arguments="",
                        finished=False,
                    )
                )

            argument_end = self._json_value_end(current_text, self._stream_mistral_arguments_start or 0, len(current_text))
            emit_end = argument_end if argument_end is not None else len(current_text)
            argument_cursor = self._stream_mistral_arguments_cursor or self._stream_mistral_arguments_start or emit_end
            if emit_end > argument_cursor:
                tool_call_payloads.append(
                    self._streaming_tool_payload(
                        index=self._stream_emitted_call_count,
                        call_id=self._stream_mistral_call_id or f"call_{self._stream_emitted_call_count}",
                        name=None,
                        arguments=current_text[argument_cursor:emit_end],
                        finished=False,
                    )
                )
                self._stream_mistral_arguments_cursor = emit_end

            if argument_end is None:
                cursor = emit_end
                break

            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_mistral_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments="",
                    finished=True,
                )
            )
            self._stream_emitted_call_count += 1
            self._reset_mistral_stream_state()
            cursor = argument_end

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _extract_raw_json_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        if not self._tool_schemas and not self._stream_request:
            return False, None

        if not self._stream_raw_json_active:
            content_start = self._stream_cursor
            cursor = content_start
            while cursor < len(current_text) and current_text[cursor].isspace():
                cursor += 1
            if cursor >= len(current_text):
                return False, None
            if current_text[cursor] not in {"{", "["}:
                return False, None
            self._stream_raw_json_active = True
            self._stream_raw_json_content_start = content_start
            self._stream_raw_json_start = cursor

        raw_start = self._stream_raw_json_start
        raw_end = self._json_value_end(current_text, raw_start, len(current_text))
        if raw_end is None:
            self._stream_cursor = self._stream_raw_json_content_start
            return True, None

        raw_text = current_text[raw_start:raw_end]
        tool_calls = self._parse_raw_json_tool_calls(raw_text)
        content_start = self._stream_raw_json_content_start
        self._stream_raw_json_active = False
        self._stream_raw_json_content_start = 0
        self._stream_raw_json_start = 0
        self._stream_cursor = raw_end

        if not tool_calls:
            content = current_text[content_start:raw_end]
            return True, {"content": content} if content else None

        start_index = self._stream_emitted_call_count
        self._stream_emitted_call_count += len(tool_calls)
        return True, {
            "tool_calls": [
                {
                    "index": start_index + index,
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": _arguments_as_string(tool_call.get("arguments")),
                    },
                    "finished": True,
                }
                for index, tool_call in enumerate(tool_calls)
            ]
        }

    def _extract_json_tool_call_streaming(self, current_text: str) -> tuple[bool, dict[str, Any] | None]:
        cursor = self._stream_cursor
        handled = self._stream_json_active
        content_parts: list[str] = []
        tool_call_payloads: list[dict[str, Any]] = []

        while cursor < len(current_text):
            if not self._stream_json_active:
                start = current_text.find("<tool_call>", cursor)
                if start == -1:
                    if not handled:
                        return False, None
                    safe_end = len(current_text)
                    partial_len = self._partial_marker_suffix_len(current_text[cursor:])
                    if partial_len:
                        safe_end -= partial_len
                    if safe_end > cursor:
                        content_parts.append(current_text[cursor:safe_end])
                        cursor = safe_end
                    break
                if start > cursor:
                    content_parts.append(current_text[cursor:start])
                payload_start = start + len("<tool_call>")
                while payload_start < len(current_text) and current_text[payload_start].isspace():
                    payload_start += 1
                if payload_start >= len(current_text) or current_text[payload_start] != "{":
                    break
                self._stream_json_active = True
                self._stream_json_call_id = generate_tool_id()
                self._stream_json_payload_start = payload_start
                self._stream_json_arguments_start = None
                self._stream_json_arguments_cursor = None
                self._stream_json_function_name = None
                self._stream_json_name_emitted = False
                cursor = payload_start
                handled = True

            closing_start = current_text.find("</tool_call>", self._stream_json_payload_start)
            payload_end = closing_start if closing_start != -1 else len(current_text)
            payload_text = current_text[self._stream_json_payload_start:payload_end]

            if not self._stream_json_name_emitted:
                name = self._json_tool_name(payload_text)
                if name is not None:
                    self._stream_json_function_name = name
                    self._stream_json_name_emitted = True
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_json_call_id or f"call_{self._stream_emitted_call_count}",
                            name=name,
                            arguments="",
                            finished=False,
                        )
                    )

            if self._stream_json_arguments_start is None:
                argument_start = self._json_value_start_for_key(current_text, self._stream_json_payload_start, payload_end, "arguments")
                if argument_start is not None and current_text[argument_start] == "{":
                    self._stream_json_arguments_start = argument_start
                    self._stream_json_arguments_cursor = argument_start

            if self._stream_json_arguments_start is not None:
                argument_end = self._json_value_end(current_text, self._stream_json_arguments_start, payload_end)
                emit_end = argument_end if argument_end is not None else payload_end
                argument_cursor = self._stream_json_arguments_cursor or self._stream_json_arguments_start
                if emit_end > argument_cursor:
                    tool_call_payloads.append(
                        self._streaming_tool_payload(
                            index=self._stream_emitted_call_count,
                            call_id=self._stream_json_call_id or f"call_{self._stream_emitted_call_count}",
                            name=None,
                            arguments=current_text[argument_cursor:emit_end],
                            finished=False,
                        )
                    )
                    self._stream_json_arguments_cursor = emit_end

            if closing_start == -1:
                cursor = payload_end
                break

            tool_call_payloads.append(
                self._streaming_tool_payload(
                    index=self._stream_emitted_call_count,
                    call_id=self._stream_json_call_id or f"call_{self._stream_emitted_call_count}",
                    name=None,
                    arguments="",
                    finished=True,
                )
            )
            self._stream_emitted_call_count += 1
            self._reset_json_stream_state()
            cursor = closing_start + len("</tool_call>")

        self._stream_cursor = cursor
        if not content_parts and not tool_call_payloads:
            return True, None
        result: dict[str, Any] = {}
        if content_parts:
            result["content"] = "".join(content_parts)
        if tool_call_payloads:
            result["tool_calls"] = tool_call_payloads
        return True, result

    def _next_xml_stream_tag(self, text: str, start: int) -> tuple[str, int] | None:
        candidates = [
            ("parameter", text.find("<parameter=", start)),
            ("function_end", text.find("</function>", start)),
            ("tool_call_end", text.find("</tool_call>", start)),
        ]
        present = [(tag, index) for tag, index in candidates if index != -1]
        if not present:
            return None
        return min(present, key=lambda item: item[1])

    def _xml_argument_fragment(self, name: str, raw_value: str) -> str:
        value = self._coerce_xml_argument_value(raw_value.strip(), self._parameter_schema(self._stream_xml_function_name, name))
        prefix = "{" if not self._stream_xml_arguments_started else ", "
        self._stream_xml_arguments_started = True
        return f"{prefix}{json.dumps(name, ensure_ascii=True)}: {json.dumps(value, ensure_ascii=True)}"

    def _minimax_argument_fragment(self, name: str, raw_value: str) -> str:
        stripped = raw_value.strip()
        parsed = self._parse_json(stripped)
        value = parsed if parsed is not None else self._coerce_xml_argument_value(stripped, self._parameter_schema(self._stream_minimax_function_name, name))
        prefix = "{" if not self._stream_minimax_arguments_started else ", "
        self._stream_minimax_arguments_started = True
        return f"{prefix}{json.dumps(name, ensure_ascii=True)}: {json.dumps(value, ensure_ascii=True)}"

    def _glm_argument_fragment(self, name: str, raw_value: str) -> str:
        stripped = raw_value.strip()
        parsed = self._parse_json(stripped)
        value = parsed if parsed is not None else self._coerce_xml_argument_value(stripped, self._parameter_schema(self._stream_glm_function_name, name))
        prefix = "{" if not self._stream_glm_arguments_started else ", "
        self._stream_glm_arguments_started = True
        return f"{prefix}{json.dumps(name, ensure_ascii=True)}: {json.dumps(value, ensure_ascii=True)}"

    def _parameter_schema(self, function_name: str | None, name: str) -> dict[str, Any] | None:
        if not function_name:
            return None
        schema = self._tool_schemas.get(function_name)
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None
        value = properties.get(name)
        return value if isinstance(value, dict) else None

    def _reset_xml_stream_state(self) -> None:
        self._stream_xml_active = False
        self._stream_xml_call_id = None
        self._stream_xml_function_name = None
        self._stream_xml_arguments_started = False
        self._stream_xml_wrapped = False

    def _reset_json_stream_state(self) -> None:
        self._stream_json_active = False
        self._stream_json_call_id = None
        self._stream_json_function_name = None
        self._stream_json_payload_start = 0
        self._stream_json_arguments_start = None
        self._stream_json_arguments_cursor = None
        self._stream_json_name_emitted = False

    def _reset_bracket_stream_state(self) -> None:
        self._stream_bracket_active = False
        self._stream_bracket_call_id = None
        self._stream_bracket_function_name = None
        self._stream_bracket_arguments_start = None
        self._stream_bracket_arguments_cursor = None

    def _reset_mistral_stream_state(self) -> None:
        self._stream_mistral_active = False
        self._stream_mistral_call_id = None
        self._stream_mistral_function_name = None
        self._stream_mistral_payload_start = 0
        self._stream_mistral_arguments_start = None
        self._stream_mistral_arguments_cursor = None
        self._stream_mistral_name_emitted = False
        self._stream_mistral_array_mode = False
        self._stream_mistral_array_cursor = None

    def _reset_deepseek_stream_state(self) -> None:
        self._stream_deepseek_active = False
        self._reset_deepseek_call_state()

    def _reset_deepseek_call_state(self) -> None:
        self._stream_deepseek_call_active = False
        self._stream_deepseek_call_id = None
        self._stream_deepseek_payload_start = 0
        self._stream_deepseek_function_name = None
        self._stream_deepseek_arguments_start = None
        self._stream_deepseek_arguments_cursor = None
        self._stream_deepseek_name_emitted = False

    def _reset_minimax_stream_state(self) -> None:
        self._stream_minimax_active = False
        self._stream_minimax_wrapper_active = False
        self._reset_minimax_call_state()

    def _reset_minimax_call_state(self) -> None:
        self._stream_minimax_call_active = False
        self._stream_minimax_call_id = None
        self._stream_minimax_function_name = None
        self._stream_minimax_arguments_started = False

    def _start_minimax_stream_call(self, match: re.Match[str]) -> None:
        self._stream_minimax_call_active = True
        self._stream_minimax_call_id = generate_tool_id()
        self._stream_minimax_function_name = match.group(1).strip()
        self._stream_minimax_arguments_started = False

    def _minimax_name_payload(self) -> dict[str, Any]:
        return self._streaming_tool_payload(
            index=self._stream_emitted_call_count,
            call_id=self._stream_minimax_call_id or f"call_{self._stream_emitted_call_count}",
            name=self._stream_minimax_function_name,
            arguments="",
            finished=False,
        )

    def _reset_glm_stream_state(self) -> None:
        self._stream_glm_active = False
        self._stream_glm_call_id = None
        self._stream_glm_payload_start = 0
        self._stream_glm_function_name = None
        self._stream_glm_arguments_started = False

    def _glm_name_end(self, text: str, start: int, end: int) -> int | None:
        candidates = [index for index in (text.find("\n", start, end), text.find(self.GLM_ARG_KEY_START, start, end)) if index != -1]
        if candidates:
            return min(candidates)
        if end < len(text) and text.startswith("</tool_call>", end):
            return end
        return end if self._stream_glm_payload_start < end and "</tool_call>" in text[start:] else None

    def _parse_glm_argument(self, text: str, start: int) -> tuple[str, str, int] | None:
        if not text.startswith(self.GLM_ARG_KEY_START, start):
            return None
        key_start = start + len(self.GLM_ARG_KEY_START)
        key_end = text.find(self.GLM_ARG_KEY_END, key_start)
        if key_end == -1:
            return None
        value_start_tag = text.find(self.GLM_ARG_VALUE_START, key_end + len(self.GLM_ARG_KEY_END))
        if value_start_tag == -1:
            return None
        value_start = value_start_tag + len(self.GLM_ARG_VALUE_START)
        value_end = text.find(self.GLM_ARG_VALUE_END, value_start)
        if value_end == -1:
            return None
        return text[key_start:key_end].strip(), text[value_start:value_end], value_end + len(self.GLM_ARG_VALUE_END)

    def _next_bracket_stream_start(self, text: str, start: int) -> tuple[int, str, int] | None:
        matches: list[tuple[int, str, int]] = []
        qwen_match = self.QWEN_BRACKET_STREAM_START_PATTERN.search(text, start)
        if qwen_match is not None:
            matches.append((qwen_match.start(), qwen_match.group(1), qwen_match.end() - 1))
        bare_match = self.BARE_BRACKET_STREAM_START_PATTERN.search(text, start)
        if bare_match is not None:
            matches.append((bare_match.start(), bare_match.group(1), bare_match.end() - 1))
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])

    @staticmethod
    def _earliest_index(*values: int) -> int | None:
        present = [value for value in values if value != -1]
        return min(present) if present else None

    def _deepseek_name_and_argument_start(self, text: str, start: int, end: int) -> tuple[str | None, int | None]:
        sep_start = text.find(self.DEEPSEEK_TOOL_SEP, start, end)
        name_start = sep_start + len(self.DEEPSEEK_TOOL_SEP) if sep_start != -1 else start

        fence_start = text.find("```json", name_start, end)
        fence_len = len("```json")
        if fence_start == -1:
            fence_start = text.find("```", name_start, end)
            fence_len = len("```")
        if fence_start != -1:
            name = text[name_start:fence_start].strip()
            argument_start = fence_start + fence_len
            while argument_start < end and text[argument_start].isspace():
                argument_start += 1
            if argument_start < end and text[argument_start] in {"{", "["}:
                return (name or None), argument_start
            return (name or None), None

        json_starts = [idx for idx in (text.find("{", name_start, end), text.find("[", name_start, end)) if idx != -1]
        if json_starts:
            argument_start = min(json_starts)
            name = text[name_start:argument_start].strip()
            return (name or None), argument_start

        line_end = text.find("\n", name_start, end)
        if line_end != -1:
            name = text[name_start:line_end].strip()
            return (name or None), None
        return None, None

    @staticmethod
    def _json_tool_name(payload_text: str) -> str | None:
        match = re.search(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', payload_text)
        if match is None:
            return None
        try:
            value = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _json_value_start_for_key(text: str, start: int, end: int, key: str) -> int | None:
        match = re.search(rf'"{re.escape(key)}"\s*:', text[start:end])
        if match is None:
            return None
        cursor = start + match.end()
        while cursor < end and text[cursor].isspace():
            cursor += 1
        return cursor if cursor < end else None

    @staticmethod
    def _json_value_end(text: str, start: int, end: int) -> int | None:
        if start >= end or text[start] not in {"{", "["}:
            return None
        expected_closers: list[str] = []
        in_string = False
        escaped = False
        for cursor in range(start, end):
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                expected_closers.append("}")
                continue
            if char == "[":
                expected_closers.append("]")
                continue
            if char in {"}", "]"}:
                if not expected_closers or expected_closers[-1] != char:
                    return None
                expected_closers.pop()
                if not expected_closers:
                    return cursor + 1
        return None

    @staticmethod
    def _coerce_xml_argument_value(raw_value: str, schema: dict[str, Any] | None = None) -> Any:
        expected_type = AutoToolParser._schema_type(schema)
        if expected_type == "string":
            return raw_value
        if raw_value.lower() == "null":
            return None
        if expected_type in {"object", "array"}:
            parsed = AutoToolParser._parse_json(raw_value)
            return parsed if parsed is not None else raw_value
        if expected_type == "integer":
            try:
                return int(raw_value)
            except ValueError:
                return raw_value
        if expected_type == "number":
            try:
                numeric = float(raw_value)
            except ValueError:
                return raw_value
            return int(numeric) if numeric.is_integer() else numeric
        if expected_type == "boolean":
            lowered = raw_value.lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            return raw_value

        parsed = AutoToolParser._parse_json(raw_value)
        if parsed is not None:
            return parsed
        if raw_value.lower() == "true":
            return True
        if raw_value.lower() == "false":
            return False
        if raw_value.lower() == "null":
            return None
        try:
            return int(raw_value)
        except ValueError:
            pass
        try:
            return float(raw_value)
        except ValueError:
            return raw_value

    @staticmethod
    def _schema_type(schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None
        value = schema.get("type")
        if isinstance(value, list):
            value = next((item for item in value if isinstance(item, str) and item != "null"), None)
        if not isinstance(value, str):
            return None
        if value in {"int", "uint", "long", "short"} or value.startswith(("int", "uint")):
            return "integer"
        if value.startswith(("num", "float")):
            return "number"
        if value in {"bool", "binary"}:
            return "boolean"
        if value in {"str", "text", "varchar", "char", "enum"}:
            return "string"
        if value in {"object", "array", "integer", "number", "boolean", "string"}:
            return value
        return "string"

    @staticmethod
    def _tool_schemas_from_request(request: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if not isinstance(request, dict):
            return {}
        schemas: dict[str, dict[str, Any]] = {}
        tools = request.get("tools")
        if not isinstance(tools, list):
            return schemas
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                parameters = function.get("parameters")
            else:
                name = tool.get("name")
                parameters = tool.get("parameters") or tool.get("input_schema")
            if isinstance(name, str) and isinstance(parameters, dict):
                schemas[name] = parameters
        return schemas

    @staticmethod
    def _streaming_tool_payload(
        *,
        index: int,
        call_id: str,
        name: str | None,
        arguments: str,
        finished: bool,
    ) -> dict[str, Any]:
        function: dict[str, Any] = {"arguments": arguments}
        if name is not None:
            function["name"] = name
        return {
            "index": index,
            "id": call_id,
            "type": "function",
            "function": function,
            "finished": finished,
        }

    def _new_streaming_tool_calls(self, text: str) -> list[dict[str, Any]]:
        parsed = self.extract_tool_calls(text)
        if not parsed.tools_called:
            return []
        new_calls = parsed.tool_calls[self._stream_emitted_call_count :]
        if not new_calls:
            return []
        start_index = self._stream_emitted_call_count
        self._stream_emitted_call_count += len(new_calls)
        return [
            {
                "index": start_index + index,
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": _arguments_as_string(tool_call.get("arguments")),
                },
            }
            for index, tool_call in enumerate(new_calls)
        ]

    def _next_complete_tool_span(self, text: str, start: int) -> tuple[int, int] | None:
        spans: list[tuple[int, int]] = []
        for pattern in (
            self.MINIMAX_PATTERN,
            self.NEMOTRON_PATTERN,
            self.QWEN_BRACKET_PATTERN,
            self.BARE_BRACKET_PATTERN,
            self.LLAMA_PATTERN,
            self.QWEN_XML_PATTERN,
        ):
            match = pattern.search(text, start)
            if match is not None:
                spans.append(match.span())

        mistral_start = text.find(self.MISTRAL_TOKEN, start)
        if mistral_start != -1:
            mistral_end = self._mistral_complete_span_end(text, mistral_start)
            if mistral_end is not None:
                spans.append((mistral_start, mistral_end))

        if not spans:
            return None
        return min(spans, key=lambda span: (span[0], span[1]))

    def _mistral_complete_span_end(self, text: str, start: int) -> int | None:
        payload_start = start + len(self.MISTRAL_TOKEN)
        while payload_start < len(text) and text[payload_start].isspace():
            payload_start += 1
        if payload_start >= len(text):
            return None

        if text[payload_start] == "[":
            return self._json_value_end(text, payload_start, len(text))

        argument_start = text.find("{", payload_start)
        if argument_start == -1:
            return None
        name = text[payload_start:argument_start].strip()
        if not name:
            return None
        return self._json_value_end(text, argument_start, len(text))

    def _next_incomplete_tool_start(self, text: str, start: int) -> int | None:
        candidates = [idx for marker in self.STREAMING_MARKERS if (idx := text.find(marker, start)) != -1]
        bare_match = self.BARE_BRACKET_MARKER_PATTERN.search(text, start)
        if bare_match is not None:
            candidates.append(bare_match.start())
        partial_match = self.BARE_BRACKET_PARTIAL_PATTERN.search(text, start)
        if partial_match is not None:
            candidates.append(partial_match.start())
        if not candidates:
            return None
        return min(candidates)

    def _partial_marker_suffix_len(self, text: str) -> int:
        partial_len = 0
        markers = self.STREAMING_MARKERS + ("[Calling tool", "<minimax", "<tool", "<function", "[TOOL", "[")
        for marker in markers:
            for size in range(1, min(len(marker), len(text)) + 1):
                if text.endswith(marker[:size]):
                    partial_len = max(partial_len, size)
        bare_match = self.BARE_BRACKET_PARTIAL_PATTERN.search(text)
        if bare_match is not None:
            partial_len = max(partial_len, len(text) - bare_match.start())
        return partial_len

    def _looks_like_unclosed_tool_block(self, text: str) -> bool:
        stripped = text.lstrip()
        if any(stripped.startswith(marker) for marker in self.STREAMING_MARKERS):
            return True
        return self.BARE_BRACKET_MARKER_PATTERN.match(stripped) is not None

    @staticmethod
    def _looks_like_raw_json_tool_protocol(text: str) -> bool:
        return re.search(r'"(?:assistant_text|tool_calls)"\s*:', text) is not None

    def _extract_deepseek_tool_calls(self, text: str) -> tuple[str | None, list[dict[str, Any]]]:
        calls_start = text.find(self.DEEPSEEK_CALLS_START)
        call_start = text.find(self.DEEPSEEK_CALL_START)
        start = self._earliest_index(calls_start, call_start)
        if start is None:
            return text, []

        content = text[:start].strip() or None
        cursor = start
        if text.startswith(self.DEEPSEEK_CALLS_START, cursor):
            cursor += len(self.DEEPSEEK_CALLS_START)

        tool_calls: list[dict[str, Any]] = []
        while cursor < len(text):
            if text.startswith(self.DEEPSEEK_CALLS_END, cursor):
                break
            next_call = text.find(self.DEEPSEEK_CALL_START, cursor)
            if next_call == -1:
                break
            call_payload_start = next_call + len(self.DEEPSEEK_CALL_START)
            call_end = text.find(self.DEEPSEEK_CALL_END, call_payload_start)
            if call_end == -1:
                break
            parsed = self._parse_deepseek_call_payload(text[call_payload_start:call_end])
            if parsed is not None:
                name, arguments = parsed
                tool_calls.append(self._tool_call(name, arguments))
            cursor = call_end + len(self.DEEPSEEK_CALL_END)

        return content, tool_calls

    def _parse_deepseek_call_payload(self, payload: str) -> tuple[str, str] | None:
        name, argument_start = self._deepseek_name_and_argument_start(payload, 0, len(payload))
        if not name or argument_start is None:
            return None
        argument_end = self._json_value_end(payload, argument_start, len(payload))
        if argument_end is None:
            return None
        return name, payload[argument_start:argument_end].strip()

    def _extract_glm_tool_calls(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        tool_calls: list[dict[str, Any]] = []
        for match in self.GLM_TOOL_CALL_PATTERN.finditer(text):
            parsed = self._parse_glm_tool_body(match.group(1).strip())
            if parsed is None:
                continue
            name, arguments = parsed
            tool_calls.append(self._tool_call(name, arguments))
        if not tool_calls:
            return text, []

        return "", tool_calls

    def _parse_glm_tool_body(self, body: str) -> tuple[str, dict[str, Any]] | None:
        if not body or body.startswith(("{", "<function=")):
            return None
        name_end = self._glm_name_end_for_body(body)
        name = body[:name_end].strip()
        if not name or "<" in name or name.startswith("{"):
            return None
        arguments: dict[str, Any] = {}
        for key, raw_value in self.GLM_ARG_PATTERN.findall(body[name_end:]):
            key = key.strip()
            if not key:
                continue
            stripped = raw_value.strip()
            parsed = self._parse_json(stripped)
            arguments[key] = parsed if parsed is not None else stripped
        return name, arguments

    def _glm_name_end_for_body(self, body: str) -> int:
        candidates = [index for index in (body.find("\n"), body.find(self.GLM_ARG_KEY_START)) if index != -1]
        return min(candidates) if candidates else len(body)

    def _extract_minimax_invokes(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for name, params_block in self.MINIMAX_INVOKE_PATTERN.findall(text):
            params: dict[str, Any] = {}
            for param_name, value in self.MINIMAX_PARAM_PATTERN.findall(params_block):
                parsed_value = self._parse_json(value.strip())
                params[param_name.strip()] = parsed_value if parsed_value is not None else value.strip()
            if params:
                tool_calls.append(self._tool_call(name, params))
        return tool_calls

    def _extend_from_mistral(self, raw: str, tool_calls: list[dict[str, Any]]) -> None:
        if not raw:
            return
        if not raw.startswith("[") and "{" in raw:
            name, args = raw.split("{", 1)
            tool_calls.append(self._tool_call(name.strip(), "{" + args))
            return
        parsed = self._parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    tool_calls.append(self._tool_call(item["name"], item.get("arguments", {})))

    def _parse_raw_json_tool_calls(self, text: str) -> list[dict[str, Any]]:
        parsed = self._parse_json(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("tool_calls"), list):
                calls: list[dict[str, Any]] = []
                for item in parsed["tool_calls"]:
                    if self._looks_like_tool_call(item):
                        calls.append(self._tool_call(item["name"], item["arguments"]))
                return calls
            if self._looks_like_tool_call(parsed):
                return [self._tool_call(parsed["name"], parsed["arguments"])]
        if isinstance(parsed, list):
            calls = []
            for item in parsed:
                if self._looks_like_tool_call(item):
                    calls.append(self._tool_call(item["name"], item["arguments"]))
            return calls
        return []

    @staticmethod
    def _tool_call(name: str, arguments: Any) -> dict[str, Any]:
        return {
            "id": generate_tool_id(),
            "name": name.strip(),
            "arguments": arguments,
        }

    @staticmethod
    def _parse_json(text: str) -> Any:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _looks_like_tool_call(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        name = value.get("name")
        arguments = value.get("arguments")
        return isinstance(name, str) and bool(name) and isinstance(arguments, (dict, str))
