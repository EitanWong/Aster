from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from aster.core.errors import AsterError
from aster.inference.reasoning_parsers import parse_reasoning_output
from aster.inference.structured_schema import normalize_json_schema
from aster.inference.tool_parsers import AutoToolParser

_REPAIR_FAILED = object()


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str | None
    parameters: dict[str, Any]


@dataclass(slots=True)
class ToolChoice:
    mode: str = "auto"
    name: str | None = None


@dataclass(slots=True)
class FeaturePlan:
    mode: str = "plain"
    tools: list[ToolSpec] = field(default_factory=list)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    structured_schema: dict[str, Any] | None = None
    structured_name: str | None = None
    allow_parallel_tool_calls: bool = True


@dataclass(slots=True)
class ToolCallResult:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class DecodedLocalOutput:
    mode: str
    raw_text: str
    assistant_text: str | None = None
    reasoning_text: str | None = None
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    structured_data: Any | None = None


class StreamingJsonFenceStripper:
    _OPENINGS = ("```json\n", "```json", "```\n", "```")
    _TAIL_HOLDBACK = 5

    def __init__(self) -> None:
        self._buffer = ""
        self._past_opening = False

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buffer += delta
        if not self._past_opening:
            stripped = self._buffer.lstrip()
            if not stripped:
                return ""
            if any(opening.startswith(stripped) and len(stripped) < len(opening) for opening in self._OPENINGS):
                return ""
            opening = next((candidate for candidate in self._OPENINGS if stripped.startswith(candidate)), None)
            self._buffer = stripped[len(opening) :].lstrip() if opening is not None else stripped
            self._past_opening = True

        buffer = self._buffer
        tail_start = len(buffer)
        while tail_start > 0 and (buffer[tail_start - 1] == "`" or buffer[tail_start - 1].isspace()):
            tail_start -= 1
        safe_end = min(tail_start, len(buffer) - self._TAIL_HOLDBACK)
        if safe_end <= 0:
            return ""
        emitted = buffer[:safe_end]
        self._buffer = buffer[safe_end:]
        return emitted

    def finalize(self) -> str:
        tail = self._buffer
        self._buffer = ""
        if not tail:
            return ""
        if not self._past_opening:
            stripped = tail.lstrip()
            if any(opening.startswith(stripped) and len(stripped) < len(opening) for opening in self._OPENINGS):
                stripped = ""
            else:
                opening = next((candidate for candidate in self._OPENINGS if stripped.startswith(candidate)), None)
                if opening is not None:
                    stripped = stripped[len(opening) :].lstrip()
            tail = stripped
            self._past_opening = True

        stripped_tail = tail.rstrip()
        for closing in ("\n```", "```"):
            if stripped_tail.endswith(closing):
                return stripped_tail[: -len(closing)].rstrip()
        return tail


def apply_feature_plan(
    messages: list[dict[str, str]],
    plan: FeaturePlan,
) -> list[dict[str, str]]:
    if plan.mode == "plain":
        return messages
    augmented = list(messages)
    system_message = _feature_system_prompt(plan)
    if augmented and augmented[0].get("role") == "system":
        augmented[0] = {
            "role": "system",
            "content": f"{system_message}\n\nExisting system instructions:\n{augmented[0].get('content', '')}".strip(),
        }
    else:
        augmented.insert(0, {"role": "system", "content": system_message})
    return augmented


def decode_local_output(text: str, plan: FeaturePlan) -> DecodedLocalOutput:
    reasoning_text, output_text = _extract_reasoning_text(text)
    if plan.mode == "plain":
        return DecodedLocalOutput(
            mode="plain",
            raw_text=text,
            assistant_text=output_text,
            reasoning_text=reasoning_text,
        )
    if plan.mode == "tools":
        try:
            payload = _extract_json_payload(output_text)
        except AsterError:
            decoded = _decode_auto_tool_output(output_text, plan, raw_text=text)
            decoded.reasoning_text = reasoning_text
            if plan.tool_choice.mode in {"auto", "none"}:
                return decoded
            if decoded.tool_calls:
                return decoded
            raise
        try:
            return _decode_tool_output(text, payload, plan, reasoning_text=reasoning_text)
        except AsterError as exc:
            decoded = _decode_auto_tool_output(output_text, plan, raw_text=text)
            decoded.reasoning_text = reasoning_text
            if decoded.tool_calls:
                return decoded
            raise exc
    payload = _extract_json_payload(output_text)
    if plan.mode == "structured":
        _validate_schema(payload, plan.structured_schema or {})
        return DecodedLocalOutput(
            mode="structured",
            raw_text=text,
            assistant_text=json.dumps(payload, ensure_ascii=True),
            reasoning_text=reasoning_text,
            structured_data=payload,
        )
    raise AsterError(code="invalid_feature_mode", message=f"Unknown feature mode '{plan.mode}'", status_code=500)


def validate_structured_output_text(text: str, plan: FeaturePlan, *, allow_repair: bool = True) -> tuple[str, Any]:
    if plan.mode != "structured":
        raise AsterError(code="invalid_feature_mode", message="Structured validation requires a structured feature plan.", status_code=500)
    payload = _extract_json_payload(text, allow_repair=allow_repair)
    _validate_schema(payload, plan.structured_schema or {})
    return json.dumps(payload, ensure_ascii=True), payload


def parse_openai_tools(value: Any) -> tuple[list[ToolSpec], ToolChoice, bool]:
    tools: list[ToolSpec] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function":
            raise AsterError(code="unsupported_tool_type", message="Only function tools are supported by the local runtime.", status_code=400)
        function_data = item.get("function")
        if not isinstance(function_data, dict):
            continue
        tools.append(
            ToolSpec(
                name=str(function_data.get("name") or ""),
                description=function_data.get("description") if isinstance(function_data.get("description"), str) else None,
                parameters=function_data.get("parameters") if isinstance(function_data.get("parameters"), dict) else {},
            )
        )
    return tools, ToolChoice(), True


def parse_openai_tool_choice(value: Any) -> ToolChoice:
    if value is None:
        return ToolChoice()
    if isinstance(value, str):
        if value in {"auto", "none", "required"}:
            return ToolChoice(mode=value)
        return ToolChoice(mode="named", name=value)
    if isinstance(value, dict):
        function_data = value.get("function")
        if isinstance(function_data, dict) and isinstance(function_data.get("name"), str):
            return ToolChoice(mode="named", name=function_data["name"])
        if isinstance(value.get("type"), str):
            value_type = value["type"]
            if value_type == "function" and isinstance(value.get("name"), str):
                return ToolChoice(mode="named", name=value["name"])
            if value_type in {"auto", "none", "required"}:
                return ToolChoice(mode=value_type)
    raise AsterError(code="invalid_tool_choice", message="Unsupported tool_choice value.", status_code=400)


def parse_anthropic_tools(value: Any) -> tuple[list[ToolSpec], ToolChoice, bool]:
    tools: list[ToolSpec] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        tools.append(
            ToolSpec(
                name=str(item.get("name") or ""),
                description=item.get("description") if isinstance(item.get("description"), str) else None,
                parameters=item.get("input_schema") if isinstance(item.get("input_schema"), dict) else {},
            )
        )
    return tools, ToolChoice(), True


def parse_anthropic_tool_choice(value: Any) -> ToolChoice:
    if value is None:
        return ToolChoice()
    if isinstance(value, dict):
        choice_type = value.get("type")
        if choice_type == "auto":
            return ToolChoice(mode="auto")
        if choice_type == "any":
            return ToolChoice(mode="required")
        if choice_type == "tool" and isinstance(value.get("name"), str):
            return ToolChoice(mode="named", name=value["name"])
    raise AsterError(code="invalid_tool_choice", message="Unsupported Anthropic tool_choice value.", status_code=400)


def parse_gemini_tools(value: Any) -> tuple[list[ToolSpec], ToolChoice, bool]:
    tools: list[ToolSpec] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        for declaration in _as_list(item.get("functionDeclarations")):
            if not isinstance(declaration, dict):
                continue
            tools.append(
                ToolSpec(
                    name=str(declaration.get("name") or ""),
                    description=declaration.get("description") if isinstance(declaration.get("description"), str) else None,
                    parameters=declaration.get("parameters") if isinstance(declaration.get("parameters"), dict) else {},
                )
            )
    return tools, ToolChoice(), True


def parse_gemini_tool_config(value: Any) -> ToolChoice:
    if not isinstance(value, dict):
        return ToolChoice()
    config = value.get("functionCallingConfig")
    if not isinstance(config, dict):
        return ToolChoice()
    mode = config.get("mode")
    if mode == "AUTO":
        return ToolChoice(mode="auto")
    if mode == "NONE":
        return ToolChoice(mode="none")
    if mode == "ANY":
        return ToolChoice(mode="required")
    if isinstance(config.get("allowedFunctionNames"), list) and len(config["allowedFunctionNames"]) == 1:
        return ToolChoice(mode="named", name=str(config["allowedFunctionNames"][0]))
    raise AsterError(code="invalid_tool_choice", message="Unsupported Gemini toolConfig.functionCallingConfig.", status_code=400)


def parse_cohere_tools(value: Any) -> tuple[list[ToolSpec], ToolChoice, bool]:
    tools: list[ToolSpec] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        tools.append(
            ToolSpec(
                name=str(item.get("name") or ""),
                description=item.get("description") if isinstance(item.get("description"), str) else None,
                parameters=item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
            )
        )
    return tools, ToolChoice(), True


def parse_bedrock_tools(value: Any) -> tuple[list[ToolSpec], ToolChoice, bool]:
    if not isinstance(value, dict):
        return [], ToolChoice(), True
    tools: list[ToolSpec] = []
    for item in _as_list(value.get("tools")):
        if not isinstance(item, dict):
            continue
        spec = item.get("toolSpec")
        if not isinstance(spec, dict):
            continue
        input_schema = spec.get("inputSchema")
        schema_json = input_schema.get("json") if isinstance(input_schema, dict) and isinstance(input_schema.get("json"), dict) else {}
        tools.append(
            ToolSpec(
                name=str(spec.get("name") or ""),
                description=spec.get("description") if isinstance(spec.get("description"), str) else None,
                parameters=schema_json,
            )
        )
    tool_choice = ToolChoice()
    choice = value.get("toolChoice")
    if isinstance(choice, dict):
        if "auto" in choice:
            tool_choice = ToolChoice(mode="auto")
        elif "any" in choice:
            tool_choice = ToolChoice(mode="required")
        elif "tool" in choice and isinstance(choice["tool"], dict) and isinstance(choice["tool"].get("name"), str):
            tool_choice = ToolChoice(mode="named", name=choice["tool"]["name"])
    return tools, tool_choice, True


def parse_structured_schema(value: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(value, dict):
        raise AsterError(code="invalid_response_format", message="Structured output schema must be an object.", status_code=400)
    if value.get("type") == "json_schema":
        schema = value.get("json_schema")
        if not isinstance(schema, dict):
            raise AsterError(code="invalid_response_format", message="json_schema payload must be an object.", status_code=400)
        inner_schema = schema.get("schema")
        if not isinstance(inner_schema, dict):
            raise AsterError(code="invalid_response_format", message="json_schema.schema must be an object.", status_code=400)
        return inner_schema, schema.get("name") if isinstance(schema.get("name"), str) else None
    return value, None


def parse_openai_responses_text_format(value: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(value, dict):
        raise AsterError(code="invalid_text_format", message="text must be an object when structured output is requested.", status_code=400)
    format_value = value.get("format")
    if not isinstance(format_value, dict):
        raise AsterError(code="invalid_text_format", message="text.format must be an object.", status_code=400)
    if format_value.get("type") != "json_schema":
        raise AsterError(code="unsupported_text_format", message="Only json_schema text.format is supported.", status_code=400)
    schema = format_value.get("schema")
    if not isinstance(schema, dict):
        raise AsterError(code="invalid_text_format", message="text.format.schema must be an object.", status_code=400)
    return schema, format_value.get("name") if isinstance(format_value.get("name"), str) else None


def parse_gemini_structured_schema(generation_config: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(generation_config, dict):
        raise AsterError(code="invalid_generation_config", message="generationConfig must be an object.", status_code=400)
    schema = generation_config.get("responseSchema")
    if not isinstance(schema, dict):
        raise AsterError(code="invalid_generation_config", message="generationConfig.responseSchema must be an object.", status_code=400)
    return schema, None


def build_tool_plan(
    *,
    tools: list[ToolSpec],
    tool_choice: ToolChoice,
    allow_parallel_tool_calls: bool = True,
) -> FeaturePlan:
    if not tools:
        return FeaturePlan()
    return FeaturePlan(
        mode="tools",
        tools=tools,
        tool_choice=tool_choice,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
    )


def tool_request_from_plan(plan: FeaturePlan) -> dict[str, Any]:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in plan.tools
        ]
    }


def validate_tool_call_arguments(tool_call: ToolCallResult, plan: FeaturePlan) -> None:
    tool = next((item for item in plan.tools if item.name == tool_call.name), None)
    if tool is None or not tool.parameters:
        return
    try:
        _validate_schema(tool_call.arguments, tool.parameters, path="$")
    except AsterError as exc:
        raise AsterError(
            code="tool_arguments_invalid",
            message=f"Tool '{tool_call.name}' arguments failed validation: {exc.message}",
            status_code=422,
            details={"tool_name": tool_call.name},
        ) from exc


def build_structured_plan(schema: dict[str, Any], *, name: str | None = None) -> FeaturePlan:
    return FeaturePlan(
        mode="structured",
        structured_schema=normalize_json_schema(schema),
        structured_name=name,
    )


def _feature_system_prompt(plan: FeaturePlan) -> str:
    if plan.mode == "tools":
        tools_payload = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in plan.tools
        ]
        choice_instruction = {
            "auto": "Use tools only when they are needed.",
            "none": "Do not call any tool; respond with assistant_text only.",
            "required": "You must produce at least one tool call.",
            "named": f"You must call the tool named '{plan.tool_choice.name}'.",
        }.get(plan.tool_choice.mode, "Use tools when appropriate.")
        parallel_instruction = (
            "You may emit multiple tool calls in one response."
            if plan.allow_parallel_tool_calls
            else "You must emit at most one tool call."
        )
        schema = {
            "assistant_text": "string or null",
            "tool_calls": [{"name": "string", "arguments": "object"}],
        }
        return (
            "You are in tool-calling mode.\n"
            "Respond with JSON only. Do not include markdown, prose outside JSON, or code fences.\n"
            f"{choice_instruction}\n"
            f"{parallel_instruction}\n"
            f"Return an object with this shape: {json.dumps(schema, ensure_ascii=True)}\n"
            f"Available tools: {json.dumps(tools_payload, ensure_ascii=True)}"
        )
    if plan.mode == "structured":
        return (
            "You must respond with valid JSON only.\n"
            "Do not include markdown, explanations, or code fences.\n"
            f"The JSON must satisfy this schema: {json.dumps(plan.structured_schema or {}, ensure_ascii=True)}"
        )
    return ""


def _decode_tool_output(
    raw_text: str,
    payload: Any,
    plan: FeaturePlan,
    *,
    reasoning_text: str | None = None,
) -> DecodedLocalOutput:
    if not isinstance(payload, dict):
        raise AsterError(code="tool_output_invalid", message="Tool-calling output must be a JSON object.", status_code=422)
    if "tool_calls" not in payload and {"name", "arguments"}.issubset(payload):
        return _decode_auto_tool_output(raw_text, plan, reasoning_text=reasoning_text)
    assistant_text = payload.get("assistant_text")
    tool_calls_payload = payload.get("tool_calls")
    if assistant_text is not None and not isinstance(assistant_text, str):
        raise AsterError(code="tool_output_invalid", message="assistant_text must be a string or null.", status_code=422)
    if not isinstance(tool_calls_payload, list):
        raise AsterError(code="tool_output_invalid", message="tool_calls must be an array.", status_code=422)
    allowed_tools = {tool.name for tool in plan.tools}
    tool_calls: list[ToolCallResult] = []
    for item in tool_calls_payload:
        if not isinstance(item, dict):
            raise AsterError(code="tool_output_invalid", message="Each tool call must be an object.", status_code=422)
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(name, str) or name not in allowed_tools:
            raise AsterError(code="tool_output_invalid", message=f"Unknown tool '{name}'.", status_code=422)
        if not isinstance(arguments, dict):
            raise AsterError(code="tool_output_invalid", message="Tool call arguments must be a JSON object.", status_code=422)
        tool_calls.append(ToolCallResult(call_id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))
    _validate_tool_choice(tool_calls, plan)
    return DecodedLocalOutput(
        mode="tools",
        raw_text=raw_text,
        assistant_text=assistant_text,
        reasoning_text=reasoning_text,
        tool_calls=tool_calls,
    )


def _decode_auto_tool_output(
    text: str,
    plan: FeaturePlan,
    *,
    raw_text: str | None = None,
    reasoning_text: str | None = None,
) -> DecodedLocalOutput:
    parsed = AutoToolParser().extract_tool_calls(text)
    stored_raw_text = raw_text or text
    if not parsed.tools_called:
        _validate_tool_choice([], plan)
        return DecodedLocalOutput(
            mode="tools",
            raw_text=stored_raw_text,
            assistant_text=text,
            reasoning_text=reasoning_text,
            tool_calls=[],
        )

    allowed_tools = {tool.name for tool in plan.tools}
    tool_calls: list[ToolCallResult] = []
    for item in parsed.tool_calls:
        name = item.get("name")
        if not isinstance(name, str) or name not in allowed_tools:
            raise AsterError(code="tool_output_invalid", message=f"Unknown tool '{name}'.", status_code=422)
        tool_calls.append(
            ToolCallResult(
                call_id=str(item.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                name=name,
                arguments=_coerce_tool_arguments(item.get("arguments")),
            )
        )
    _validate_tool_choice(tool_calls, plan)
    return DecodedLocalOutput(
        mode="tools",
        raw_text=stored_raw_text,
        assistant_text=parsed.content,
        reasoning_text=reasoning_text,
        tool_calls=tool_calls,
    )


def _extract_reasoning_text(text: str) -> tuple[str | None, str]:
    parsed = parse_reasoning_output(text)
    return parsed.reasoning_content or None, parsed.content


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AsterError(
                code="tool_output_invalid",
                message="Tool call arguments must be a JSON object.",
                status_code=422,
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise AsterError(
        code="tool_output_invalid",
        message="Tool call arguments must be a JSON object.",
        status_code=422,
    )


def _validate_tool_choice(tool_calls: list[ToolCallResult], plan: FeaturePlan) -> None:
    if plan.tool_choice.mode == "none" and tool_calls:
        raise AsterError(code="tool_choice_violation", message="The model returned tool calls despite tool_choice='none'.", status_code=422)
    if plan.tool_choice.mode == "required" and not tool_calls:
        raise AsterError(code="tool_choice_violation", message="The model did not return a required tool call.", status_code=422)
    if plan.tool_choice.mode == "named":
        if len(tool_calls) != 1 or tool_calls[0].name != plan.tool_choice.name:
            raise AsterError(code="tool_choice_violation", message=f"The model must call tool '{plan.tool_choice.name}'.", status_code=422)
    if not plan.allow_parallel_tool_calls and len(tool_calls) > 1:
        raise AsterError(code="tool_choice_violation", message="Parallel tool calls are disabled for this request.", status_code=422)


def _extract_json_payload(text: str, *, allow_repair: bool = True) -> Any:
    stripped = text.strip()
    if not stripped:
        raise AsterError(code="empty_model_output", message="The local model returned an empty response.", status_code=422)
    for candidate in _json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    if allow_repair:
        for fragment in _truncated_json_fragments(stripped):
            repaired = _repair_truncated_json(fragment)
            if repaired is not _REPAIR_FAILED:
                return repaired
    raise AsterError(code="invalid_json_output", message="The local model did not return valid JSON.", status_code=422)


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    if "```" in text:
        blocks = text.split("```")
        for block in blocks:
            cleaned = block.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned:
                candidates.append(cleaned)
    for index, char in enumerate(text):
        if char in {"{", "["}:
            candidate = _scan_balanced_json(text, index)
            if candidate is not None:
                candidates.append(candidate)
    start_object = text.find("{")
    end_object = text.rfind("}")
    if start_object != -1 and end_object != -1 and end_object > start_object:
        candidates.append(text[start_object : end_object + 1])
    start_array = text.find("[")
    end_array = text.rfind("]")
    if start_array != -1 and end_array != -1 and end_array > start_array:
        candidates.append(text[start_array : end_array + 1])
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _scan_balanced_json(text: str, start: int) -> str | None:
    if start < 0 or start >= len(text) or text[start] not in {"{", "["}:
        return None

    expected_closers: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for index in range(start, len(text)):
        char = text[index]
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
        if char in pairs:
            expected_closers.append(pairs[char])
            continue
        if char in {"}", "]"}:
            if not expected_closers or expected_closers[-1] != char:
                return None
            expected_closers.pop()
            if not expected_closers:
                return text[start : index + 1]
    return None


def _truncated_json_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    fenced_match = re.search(r"```(?:json)?\s*\n?([\s\S]*)$", text)
    if fenced_match is not None:
        fenced = fenced_match.group(1).strip()
        if fenced.endswith("```"):
            fenced = fenced[:-3].strip()
        if fenced:
            fragments.append(fenced)
    for index, char in enumerate(text):
        if char in {"{", "["}:
            fragments.append(text[index:].strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment not in seen:
            seen.add(fragment)
            deduped.append(fragment)
    return deduped


def _repair_truncated_json(fragment: str) -> Any:
    if not fragment:
        return _REPAIR_FAILED

    opener_stack: list[str] = []
    in_string = False
    escaped = False
    for char in fragment:
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
        elif char in {"{", "["}:
            opener_stack.append(char)
        elif char in {"}", "]"} and opener_stack:
            expected = "{" if char == "}" else "["
            if opener_stack[-1] == expected:
                opener_stack.pop()

    if not opener_stack and not in_string:
        return _REPAIR_FAILED

    def close_json(text: str) -> str:
        for opener in reversed(opener_stack):
            text += "}" if opener == "{" else "]"
        return text

    base = fragment
    if in_string:
        if escaped:
            base = base[:-1]
        base += '"'

    candidates = [close_json(base)]
    stripped_separator = re.sub(r"[,:\s]+$", "", base)
    if stripped_separator != base:
        candidates.append(close_json(stripped_separator))
    if opener_stack and opener_stack[-1] == "{":
        without_dangling_key = re.sub(r',?\s*"[^"]*"\s*:?\s*$', "", stripped_separator)
        if without_dangling_key != stripped_separator:
            candidates.append(close_json(without_dangling_key))
    without_partial_scalar = re.sub(
        r",?\s*(?:-?\d+(?:\.\d*)?(?:[eE][+-]?\d*)?|t|tr|tru|f|fa|fal|fals|n|nu|nul)$",
        "",
        stripped_separator,
    )
    if without_partial_scalar != stripped_separator:
        candidates.append(close_json(without_partial_scalar))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return _REPAIR_FAILED


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for index, child_schema in enumerate(all_of):
            if isinstance(child_schema, dict):
                _validate_schema(value, child_schema, path=f"{path}.allOf[{index}]")

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(_schema_accepts(value, child_schema, path=path) for child_schema in any_of if isinstance(child_schema, dict)):
            raise AsterError(code="structured_output_invalid", message=f"{path} does not match any allowed schema.", status_code=422)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(
            1
            for child_schema in one_of
            if isinstance(child_schema, dict) and _schema_accepts(value, child_schema, path=path)
        )
        if matches != 1:
            raise AsterError(code="structured_output_invalid", message=f"{path} must match exactly one allowed schema.", status_code=422)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        allowed_types = schema_type
    elif isinstance(schema_type, str):
        allowed_types = [schema_type]
    else:
        allowed_types = []

    if allowed_types and not any(_matches_type(value, schema_name) for schema_name in allowed_types):
        raise AsterError(code="structured_output_invalid", message=f"{path} does not match required type {allowed_types}.", status_code=422)

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise AsterError(code="structured_output_invalid", message=f"{path} is not one of the allowed enum values.", status_code=422)

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and not isinstance(min_length, bool) and len(value) < min_length:
            raise AsterError(code="structured_output_invalid", message=f"{path} must contain at least {min_length} characters.", status_code=422)
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and not isinstance(max_length, bool) and len(value) > max_length:
            raise AsterError(code="structured_output_invalid", message=f"{path} must contain at most {max_length} characters.", status_code=422)
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise AsterError(code="structured_output_invalid", message=f"{path} does not match required pattern.", status_code=422)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and not isinstance(minimum, bool) and value < minimum:
            raise AsterError(code="structured_output_invalid", message=f"{path} must be greater than or equal to {minimum}.", status_code=422)
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and value > maximum:
            raise AsterError(code="structured_output_invalid", message=f"{path} must be less than or equal to {maximum}.", status_code=422)
        exclusive_minimum = schema.get("exclusiveMinimum")
        if (
            isinstance(exclusive_minimum, (int, float))
            and not isinstance(exclusive_minimum, bool)
            and value <= exclusive_minimum
        ):
            raise AsterError(code="structured_output_invalid", message=f"{path} must be greater than {exclusive_minimum}.", status_code=422)
        exclusive_maximum = schema.get("exclusiveMaximum")
        if (
            isinstance(exclusive_maximum, (int, float))
            and not isinstance(exclusive_maximum, bool)
            and value >= exclusive_maximum
        ):
            raise AsterError(code="structured_output_invalid", message=f"{path} must be less than {exclusive_maximum}.", status_code=422)

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    raise AsterError(code="structured_output_invalid", message=f"{path}.{key} is required.", status_code=422)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    _validate_schema(value[key], child_schema, path=f"{path}.{key}")
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False and isinstance(properties, dict):
            extra_keys = set(value) - set(properties)
            if extra_keys:
                raise AsterError(code="structured_output_invalid", message=f"{path} contains unexpected keys: {sorted(extra_keys)}", status_code=422)
    elif isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and not isinstance(min_items, bool) and len(value) < min_items:
            raise AsterError(code="structured_output_invalid", message=f"{path} must contain at least {min_items} items.", status_code=422)
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and not isinstance(max_items, bool) and len(value) > max_items:
            raise AsterError(code="structured_output_invalid", message=f"{path} must contain at most {max_items} items.", status_code=422)
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_schema(item, items, path=f"{path}[{index}]")


def _schema_accepts(value: Any, schema: dict[str, Any], *, path: str) -> bool:
    try:
        _validate_schema(value, schema, path=path)
    except AsterError:
        return False
    return True


def _matches_type(value: Any, schema_type: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(schema_type, True)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
