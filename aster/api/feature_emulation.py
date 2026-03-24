from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from aster.core.errors import AsterError


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
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    structured_data: Any | None = None


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
    if plan.mode == "plain":
        return DecodedLocalOutput(mode="plain", raw_text=text, assistant_text=text)
    if plan.mode == "tools":
        try:
            payload = _extract_json_payload(text)
        except AsterError:
            if plan.tool_choice.mode in {"auto", "none"}:
                return DecodedLocalOutput(mode="tools", raw_text=text, assistant_text=text, tool_calls=[])
            raise
        return _decode_tool_output(text, payload, plan)
    payload = _extract_json_payload(text)
    if plan.mode == "structured":
        _validate_schema(payload, plan.structured_schema or {})
        return DecodedLocalOutput(mode="structured", raw_text=text, assistant_text=json.dumps(payload, ensure_ascii=True), structured_data=payload)
    raise AsterError(code="invalid_feature_mode", message=f"Unknown feature mode '{plan.mode}'", status_code=500)


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


def build_structured_plan(schema: dict[str, Any], *, name: str | None = None) -> FeaturePlan:
    return FeaturePlan(mode="structured", structured_schema=schema, structured_name=name)


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


def _decode_tool_output(raw_text: str, payload: Any, plan: FeaturePlan) -> DecodedLocalOutput:
    if not isinstance(payload, dict):
        raise AsterError(code="tool_output_invalid", message="Tool-calling output must be a JSON object.", status_code=422)
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
    return DecodedLocalOutput(mode="tools", raw_text=raw_text, assistant_text=assistant_text, tool_calls=tool_calls)


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


def _extract_json_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise AsterError(code="empty_model_output", message="The local model returned an empty response.", status_code=422)
    for candidate in _json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
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


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
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
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_schema(item, items, path=f"{path}[{index}]")


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
