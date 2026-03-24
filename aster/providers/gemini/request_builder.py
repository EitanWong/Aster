from __future__ import annotations

from typing import Any

from aster.core.canonical import (
    CanonicalRequest,
    ContentPartType,
    MessageRole,
    ProviderHttpRequest,
    ProviderRef,
    ProviderRequestContext,
)
from aster.core.provider_errors import UnsupportedProviderFeatureError
from aster.providers._shared import non_system_messages, system_message_text
from aster.providers.gemini.tool_mapper import build_tools
from aster.providers.openai.request_builder import build_chat_request


def build_generate_content_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    body: dict[str, Any] = {
        "contents": [_message_to_gemini_content(message) for message in non_system_messages(request.messages)],
    }
    if request.tools:
        body["tools"] = build_tools(request.tools)
    generation_config: dict[str, Any] = {}
    if request.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_output_tokens
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.stop is not None:
        generation_config["stopSequences"] = [request.stop] if isinstance(request.stop, str) else request.stop
    if request.structured_output_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = request.structured_output_schema
    if generation_config:
        body["generationConfig"] = generation_config
    system_prompt = system_message_text(request.messages)
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if request.tool_choice is not None:
        body["toolConfig"] = {"functionCallingConfig": _tool_choice(request.tool_choice)}
    if request.reasoning:
        thinking_budget = request.reasoning.get("thinking_budget")
        if thinking_budget is not None:
            body.setdefault("generationConfig", {})
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body.update(request.provider_options)
    path = (
        f"/v1beta/models/{request.model.model_id}:streamGenerateContent"
        if request.stream
        else f"/v1beta/models/{request.model.model_id}:generateContent"
    )
    query = {"alt": "sse"} if request.stream else {}
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": context.api_key if context and context.api_key else "",
    }
    if context:
        headers.update(context.extra_headers)
    return ProviderHttpRequest(
        method="POST",
        path=path,
        headers={key: value for key, value in headers.items() if value},
        query=query,
        json_body=body,
    )


def build_openai_chat_request(
    provider: ProviderRef,
    request: CanonicalRequest,
    context: ProviderRequestContext | None,
) -> ProviderHttpRequest:
    http_request = build_chat_request(provider, request, context)
    http_request.path = "/v1beta/openai/chat/completions"
    return http_request


def _message_to_gemini_content(message: Any) -> dict[str, Any]:
    role = "model" if message.role is MessageRole.ASSISTANT else "user"
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if part.type in {ContentPartType.TEXT, ContentPartType.INPUT_TEXT, ContentPartType.OUTPUT_TEXT}:
            parts.append({"text": part.text or ""})
            continue
        if part.type is ContentPartType.INPUT_IMAGE and "gemini_native" in part.provider_extensions.values:
            parts.append(dict(part.provider_extensions.values["gemini_native"]))
            continue
        if "gemini_native" in part.provider_extensions.values:
            parts.append(dict(part.provider_extensions.values["gemini_native"]))
            continue
        raise UnsupportedProviderFeatureError.build(
            provider=None,
            category="unsupported_feature",
            code="gemini_content_part_unsupported",
            message=f"Gemini content part '{part.type.value}' requires gemini_native extension data.",
        )
    for tool_call in message.tool_calls:
        parts.append({"functionCall": {"name": tool_call.name, "args": tool_call.arguments or {}}})
    if message.role is MessageRole.TOOL or message.tool_results:
        role = "user"
        parts = []
        for result in message.tool_results:
            parts.append(
                {
                    "functionResponse": {
                        "name": result.name or result.call_id,
                        "response": result.output if isinstance(result.output, dict) else {"content": result.output},
                    }
                }
            )
    return {"role": role, "parts": parts}


def _tool_choice(choice: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(choice, dict):
        return choice
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    return {"mode": "AUTO"}
