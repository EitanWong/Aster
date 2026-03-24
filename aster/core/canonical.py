from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderName(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    MISTRAL = "mistral"
    COHERE = "cohere"
    XAI = "xai"


class CapabilitySupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    MODEL_DEPENDENT = "model_dependent"
    BETA = "beta"
    PROVIDER_SPECIFIC = "provider_specific"


class MessageRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPartType(StrEnum):
    TEXT = "text"
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    INPUT_IMAGE = "input_image"
    INPUT_AUDIO = "input_audio"
    FILE = "file"
    JSON = "json"
    THINKING = "thinking"
    REFUSAL = "refusal"
    TOOL_RESULT = "tool_result"
    PROVIDER_NATIVE = "provider_native"


class ErrorCategory(StrEnum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    VALIDATION = "validation"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    PROVIDER = "provider"
    STREAM = "stream"
    PAYLOAD = "payload"


class ProviderExtensionData(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    def merged(self, other: dict[str, Any] | None = None) -> "ProviderExtensionData":
        payload = dict(self.values)
        if other:
            payload.update(other)
        return ProviderExtensionData(values=payload)


class ProviderRef(BaseModel):
    name: ProviderName
    api_family: str
    api_version: str | None = None
    region: str | None = None
    base_url: str | None = None
    extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")

    @property
    def adapter_id(self) -> str:
        return f"{self.name.value}.{self.api_family}"


class ModelRef(BaseModel):
    provider: ProviderRef
    model_id: str
    snapshot: str | None = None
    deployment: str | None = None
    extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_values(
        cls,
        provider: ProviderName | str,
        api_family: str,
        model_id: str,
        *,
        api_version: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        snapshot: str | None = None,
        deployment: str | None = None,
        extensions: dict[str, Any] | None = None,
    ) -> "ModelRef":
        provider_name = provider if isinstance(provider, ProviderName) else ProviderName(provider)
        return cls(
            provider=ProviderRef(
                name=provider_name,
                api_family=api_family,
                api_version=api_version,
                region=region,
                base_url=base_url,
            ),
            model_id=model_id,
            snapshot=snapshot,
            deployment=deployment,
            extensions=ProviderExtensionData(values=extensions or {}),
        )


class CanonicalContentPart(BaseModel):
    type: ContentPartType
    text: str | None = None
    mime_type: str | None = None
    image_url: str | None = None
    audio_format: str | None = None
    file_id: str | None = None
    file_name: str | None = None
    data: str | None = None
    json_value: Any | None = None
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class CanonicalToolDefinition(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class CanonicalToolCall(BaseModel):
    call_id: str
    name: str
    arguments_json: str | None = None
    arguments: dict[str, Any] | None = None
    status: str | None = None
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class CanonicalToolResult(BaseModel):
    call_id: str
    output: Any
    name: str | None = None
    is_error: bool = False
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class CanonicalMessage(BaseModel):
    role: MessageRole
    content: list[CanonicalContentPart] = Field(default_factory=list)
    name: str | None = None
    tool_calls: list[CanonicalToolCall] = Field(default_factory=list)
    tool_results: list[CanonicalToolResult] = Field(default_factory=list)
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class CanonicalUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class RawProviderEnvelope(BaseModel):
    provider: ProviderName
    api_family: str
    status_code: int | None = None
    event_name: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] | str | None = None

    model_config = ConfigDict(extra="allow")


class CanonicalRequest(BaseModel):
    model: ModelRef
    messages: list[CanonicalMessage] = Field(default_factory=list)
    tools: list[CanonicalToolDefinition] = Field(default_factory=list)
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    background: bool | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    conversation_id: str | None = None
    structured_output_schema: dict[str, Any] | None = None
    structured_output_name: str | None = None
    parallel_tool_calls: bool | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_structured_output(self) -> "CanonicalRequest":
        if self.structured_output_name and not self.structured_output_schema:
            raise ValueError("structured_output_name requires structured_output_schema")
        return self


class CanonicalResponseChunk(BaseModel):
    provider: ProviderRef
    response_id: str | None = None
    event_type: str
    delta_text: str | None = None
    tool_call: CanonicalToolCall | None = None
    tool_result: CanonicalToolResult | None = None
    usage: CanonicalUsage | None = None
    finish_reason: str | None = None
    output_index: int | None = None
    content_index: int | None = None
    sequence_number: int | None = None
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)
    raw_provider_envelope: RawProviderEnvelope | None = None

    model_config = ConfigDict(extra="allow")


class CanonicalFinalResponse(BaseModel):
    provider: ProviderRef
    model: ModelRef
    response_id: str
    status: str | None = None
    output: list[CanonicalMessage] = Field(default_factory=list)
    output_text: str = ""
    tool_calls: list[CanonicalToolCall] = Field(default_factory=list)
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)
    finish_reason: str | None = None
    request_id: str | None = None
    rate_limit_headers: dict[str, str] = Field(default_factory=dict)
    provider_version: str | None = None
    provider_beta_headers: list[str] = Field(default_factory=list)
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)
    raw_provider_envelope: RawProviderEnvelope | None = None

    model_config = ConfigDict(extra="allow")


class CanonicalError(BaseModel):
    provider: ProviderRef | None = None
    category: ErrorCategory
    code: str
    message: str
    status_code: int | None = None
    retryable: bool = False
    request_id: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)
    raw_provider_envelope: RawProviderEnvelope | None = None

    model_config = ConfigDict(extra="allow")


class CanonicalStreamEvent(BaseModel):
    provider: ProviderRef
    event_type: str
    response_id: str | None = None
    chunk: CanonicalResponseChunk | None = None
    error: CanonicalError | None = None
    raw_provider_envelope: RawProviderEnvelope | None = None

    model_config = ConfigDict(extra="allow")


class ProviderFeatureFlags(BaseModel):
    text_input: CapabilitySupport = CapabilitySupport.SUPPORTED
    text_output: CapabilitySupport = CapabilitySupport.SUPPORTED
    multimodal_input: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    multimodal_output: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    streaming: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    structured_outputs: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    user_defined_tools: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    built_in_tools: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    parallel_tool_calls: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    conversation_state: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    background_mode: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    reasoning_controls: CapabilitySupport = CapabilitySupport.UNSUPPORTED
    deferred_execution: CapabilitySupport = CapabilitySupport.UNSUPPORTED

    model_config = ConfigDict(extra="allow")


class ProviderCapabilities(BaseModel):
    provider: ProviderRef
    auth_scheme: str
    endpoint_family: str
    features: ProviderFeatureFlags = Field(default_factory=ProviderFeatureFlags)
    required_headers: list[str] = Field(default_factory=list)
    beta_headers: list[str] = Field(default_factory=list)
    versioning_strategy: str | None = None
    supported_stop_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    model_constraints: list[str] = Field(default_factory=list)
    extension_fields: list[str] = Field(default_factory=list)
    docs_urls: tuple[str, ...] = ()

    model_config = ConfigDict(extra="allow")


class ProviderRequestContext(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    region: str | None = None
    anthropic_version: str | None = None
    anthropic_beta: list[str] = Field(default_factory=list)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    use_sigv4: bool = False

    model_config = ConfigDict(extra="allow")


class ProviderHttpRequest(BaseModel):
    method: str
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    json_body: dict[str, Any] = Field(default_factory=dict)
    query: dict[str, str] = Field(default_factory=dict)
    requires_sigv4: bool = False
    provider_extensions: ProviderExtensionData = Field(default_factory=ProviderExtensionData)

    model_config = ConfigDict(extra="allow")


class ProviderStreamEvent(BaseModel):
    event: str | None = None
    data: dict[str, Any] | str | None = None
    sequence_number: int | None = None
    raw: dict[str, Any] | str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


ContentPart = CanonicalContentPart
ToolDefinition = CanonicalToolDefinition
ToolCall = CanonicalToolCall
ToolResult = CanonicalToolResult
Message = CanonicalMessage
Usage = CanonicalUsage
ResponseChunk = CanonicalResponseChunk
FinalResponse = CanonicalFinalResponse
