from __future__ import annotations

import pytest
from pydantic import ValidationError

from aster.core.canonical import CanonicalMessage, CanonicalRequest, MessageRole, ModelRef
from aster.providers import build_default_provider_registry
from aster.testing.fixtures import load_fixture


def test_canonical_request_rejects_named_structured_output_without_schema() -> None:
    with pytest.raises(ValidationError):
        CanonicalRequest(
            model=ModelRef.from_values("openai", "responses", "gpt-5.1"),
            messages=[CanonicalMessage(role=MessageRole.USER)],
            structured_output_name="weather",
        )


def test_default_provider_registry_contains_expected_adapters() -> None:
    registry = build_default_provider_registry()
    assert "openai.responses" in registry.ids()
    assert "anthropic.messages" in registry.ids()
    assert "gemini.generate_content" in registry.ids()


def test_fixture_loader_reads_json_fixtures() -> None:
    fixture = load_fixture("providers", "openai", "responses_response.json")
    assert fixture["id"] == "resp_123"
