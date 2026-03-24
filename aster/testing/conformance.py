from __future__ import annotations

from aster.core.canonical import CanonicalFinalResponse, ProviderHttpRequest


def assert_json_body_contains(request: ProviderHttpRequest, key: str) -> None:
    assert key in request.json_body, f"expected '{key}' in request body"


def assert_preserves_extension(final_response: CanonicalFinalResponse, key: str) -> None:
    assert key in final_response.provider_extensions.values, f"missing provider extension '{key}'"
