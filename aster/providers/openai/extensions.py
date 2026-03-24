from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import ProviderExtensionData
from aster.providers._shared import extract_extensions
from aster.providers.openai.models import CHAT_KNOWN_KEYS, RESPONSES_KNOWN_KEYS


def response_extensions(payload: Mapping[str, Any]) -> ProviderExtensionData:
    return extract_extensions(payload, RESPONSES_KNOWN_KEYS)


def chat_extensions(payload: Mapping[str, Any]) -> ProviderExtensionData:
    return extract_extensions(payload, CHAT_KNOWN_KEYS)
