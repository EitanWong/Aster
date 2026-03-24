from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import ProviderExtensionData
from aster.providers._shared import extract_extensions
from aster.providers.gemini.models import KNOWN_KEYS


def response_extensions(payload: Mapping[str, Any]) -> ProviderExtensionData:
    return extract_extensions(payload, KNOWN_KEYS)
