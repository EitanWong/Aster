from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aster.core.canonical import ProviderExtensionData
from aster.providers._shared import extract_extensions


def chat_extensions(payload: Mapping[str, Any]) -> ProviderExtensionData:
    return extract_extensions(payload, {"id", "message", "usage", "finish_reason"})
