from __future__ import annotations

from aster.core.canonical import ModelRef
from aster.core.contracts import ProviderAdapter
from aster.runtime.provider_registry import ProviderRegistry


class ProviderRouter:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    def resolve(self, model: ModelRef) -> ProviderAdapter:
        return self.registry.get(model.provider.name.value, model.provider.api_family)
