from __future__ import annotations

from dataclasses import dataclass, field

from aster.core.contracts import ProviderAdapter


@dataclass(slots=True)
class ProviderRegistry:
    _adapters: dict[str, ProviderAdapter] = field(default_factory=dict)

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.adapter_id] = adapter

    def get(self, provider: str, api_family: str) -> ProviderAdapter:
        adapter_id = f"{provider}.{api_family}"
        if adapter_id not in self._adapters:
            available = ", ".join(sorted(self._adapters))
            raise KeyError(f"Unknown provider adapter '{adapter_id}'. Available: {available}")
        return self._adapters[adapter_id]

    def all(self) -> list[ProviderAdapter]:
        return [self._adapters[key] for key in sorted(self._adapters)]

    def ids(self) -> list[str]:
        return sorted(self._adapters)
