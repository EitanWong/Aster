from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ModelResidencyState(StrEnum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    path: str
    estimated_bytes: int = 0
    fingerprint: str | None = None


@dataclass(slots=True)
class ResidentModel:
    spec: ModelSpec
    state: ModelResidencyState = ModelResidencyState.UNLOADED
    value: Any | None = None
    loaded_bytes: int = 0
    active_leases: int = 0
    last_used_at: float = field(default_factory=time.monotonic)
    failure: str | None = None


@dataclass(slots=True)
class ModelLease:
    registry: ModelRegistry
    name: str
    model: ResidentModel
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self.registry.release(self.name)
        self._released = True

    def __enter__(self) -> ResidentModel:
        return self.model

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


class ModelRegistry:
    def __init__(self, *, memory_budget_bytes: int) -> None:
        self.memory_budget_bytes = memory_budget_bytes
        self._models: dict[str, ResidentModel] = {}
        self._loaded_bytes = 0

    @property
    def loaded_bytes(self) -> int:
        return self._loaded_bytes

    def register(self, spec: ModelSpec) -> None:
        current = self._models.get(spec.name)
        if current is not None and current.active_leases > 0:
            raise RuntimeError(f"Cannot replace leased model {spec.name!r}")
        self._models[spec.name] = ResidentModel(spec=spec)

    def get(self, name: str) -> ResidentModel | None:
        return self._models.get(name)

    def acquire(self, name: str, *, loaded_value: Any | None = None, loaded_bytes: int | None = None) -> ModelLease:
        resident = self._models.get(name)
        if resident is None:
            raise KeyError(name)
        if resident.state == ModelResidencyState.FAILED:
            raise RuntimeError(resident.failure or f"Model {name!r} is failed")
        if resident.state == ModelResidencyState.UNLOADED:
            self._mark_loaded(
                resident,
                value=loaded_value,
                loaded_bytes=loaded_bytes if loaded_bytes is not None else resident.spec.estimated_bytes,
            )
        resident.active_leases += 1
        resident.last_used_at = time.monotonic()
        self.evict_idle()
        return ModelLease(registry=self, name=name, model=resident)

    def release(self, name: str) -> None:
        resident = self._models.get(name)
        if resident is None:
            return
        resident.active_leases = max(resident.active_leases - 1, 0)
        resident.last_used_at = time.monotonic()
        self.evict_idle()

    def evict_idle(self) -> list[str]:
        evicted: list[str] = []
        while self._loaded_bytes > self.memory_budget_bytes:
            victim = self._select_idle_victim()
            if victim is None:
                break
            self._mark_unloaded(victim)
            evicted.append(victim.spec.name)
        return evicted

    def mark_failed(self, name: str, error: BaseException) -> None:
        resident = self._models.get(name)
        if resident is None:
            raise KeyError(name)
        if resident.state == ModelResidencyState.LOADED:
            self._mark_unloaded(resident)
        resident.state = ModelResidencyState.FAILED
        resident.failure = str(error)

    def _mark_loaded(self, resident: ResidentModel, *, value: Any | None, loaded_bytes: int) -> None:
        resident.value = value
        resident.loaded_bytes = max(loaded_bytes, 0)
        resident.state = ModelResidencyState.LOADED
        resident.failure = None
        self._loaded_bytes += resident.loaded_bytes

    def _mark_unloaded(self, resident: ResidentModel) -> None:
        self._loaded_bytes = max(self._loaded_bytes - resident.loaded_bytes, 0)
        resident.value = None
        resident.loaded_bytes = 0
        resident.state = ModelResidencyState.UNLOADED

    def _select_idle_victim(self) -> ResidentModel | None:
        candidates = [
            model
            for model in self._models.values()
            if model.state == ModelResidencyState.LOADED and model.active_leases == 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda model: model.last_used_at)
