from __future__ import annotations

from aster.inference.model_registry import ModelRegistry, ModelResidencyState, ModelSpec


def test_model_registry_lease_pins_model_from_eviction() -> None:
    registry = ModelRegistry(memory_budget_bytes=100)
    registry.register(ModelSpec(name="a", path="/models/a", estimated_bytes=80))
    registry.register(ModelSpec(name="b", path="/models/b", estimated_bytes=80))

    lease = registry.acquire("a", loaded_value="model-a")
    registry.acquire("b", loaded_value="model-b").release()

    model_a = registry.get("a")
    model_b = registry.get("b")
    assert model_a is not None
    assert model_b is not None
    assert model_a.state == ModelResidencyState.LOADED
    assert model_b.state == ModelResidencyState.UNLOADED

    lease.release()
    evicted = registry.evict_idle()

    assert evicted == []
    assert registry.loaded_bytes <= registry.memory_budget_bytes


def test_model_registry_evicts_lru_idle_model_under_budget_pressure() -> None:
    registry = ModelRegistry(memory_budget_bytes=100)
    registry.register(ModelSpec(name="a", path="/models/a", estimated_bytes=70))
    registry.register(ModelSpec(name="b", path="/models/b", estimated_bytes=70))

    registry.acquire("a").release()
    registry.acquire("b").release()

    model_a = registry.get("a")
    model_b = registry.get("b")
    assert model_a is not None
    assert model_b is not None
    assert model_a.state == ModelResidencyState.UNLOADED
    assert model_b.state == ModelResidencyState.LOADED
    assert registry.loaded_bytes == 70
