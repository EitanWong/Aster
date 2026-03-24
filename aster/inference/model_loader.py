from __future__ import annotations

from dataclasses import dataclass

from aster.core.config import ModelSettings, SpeculativeSettings


@dataclass(slots=True)
class LoadedModel:
    name: str
    path: str
    runtime: str


class ModelLoader:
    def __init__(self, model: ModelSettings, speculative: SpeculativeSettings) -> None:
        self.model = model
        self.speculative = speculative

    def load_target(self) -> LoadedModel:
        return LoadedModel(self.model.name, self.model.path, self.model.runtime)

    def load_draft(self) -> LoadedModel:
        return LoadedModel(self.speculative.draft_name, self.speculative.draft_path, self.model.runtime)
