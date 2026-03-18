from __future__ import annotations

from dataclasses import dataclass

from aster.core.config import ModelSettings


@dataclass(slots=True)
class LoadedModel:
    name: str
    path: str
    runtime: str


class ModelLoader:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def load_target(self) -> LoadedModel:
        return LoadedModel(self.settings.name, self.settings.path, self.settings.runtime)

    def load_draft(self) -> LoadedModel:
        return LoadedModel(self.settings.draft_name, self.settings.draft_path, self.settings.runtime)
