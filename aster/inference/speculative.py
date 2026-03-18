from __future__ import annotations

from dataclasses import dataclass

from aster.core.config import SpeculativeSettings


@dataclass(slots=True)
class SpeculativeResult:
    enabled: bool
    proposed_tokens: int
    accepted_tokens: int

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens


class SpeculativeController:
    def __init__(self, settings: SpeculativeSettings) -> None:
        self.settings = settings
        self._recent_acceptance: list[float] = []

    def should_enable(
        self,
        request_class: str,
        *,
        policy_enabled: bool = True,
        force_disable: bool = False,
    ) -> bool:
        _ = request_class
        if force_disable:
            return False
        if not policy_enabled:
            return False
        if self._recent_acceptance:
            recent = self._recent_acceptance[-8:]
            if sum(recent) / len(recent) < self.settings.min_acceptance_rate:
                return False
        return True

    def record(self, result: SpeculativeResult) -> None:
        self._recent_acceptance.append(result.acceptance_rate)
        if len(self._recent_acceptance) > 256:
            self._recent_acceptance = self._recent_acceptance[-128:]
