from __future__ import annotations

from dataclasses import dataclass

from aster.inference.speculative import SpeculativeController, SpeculativeResult


@dataclass(slots=True)
class VerificationOutcome:
    accepted_tokens: int
    rolled_back_tokens: int


class SpeculativePipeline:
    def __init__(self, controller: SpeculativeController) -> None:
        self.controller = controller

    def run_cycle(self, draft_tokens: int, acceptance_hint: float) -> tuple[SpeculativeResult, VerificationOutcome]:
        accepted = int(draft_tokens * acceptance_hint)
        outcome = VerificationOutcome(
            accepted_tokens=accepted,
            rolled_back_tokens=max(0, draft_tokens - accepted),
        )
        result = SpeculativeResult(enabled=True, proposed_tokens=draft_tokens, accepted_tokens=accepted)
        self.controller.record(result)
        return result, outcome
