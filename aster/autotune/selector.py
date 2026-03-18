from __future__ import annotations

from pathlib import Path

from aster.autotune.benchmark import BenchmarkSuite
from aster.autotune.profiles import TuningProfile
from aster.core.config import RuntimeSettings
from aster.scheduler.policy_engine import PolicyEngine
from aster.telemetry.logging import get_logger
from aster.telemetry.metrics import MetricsRegistry


class PolicySelector:
    def __init__(self, settings: RuntimeSettings, metrics: MetricsRegistry, policy_engine: PolicyEngine) -> None:
        self.settings = settings
        self.metrics = metrics
        self.policy_engine = policy_engine
        self.logger = get_logger(__name__)

    async def startup_select(self) -> TuningProfile:
        existing = TuningProfile.from_path(self.settings.autotune.profile_path)
        if existing is not None:
            self._apply(existing)
            self.logger.info("loaded_autotune_profile")
            return existing

        suite = BenchmarkSuite(self.settings, self.metrics, self.policy_engine)
        ranked = await suite.run(
            mode="quick",
            candidate_timeout_seconds=60.0,
            shortlist_size=3,
            progress=False,
        )
        if not ranked:
            fallback = TuningProfile(
                name="fallback-default-stable",
                speculative_enabled=False,
                draft_tokens=2,
                batch_window_ms=self.settings.batch.min_batch_window_ms,
                max_batch_size=min(2, self.settings.batch.max_batch_size),
                stream_flush_ms=12.0,
                score=0.0,
            )
            self._apply(fallback)
            return fallback

        winner = ranked[0]
        profile = TuningProfile(
            name="startup-selected",
            speculative_enabled=winner.speculative_enabled,
            draft_tokens=winner.draft_tokens,
            batch_window_ms=winner.batch_window_ms,
            max_batch_size=winner.max_batch_size,
            stream_flush_ms=winner.stream_flush_ms,
            score=winner.total_score,
        )
        path = Path(self.settings.autotune.profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.to_json())
        self._apply(profile)
        self.logger.info("selected_autotune_profile")
        return profile

    def _apply(self, profile: TuningProfile) -> None:
        self.policy_engine.update(
            speculative_enabled=profile.speculative_enabled,
            speculative_draft_tokens=profile.draft_tokens,
            batch_window_ms=profile.batch_window_ms,
            max_batch_size=profile.max_batch_size,
            stream_flush_ms=profile.stream_flush_ms,
        )
