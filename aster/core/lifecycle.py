from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from aster.api.routes import build_router
from aster.autotune.selector import PolicySelector
from aster.cache.paged_kv_cache import PagedKVCache
from aster.cache.prefix_cache import PrefixCache
from aster.core.config import RuntimeSettings, load_settings
from aster.inference.engine import InferenceEngine
from aster.scheduler.policy_engine import PolicyEngine
from aster.scheduler.scheduler import RequestScheduler
from aster.telemetry.logging import configure_logging, get_logger
from aster.telemetry.metrics import MetricsRegistry
from aster.workers.supervisor import WorkerSupervisor


@dataclass(slots=True)
class Container:
    settings: RuntimeSettings
    metrics: MetricsRegistry
    kv_cache: PagedKVCache
    prefix_cache: PrefixCache
    policy_engine: PolicyEngine
    inference_engine: InferenceEngine
    scheduler: RequestScheduler
    supervisor: WorkerSupervisor
    audio: AudioServiceContainer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    logger = get_logger(__name__)
    await container.supervisor.attach_scheduler(container.scheduler)
    await container.supervisor.start()
    if container.settings.autotune.enabled and container.settings.autotune.startup_warmup:
        await PolicySelector(container.settings, container.metrics, container.policy_engine).startup_select()
    logger.info("application_started")
    yield
    await container.supervisor.stop()
    logger.info("application_stopped")


def create_application(config_path: str) -> FastAPI:
    settings = load_settings(config_path)
    configure_logging(settings)
    metrics = MetricsRegistry(settings.telemetry.metrics_namespace)
    kv_cache = PagedKVCache(settings.cache, metrics)
    prefix_cache = PrefixCache(settings.cache, metrics)
    policy_engine = PolicyEngine(settings)
    inference_engine = InferenceEngine(settings, metrics, kv_cache, prefix_cache, policy_engine)
    scheduler = RequestScheduler(settings, metrics, inference_engine, policy_engine)
    supervisor = WorkerSupervisor(settings, metrics, inference_engine)
    container = Container(
        settings=settings,
        metrics=metrics,
        kv_cache=kv_cache,
        prefix_cache=prefix_cache,
        policy_engine=policy_engine,
        inference_engine=inference_engine,
        scheduler=scheduler,
        supervisor=supervisor,
    )
    app = FastAPI(title="Aster", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.include_router(build_router())
    return app
er())
    return app
p.include_router(build_router())
    return app
