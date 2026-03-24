from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from aster.api.routes import build_router
from aster.audio.factory import create_asr_service, create_tts_service
from aster.audio.service import AudioServiceContainer
from aster.autotune.selector import PolicySelector
from aster.cache.paged_kv_cache import PagedKVCache
from aster.cache.prefix_cache import PrefixCache
from aster.core.config import RuntimeSettings, load_settings
from aster.inference.engine import InferenceEngine
from aster.runtime.tools import ToolRegistry, build_default_tool_registry
from aster.scheduler.policy_engine import PolicyEngine
from aster.scheduler.scheduler import RequestScheduler
from aster.telemetry.logging import configure_logging, get_logger
from aster.telemetry.metrics import MetricsRegistry
from aster.vllm_sidecar import VLLMSidecarManager
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
    tool_registry: ToolRegistry
    vllm_sidecar: VLLMSidecarManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    logger = get_logger(__name__)
    # Start the vLLM-MLX sidecar first (no-op when runtime != vllm_mlx or remote)
    await container.vllm_sidecar.start()
    await container.supervisor.attach_scheduler(container.scheduler)
    await container.supervisor.start()
    # Pre-warm expensive resources (tokenizer, etc.) concurrently while the
    # server is already accepting requests — non-blocking, best-effort.
    asyncio.create_task(container.inference_engine.warmup(), name="aster-warmup")
    if container.settings.autotune.enabled and container.settings.autotune.startup_warmup:
        await PolicySelector(container.settings, container.metrics, container.policy_engine).startup_select()
    logger.info("application_started")
    yield
    await container.supervisor.stop()
    await container.inference_engine.aclose()
    # Stop the sidecar after Aster has fully shut down
    await container.vllm_sidecar.stop()
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
    vllm_sidecar = VLLMSidecarManager(settings)

    asr_service = create_asr_service(settings.audio.asr)
    tts_service = create_tts_service(settings.audio.tts)
    audio = AudioServiceContainer(asr=asr_service, tts=tts_service)
    tool_registry = build_default_tool_registry()

    container = Container(
        settings=settings,
        metrics=metrics,
        kv_cache=kv_cache,
        prefix_cache=prefix_cache,
        policy_engine=policy_engine,
        inference_engine=inference_engine,
        scheduler=scheduler,
        supervisor=supervisor,
        audio=audio,
        tool_registry=tool_registry,
        vllm_sidecar=vllm_sidecar,
    )
    app = FastAPI(title="Aster", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    app.include_router(build_router())
    return app
