from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from aster.api.middleware import install_api_middleware
from aster.api.routes import build_router
from aster.audio.factory import create_asr_service, create_tts_service
from aster.audio.service import AudioServiceContainer
from aster.core.config import RuntimeSettings, load_settings
from aster.inference.engine import InferenceEngine
from typing import Any
from aster.runtime.tools import ToolRegistry, build_default_tool_registry
from aster.telemetry.logging import configure_logging, get_logger
from aster.telemetry.metrics import MetricsRegistry


@dataclass(slots=True)
class Container:
    settings: RuntimeSettings
    metrics: MetricsRegistry
    inference_engine: Any
    audio: AudioServiceContainer
    tool_registry: ToolRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    logger = get_logger(__name__)
    await container.inference_engine.start()
    asyncio.create_task(container.inference_engine.warmup(), name="aster-warmup")
    if container.settings.deprecation_warnings:
        logger.warning(
            "deprecated_config_detected",
            extra={"warnings": list(container.settings.deprecation_warnings)},
        )
    logger.info("application_started")
    yield
    await container.inference_engine.aclose()
    logger.info("application_stopped")


def create_application_from_settings(settings: RuntimeSettings) -> FastAPI:
    configure_logging(settings)
    metrics = MetricsRegistry(settings.telemetry.metrics_namespace)
    if settings.engine.engine_type == "batched":
        from aster.inference.batched_engine import BatchedEngine
        inference_engine = BatchedEngine(settings, metrics)
    else:
        inference_engine = InferenceEngine(settings, metrics)
    asr_service = create_asr_service(settings.audio.asr)
    tts_service = create_tts_service(settings.audio.tts)
    audio = AudioServiceContainer(asr=asr_service, tts=tts_service)
    tool_registry = build_default_tool_registry()

    container = Container(
        settings=settings,
        metrics=metrics,
        inference_engine=inference_engine,
        audio=audio,
        tool_registry=tool_registry,
    )
    app = FastAPI(title="Aster", version="0.1.0", lifespan=lifespan)
    app.state.container = container
    install_api_middleware(app, settings.api)
    app.include_router(
        build_router(responses_store_max_entries=settings.api.responses_store_max_entries)
    )
    return app


def create_application(config_path: str) -> FastAPI:
    return create_application_from_settings(load_settings(config_path))
