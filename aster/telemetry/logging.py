from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from aster.core.config import RuntimeSettings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        builtin = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
            "relativeCreated", "thread", "threadName", "processName", "process", "message",
        }
        for key, value in record.__dict__.items():
            if key not in builtin and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: RuntimeSettings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if settings.telemetry.json_logs else logging.Formatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.logging.level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
