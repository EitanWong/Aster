from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AsterError(Exception):
    code: str
    message: str
    status_code: int = 500
    details: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "type": self.code,
                "message": self.message,
                "details": self.details or {},
            }
        }


class ConfigurationError(AsterError):
    pass


class OverloadedError(AsterError):
    pass


class WorkerUnavailableError(AsterError):
    pass
