from __future__ import annotations

import sys

from aster.core.config import RuntimeSettings


def build_aster_process_title(settings: RuntimeSettings) -> str:
    """``aster@<port>``"""
    return f"aster@{settings.api.port}"


def build_vllm_process_title(*, port: int) -> str:
    """``vllm-mlx@<port>``"""
    return f"vllm-mlx@{port}"


def set_process_title(title: str) -> None:
    try:
        import setproctitle
        setproctitle.setproctitle(title)
    except Exception:
        sys.argv[0] = title
