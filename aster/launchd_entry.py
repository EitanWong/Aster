from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import httpx
import uvicorn

from aster.core.config import RuntimeSettings, load_settings
from aster.core.lifecycle import create_application
from aster.core.process_title import build_aster_process_title, set_process_title


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "run"
LOG_DIR = PROJECT_ROOT / "logs"
VLLM_PID_FILE = RUN_DIR / "vllm-mlx.pid"
VLLM_LOG_FILE = LOG_DIR / "vllm-mlx.log"

_STARTED_VLLM: subprocess.Popen[bytes] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="launchd entrypoint for Aster")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    settings = load_settings(args.config)
    set_process_title(build_aster_process_title(settings))

    _install_signal_handlers()
    _start_vllm_if_needed(settings, args.config)

    app = create_application(args.config)
    runtime_settings = app.state.container.settings
    uvicorn.run(
        app,
        host=runtime_settings.api.host,
        port=runtime_settings.api.port,
        log_level=runtime_settings.logging.level.lower(),
    )


def _install_signal_handlers() -> None:
    atexit.register(_stop_started_vllm)

    def _handle_signal(signum: int, _frame: object) -> None:
        _stop_started_vllm()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _start_vllm_if_needed(settings: RuntimeSettings, config_path: str) -> None:
    global _STARTED_VLLM

    if settings.model.runtime != "vllm_mlx":
        return
    if not _is_local_base_url(settings.vllm_mlx.base_url):
        return
    if _port_open(_vllm_host(settings), _vllm_port(settings)):
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with VLLM_LOG_FILE.open("ab") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "aster.vllm_sidecar", "--config", str(Path(config_path).resolve())],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    _STARTED_VLLM = proc
    VLLM_PID_FILE.write_text(str(proc.pid))
    _wait_for_vllm(settings)


def _wait_for_vllm(settings: RuntimeSettings) -> None:
    health_url = settings.vllm_mlx.base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if _STARTED_VLLM is not None and _STARTED_VLLM.poll() is not None:
            raise RuntimeError(f"vLLM-MLX exited early with code {_STARTED_VLLM.returncode}")
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("Timed out waiting for vLLM-MLX health endpoint")


def _stop_started_vllm() -> None:
    global _STARTED_VLLM

    proc = _STARTED_VLLM
    _STARTED_VLLM = None
    if proc is None:
        return
    if proc.poll() is not None:
        if VLLM_PID_FILE.exists():
            VLLM_PID_FILE.unlink(missing_ok=True)
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    VLLM_PID_FILE.unlink(missing_ok=True)


def _is_local_base_url(base_url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _vllm_host(settings: RuntimeSettings) -> str:
    from urllib.parse import urlparse

    return urlparse(settings.vllm_mlx.base_url).hostname or "127.0.0.1"


def _vllm_port(settings: RuntimeSettings) -> int:
    from urllib.parse import urlparse

    parsed = urlparse(settings.vllm_mlx.base_url)
    if parsed.port is not None:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


if __name__ == "__main__":
    main()
