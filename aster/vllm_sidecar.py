"""vLLM-MLX sidecar lifecycle management.

``VLLMSidecarManager`` is the only public API.  It spawns vllm-mlx as a
**child process** of Aster, waits until the sidecar passes a /health check,
and terminates it cleanly on shutdown.

From the user's perspective there is only one entry-point (``aster@<port>``).
The sidecar is an implementation detail — its own process title is set to
``vllm-mlx@<port>`` so it is identifiable when needed but clearly subordinate.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import httpx

from aster.core.process_title import build_vllm_process_title
from aster.telemetry.logging import get_logger

if TYPE_CHECKING:
    from aster.core.config import RuntimeSettings


class VLLMSidecarManager:
    """Owns the vllm-mlx child-process lifecycle on behalf of Aster.

    Lifecycle:
      start() → spawn, wait for /health (blocks the Aster lifespan until ready)
      stop()  → SIGTERM → wait → SIGKILL if needed
    """

    _HEALTH_POLL_INTERVAL: float = 1.0    # s between health probes
    _HEALTH_TIMEOUT: float = 300.0         # s before giving up on startup
    _STOP_GRACEFUL_TIMEOUT: float = 15.0   # s before escalating to SIGKILL

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._process: subprocess.Popen[bytes] | None = None
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_manage(self) -> bool:
        """True when Aster should own the sidecar (local vllm_mlx runtime)."""
        if self._settings.model.runtime != "vllm_mlx":
            return False
        host = _host(self._settings.vllm_mlx.base_url)
        return host in ("127.0.0.1", "localhost", "::1")

    async def start(self) -> None:
        """Spawn the sidecar and block until it is healthy."""
        if not self.should_manage():
            return
        if self._process is not None and self._process.poll() is None:
            self._log.info("vllm_sidecar_already_running", extra={"pid": self._process.pid})
            return

        cmd = self._build_cmd()
        self._log.info("vllm_sidecar_starting")
        # start_new_session=True puts the child in its own process group.
        # This means os.killpg() on stop() reaches every subprocess spawned
        # by the bootstrap (uvicorn workers, engine threads, etc.) — not just
        # the top-level python -c process.
        self._process = subprocess.Popen(cmd, start_new_session=True)
        self._log.info("vllm_sidecar_spawned", extra={"pid": self._process.pid})
        await self._wait_healthy()

    async def stop(self) -> None:
        """Gracefully terminate the sidecar process group."""
        if self._process is None:
            return
        proc, self._process = self._process, None
        if proc.poll() is not None:
            return  # already gone

        pid = proc.pid
        self._log.info("vllm_sidecar_stopping", extra={"pid": pid})

        # Send SIGTERM to the entire process group so uvicorn workers,
        # Metal engine threads, etc. all receive the signal — not only the
        # top-level python -c bootstrap process.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone between poll() and killpg()

        # Wait in a real OS thread so this survives asyncio loop shutdown
        # (the lifespan finally-block runs while the loop is still closing).
        def _wait() -> int | None:
            try:
                proc.wait(timeout=self._STOP_GRACEFUL_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
            return proc.poll()

        rc = await asyncio.to_thread(_wait)

        if rc is None:
            # Graceful window expired — escalate to SIGKILL on the group.
            self._log.warning("vllm_sidecar_sigkill", extra={"pid": pid})
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

        self._log.info("vllm_sidecar_stopped", extra={"pid": pid})

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def _build_cmd(self) -> list[str]:
        """Build the argv list for the vllm-mlx child process.

        The child runs a tiny Python bootstrap (via ``python -c``) that:
          1. Sets the process title with setproctitle *before* importing vllm.
          2. Calls vllm_mlx.cli.main() with the correct sys.argv.

        This guarantees ``ps`` shows ``vllm-mlx@<port>`` regardless of how
        the OS handles argv[0] for script-based entry-points.
        """
        s = self._settings
        title = build_vllm_process_title(port=_port(s.vllm_mlx.base_url))

        vllm_argv = [
            "vllm-mlx", "serve", s.model.path,
            "--host", _host(s.vllm_mlx.base_url),
            "--port", str(_port(s.vllm_mlx.base_url)),
            "--max-num-seqs", str(s.batch.max_batch_size),
            "--prefill-batch-size", str(s.batch.prefill_batch_size),
            "--completion-batch-size", str(s.batch.decode_batch_size),
            "--timeout", str(s.vllm_mlx.timeout_seconds),
            "--cache-memory-percent", str(s.vllm_mlx.cache_memory_percent),
            "--stream-interval", str(s.vllm_mlx.stream_interval),
        ]
        if s.vllm_mlx.chunked_prefill_tokens > 0:
            vllm_argv += ["--chunked-prefill-tokens", str(s.vllm_mlx.chunked_prefill_tokens)]
        if s.vllm_mlx.continuous_batching:
            vllm_argv.append("--continuous-batching")
        if s.vllm_mlx.use_paged_cache:
            vllm_argv.append("--use-paged-cache")
        vllm_argv.append(
            "--enable-prefix-cache" if s.vllm_mlx.enable_prefix_cache
            else "--disable-prefix-cache"
        )
        emb = s.embeddings.model_path or s.embeddings.model
        if emb:
            vllm_argv += ["--embedding-model", emb]
        if s.vllm_mlx.reasoning_parser:
            vllm_argv += ["--reasoning-parser", s.vllm_mlx.reasoning_parser]
        if s.vllm_mlx.api_key:
            vllm_argv += ["--api-key", s.vllm_mlx.api_key]

        # Tiny bootstrap: set title then hand off to vllm_mlx.cli.main
        bootstrap = (
            "import sys, setproctitle as _sp\n"
            f"_sp.setproctitle({title!r})\n"
            f"sys.argv = {vllm_argv!r}\n"
            "from vllm_mlx.cli import main\n"
            "raise SystemExit(main() or 0)\n"
        )
        return [sys.executable, "-c", bootstrap]

    # ------------------------------------------------------------------
    # Health polling
    # ------------------------------------------------------------------

    async def _wait_healthy(self) -> None:
        health_url = self._settings.vllm_mlx.base_url.rstrip("/") + "/health"
        deadline = time.monotonic() + self._HEALTH_TIMEOUT

        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                # Bail out early if the child already died
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(
                        f"vllm-mlx exited (code {self._process.returncode}) "
                        "before passing health check — check logs"
                    )
                try:
                    resp = await client.get(health_url)
                    if resp.status_code < 500:
                        elapsed = round(time.monotonic() - (deadline - self._HEALTH_TIMEOUT), 1)
                        self._log.info("vllm_sidecar_ready", extra={"elapsed_s": elapsed})
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(self._HEALTH_POLL_INTERVAL)

        raise RuntimeError(
            f"vllm-mlx did not become healthy within {self._HEALTH_TIMEOUT}s"
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _host(base_url: str) -> str:
    return urlparse(base_url).hostname or "127.0.0.1"


def _port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running vllm-mlx as a standalone sidecar.

    This is used by launchd_entry.py to start vllm-mlx before Aster.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Start vLLM-MLX sidecar")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    from aster.core.config import load_settings

    settings = load_settings(args.config)

    # Build and execute the vllm-mlx command
    manager = VLLMSidecarManager(settings)
    cmd = manager._build_cmd()

    # Exec the vllm-mlx process (replaces current process)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
