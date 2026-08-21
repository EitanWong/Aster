#!/usr/bin/env python3
"""Read-only host and child-process telemetry for benchmark envelopes."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from statistics import fmean
from typing import Any

import psutil


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_average() -> list[float] | None:
    try:
        values = os.getloadavg()
    except (AttributeError, OSError):
        return None
    return [float(value) for value in values]


def _process_snapshot(process: psutil.Process) -> dict[str, Any]:
    try:
        memory = process.memory_info()
        cpu_times = process.cpu_times()
        return {
            "pid": int(process.pid),
            "rss_bytes": int(memory.rss),
            "vms_bytes": int(memory.vms),
            "cpu_user_seconds": float(cpu_times.user),
            "cpu_system_seconds": float(cpu_times.system),
        }
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as error:
        return {
            "pid": int(process.pid),
            "rss_bytes": None,
            "vms_bytes": None,
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
            "error": type(error).__name__,
        }


def capture_host_state() -> dict[str, Any]:
    """Capture one host/process snapshot outside the timed child interval."""

    process = psutil.Process()
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "schema_version": 1,
        "captured_utc": _now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process": _process_snapshot(process),
        "system": {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_bytes": int(memory.total),
            "memory_available_bytes": int(memory.available),
            "memory_available_percent": 100.0 * float(memory.available) / float(memory.total),
            "memory_used_bytes": int(memory.used),
            "swap_total_bytes": int(swap.total),
            "swap_used_bytes": int(swap.used),
            "load_average": _load_average(),
        },
    }


def probe_command(command: tuple[str, ...], *, timeout_seconds: float = 1.5) -> dict[str, Any]:
    """Run one read-only capability probe and preserve unavailable reasons."""

    if not command:
        raise ValueError("probe command must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("probe timeout must be positive")
    executable = shutil.which(command[0])
    base = {"command": list(command), "exit_code": None}
    if executable is None:
        return {**base, "status": "unavailable", "reason": "command-not-found"}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            (executable, *command[1:]),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            **base,
            "status": "unavailable",
            "reason": "timeout",
            "elapsed_seconds": time.perf_counter() - started,
        }
    except OSError as error:
        return {
            **base,
            "status": "unavailable",
            "reason": type(error).__name__,
            "elapsed_seconds": time.perf_counter() - started,
        }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result = {
        **base,
        "exit_code": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
    }
    if completed.returncode != 0:
        return {**result, "status": "unavailable", "reason": f"exit-{completed.returncode}"}
    return {**result, "status": "available"}


def probe_thermal_power(*, timeout_seconds: float = 1.5) -> dict[str, Any]:
    """Probe macOS telemetry commands without assuming privileged access."""

    return {
        "schema_version": 1,
        "captured_utc": _now(),
        "probes": {
            "powermetrics": probe_command(
                ("powermetrics", "-n", "1", "-i", "100", "--show-process-energy"),
                timeout_seconds=timeout_seconds,
            ),
            "pmset_thermal": probe_command(
                ("pmset", "-g", "thermlog"), timeout_seconds=timeout_seconds
            ),
            "memory_pressure": probe_command(
                ("memory_pressure", "-Q"), timeout_seconds=timeout_seconds
            ),
        },
    }


class ProcessTelemetrySampler:
    """Sample one child process and host counters from outside the timed path."""

    def __init__(self, pid: int, *, interval_seconds: float = 0.05) -> None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval must be positive")
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self._process: psutil.Process | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []

    def _sample(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            memory = process.memory_info()
            cpu_percent = float(process.cpu_percent(None))
            system_cpu_percent = float(psutil.cpu_percent(None))
            system_memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            self._samples.append(
                {
                    "captured_monotonic": time.monotonic(),
                    "rss_bytes": int(memory.rss),
                    "vms_bytes": int(memory.vms),
                    "cpu_percent": max(0.0, cpu_percent),
                    "system_cpu_percent": max(0.0, system_cpu_percent),
                    "system_available_memory_bytes": int(system_memory.available),
                    "system_available_memory_percent": (
                        100.0 * float(system_memory.available) / float(system_memory.total)
                    ),
                    "system_swap_used_bytes": int(swap.used),
                    "load_average": _load_average(),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as error:
            name = type(error).__name__
            if name not in self._errors:
                self._errors.append(name)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        try:
            self._process = psutil.Process(self.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as error:
            self._errors.append(type(error).__name__)
            return
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2 + 1.0)
        self._sample()
        if not self._samples:
            return {
                "schema_version": 1,
                "status": "unavailable",
                "pid": self.pid,
                "sample_count": 0,
                "errors": list(self._errors),
            }
        rss = [int(sample["rss_bytes"]) for sample in self._samples]
        cpu = [float(sample["cpu_percent"]) for sample in self._samples]
        system_cpu = [float(sample["system_cpu_percent"]) for sample in self._samples]
        available = [int(sample["system_available_memory_bytes"]) for sample in self._samples]
        available_percent = [
            float(sample["system_available_memory_percent"]) for sample in self._samples
        ]
        swap = [int(sample["system_swap_used_bytes"]) for sample in self._samples]
        loads = [
            float(values[0])
            for sample in self._samples
            if isinstance(values := sample.get("load_average"), list) and values
        ]
        return {
            "schema_version": 1,
            "status": "complete",
            "pid": self.pid,
            "sample_count": len(self._samples),
            "rss_before_bytes": rss[0],
            "peak_rss_bytes": max(rss),
            "rss_after_bytes": rss[-1],
            "cpu_percent_avg": fmean(cpu),
            "cpu_percent_max": max(cpu),
            "system_cpu_percent_avg": fmean(system_cpu),
            "system_cpu_percent_max": max(system_cpu),
            "system_available_memory_min_bytes": min(available),
            "system_available_memory_min_percent": min(available_percent),
            "system_swap_used_before_bytes": swap[0],
            "system_swap_used_max_bytes": max(swap),
            "system_swap_used_after_bytes": swap[-1],
            "load_average_one_min_max": max(loads) if loads else None,
            "errors": list(self._errors),
        }
