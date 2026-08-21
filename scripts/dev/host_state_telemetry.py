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
from statistics import fmean, median
from typing import Any

import psutil

QUIESCENCE_SCHEMA_VERSION = 1
EXTERNAL_CPU_FORMULA = "max(0, system_cpu_percent - child_cpu_percent / logical_cpu_count)"


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


def capture_quiescence_sample() -> dict[str, Any]:
    """Capture the minimal host state needed before launching a benchmark child."""

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "schema_version": QUIESCENCE_SCHEMA_VERSION,
        "captured_utc": _now(),
        "captured_monotonic": time.monotonic(),
        "system_cpu_percent": max(0.0, float(psutil.cpu_percent(None))),
        "system_available_memory_bytes": int(memory.available),
        "system_available_memory_percent": 100.0 * float(memory.available) / float(memory.total),
        "system_swap_used_bytes": int(swap.used),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_quiescence_window(
    samples: list[dict[str, Any]],
    *,
    min_samples: int,
    cpu_median_limit: float,
    cpu_p95_limit: float,
    memory_percent_floor: float,
) -> dict[str, Any]:
    """Evaluate one rolling pre-launch host window without discarding any samples."""

    if min_samples <= 0:
        raise ValueError("minimum samples must be positive")
    if cpu_median_limit < 0.0 or cpu_p95_limit < cpu_median_limit:
        raise ValueError("CPU limits must be non-negative and ordered")
    if not 0.0 < memory_percent_floor <= 100.0:
        raise ValueError("memory floor must be in (0, 100]")

    failure_reasons: list[str] = []
    if len(samples) < min_samples:
        return {
            "schema_version": QUIESCENCE_SCHEMA_VERSION,
            "passed": False,
            "sample_count": len(samples),
            "cpu_median_percent": None,
            "cpu_p95_percent": None,
            "memory_available_min_percent": None,
            "swap_stable": False,
            "swap_used_before_bytes": None,
            "swap_used_after_bytes": None,
            "failure_reasons": ["insufficient-samples"],
        }

    cpu = [float(sample["system_cpu_percent"]) for sample in samples]
    memory = [float(sample["system_available_memory_percent"]) for sample in samples]
    swap = [int(sample["system_swap_used_bytes"]) for sample in samples]
    cpu_median = float(median(cpu))
    cpu_p95 = _percentile(cpu, 0.95)
    memory_min = min(memory)
    swap_stable = min(swap) == max(swap)
    if cpu_median > cpu_median_limit:
        failure_reasons.append("cpu-median-above-limit")
    if cpu_p95 > cpu_p95_limit:
        failure_reasons.append("cpu-p95-above-limit")
    if memory_min < memory_percent_floor:
        failure_reasons.append("memory-below-floor")
    if not swap_stable:
        failure_reasons.append("swap-changed")
    return {
        "schema_version": QUIESCENCE_SCHEMA_VERSION,
        "passed": not failure_reasons,
        "sample_count": len(samples),
        "cpu_median_percent": cpu_median,
        "cpu_p95_percent": cpu_p95,
        "memory_available_min_percent": memory_min,
        "swap_stable": swap_stable,
        "swap_used_before_bytes": swap[0],
        "swap_used_after_bytes": swap[-1],
        "failure_reasons": failure_reasons,
    }


def await_quiescent_host(
    *,
    sample_interval_seconds: float,
    min_samples: int,
    max_wait_seconds: float,
    cpu_median_limit: float = 6.0,
    cpu_p95_limit: float = 12.0,
    memory_percent_floor: float = 20.0,
    sample_fn: Any = capture_quiescence_sample,
    sleep_fn: Any = time.sleep,
    clock_fn: Any = time.monotonic,
) -> dict[str, Any]:
    """Wait for a declared rolling host contract and retain every attempted window."""

    if sample_interval_seconds < 0.0:
        raise ValueError("sample interval must be non-negative")
    if max_wait_seconds < 0.0:
        raise ValueError("maximum wait must be non-negative")

    policy = {
        "schema_version": QUIESCENCE_SCHEMA_VERSION,
        "sample_interval_seconds": float(sample_interval_seconds),
        "window_samples": int(min_samples),
        "max_wait_seconds": float(max_wait_seconds),
        "cpu_median_limit": float(cpu_median_limit),
        "cpu_p95_limit": float(cpu_p95_limit),
        "memory_percent_floor": float(memory_percent_floor),
    }
    psutil.cpu_percent(None)
    started = clock_fn()
    samples: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    while True:
        sample = dict(sample_fn())
        samples.append(sample)
        if len(samples) >= min_samples:
            start_index = len(samples) - min_samples
            window = summarize_quiescence_window(
                samples[start_index:],
                min_samples=min_samples,
                cpu_median_limit=cpu_median_limit,
                cpu_p95_limit=cpu_p95_limit,
                memory_percent_floor=memory_percent_floor,
            )
            window["sample_start_index"] = start_index
            window["sample_end_index"] = len(samples) - 1
            windows.append(window)
            if window["passed"]:
                return {
                    "schema_version": QUIESCENCE_SCHEMA_VERSION,
                    "status": "admitted",
                    "timed_out": False,
                    "wait_seconds": clock_fn() - started,
                    "policy": policy,
                    "samples": samples,
                    "windows": windows,
                    "admitted_window": window,
                }
        if clock_fn() - started >= max_wait_seconds:
            if not windows:
                timeout_window = summarize_quiescence_window(
                    samples,
                    min_samples=min_samples,
                    cpu_median_limit=cpu_median_limit,
                    cpu_p95_limit=cpu_p95_limit,
                    memory_percent_floor=memory_percent_floor,
                )
                timeout_window["sample_start_index"] = 0
                timeout_window["sample_end_index"] = len(samples) - 1
                windows.append(timeout_window)
            return {
                "schema_version": QUIESCENCE_SCHEMA_VERSION,
                "status": "timeout",
                "timed_out": True,
                "wait_seconds": clock_fn() - started,
                "policy": policy,
                "samples": samples,
                "windows": windows,
                "admitted_window": None,
            }
        if sample_interval_seconds > 0.0:
            sleep_fn(sample_interval_seconds)


def estimate_external_cpu_percent(
    system_cpu_percent: float, child_cpu_percent: float, logical_cpu_count: int
) -> float:
    """Estimate non-child CPU from aligned host and child samples."""

    if logical_cpu_count <= 0:
        raise ValueError("logical CPU count must be positive")
    return max(0.0, float(system_cpu_percent) - float(child_cpu_percent) / logical_cpu_count)


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
        self._logical_cpu_count = int(psutil.cpu_count(logical=True) or 0)

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
            child_cpu_percent = max(0.0, cpu_percent)
            system_cpu_percent = max(0.0, system_cpu_percent)
            sample = {
                "captured_monotonic": time.monotonic(),
                "rss_bytes": int(memory.rss),
                "vms_bytes": int(memory.vms),
                "cpu_percent": child_cpu_percent,
                "system_cpu_percent": system_cpu_percent,
                "system_available_memory_bytes": int(system_memory.available),
                "system_available_memory_percent": (
                    100.0 * float(system_memory.available) / float(system_memory.total)
                ),
                "system_swap_used_bytes": int(swap.used),
                "load_average": _load_average(),
            }
            if self._logical_cpu_count > 0:
                sample["estimated_external_cpu_percent"] = estimate_external_cpu_percent(
                    system_cpu_percent,
                    child_cpu_percent,
                    self._logical_cpu_count,
                )
            else:
                sample["estimated_external_cpu_percent"] = None
                if "logical-cpu-count-unavailable" not in self._errors:
                    self._errors.append("logical-cpu-count-unavailable")
            self._samples.append(sample)
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
        external_cpu = [
            float(sample["estimated_external_cpu_percent"])
            for sample in self._samples
            if sample.get("estimated_external_cpu_percent") is not None
        ]
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
            "logical_cpu_count": self._logical_cpu_count,
            "external_cpu_formula": EXTERNAL_CPU_FORMULA,
            "estimated_external_cpu_percent_median": float(median(external_cpu))
            if external_cpu
            else None,
            "estimated_external_cpu_percent_p95": _percentile(external_cpu, 0.95)
            if external_cpu
            else None,
            "samples": self._samples,
            "errors": list(self._errors),
        }
