from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/dev/host_state_telemetry.py"


def load_telemetry() -> ModuleType:
    spec = importlib.util.spec_from_file_location("host_state_telemetry", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_snapshot_exposes_explicit_process_and_system_fields() -> None:
    telemetry = load_telemetry()

    snapshot = telemetry.capture_host_state()

    assert snapshot["schema_version"] == 1
    assert snapshot["process"]["pid"] > 0
    assert snapshot["process"]["rss_bytes"] > 0
    assert snapshot["system"]["memory_total_bytes"] > 0
    assert snapshot["system"]["memory_available_bytes"] > 0
    assert 0.0 < snapshot["system"]["memory_available_percent"] <= 100.0
    assert snapshot["system"]["swap_used_bytes"] >= 0
    assert len(snapshot["system"]["load_average"]) == 3


def test_missing_probe_is_explicitly_unavailable() -> None:
    telemetry = load_telemetry()

    result = telemetry.probe_command(("aster-command-that-does-not-exist",), timeout_seconds=0.1)

    assert result["status"] == "unavailable"
    assert result["reason"] == "command-not-found"
    assert result["exit_code"] is None


def test_process_sampler_records_child_and_system_samples() -> None:
    telemetry = load_telemetry()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"])
    sampler = telemetry.ProcessTelemetrySampler(child.pid, interval_seconds=0.02)
    sampler.start()
    assert child.wait(timeout=2.0) == 0
    result = sampler.finish()

    assert result["status"] == "complete"
    assert result["pid"] == child.pid
    assert result["sample_count"] >= 1
    assert result["peak_rss_bytes"] >= result["rss_before_bytes"] > 0
    assert result["cpu_percent_max"] >= 0.0
    assert result["system_cpu_percent_avg"] >= 0.0
    assert result["system_cpu_percent_max"] >= result["system_cpu_percent_avg"]
    assert result["system_available_memory_min_bytes"] > 0
    assert 0.0 < result["system_available_memory_min_percent"] <= 100.0
    assert result["system_swap_used_max_bytes"] >= 0


def test_sampler_rejects_non_positive_interval() -> None:
    telemetry = load_telemetry()

    with pytest.raises(ValueError, match="interval"):
        telemetry.ProcessTelemetrySampler(1, interval_seconds=0)


def _quiescence_sample(*, cpu: float = 4.0, memory: float = 80.0, swap: int = 10) -> dict:
    return {
        "captured_monotonic": 1.0,
        "system_cpu_percent": cpu,
        "system_available_memory_percent": memory,
        "system_swap_used_bytes": swap,
    }


def test_quiescence_window_requires_cpu_memory_and_stable_swap() -> None:
    telemetry = load_telemetry()

    passed = telemetry.summarize_quiescence_window(
        [_quiescence_sample() for _ in range(20)],
        min_samples=20,
        cpu_median_limit=6.0,
        cpu_p95_limit=12.0,
        memory_percent_floor=20.0,
    )

    assert passed["passed"] is True
    assert passed["sample_count"] == 20
    assert passed["cpu_median_percent"] == 4.0
    assert passed["cpu_p95_percent"] == 4.0
    assert passed["memory_available_min_percent"] == 80.0
    assert passed["swap_stable"] is True
    assert passed["failure_reasons"] == []

    failed = telemetry.summarize_quiescence_window(
        [_quiescence_sample(cpu=20.0, memory=10.0, swap=index) for index in range(20)],
        min_samples=20,
        cpu_median_limit=6.0,
        cpu_p95_limit=12.0,
        memory_percent_floor=20.0,
    )

    assert failed["passed"] is False
    assert failed["swap_stable"] is False
    assert failed["failure_reasons"] == [
        "cpu-median-above-limit",
        "cpu-p95-above-limit",
        "memory-below-floor",
        "swap-changed",
    ]


def test_await_quiescent_host_retains_rejected_and_admitted_windows() -> None:
    telemetry = load_telemetry()
    samples = iter(
        [_quiescence_sample(cpu=20.0) for _ in range(20)]
        + [_quiescence_sample(cpu=4.0) for _ in range(20)]
    )

    result = telemetry.await_quiescent_host(
        sample_interval_seconds=0.0,
        min_samples=20,
        max_wait_seconds=1.0,
        sample_fn=lambda: next(samples),
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "admitted"
    assert len(result["samples"]) == 39
    assert len(result["windows"]) == 20
    assert result["windows"][0]["passed"] is False
    assert result["windows"][-1]["passed"] is True
    assert result["admitted_window"]["sample_start_index"] == 19


def test_await_quiescent_host_retains_timeout_without_replacement() -> None:
    telemetry = load_telemetry()

    result = telemetry.await_quiescent_host(
        sample_interval_seconds=0.0,
        min_samples=2,
        max_wait_seconds=0.0,
        sample_fn=lambda: _quiescence_sample(cpu=20.0),
        sleep_fn=lambda _: None,
    )

    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["samples"]
    assert result["windows"]
    assert result["admitted_window"] is None


def test_external_cpu_estimate_is_sample_aligned_and_clamped() -> None:
    telemetry = load_telemetry()

    assert telemetry.estimate_external_cpu_percent(20.0, 100.0, 10) == 10.0
    assert telemetry.estimate_external_cpu_percent(20.0, 300.0, 10) == 0.0
    with pytest.raises(ValueError, match="logical CPU"):
        telemetry.estimate_external_cpu_percent(20.0, 1.0, 0)
