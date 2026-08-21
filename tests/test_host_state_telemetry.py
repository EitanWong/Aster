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
