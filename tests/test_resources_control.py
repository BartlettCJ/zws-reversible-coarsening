from __future__ import annotations

from pathlib import Path

from zws_nullity_lab.control import RunControl
from zws_nullity_lab.resources import GpuProcess, ResourceGovernor, ResourceSnapshot


def _snapshot(**overrides: object) -> ResourceSnapshot:
    values = {
        "sampled_utc": "2026-08-24T00:00:00+00:00",
        "cpu_percent": 10.0,
        "available_ram_gib": 40.0,
        "gpu_index": 0,
        "gpu_name": "RTX 3060",
        "gpu_utilization_percent": 2.0,
        "gpu_memory_used_mib": 600.0,
        "gpu_memory_total_mib": 12288.0,
        "gpu_temperature_c": 45.0,
        "external_gpu_processes": (),
        "external_gpu_memory_mib": 0.0,
        "user_idle_seconds": 120.0,
        "nvidia_smi_available": True,
    }
    values.update(overrides)
    return ResourceSnapshot(**values)  # type: ignore[arg-type]


def test_resource_governor_yields_to_external_gpu_process() -> None:
    governor = ResourceGovernor(
        {
            "enabled": True,
            "yield_for_external_gpu_processes": True,
            "external_gpu_memory_mib": 1536,
            "cpu_busy_percent": 78,
            "cpu_resume_percent": 55,
            "min_available_ram_gib": 8,
            "yield_for_gpu_utilization": True,
            "gpu_busy_utilization_percent": 40,
            "gpu_resume_utilization_percent": 20,
            "gpu_busy_temperature_c": 83,
            "gpu_resume_temperature_c": 76,
        }
    )
    busy = _snapshot(
        external_gpu_processes=(GpuProcess(999, "blender.exe", 4096.0),),
        external_gpu_memory_mib=4096.0,
    )
    decision = governor.decide(busy)
    assert not decision.runnable
    assert any("external CUDA process" in reason for reason in decision.reasons)


def test_file_controls_round_trip(tmp_path: Path) -> None:
    control = RunControl(tmp_path)
    control.request("pause")
    assert control.read().pause
    control.request("resume")
    assert not control.read().pause
    control.request("hibernate")
    assert control.read().hibernate
    control.request("resume")
    assert not control.read().hibernate
    control.request("stop")
    assert control.read().stop
    control.request("clear-stop")
    assert not control.read().stop
