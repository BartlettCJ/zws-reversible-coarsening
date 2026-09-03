from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import torch


def _system_memory_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total_phys)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return None


def _command_version(command: str, args: list[str]) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=8, check=False
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return text[0] if text else executable
    except Exception:
        return executable



def recommend_config(
    cuda_devices: list[dict[str, Any]],
    *,
    memory_gib: float | None = None,
    platform_name: str | None = None,
) -> str:
    """Choose the conservative supplied profile for the detected primary GPU.

    The RTX 3060 12 GB Windows profile is explicit because that machine was
    requested as a supported target. Other GPUs retain the original VRAM-tier
    policy. System RAM is advisory; it is recorded and warned on rather than
    silently changing scientific parameters.
    """
    if not cuda_devices:
        return "configs/smoke.yaml"
    platform_value = (platform_name or sys.platform).lower()
    primary = cuda_devices[0]
    primary_name = str(primary.get("name", "")).lower()
    primary_vram = float(primary.get("memory_gib", 0.0) or 0.0)
    if (
        (platform_value.startswith("win") or "windows" in platform_value)
        and "3060" in primary_name
        and primary_vram >= 10.0
    ):
        return "configs/rtx3060_12gb_windows.yaml"
    smallest_vram = min(float(device.get("memory_gib", 0.0) or 0.0) for device in cuda_devices)
    return "configs/deep.yaml" if smallest_vram >= 14.0 else "configs/workstation.yaml"

def doctor_report() -> dict[str, Any]:
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_gib": round(props.total_memory / 2**30, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    memory = _system_memory_bytes()
    report: dict[str, Any] = {
        "schema": "zws-nullity-doctor-v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(memory / 2**30, 2) if memory else None,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "tools": {
            "git": _command_version("git", ["--version"]),
            "cmake": _command_version("cmake", ["--version"]),
            "nvidia_smi": _command_version("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]),
            "eshkol_run": _command_version("eshkol-run", ["--version"]),
        },
    }
    report["recommended_config"] = recommend_config(
        cuda_devices,
        memory_gib=report["memory_gib"],
        platform_name=sys.platform,
    )
    warnings: list[str] = []
    if report["recommended_config"] == "configs/rtx3060_12gb_windows.yaml":
        if report["memory_gib"] is not None and float(report["memory_gib"]) < 48.0:
            warnings.append(
                "RTX 3060 12 GB profile expects approximately 64 GB system RAM; detected less than 48 GiB."
            )
        report["hardware_profile"] = "windows-rtx3060-12gb-64gb"
    elif cuda_devices:
        report["hardware_profile"] = "generic-cuda"
    else:
        report["hardware_profile"] = "cpu-smoke"
    report["warnings"] = warnings
    report["ready_for_gpu_search"] = bool(torch.cuda.is_available() and cuda_devices)
    return report


def write_doctor_report(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(doctor_report(), indent=2, sort_keys=True), encoding="utf-8")
    return destination
