from __future__ import annotations

from datetime import datetime, timezone
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import psutil
import scipy
import torch

from .rule import ReversibleRule


def _nvidia_smi() -> dict[str, Any] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "returncode": result.returncode}
    devices = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4:
            devices.append(
                {
                    "name": parts[0],
                    "memory_total_mib": float(parts[1]),
                    "driver_version": parts[2],
                    "temperature_c": float(parts[3]),
                }
            )
    return {"devices": devices}


def environment_report(rule_path: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "zws-reversible-null-phase-doctor-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gib": psutil.virtual_memory().total / (1024**3),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gib": torch.cuda.get_device_properties(index).total_memory / (1024**3),
            }
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "nvidia_smi": _nvidia_smi(),
        "eshkol_run": shutil.which("eshkol-run"),
    }
    if rule_path:
        rule = ReversibleRule.load(rule_path)
        audit = rule.audit()
        pairs = rule.reactive_pairs_relative_to_rotation()
        sibling = rule.closest_qm_preserving_sibling()
        payload["rule"] = {
            "path": rule_path,
            "name": rule.name,
            "audit": audit.__dict__,
            "reactive_pair_count": len(pairs),
            "qm_sibling_audit": sibling.audit().__dict__,
            "qm_sibling_identical_entries": sibling.metadata.get("entries_identical_to_source"),
        }
    return payload
