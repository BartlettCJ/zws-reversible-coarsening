from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

import psutil

from .config import ResourceConfig


@dataclass(frozen=True)
class PressureSample:
    utc: str
    cpu_percent: float
    available_ram_gib: float
    gpu_temperature_c: float | None
    external_gpu_memory_mib: float | None
    reasons: tuple[str, ...]


def _nvidia_rows() -> list[tuple[int, float]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, float]] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return rows


def _gpu_temperature() -> float | None:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def sample_pressure(policy: ResourceConfig) -> PressureSample:
    cpu = float(psutil.cpu_percent(interval=0.2))
    ram = float(psutil.virtual_memory().available / (1024**3))
    temperature = _gpu_temperature()
    rows = _nvidia_rows()
    external = float(sum(mem for pid, mem in rows if pid != os.getpid())) if rows else None
    reasons: list[str] = []
    if policy.enabled:
        if cpu >= policy.max_cpu_percent:
            reasons.append(f"cpu={cpu:.1f}%")
        if ram <= policy.min_available_ram_gib:
            reasons.append(f"available_ram={ram:.2f}GiB")
        if temperature is not None and temperature >= policy.max_gpu_temperature_c:
            reasons.append(f"gpu_temperature={temperature:.1f}C")
        if external is not None and external >= policy.max_external_gpu_memory_mib:
            reasons.append(f"external_gpu_memory={external:.0f}MiB")
    return PressureSample(
        utc=datetime.now(timezone.utc).isoformat(),
        cpu_percent=cpu,
        available_ram_gib=ram,
        gpu_temperature_c=temperature,
        external_gpu_memory_mib=external,
        reasons=tuple(reasons),
    )


def append_resource_log(path: str | Path, sample: PressureSample) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "utc",
                    "cpu_percent",
                    "available_ram_gib",
                    "gpu_temperature_c",
                    "external_gpu_memory_mib",
                    "reasons",
                ]
            )
        writer.writerow(
            [
                sample.utc,
                sample.cpu_percent,
                sample.available_ram_gib,
                sample.gpu_temperature_c,
                sample.external_gpu_memory_mib,
                ";".join(sample.reasons),
            ]
        )


def wait_for_quiet(policy: ResourceConfig, *, log_path: str | Path, stop_file: Path | None = None) -> None:
    if not policy.enabled:
        return
    while True:
        if stop_file is not None and stop_file.exists():
            return
        sample = sample_pressure(policy)
        append_resource_log(log_path, sample)
        if not sample.reasons:
            return
        time.sleep(policy.sleep_seconds)
