from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil

from .atomic import atomic_write_json


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    memory_mib: float | None


@dataclass(frozen=True)
class ResourceSnapshot:
    sampled_utc: str
    cpu_percent: float
    available_ram_gib: float
    gpu_index: int | None
    gpu_name: str | None
    gpu_utilization_percent: float | None
    gpu_memory_used_mib: float | None
    gpu_memory_total_mib: float | None
    gpu_temperature_c: float | None
    external_gpu_processes: tuple[GpuProcess, ...]
    external_gpu_memory_mib: float | None
    user_idle_seconds: float | None
    nvidia_smi_available: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["external_gpu_processes"] = [asdict(item) for item in self.external_gpu_processes]
        return value


@dataclass(frozen=True)
class ResourceDecision:
    runnable: bool
    reasons: tuple[str, ...]
    snapshot: ResourceSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "runnable": self.runnable,
            "reasons": list(self.reasons),
            "snapshot": self.snapshot.to_dict(),
        }


def _float_or_none(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _run_nvidia_smi(arguments: list[str]) -> list[str] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _windows_user_idle_seconds() -> float | None:
    if sys.platform != "win32":
        return None
    try:
        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        elapsed_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
        return max(0.0, float(elapsed_ms) / 1000.0)
    except Exception:
        return None


def set_background_priority(*, leave_logical_cores: int = 0) -> dict[str, Any]:
    """Best-effort low-impact process policy; failure is reported, never fatal."""
    report: dict[str, Any] = {"priority_changed": False, "affinity_changed": False}
    process = psutil.Process()
    try:
        if sys.platform == "win32":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(max(5, int(process.nice())))
        report["priority_changed"] = True
    except (psutil.Error, OSError, PermissionError) as exc:
        report["priority_error"] = f"{type(exc).__name__}: {exc}"
    try:
        affinity = process.cpu_affinity()
        keep = max(1, len(affinity) - max(0, int(leave_logical_cores)))
        if keep < len(affinity):
            process.cpu_affinity(affinity[:keep])
            report["affinity_changed"] = True
            report["affinity"] = affinity[:keep]
        else:
            report["affinity"] = affinity
    except (AttributeError, psutil.Error, OSError, PermissionError) as exc:
        report["affinity_error"] = f"{type(exc).__name__}: {exc}"
    return report


class ResourceGovernor:
    """Cooperative detector for external CPU/GPU/RAM pressure.

    It is deliberately checked only at safe chunk boundaries. The worker can
    then checkpoint, release CUDA caches, sleep, or exit for supervisor-managed
    hibernation without corrupting a generation.
    """

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        run_directory: str | Path | None = None,
        own_pid: int | None = None,
    ) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.gpu_index = int(self.config.get("gpu_index", 0))
        self.own_pid = int(own_pid or os.getpid())
        self.run_directory = Path(run_directory).resolve() if run_directory else None
        self.log_path = self.run_directory / "resource_log.csv" if self.run_directory else None
        self.state_path = self.run_directory / "resource_state.json" if self.run_directory else None
        self._busy_since: float | None = None
        self._busy_samples = 0
        self._idle_samples = 0

    @property
    def mode(self) -> str:
        return str(self.config.get("mode", "sleep")).lower()

    @property
    def poll_seconds(self) -> float:
        return max(0.5, float(self.config.get("poll_seconds", 5.0)))

    @property
    def hibernate_after_seconds(self) -> float:
        return max(0.0, float(self.config.get("hibernate_after_seconds", 900.0)))

    def _excluded_pids(self) -> set[int]:
        result = {self.own_pid}
        try:
            process = psutil.Process(self.own_pid)
            result.update(child.pid for child in process.children(recursive=True))
        except psutil.Error:
            pass
        return result

    def sample(self, *, cpu_interval: float | None = None) -> ResourceSnapshot:
        interval = float(
            self.config.get("sample_interval_seconds", 0.75)
            if cpu_interval is None
            else cpu_interval
        )
        cpu = float(psutil.cpu_percent(interval=max(0.0, interval)))
        available_ram = float(psutil.virtual_memory().available / 2**30)
        idle = _windows_user_idle_seconds()

        gpu_lines = _run_nvidia_smi(
            [
                f"--id={self.gpu_index}",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        gpu_index: int | None = None
        gpu_name: str | None = None
        gpu_util: float | None = None
        gpu_used: float | None = None
        gpu_total: float | None = None
        gpu_temp: float | None = None
        if gpu_lines:
            fields = [field.strip() for field in gpu_lines[0].split(",")]
            if len(fields) >= 6:
                try:
                    gpu_index = int(fields[0])
                except ValueError:
                    gpu_index = self.gpu_index
                gpu_name = fields[1]
                gpu_util = _float_or_none(fields[2])
                gpu_used = _float_or_none(fields[3])
                gpu_total = _float_or_none(fields[4])
                gpu_temp = _float_or_none(fields[5])

        process_lines = _run_nvidia_smi(
            [
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        excluded = self._excluded_pids()
        external: list[GpuProcess] = []
        own_reported_memory = 0.0
        if process_lines is not None:
            for line in process_lines:
                fields = [field.strip() for field in line.split(",", 2)]
                if len(fields) < 2:
                    continue
                try:
                    pid = int(fields[0])
                except ValueError:
                    continue
                memory = _float_or_none(fields[2]) if len(fields) >= 3 else None
                if pid in excluded:
                    if memory is not None:
                        own_reported_memory += memory
                    continue
                external.append(GpuProcess(pid=pid, name=fields[1], memory_mib=memory))
        reported_external = sum(item.memory_mib or 0.0 for item in external)
        external_memory: float | None
        if gpu_used is None:
            external_memory = reported_external if process_lines is not None else None
        else:
            driver_allowance = float(self.config.get("gpu_driver_allowance_mib", 768.0))
            residual = max(0.0, gpu_used - own_reported_memory - driver_allowance)
            external_memory = max(reported_external, residual)

        return ResourceSnapshot(
            sampled_utc=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu,
            available_ram_gib=available_ram,
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mib=gpu_used,
            gpu_memory_total_mib=gpu_total,
            gpu_temperature_c=gpu_temp,
            external_gpu_processes=tuple(external),
            external_gpu_memory_mib=external_memory,
            user_idle_seconds=idle,
            nvidia_smi_available=gpu_lines is not None,
        )

    def decide(self, snapshot: ResourceSnapshot, *, resuming: bool = False) -> ResourceDecision:
        if not self.enabled:
            return ResourceDecision(True, (), snapshot)
        reasons: list[str] = []
        cpu_threshold = float(
            self.config.get("cpu_resume_percent", 55.0)
            if resuming
            else self.config.get("cpu_busy_percent", 78.0)
        )
        if snapshot.cpu_percent >= cpu_threshold:
            reasons.append(f"system CPU {snapshot.cpu_percent:.1f}% >= {cpu_threshold:.1f}%")
        min_ram = float(self.config.get("min_available_ram_gib", 8.0))
        if snapshot.available_ram_gib < min_ram:
            reasons.append(
                f"available RAM {snapshot.available_ram_gib:.2f} GiB < {min_ram:.2f} GiB"
            )
        if bool(self.config.get("yield_for_external_gpu_processes", True)) and snapshot.external_gpu_processes:
            names = ", ".join(f"{item.name}({item.pid})" for item in snapshot.external_gpu_processes[:4])
            reasons.append(f"external CUDA process detected: {names}")
        external_limit = float(self.config.get("external_gpu_memory_mib", 1536.0))
        if (
            snapshot.external_gpu_memory_mib is not None
            and snapshot.external_gpu_memory_mib >= external_limit
        ):
            reasons.append(
                f"external GPU memory {snapshot.external_gpu_memory_mib:.0f} MiB >= {external_limit:.0f} MiB"
            )
        util_threshold = float(
            self.config.get("gpu_resume_utilization_percent", 20.0)
            if resuming
            else self.config.get("gpu_busy_utilization_percent", 40.0)
        )
        if (
            bool(self.config.get("yield_for_gpu_utilization", True))
            and snapshot.gpu_utilization_percent is not None
            and snapshot.gpu_utilization_percent >= util_threshold
        ):
            reasons.append(
                f"GPU utilization {snapshot.gpu_utilization_percent:.0f}% >= {util_threshold:.0f}%"
            )
        temp_threshold = float(
            self.config.get("gpu_resume_temperature_c", 76.0)
            if resuming
            else self.config.get("gpu_busy_temperature_c", 83.0)
        )
        if (
            snapshot.gpu_temperature_c is not None
            and snapshot.gpu_temperature_c >= temp_threshold
        ):
            reasons.append(
                f"GPU temperature {snapshot.gpu_temperature_c:.0f} C >= {temp_threshold:.0f} C"
            )
        if bool(self.config.get("yield_when_user_active", False)):
            required_idle = float(self.config.get("required_user_idle_seconds", 90.0))
            if snapshot.user_idle_seconds is not None and snapshot.user_idle_seconds < required_idle:
                reasons.append(
                    f"Windows user idle {snapshot.user_idle_seconds:.0f}s < {required_idle:.0f}s"
                )
        return ResourceDecision(not reasons, tuple(reasons), snapshot)

    def observe(self) -> ResourceDecision:
        decision = self.decide(self.sample(), resuming=self._busy_since is not None)
        now = time.monotonic()
        busy_confirm = max(1, int(self.config.get("busy_confirm_samples", 2)))
        idle_confirm = max(1, int(self.config.get("idle_confirm_samples", 3)))
        if decision.runnable:
            self._idle_samples += 1
            self._busy_samples = 0
            if self._busy_since is not None and self._idle_samples < idle_confirm:
                decision = ResourceDecision(
                    False,
                    (f"waiting for {idle_confirm} consecutive idle samples",),
                    decision.snapshot,
                )
            elif self._idle_samples >= idle_confirm:
                self._busy_since = None
        else:
            self._busy_samples += 1
            self._idle_samples = 0
            if self._busy_samples < busy_confirm:
                decision = ResourceDecision(
                    True,
                    (f"busy sample {self._busy_samples}/{busy_confirm} not yet confirmed",),
                    decision.snapshot,
                )
            elif self._busy_since is None:
                self._busy_since = now
        self._log(decision)
        return decision

    def busy_seconds(self) -> float:
        if self._busy_since is None:
            return 0.0
        return max(0.0, time.monotonic() - self._busy_since)

    def should_hibernate(self) -> bool:
        return self.mode == "hibernate" and self.busy_seconds() >= self.hibernate_after_seconds

    def _log(self, decision: ResourceDecision) -> None:
        if self.state_path is not None:
            atomic_write_json(
                self.state_path,
                {
                    **decision.to_dict(),
                    "busy_seconds": self.busy_seconds(),
                    "mode": self.mode,
                },
            )
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sampled_utc",
                    "runnable",
                    "reasons",
                    "cpu_percent",
                    "available_ram_gib",
                    "gpu_utilization_percent",
                    "gpu_memory_used_mib",
                    "external_gpu_memory_mib",
                    "gpu_temperature_c",
                    "external_gpu_processes",
                    "user_idle_seconds",
                ],
            )
            if new:
                writer.writeheader()
            snapshot = decision.snapshot
            writer.writerow(
                {
                    "sampled_utc": snapshot.sampled_utc,
                    "runnable": decision.runnable,
                    "reasons": " | ".join(decision.reasons),
                    "cpu_percent": snapshot.cpu_percent,
                    "available_ram_gib": snapshot.available_ram_gib,
                    "gpu_utilization_percent": snapshot.gpu_utilization_percent,
                    "gpu_memory_used_mib": snapshot.gpu_memory_used_mib,
                    "external_gpu_memory_mib": snapshot.external_gpu_memory_mib,
                    "gpu_temperature_c": snapshot.gpu_temperature_c,
                    "external_gpu_processes": "; ".join(
                        f"{item.name}:{item.pid}:{item.memory_mib}" for item in snapshot.external_gpu_processes
                    ),
                    "user_idle_seconds": snapshot.user_idle_seconds,
                }
            )
