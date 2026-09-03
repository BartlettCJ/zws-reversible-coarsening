from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import threading
from typing import Callable

from .atomic import atomic_write_json


class WorkloadPaused(RuntimeError):
    """Raised after a safe checkpoint when the worker should exit resumably."""


class WorkloadHibernate(RuntimeError):
    """Raised after a safe checkpoint so a supervisor can release the worker."""


class WorkloadStop(RuntimeError):
    """Raised after a safe checkpoint for an intentional terminal stop."""


@dataclass(frozen=True)
class ControlFlags:
    pause: bool
    hibernate: bool
    stop: bool


class RunControl:
    """File-based controls that work from another shell session."""

    def __init__(self, run_directory: str | Path) -> None:
        self.root = Path(run_directory).resolve()
        self.directory = self.root / "control"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pause_file = self.directory / "PAUSE"
        self.hibernate_file = self.directory / "HIBERNATE"
        self.stop_file = self.directory / "STOP"
        self.resume_file = self.directory / "RESUME"

    def read(self) -> ControlFlags:
        if self.resume_file.exists():
            self.pause_file.unlink(missing_ok=True)
            self.hibernate_file.unlink(missing_ok=True)
            self.resume_file.unlink(missing_ok=True)
        return ControlFlags(
            pause=self.pause_file.exists(),
            hibernate=self.hibernate_file.exists(),
            stop=self.stop_file.exists(),
        )

    def request(self, action: str) -> Path:
        normalized = action.strip().lower()
        now = datetime.now(timezone.utc).isoformat()
        if normalized == "pause":
            self.resume_file.unlink(missing_ok=True)
            return atomic_write_json(self.pause_file, {"requested_utc": now, "pid": os.getpid()})
        if normalized == "hibernate":
            self.resume_file.unlink(missing_ok=True)
            return atomic_write_json(
                self.hibernate_file, {"requested_utc": now, "pid": os.getpid()}
            )
        if normalized == "stop":
            return atomic_write_json(self.stop_file, {"requested_utc": now, "pid": os.getpid()})
        if normalized == "resume":
            self.pause_file.unlink(missing_ok=True)
            self.hibernate_file.unlink(missing_ok=True)
            return atomic_write_json(self.resume_file, {"requested_utc": now, "pid": os.getpid()})
        if normalized == "clear-stop":
            self.stop_file.unlink(missing_ok=True)
            return self.stop_file
        raise ValueError(f"unsupported control action: {action}")


class SignalLatch:
    """Convert console signals into a checkpoint-at-next-safe-boundary request."""

    def __init__(self, on_signal: Callable[[int], None] | None = None) -> None:
        self.requested = threading.Event()
        self.signal_number: int | None = None
        self._previous: dict[int, object] = {}
        self._on_signal = on_signal

    def _handle(self, signum: int, _frame: object) -> None:
        self.signal_number = signum
        self.requested.set()
        if self._on_signal is not None:
            self._on_signal(signum)

    def install(self) -> "SignalLatch":
        candidates = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            candidates.append(signal.SIGBREAK)
        for signum in candidates:
            try:
                self._previous[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (OSError, ValueError):
                continue
        return self

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except (OSError, ValueError):
                continue
        self._previous.clear()

    def __enter__(self) -> "SignalLatch":
        return self.install()

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.restore()
