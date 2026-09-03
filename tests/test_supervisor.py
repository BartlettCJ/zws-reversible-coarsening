from __future__ import annotations

import json
import os
from pathlib import Path

import psutil

from zws_nullity_lab.cli import _active_supervisor_record


def test_active_supervisor_record_recognizes_current_process(tmp_path: Path) -> None:
    process = psutil.Process(os.getpid())
    value = {
        "pid": os.getpid(),
        "process_create_time": process.create_time(),
        "status": "worker_running",
    }
    (tmp_path / "supervisor.json").write_text(json.dumps(value), encoding="utf-8")
    assert _active_supervisor_record(tmp_path) == value


def test_active_supervisor_record_ignores_terminal_or_stale_record(tmp_path: Path) -> None:
    value = {
        "pid": os.getpid(),
        "process_create_time": 1.0,
        "status": "worker_running",
    }
    path = tmp_path / "supervisor.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _active_supervisor_record(tmp_path) is None
    value["process_create_time"] = psutil.Process(os.getpid()).create_time()
    value["status"] = "complete"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _active_supervisor_record(tmp_path) is None
