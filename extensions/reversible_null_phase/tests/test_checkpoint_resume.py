from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from zws_reversible_null_phase.config import StudyConfig
from zws_reversible_null_phase.runner import ManagedPause, _atomic_torch_save, _load_verified_torch, run_study


def _tiny_config(
    path: Path,
    rule_path: Path,
    *,
    seeds: list[int] | None = None,
    batch_size: int = 1,
) -> StudyConfig:
    payload = {
        "schema": "zws-reversible-null-phase-study-v1",
        "name": "pause",
        "rule_path": str(rule_path),
        "device": "cpu",
        "criterion": {
            "min_peak_scale_ratio": 0.0,
            "min_final_scale_ratio": 0.0,
            "min_peak_largest_finite_empty_fraction": 0.0,
            "initial_scale_floor": 1.0,
        },
        "nucleation": {
            "schema": "zws-reversible-null-phase-nucleation-v1",
            "observation_every": 4,
            "persistence_followups": 2,
            "fixed_readout_phases": [4, 8, 16],
            "event_window_before": 8,
            "event_window_after": 8,
            "event_crop_fraction": 0.5,
        },
        "resource": {"enabled": False},
        "stages": {
            "onset_pause": {
                "sizes": [16],
                "densities": [0.28],
                "seeds": seeds or [777],
                "steps": 16,
                "sample_every": 2,
                "topology_every": 4,
                "checkpoint_every": 4,
                "batch_size": batch_size,
                "arrangements": ["homogeneous"],
                "rule_variants": ["winner"],
                "early_fit_start": 2,
                "early_fit_end": 12,
                "shells": [[1, 2], [2, 4]],
            }
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return StudyConfig.load(path)


def test_pause_resume_matches_uninterrupted(tmp_path: Path, extension_root: Path) -> None:
    config = _tiny_config(tmp_path / "pause.yaml", extension_root / "rules" / "winner_d1fe99.json")
    paused = tmp_path / "paused"
    paused.mkdir()
    (paused / "PAUSE.request").write_text("pause", encoding="utf-8")
    with pytest.raises(ManagedPause):
        run_study(config, paused, managed=False)
    assert list((paused / "checkpoints").glob("*.pt"))
    (paused / "PAUSE.request").unlink()
    run_study(config, paused)

    uninterrupted = tmp_path / "uninterrupted"
    run_study(config, uninterrupted)
    paused_result = next((paused / "tasks" / "onset_pause").glob("*.json"))
    full_result = next((uninterrupted / "tasks" / "onset_pause").glob("*.json"))
    p = json.loads(paused_result.read_text(encoding="utf-8"))
    f = json.loads(full_result.read_text(encoding="utf-8"))
    assert p["final"]["state_sha256"] == f["final"]["state_sha256"]
    assert p["topological_series"] == f["topological_series"]
    assert p["temporal_series"] == f["temporal_series"]
    assert p["time_to_event"] == f["time_to_event"]
    assert p["time_to_event"]["T_hit"] == 4
    assert p["time_to_event"]["T_persistent"] == 4
    paused_window = np.load(paused_result.parent / paused_result.stem / "event_window.npz")
    full_window = np.load(full_result.parent / full_result.stem / "event_window.npz")
    assert np.array_equal(paused_window["phases"], full_window["phases"])
    assert np.array_equal(paused_window["states"], full_window["states"])


def test_batch_size_changes_only_execution_efficiency(
    tmp_path: Path, extension_root: Path
) -> None:
    rule_path = extension_root / "rules" / "winner_d1fe99.json"
    serial_config = _tiny_config(
        tmp_path / "serial.yaml", rule_path, seeds=[777, 778], batch_size=1
    )
    batched_config = _tiny_config(
        tmp_path / "batched.yaml", rule_path, seeds=[777, 778], batch_size=2
    )
    serial = tmp_path / "serial"
    batched = tmp_path / "batched"
    run_study(serial_config, serial)
    run_study(batched_config, batched)

    def results_by_seed(root: Path) -> dict[int, dict[str, object]]:
        results: dict[int, dict[str, object]] = {}
        for path in (root / "tasks" / "onset_pause").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            results[int(payload["task"]["seed"])] = payload
        return results

    serial_results = results_by_seed(serial)
    batched_results = results_by_seed(batched)
    assert serial_results.keys() == batched_results.keys() == {777, 778}
    for seed in serial_results:
        serial_payload = serial_results[seed]
        batched_payload = batched_results[seed]
        assert serial_payload["final"] == batched_payload["final"]
        assert serial_payload["topological_series"] == batched_payload["topological_series"]
        assert serial_payload["temporal_series"] == batched_payload["temporal_series"]
        assert serial_payload["time_to_event"] == batched_payload["time_to_event"]


def test_corrupt_newest_checkpoint_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save(path, {"generation": 1, "tensor": torch.arange(4)})
    _atomic_torch_save(path, {"generation": 2, "tensor": torch.arange(5)})
    path.write_bytes(b"corrupt")
    loaded = _load_verified_torch(path)
    assert loaded["generation"] == 1


def test_checkpoint_rotation_retries_transient_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    _atomic_torch_save(path, {"generation": 1})
    _atomic_torch_save(path, {"generation": 2})
    real_replace = os.replace
    failures = 0

    def transiently_locked(source: str | Path, target: str | Path) -> None:
        nonlocal failures
        if Path(source).name == "checkpoint.prev1.pt" and failures < 2:
            failures += 1
            raise PermissionError(5, "Access is denied", str(source))
        real_replace(source, target)

    monkeypatch.setattr("zws_reversible_null_phase.runner.os.replace", transiently_locked)
    monkeypatch.setattr("zws_reversible_null_phase.runner.time.sleep", lambda _delay: None)
    _atomic_torch_save(path, {"generation": 3})

    assert failures == 2
    assert _load_verified_torch(path)["generation"] == 3
