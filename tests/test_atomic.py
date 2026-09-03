from __future__ import annotations

from pathlib import Path

import torch

import zws_nullity_lab.atomic as atomic


def test_atomic_write_retries_transient_windows_style_permission_error(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "state.json"
    real_replace = atomic.os.replace
    calls = {"count": 0}

    def flaky_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("simulated transient sharing violation")
        return real_replace(source, target)

    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    atomic.atomic_write_json(destination, {"safe": True})
    assert destination.read_text(encoding="utf-8").strip().endswith("}")
    assert calls["count"] == 2


def test_atomic_torch_save_flushes_and_round_trips_on_windows(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    payload = {"generation": 3, "population": torch.arange(8)}

    atomic.atomic_torch_save(destination, payload)

    restored = torch.load(destination, map_location="cpu", weights_only=False)
    assert restored["generation"] == 3
    assert torch.equal(restored["population"], payload["population"])
