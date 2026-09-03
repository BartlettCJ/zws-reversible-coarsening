from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zws_reversible_null_phase.atomic import atomic_write_json, sha256_file
from zws_reversible_null_phase import history_media
from zws_reversible_null_phase.history_erasure import state_sha256
from zws_reversible_null_phase.history_media import (
    choose_exact_record,
    exact_state_metadata,
    index_task_states,
    load_exact_npz,
    verify_sidecar,
)


def _write_state(path: Path, state: np.ndarray, *, phase: int, elapsed: int) -> None:
    np.savez_compressed(
        path,
        state=np.asarray(state, dtype=np.uint8),
        phase=np.asarray([phase], dtype=np.int64),
        elapsed=np.asarray([elapsed], dtype=np.int64),
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def test_exact_state_metadata_and_selection(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_state(first, np.asarray([[0, 1], [2, 3]], dtype=np.uint8), phase=10, elapsed=0)
    _write_state(second, np.asarray([[1, 1], [2, 2]], dtype=np.uint8), phase=18, elapsed=8)
    records = [exact_state_metadata(first), exact_state_metadata(second)]
    assert records[0]["histogram"] == [1, 1, 1, 1]
    assert records[0]["Q"] == 0
    assert records[0]["M"] == 3
    assert choose_exact_record(records, elapsed=8)["path"] == str(second.resolve())
    assert choose_exact_record(records, elapsed=7)["path"] == str(first.resolve())


def test_exact_state_sidecar_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    _write_state(path, np.zeros((4, 4), dtype=np.uint8), phase=0, elapsed=0)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_sidecar(path)


def test_invalid_state_value_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "state.npz"
    _write_state(path, np.asarray([[4]], dtype=np.uint8), phase=0, elapsed=0)
    with pytest.raises(RuntimeError, match="invalid exact state"):
        load_exact_npz(path)


def test_state_index_ignores_verified_nonstate_npz_archives(tmp_path: Path) -> None:
    state_dir = tmp_path / "task" / "states"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "event_phase_000008.npz"
    _write_state(
        state_path,
        np.asarray([[0, 1], [2, 3]], dtype=np.uint8),
        phase=8,
        elapsed=8,
    )
    structure_path = state_dir / "event_phase_000008_structure_factor.npz"
    np.savez_compressed(structure_path, radial=np.asarray([1.0, 0.5]))
    structure_path.with_suffix(structure_path.suffix + ".sha256").write_text(
        f"{sha256_file(structure_path)}  {structure_path.name}\n", encoding="utf-8"
    )

    records = index_task_states(tmp_path / "task")

    assert len(records) == 1
    assert Path(records[0]["path"]).name == state_path.name


def test_media_replay_batches_sources_and_verifies_each_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(
        rule_path=tmp_path / "rule.json",
        rule_sha256_uint8=history_media.RULE_SHA256,
        device="cpu",
        fixed_elapsed_phases=[0, 2],
        source_horizon=2,
    )

    class FakeRule:
        table = torch.zeros(256, dtype=torch.uint8)

        def assert_valid(self) -> None:
            return None

        def sha256(self) -> str:
            return history_media.RULE_SHA256

    monkeypatch.setattr(history_media.HistoryConfig, "load", lambda _: config)
    monkeypatch.setattr(history_media.ReversibleRule, "load", lambda _: FakeRule())
    calls: list[int] = []

    def fake_step(states: torch.Tensor, table: torch.Tensor, elapsed: int) -> torch.Tensor:
        calls.append(states.shape[0])
        assert table.shape[0] == states.shape[0]
        return (states + 1) % 4

    monkeypatch.setattr(history_media, "step_batched", fake_step)
    for index, source_id in enumerate(("s1", "s2")):
        initial = np.full((2, 2), index, dtype=np.uint8)
        final = (initial + 2) % 4
        state_path = tmp_path / f"{source_id}_branch.npz"
        _write_state(state_path, initial, phase=0, elapsed=-1)
        source_dir = tmp_path / "sources" / source_id
        source_dir.mkdir(parents=True)
        atomic_write_json(
            source_dir / "source_result.json",
            {
                "branch_points": {
                    "B": {
                        "eligible": True,
                        "phase": 0,
                        "path": str(state_path),
                        "state_sha256": state_sha256(initial),
                    }
                },
                "series": [
                    {"phase": 0, "state_sha256": state_sha256(initial)},
                    {"phase": 2, "state_sha256": state_sha256(final)},
                ],
            },
        )
    output = history_media.reconstruct_untouched_media_states(
        tmp_path / "config.yaml",
        tmp_path,
        ["s1", "s2"],
        elapsed_targets=[0, 2],
    )
    assert calls == [2, 2]
    assert set(output) == {"s1", "s2"}
    assert all([record["elapsed"] for record in records] == [0, 2] for records in output.values())
