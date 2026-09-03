from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from zws_reversible_null_phase import eshkol
from zws_reversible_null_phase.eshkol import export_parity_case, generate_eshkol_parity_source
from zws_reversible_null_phase.initial import inventory_for_density, make_state
from zws_reversible_null_phase.reaction_network import reaction_catalog, summarize_flux
from zws_reversible_null_phase.rule import ReversibleRule


def test_reaction_catalog_preserves_q(winner: ReversibleRule) -> None:
    catalog = reaction_catalog(winner)
    assert len(catalog) == 256
    assert all(entry.delta_q == 0 for entry in catalog)
    assert sum(entry.reactive_pair_index is not None for entry in catalog) == 24
    hist = np.ones(256, dtype=np.int64)
    flux = summarize_flux(winner, hist)
    assert flux["total_blocks"] == 256
    assert flux["reactive_visits"] == 24


def test_eshkol_source_is_complete(tmp_path: Path, winner: ReversibleRule) -> None:
    initial = make_state(8, 8, inventory_for_density(8, 8, 0.28), seed=11)
    source = generate_eshkol_parity_source(winner, initial, steps=8)
    assert "AUDIT_OK" in source
    assert "step-state" in source
    assert "FINAL" in source
    assert "(define rule (vector " in source
    assert "(define rule #(" not in source
    path = tmp_path / "parity.esk"
    expected = export_parity_case(winner, path, size=8, density=0.28, seed=11, steps=8)
    assert path.exists()
    assert path.with_suffix(".expected.json").exists()
    assert len(expected["final"]) == 64


def test_eshkol_process_failure_is_checksummed(
    tmp_path: Path, winner: ReversibleRule, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "parity.esk"
    export_parity_case(winner, path, size=8, density=0.28, seed=11, steps=8)
    failed = subprocess.CompletedProcess(
        ["eshkol-run", "--run", str(path)], returncode=1, stdout="", stderr="missing SDK"
    )
    monkeypatch.setattr(eshkol.subprocess, "run", lambda *args, **kwargs: failed)

    with pytest.raises(RuntimeError, match="exited 1"):
        eshkol.run_eshkol_parity(path)

    failure_path = path.with_suffix(".parity-failure.json")
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["stderr"] == "missing SDK"
    assert payload["command"] == ["eshkol-run", "--run", str(path)]
    assert failure_path.with_suffix(failure_path.suffix + ".sha256").exists()


def test_eshkol_exact_success_removes_stale_failure(
    tmp_path: Path, winner: ReversibleRule, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "parity.esk"
    expected = export_parity_case(winner, path, size=8, density=0.28, seed=11, steps=8)
    failure_path = path.with_suffix(".parity-failure.json")
    failure_path.write_text("{}", encoding="utf-8")
    failure_path.with_suffix(failure_path.suffix + ".sha256").write_text("stale", encoding="utf-8")
    stdout = "AUDIT_OK\nQ 0\nM 18\nFINAL " + " ".join(str(x) for x in expected["final"]) + "\n"
    completed = subprocess.CompletedProcess(
        ["eshkol-run", "--run", str(path)], returncode=0, stdout=stdout, stderr=""
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return completed

    monkeypatch.setattr(eshkol.subprocess, "run", fake_run)
    payload = eshkol.run_eshkol_parity(path)

    assert calls == [["eshkol-run", "--run", str(path)]]
    assert payload["passed"] is True
    assert payload["mismatch_count"] == 0
    assert payload["actual_final_sha256"] == expected["final_sha256"]
    assert not failure_path.exists()
    assert not failure_path.with_suffix(failure_path.suffix + ".sha256").exists()
