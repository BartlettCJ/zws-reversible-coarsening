from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zws_nullity_lab import homogeneous_campaign
from zws_nullity_lab.homogeneous_campaign import load_and_verify_manifest, run_campaign
from zws_nullity_lab.homogeneous_ic import (
    ARM_GRADIENT,
    ARM_HOMOGENEOUS,
    ARM_LOCALIZED,
    MATCHED_ARMS,
    construct_initial_state,
    deterministic_seed,
    state_inventory,
    tensor_digest,
)
from zws_nullity_lab.rules import ReversibleRule


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sidecar(path: Path) -> str:
    digest = _sha(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _fixture_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    experiment = root / "experiments" / "fixture"
    experiment.mkdir(parents=True)
    preregistration = experiment / "PREREGISTRATION.md"
    preregistration.write_text("frozen fixture\n", encoding="utf-8")
    rule_path = experiment / "winner.json"
    rule = ReversibleRule.identity()
    rule.save(rule_path)
    rule_digest = hashlib.sha256(bytes(int(value) for value in rule.table.tolist())).hexdigest()

    stage_id = "stage1_0064"
    rows = []
    for seed_index, inventory_seed in enumerate((101, 202)):
        for arm_id in MATCHED_ARMS:
            arrangement_seed = deterministic_seed(f"fixture:{arm_id}:{seed_index}")
            state = construct_initial_state(
                arm_id,
                height=64,
                width=64,
                inventory_seed=inventory_seed,
                arrangement_seed=arrangement_seed,
                localized_fraction=0.5,
            )
            inventory = state_inventory(state)
            rows.append(
                {
                    "run_id": f"{stage_id}__{arm_id}__seed{seed_index:02d}",
                    "stage_id": stage_id,
                    "arm_id": arm_id,
                    "seed_index": seed_index,
                    "inventory_seed": inventory_seed,
                    "arrangement_seed": arrangement_seed,
                    "diagnostic_seed": deterministic_seed(
                        f"fixture:diagnostic:{arm_id}:{seed_index}"
                    ),
                    "initial_state_sha256_uint8": tensor_digest(state),
                    **inventory,
                }
            )
    stage = {
        "stage_id": stage_id,
        "conditional": False,
        "world": [64, 64],
        "phases": 16,
        "sample_interval_phases": 8,
        "checkpoint_interval_phases": 8,
        "coarse_blocks": [16, 32, 64],
        "recurrence_lags_phases": [1, 4, 8],
        "early_snapshot_phase": 8,
        "initial_condition": {
            "positive_fraction": 0.12,
            "bound_fraction": 0.08,
            "localized_fraction": 0.5,
            "weak_gradient_modulation_amplitude": 0.25,
        },
        "arm_ids": list(MATCHED_ARMS),
        "inventory_seeds": [101, 202],
        "planned_runs": rows,
        "planned_run_count": len(rows),
    }
    manifest = {
        "schema": "zwslab-homogeneous-ic-manifest-v1",
        "preregistration_path": preregistration.relative_to(root).as_posix(),
        "preregistration_sha256": _sha(preregistration),
        "winner_rule_file": rule_path.relative_to(root).as_posix(),
        "winner_rule_table_sha256_uint8": rule_digest,
        "stages": {stage_id: stage},
    }
    manifest_path = experiment / "HOMOGENEOUS_IC_MANIFEST.json"
    _write_json(manifest_path, manifest)
    manifest_digest = _write_sidecar(manifest_path)
    audit = {
        "schema": "zwslab-homogeneous-ic-input-audit-v1",
        "manifest_sha256": manifest_digest,
        "all_matched_inventories_exact": True,
        "all_total_charge_zero": True,
    }
    audit_path = experiment / "INITIAL_STATE_AUDIT.json"
    _write_json(audit_path, audit)
    _write_sidecar(audit_path)
    return manifest_path


def test_manifest_verification_rejects_changed_preregistration(tmp_path: Path) -> None:
    manifest_path = _fixture_manifest(tmp_path)
    manifest, digest, root = load_and_verify_manifest(manifest_path)
    assert manifest["schema"] == "zwslab-homogeneous-ic-manifest-v1"
    assert len(digest) == 64
    (root / manifest["preregistration_path"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preregistration changed"):
        load_and_verify_manifest(manifest_path)


def test_campaign_resumes_checksummed_arm_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _fixture_manifest(tmp_path)
    output = tmp_path / "results"
    calls = 0

    def pause_after_first_boundary(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "runnable" if calls == 1 else "paused"

    monkeypatch.setattr(homogeneous_campaign, "_resource_gate", pause_after_first_boundary)
    paused = run_campaign(
        manifest_path,
        output,
        stage_id="stage1_0064",
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert paused["status"] == "paused"
    checkpoint = output / "arms" / ARM_LOCALIZED / "checkpoint.pt"
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".pt.sha256").is_file()

    monkeypatch.setattr(homogeneous_campaign, "_resource_gate", lambda *a, **k: "runnable")
    completed = run_campaign(
        manifest_path,
        output,
        stage_id="stage1_0064",
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert completed["status"] == "complete"
    assert completed["planned_run_count"] == 6
    for arm_id in (ARM_LOCALIZED, ARM_HOMOGENEOUS, ARM_GRADIENT):
        assert (output / "arms" / arm_id / "completion.json").is_file()
    trajectories = list((output / "trajectories").glob("*.json"))
    assert len(trajectories) == 6
    value = json.loads(trajectories[0].read_text(encoding="utf-8"))
    assert value["samples"][-1]["tick"] == 16
    assert "low_wavenumber" in value["samples"][-1]
