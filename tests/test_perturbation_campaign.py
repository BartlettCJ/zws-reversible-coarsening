from __future__ import annotations

import hashlib
import json
from pathlib import Path

from zws_nullity_lab.atomic import atomic_write_json, atomic_write_text
from zws_nullity_lab.control import RunControl
from zws_nullity_lab.followup_campaign import rule_table_digest
from zws_nullity_lab.perturbation_campaign import (
    load_and_verify_manifest,
    run_campaign,
)
from zws_nullity_lab.rules import ReversibleRule


def _manifest(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='smoke'\n", encoding="utf-8")
    baseline_path = repo / "BASELINE.json"
    atomic_write_json(baseline_path, {"schema": "smoke-baseline"})
    rule = ReversibleRule.identity()
    rule_path = repo / "identity.json"
    rule.save(rule_path)
    branches = []
    seeds = [1001, 1002]
    for kind_index, kind in enumerate(("q_balanced_creation", "qm_state_swaps")):
        for seed_index, initial_seed in enumerate(seeds):
            run_id = f"response__identity__{kind}__seed{seed_index:02d}"
            branches.append(
                {
                    "run_id": run_id,
                    "condition_id": "identity",
                    "seed_index": seed_index,
                    "initial_seed": initial_seed,
                    "baseline_run_id": f"baseline__identity__seed{seed_index:02d}",
                    "intervention_tick": 8,
                    "intervention_kind": kind,
                    "selection_seed": 3000 + kind_index * 10 + seed_index,
                    "evaluation_ticks": [8, 16],
                    "shuffle_base_seed": 4000 + kind_index * 10 + seed_index,
                }
            )
    manifest = {
        "schema": "zwslab-perturbation-response-manifest-v1",
        "baseline_manifest_path": "BASELINE.json",
        "baseline_manifest_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "conditions": [
            {
                "condition_id": "identity",
                "conservation": "Q_and_M",
                "rule_file": "identity.json",
                "rule_table_sha256_uint8": rule_table_digest(rule),
            }
        ],
        "initial_seeds": seeds,
        "configuration": {
            "world": [64, 64],
            "phases": 16,
            "sample_interval_phases": 8,
            "checkpoint_interval_phases": 8,
            "initial_condition": {
                "construction": "balanced_random",
                "positive_fraction": 0.12,
                "bound_fraction": 0.08,
                "localized": True,
                "localized_fraction": 0.45,
            },
            "coarse_blocks": [8, 16, 32, 64],
            "recurrence_lags_phases": [1, 2, 4, 8],
            "intervention_ticks": [8],
            "intervention": {
                "patch_origin_row_column": [0, 0],
                "patch_size": 64,
                "changed_sites": 16,
                "kinds": {
                    "q_balanced_creation": {
                        "required_exact_invariants": ["Q"],
                        "exact_mass_delta": 16,
                    },
                    "qm_state_swaps": {
                        "required_exact_invariants": ["Q", "M", "all_state_counts"],
                        "exact_mass_delta": 0,
                    },
                },
            },
        },
        "planned_branches": branches,
        "planned_branch_count": len(branches),
    }
    path = repo / "PERTURBATION.json"
    atomic_write_json(path, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return path


def _refreeze(path: Path, manifest: dict[str, object]) -> None:
    atomic_write_json(path, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")


def test_perturbation_manifest_verifies_frozen_rules_and_branches(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest, digest, repo = load_and_verify_manifest(path)
    assert manifest["planned_branch_count"] == 4
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert repo == path.parent


def test_perturbation_campaign_pauses_before_allocation_then_resumes(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    output = tmp_path / "output"
    RunControl(output).request("pause")
    paused = run_campaign(
        path,
        output,
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert paused["status"] == "paused"
    assert not (output / "groups" / "identity__t00008" / "checkpoint.pt").exists()

    RunControl(output).request("resume")
    complete = run_campaign(
        path,
        output,
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert complete["status"] == "complete"
    assert complete["planned_branch_count"] == 4
    trajectories = sorted((output / "trajectories").glob("*.json"))
    assert len(trajectories) == 4
    records = [json.loads(path.read_text(encoding="utf-8")) for path in trajectories]
    assert all(record["perturbation_audit"]["observed_changed_sites"] == 16 for record in records)
    assert all(record["invariant_violation_counts"] == {"M": 0, "Q": 0} for record in records)
    assert all([sample["tick"] for sample in record["samples"]] == [8, 16] for record in records)


def test_perturbation_manifest_rejects_changed_bytes(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        load_and_verify_manifest(path)
    except RuntimeError as exc:
        assert "manifest hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("changed perturbation manifest was accepted")


def test_perturbation_manifest_requires_complete_balanced_product(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["planned_branches"] = manifest["planned_branches"][:-1]
    manifest["planned_branch_count"] -= 1
    _refreeze(path, manifest)
    try:
        load_and_verify_manifest(path)
    except ValueError as exc:
        assert "complete balanced branch product" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("incomplete perturbation branch product was accepted")


def test_perturbation_manifest_requires_saved_sample_evaluation_ticks(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["planned_branches"][0]["evaluation_ticks"] = [8, 15, 16]
    _refreeze(path, manifest)
    try:
        load_and_verify_manifest(path)
    except ValueError as exc:
        assert "invalid perturbation evaluation ticks" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("off-boundary perturbation evaluation tick was accepted")
