from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from zws_reversible_null_phase.atomic import atomic_write_json, atomic_write_text, sha256_file
from zws_reversible_null_phase.history_erasure import (
    HistoryConfig,
    SourceCalibration,
    multiscale_summary,
    payload_sha256,
)
from zws_reversible_null_phase.history_runner import (
    ManagedPause,
    _audit_branch_checkpoint_replay,
    _branch_result_path,
    _load_reusable_realized_plan,
    _metric_row,
    _run_branch_batch,
    _save_exact_state,
    _source_result_path,
)
from zws_reversible_null_phase.initial import inventory_for_density, make_state


def small_config(tmp_path: Path, extension_root: Path) -> HistoryConfig:
    return HistoryConfig(
        name="v032_test",
        rule_path=extension_root / "rules" / "winner_d1fe99.json",
        rule_sha256_uint8="d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c",
        device="cpu",
        output_root=tmp_path,
        lattice_size=512,
        initial_densities=(0.20,),
        source_seeds=(6201,),
        source_horizon=16,
        branch_horizon=16,
        observation_every=8,
        topology_every=8,
        checkpoint_every=8,
        persistence_followups=2,
        post_persistent_capture=16,
        fixed_elapsed_phases=(0, 8, 16),
        event_relative_phases=(-8, 0, 8),
        block_sizes=(2, 4, 8, 16, 32, 64, 128, 256),
        mutual_information_offsets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        angular_bins=16,
        anisotropy_k_min=1,
        anisotropy_k_max=32,
        branch_a_occupancy_fraction=0.35,
        branch_b_lead_phases=32768,
        branch_b_minimum_hit_phase=49152,
        equivalence_margin_phases=8192,
        bootstrap_replicates=10,
        extra_intervention_subset_seeds=(6201,),
        resource={"enabled": False},
        source_path=tmp_path / "unused.yaml",
    )


def prepare_branch_case(
    output: Path,
    config: HistoryConfig,
    winner,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    inventory = inventory_for_density(512, 512, 0.20)
    source_state = make_state(512, 512, inventory, seed=6201, arrangement="homogeneous")
    altered_state = source_state.roll(shifts=(0, 2), dims=(0, 1))
    source_id = "source-test"
    task_id = "branch-test"
    source_dir = output / "sources" / source_id
    branch_dir = output / "branches" / task_id
    source_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)
    source_capture = _save_exact_state(source_dir / "source.npz", source_state, phase=0, elapsed=0)
    altered_capture = _save_exact_state(branch_dir / "altered_initial.npz", altered_state, phase=0, elapsed=0)
    metric = _metric_row(source_state, phase=0)
    calibration = SourceCalibration.from_initial_metric(metric)
    source_result = {
        "schema": "test-source",
        "task": {"task_id": source_id, "horizon_phases": 16, "density": 0.20, "seed": 6201},
        "calibration": calibration.to_dict(),
        "series": [metric],
        "time_to_event": {"T_hit": None, "T_persistent": None},
    }
    atomic_write_json(_source_result_path(output, source_id), source_result)
    summary, _ = multiscale_summary(altered_state)
    audit_path = branch_dir / "intervention_audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema": "test-audit",
            "generator": {},
            "altered_multiscale": summary,
        },
    )
    task = {
        "task_id": task_id,
        "source_task_id": source_id,
        "density": 0.20,
        "source_seed": 6201,
        "branch_point": "A",
        "branch_phase": 0,
        "intervention": "global_shuffle",
        "realization": 1,
        "intervention_seed": 1,
        "source_state_path": source_capture["path"],
        "source_state_sha256": source_capture["state_sha256"],
        "initial_state_path": altered_capture["path"],
        "initial_state_sha256": altered_capture["state_sha256"],
        "intervention_audit_path": str(audit_path),
        "intervention_audit_sha256": "unused",
        "untouched_reference": {
            "R_hit": None,
            "hit_observed": False,
            "R_persistent": None,
            "persistent_observed": False,
            "censor_elapsed": 16,
        },
    }
    realized = {"payload_sha256": "test-realized", "prepared_entries": [task]}
    return realized, task


def test_interrupted_resumed_branch_is_byte_identical(
    tmp_path: Path,
    extension_root: Path,
    winner,
) -> None:
    resumed_output = tmp_path / "resumed"
    reference_output = tmp_path / "reference"
    config = small_config(tmp_path, extension_root)
    resumed_plan, resumed_task = prepare_branch_case(resumed_output, config, winner)
    atomic_write_text(resumed_output / "PAUSE.request", "pause\n")
    with pytest.raises(ManagedPause):
        _run_branch_batch(
            config,
            resumed_plan,
            resumed_output,
            [resumed_task],
            winner,
            torch.device("cpu"),
            managed=False,
        )
    assert any((resumed_output / "checkpoints").glob("branch_*.pt"))
    (resumed_output / "PAUSE.request").unlink()
    _run_branch_batch(
        config,
        resumed_plan,
        resumed_output,
        [resumed_task],
        winner,
        torch.device("cpu"),
        managed=False,
    )

    reference_plan, reference_task = prepare_branch_case(reference_output, config, winner)
    _run_branch_batch(
        config,
        reference_plan,
        reference_output,
        [reference_task],
        winner,
        torch.device("cpu"),
        managed=False,
    )
    resumed = json.loads(_branch_result_path(resumed_output, "branch-test").read_text(encoding="utf-8"))
    reference = json.loads(_branch_result_path(reference_output, "branch-test").read_text(encoding="utf-8"))
    assert resumed["final_state_sha256"] == reference["final_state_sha256"]
    assert resumed["series"] == reference["series"]
    assert resumed["strong_domain_flags"] == reference["strong_domain_flags"]


def test_verified_realized_plan_is_reused_without_regeneration(
    tmp_path: Path,
    extension_root: Path,
    winner,
) -> None:
    output = tmp_path / "reusable-plan"
    config = small_config(tmp_path, extension_root)
    _, task = prepare_branch_case(output, config, winner)
    task["intervention_audit_sha256"] = sha256_file(task["intervention_audit_path"])
    frozen_plan = {"payload_sha256": "frozen-plan"}
    manifest = {
        "payload_sha256": "frozen-manifest",
        "entries": [{"task_id": task["task_id"]}],
    }
    realized = {
        "frozen_plan_payload_sha256": frozen_plan["payload_sha256"],
        "intervention_manifest_payload_sha256": manifest["payload_sha256"],
        "source_gate": {"passed": True},
        "prepared_entries": [task],
        "ineligible_entries": [],
        "prepared_count": 1,
        "ineligible_count": 0,
    }
    realized["payload_sha256"] = payload_sha256(realized)
    atomic_write_json(output / "REALIZED_BRANCH_PLAN.json", realized)

    reused = _load_reusable_realized_plan(output, frozen_plan, manifest)
    assert reused is not None
    assert reused["payload_sha256"] == realized["payload_sha256"]


def test_regenerated_plan_requires_exact_replay_audit_before_resume(
    tmp_path: Path,
    extension_root: Path,
    winner,
) -> None:
    output = tmp_path / "lineage-repair"
    config = small_config(tmp_path, extension_root)
    original_plan, task = prepare_branch_case(output, config, winner)
    atomic_write_text(output / "PAUSE.request", "pause\n")
    with pytest.raises(ManagedPause):
        _run_branch_batch(
            config,
            original_plan,
            output,
            [task],
            winner,
            torch.device("cpu"),
            managed=False,
        )
    (output / "PAUSE.request").unlink()

    regenerated_plan = {**original_plan, "payload_sha256": "regenerated-realized"}
    with pytest.raises(RuntimeError, match="exact checkpoint replay audit"):
        _run_branch_batch(
            config,
            regenerated_plan,
            output,
            [task],
            winner,
            torch.device("cpu"),
            managed=False,
        )
    audit = _audit_branch_checkpoint_replay(
        config,
        regenerated_plan,
        output,
        [task],
        winner,
        torch.device("cpu"),
    )
    assert audit["passed"] is True
    assert audit["mismatch_counts"] == [0]
    _run_branch_batch(
        config,
        regenerated_plan,
        output,
        [task],
        winner,
        torch.device("cpu"),
        managed=False,
    )
    assert _branch_result_path(output, "branch-test").is_file()
