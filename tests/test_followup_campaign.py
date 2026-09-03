from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from zws_nullity_lab.atomic import atomic_write_json, atomic_write_text
from zws_nullity_lab.control import RunControl, SignalLatch
from zws_nullity_lab.followup_campaign import (
    load_and_verify_manifest,
    rule_table_digest,
    run_campaign,
    run_condition_batch,
)
from zws_nullity_lab.resources import ResourceGovernor
from zws_nullity_lab.rules import ReversibleRule


def _smoke_manifest(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='smoke'\n", encoding="utf-8")
    rule = ReversibleRule.identity()
    rule_path = repo / "identity.json"
    rule.save(rule_path)
    manifest_path = repo / "SMOKE_MANIFEST.json"
    manifest = {
        "schema": "zwslab-double-conservation-screen-manifest-v1",
        "conditions": [
            {
                "condition_id": "identity",
                "conservation": "Q_and_M",
                "rule_file": "identity.json",
                "rule_table_sha256_uint8": rule_table_digest(rule),
            }
        ],
        "configuration": {
            "world": [64, 64],
            "phases": 8,
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
        },
        "planned_trajectories": [
            {
                "run_id": "smoke__identity__seed00",
                "condition_id": "identity",
                "seed_index": 0,
                "initial_seed": 1001,
                "shuffle_base_seed": 2001,
            },
            {
                "run_id": "smoke__identity__seed01",
                "condition_id": "identity",
                "seed_index": 1,
                "initial_seed": 1002,
                "shuffle_base_seed": 2002,
            },
        ],
        "planned_trajectory_count": 2,
    }
    atomic_write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    atomic_write_text(manifest_path.with_suffix(".sha256"), f"{digest}  {manifest_path.name}\n")
    return manifest_path


def test_followup_manifest_verifies_rules_and_sidecar(tmp_path: Path) -> None:
    manifest_path = _smoke_manifest(tmp_path)
    manifest, digest, repo = load_and_verify_manifest(manifest_path)
    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["planned_trajectory_count"] == 2
    assert repo == manifest_path.parent


def test_followup_campaign_obeys_preflight_pause_then_resumes(tmp_path: Path) -> None:
    manifest_path = _smoke_manifest(tmp_path)
    output = tmp_path / "output"
    RunControl(output).request("pause")
    paused = run_campaign(
        manifest_path,
        output,
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert paused["status"] == "paused"
    assert not (output / "conditions" / "identity" / "checkpoint.pt").exists()

    RunControl(output).request("resume")
    complete = run_campaign(
        manifest_path,
        output,
        device="cpu",
        resource_config={"enabled": False, "below_normal_priority": False},
    )
    assert complete["status"] == "complete"
    trajectory_path = output / "trajectories" / "smoke__identity__seed00.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["required_invariants"] == ["Q", "M"]
    assert trajectory["invariant_violation_counts"] == {"M": 0, "Q": 0}
    assert trajectory["samples"][-1]["mass_invariant_violation"] is False


def test_followup_condition_resumes_a_checksummed_midrun_checkpoint(tmp_path: Path) -> None:
    manifest_path = _smoke_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["phases"] = 16
    atomic_write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    atomic_write_text(manifest_path.with_suffix(".sha256"), f"{digest}  {manifest_path.name}\n")
    manifest, digest, repo = load_and_verify_manifest(manifest_path)
    output = tmp_path / "midrun"
    control = RunControl(output)

    class PauseOnFirstObservation:
        config = {"release_cuda_cache": False}
        poll_seconds = 0.0

        def observe(self) -> SimpleNamespace:
            control.request("pause")
            return SimpleNamespace(
                runnable=False,
                reasons=("test_pause",),
                snapshot=SimpleNamespace(to_dict=lambda: {}),
            )

    with SignalLatch() as signal_latch:
        paused = run_condition_batch(
            root=output,
            repo_root=repo,
            condition=manifest["conditions"][0],
            trajectory_rows=manifest["planned_trajectories"],
            config=manifest["configuration"],
            manifest_digest=digest,
            device=torch.device("cpu"),
            governor=PauseOnFirstObservation(),  # type: ignore[arg-type]
            control=control,
            signal_latch=signal_latch,
        )
    assert paused["status"] == "paused"
    assert paused["tick"] == 8
    assert (output / "conditions" / "identity" / "checkpoint.pt.sha256").is_file()

    control.request("resume")
    with SignalLatch() as signal_latch:
        complete = run_condition_batch(
            root=output,
            repo_root=repo,
            condition=manifest["conditions"][0],
            trajectory_rows=manifest["planned_trajectories"],
            config=manifest["configuration"],
            manifest_digest=digest,
            device=torch.device("cpu"),
            governor=ResourceGovernor({"enabled": False}, run_directory=output),
            control=control,
            signal_latch=signal_latch,
        )
    assert complete["status"] == "complete"
    trajectory = json.loads(
        (output / "trajectories" / "smoke__identity__seed00.json").read_text(encoding="utf-8")
    )
    assert [sample["tick"] for sample in trajectory["samples"]] == [0, 8, 16]


def test_followup_manifest_rejects_changed_content(tmp_path: Path) -> None:
    manifest_path = _smoke_manifest(tmp_path)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        load_and_verify_manifest(manifest_path)
    except RuntimeError as exc:
        assert "manifest hash mismatch" in str(exc)
    else:  # pragma: no cover - required assertion path
        raise AssertionError("changed manifest was accepted")
