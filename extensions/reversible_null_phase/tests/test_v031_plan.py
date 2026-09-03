from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from zws_reversible_null_phase.config import StudyConfig
from zws_reversible_null_phase.initial import inventory_for_density
from zws_reversible_null_phase.tasks import generate_tasks

# Scheme-1 configuration hash (package <= 0.3.1) recorded for the v0.3.0 balanced
# profile.  It included the rule file's resolved absolute path on the machine that ran
# the campaigns, so it is reproduced by substituting the recorded path, not by
# requiring that checkout location.
V030_LEGACY_CONFIG_SHA256 = "44e2a6155637587ef350d4646b1bb558bde4fb7df381c69a55c8f57ca77bbc21"


def _recorded_plans(extension_root: Path) -> list[Path]:
    repo = extension_root.parents[1]
    candidates = [
        repo / "experiments" / "reversible_null_phase_v030_20260827" / "balanced" / "PLAN.json",
        repo / "experiments" / "reversible_null_phase_v031_nucleation_20260827" / "FROZEN_PLAN.json",
    ]
    return [path for path in candidates if path.exists()]


def test_v030_config_hash_is_location_independent(extension_root: Path, tmp_path: Path) -> None:
    config = StudyConfig.load(extension_root / "configs" / "rtx3060_balanced.yaml")
    payload = config.canonical_payload()
    assert payload["hash_scheme"] == StudyConfig.HASH_SCHEME == 2
    assert payload["rule_path"] == "../rules/winner_d1fe99.json"
    # Scheme-2 pin for the v0.3.0 balanced profile; must hold at every checkout location.
    assert config.sha256() == "5a09c303ec91dd67ac8e0adbbe15b2ba85fabd8aedcca926e61f50f0991ea5a8"
    # The same two files at another location must hash identically under scheme 2
    # and differently under the legacy scheme.
    (tmp_path / "configs").mkdir()
    (tmp_path / "rules").mkdir()
    shutil.copy(extension_root / "configs" / "rtx3060_balanced.yaml", tmp_path / "configs")
    shutil.copy(extension_root / "rules" / "winner_d1fe99.json", tmp_path / "rules")
    relocated = StudyConfig.load(tmp_path / "configs" / "rtx3060_balanced.yaml")
    assert relocated.rule_path != config.rule_path
    assert relocated.sha256() == config.sha256()
    assert relocated.legacy_sha256() != config.legacy_sha256()


def test_v030_legacy_config_hash_is_reproduced(extension_root: Path) -> None:
    plans = _recorded_plans(extension_root)
    if not plans:
        pytest.skip("no recorded plan available to supply the recorded rule path")
    # Every recorded plan names the same rule file on the original machine.
    recorded_rule_path = json.loads(plans[-1].read_text(encoding="utf-8"))["config"]["rule_path"]
    config = StudyConfig.load(extension_root / "configs" / "rtx3060_balanced.yaml")
    recorded = dataclasses.replace(config, rule_path=recorded_rule_path)
    assert recorded.legacy_sha256() == V030_LEGACY_CONFIG_SHA256
    assert "hash_scheme" not in recorded.canonical_payload(StudyConfig.LEGACY_HASH_SCHEME)


def test_recorded_plans_verify_from_their_embedded_payload(extension_root: Path) -> None:
    plans = _recorded_plans(extension_root)
    # The public release ships the v0.3.1 frozen plan; the working repository has both.
    assert plans
    for plan_path in plans:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assert StudyConfig.payload_sha256(plan["config"]) == plan["config_sha256"], plan_path.name


def test_v031_frozen_task_matrix(extension_root: Path) -> None:
    config = StudyConfig.load(extension_root / "configs" / "v031_nucleation.yaml")
    tasks = generate_tasks(config)
    assert len(tasks) == 156
    assert len({task.id for task in tasks}) == 156
    counts = {stage: sum(task.stage == stage for task in tasks) for stage in config.stages}
    assert counts == {
        "primary_homogeneous_1024": 84,
        "area_scaling_512": 24,
        "seeded_basin_1024": 48,
    }
    assert {task.seed for task in tasks if task.stage == "primary_homogeneous_1024"} == set(
        range(5101, 5113)
    )
    assert all(task.steps == 262144 for task in tasks)
    assert all(task.sample_every == task.topology_every == 2048 for task in tasks)
    for task in tasks:
        inventory = inventory_for_density(task.size, task.size, task.density)
        assert inventory.total == task.size * task.size
        assert inventory.charge == 0


def test_nucleation_block_is_additive(extension_root: Path) -> None:
    old = StudyConfig.load(extension_root / "configs" / "rtx3060_balanced.yaml")
    new = StudyConfig.load(extension_root / "configs" / "v031_nucleation.yaml")
    assert "nucleation" not in old.canonical_payload()
    assert new.canonical_payload()["nucleation"]["persistence_followups"] == 2
