from __future__ import annotations

import json
from pathlib import Path

import yaml

from zws_reversible_null_phase.atomic import atomic_write_json
from zws_reversible_null_phase.config import StudyConfig
from zws_reversible_null_phase.runner import run_study
from zws_reversible_null_phase.stage2 import (
    choose_stage2_bracket,
    freeze_stage2_addendum,
    run_stage2_continuation,
)
from zws_reversible_null_phase.tasks import generate_tasks


def test_choose_lowest_eligible_stage2_bracket() -> None:
    assert choose_stage2_bracket(
        {0.16: 0, 0.18: 0, 0.20: 1, 0.22: 3, 0.24: 5, 0.25: 8},
        {density: 12 for density in [0.16, 0.18, 0.20, 0.22, 0.24, 0.25]},
    ) == (0.20, 0.22)
    assert (
        choose_stage2_bracket(
            {0.16: 0, 0.18: 0, 0.20: 0, 0.22: 1, 0.24: 1},
            {density: 12 for density in [0.16, 0.18, 0.20, 0.22, 0.24]},
        )
        is None
    )


def test_exact_stage2_continuation_matches_uninterrupted(
    tmp_path: Path, extension_root: Path
) -> None:
    config_path = tmp_path / "source.yaml"
    payload = {
        "schema": "zws-reversible-null-phase-study-v1",
        "name": "stage2-test",
        "rule_path": str(extension_root / "rules" / "winner_d1fe99.json"),
        "device": "cpu",
        "criterion": {
            "min_peak_scale_ratio": 1e9,
            "min_final_scale_ratio": 1e9,
            "min_peak_largest_finite_empty_fraction": 1.0,
            "initial_scale_floor": 1.0,
        },
        "nucleation": {
            "schema": "zws-reversible-null-phase-nucleation-v1",
            "observation_every": 4,
            "persistence_followups": 2,
            "fixed_readout_phases": [4, 8, 12],
            "event_window_before": 8,
            "event_window_after": 8,
            "event_crop_fraction": 0.5,
            "primary_stage": "primary_homogeneous_1024",
            "stage2": {"maximum_phase": 20},
        },
        "resource": {"enabled": False},
        "stages": {
            "primary_homogeneous_1024": {
                "sizes": [16],
                "densities": [0.24, 0.26],
                "seeds": [5101],
                "steps": 12,
                "sample_every": 4,
                "topology_every": 4,
                "checkpoint_every": 4,
                "batch_size": 2,
                "arrangements": ["homogeneous"],
                "rule_variants": ["winner"],
                "early_fit_start": 4,
                "early_fit_end": 12,
                "shells": [[1, 2], [2, 4]],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = StudyConfig.load(config_path)
    run_dir = tmp_path / "run"
    run_study(config, run_dir)
    tasks = generate_tasks(config)
    experiment = tmp_path / "experiment"
    atomic_write_json(
        experiment / "FROZEN_PLAN.json",
        {
            "schema": "test",
            "config_path": str(config_path),
            "config_sha256": config.sha256(),
            "winner_raw_uint8_sha256": "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c",
            "tasks": [task.to_dict() for task in tasks],
        },
    )
    atomic_write_json(
        experiment / "PRIMARY_GATE.json",
        {
            "schema": "test",
            "status": "stage2_eligible",
            "selected_bracket": [0.24, 0.26],
            "eligible_censored_source_task_ids": [task.id for task in tasks],
            "target_phase": 20,
        },
    )
    stage2_plan = freeze_stage2_addendum(run_dir, experiment)
    assert stage2_plan["task_count"] == 2
    result = run_stage2_continuation(
        run_dir, experiment, managed=False, require_committed=False
    )
    assert result["completed"] == 2
    status = json.loads((run_dir / "stage2" / "status.json").read_text(encoding="utf-8"))
    assert status == {
        "completed": 2,
        "pending": 0,
        "state": "complete",
        "task_count": 2,
        "updated_utc": status["updated_utc"],
    }

    direct_payload = json.loads(json.dumps(payload))
    direct_payload["stages"]["primary_homogeneous_1024"]["steps"] = 20
    direct_path = tmp_path / "direct.yaml"
    direct_path.write_text(yaml.safe_dump(direct_payload), encoding="utf-8")
    direct = tmp_path / "direct"
    run_study(StudyConfig.load(direct_path), direct)
    direct_results = {}
    for path in (direct / "tasks" / "primary_homogeneous_1024").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        direct_results[(item["task"]["density"], item["task"]["seed"])] = item
    for path in (run_dir / "stage2" / "tasks" / "stage2_extension_1024").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        expected = direct_results[(item["task"]["density"], item["task"]["seed"])]
        assert item["final"]["state_sha256"] == expected["final"]["state_sha256"]
        assert item["topological_series"] == expected["topological_series"]
        assert item["temporal_series"] == expected["temporal_series"]
