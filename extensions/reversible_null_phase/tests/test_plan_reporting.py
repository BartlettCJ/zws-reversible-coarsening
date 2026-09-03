from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from zws_reversible_null_phase.config import StudyConfig
from zws_reversible_null_phase.reporting import build_report, fit_logistic
from zws_reversible_null_phase.runner import run_study
from zws_reversible_null_phase.tasks import generate_tasks


def test_task_expansion_for_ablations(tmp_path: Path, extension_root: Path) -> None:
    config_path = tmp_path / "config.yaml"
    payload = {
        "schema": "zws-reversible-null-phase-study-v1",
        "name": "plan",
        "rule_path": str(extension_root / "rules" / "winner_d1fe99.json"),
        "device": "cpu",
        "resource": {"enabled": False},
        "stages": {
            "ablation_test": {
                "sizes": [16],
                "densities": [0.28],
                "seeds": [1],
                "steps": 4,
                "sample_every": 2,
                "topology_every": 2,
                "checkpoint_every": 2,
                "rule_variants": ["winner", "single_ablations", "rotation"],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = StudyConfig.load(config_path)
    tasks = generate_tasks(config)
    assert len(tasks) == 14
    assert len({task.id for task in tasks}) == 14


def test_logistic_fit_midpoint() -> None:
    density = np.repeat(np.array([0.26, 0.27, 0.28, 0.29]), 8)
    outcome = np.concatenate([np.zeros(8), np.zeros(8), np.ones(8), np.ones(8)])
    fit = fit_logistic(density, outcome)
    assert fit.success
    assert 0.27 < fit.p50 < 0.28
    assert fit.width > 0
    assert fit.midpoint_bracketed
    assert fit.sample_min == 0.26
    assert fit.sample_max == 0.29


def test_tiny_study_resume_and_results(tmp_path: Path, extension_root: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    payload = {
        "schema": "zws-reversible-null-phase-study-v1",
        "name": "tiny",
        "rule_path": str(extension_root / "rules" / "winner_d1fe99.json"),
        "device": "cpu",
        "low_k_max": 4,
        "criterion": {
            "min_peak_scale_ratio": 1.0,
            "min_final_scale_ratio": 1.0,
            "min_peak_largest_finite_empty_fraction": 0.0,
            "initial_scale_floor": 1.0,
        },
        "resource": {"enabled": False},
        "stages": {
            "onset_tiny": {
                "sizes": [16],
                "densities": [0.26, 0.28],
                "seeds": [1, 2],
                "steps": 8,
                "sample_every": 2,
                "topology_every": 4,
                "checkpoint_every": 4,
                "batch_size": 2,
                "arrangements": ["homogeneous"],
                "rule_variants": ["winner"],
                "early_fit_start": 2,
                "early_fit_end": 8,
                "shells": [[1, 2], [2, 4]],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = StudyConfig.load(config_path)
    output = tmp_path / "run"
    first = run_study(config, output)
    assert first["completed"] == 4
    results = list((output / "tasks" / "onset_tiny").glob("*.json"))
    assert len(results) == 4
    # A second invocation must skip verified completed tasks rather than alter them.
    mtimes = {path.name: path.stat().st_mtime_ns for path in results}
    second = run_study(config, output)
    assert second["completed"] == 4
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in results}
    payload = json.loads(results[0].read_text(encoding="utf-8"))
    assert payload["rule"]["sha256"].startswith("d1fe99")
    report = build_report(output)
    assert report["trajectories"] == 4
    report_text = (output / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "Exact-charge violations recorded in compact results: **0**" in report_text
    trajectories = (output / "report" / "TRAJECTORIES.csv").read_text(encoding="utf-8")
    assert "initial_charge,charge_violation" in trajectories
    dashboard = (output / "report" / "dashboard.html").read_text(encoding="utf-8")
    assert 'id="dashboard-data"' in dashboard
    assert "fetch('dashboard_data.json')" not in dashboard
