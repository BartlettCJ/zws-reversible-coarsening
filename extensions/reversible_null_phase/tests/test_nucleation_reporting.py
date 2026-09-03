from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from zws_reversible_null_phase.atomic import atomic_write_json, sha256_file
from zws_reversible_null_phase.config import StudyConfig
from zws_reversible_null_phase.nucleation_reporting import (
    _group_summary,
    _km_rows,
    _mechanism_label,
    build_nucleation_report,
)
from zws_reversible_null_phase.runner import run_study
from zws_reversible_null_phase.tasks import generate_tasks


def test_km_censoring_and_identifiable_median() -> None:
    frame = pd.DataFrame(
        [
            {
                "task_id": f"t{index}",
                "stage": "primary_homogeneous_1024",
                "size": 1024,
                "density": 0.24,
                "arrangement": "homogeneous",
                "hit_observed": event,
                "persistent_observed": event,
                "right_censored_T_hit": not event,
                "T_hit": time if event else None,
                "T_persistent": time if event else None,
                "event_or_censor_time": time,
                "phase_exposure": time,
                "horizon": 100,
                "transformed_by_32768": event,
                "transformed_by_65536": event,
                "transformed_by_131072": event,
                "transformed_by_262144": event,
            }
            for index, (time, event) in enumerate(
                [(20, True), (40, True), (60, True), (100, False)]
            )
        ]
    )
    km = pd.DataFrame(_km_rows(frame))
    summary = _group_summary(frame, km)[0]
    assert summary["median_T_hit"] == 40
    assert summary["median_T_persistent"] == 40
    assert summary["q25_T_persistent"] == 20
    assert summary["q75_T_persistent"] == 60
    assert summary["events"] == 3
    assert summary["right_censored"] == 1
    assert summary["restricted_mean_event_free_time"] == 55
    assert summary["restricted_mean_event_free_time_common_horizon"] == 55
    assert summary["restricted_mean_common_horizon"] == 100

    sparse_persistence = frame.copy()
    sparse_persistence["persistent_observed"] = [True, False, False, False]
    sparse_persistence["T_persistent"] = [20, None, None, None]
    sparse_summary = _group_summary(
        sparse_persistence, pd.DataFrame(_km_rows(sparse_persistence))
    )[0]
    assert sparse_summary["median_T_persistent"] is None
    assert sparse_summary["median_T_persistent_lower_bound"] == 100
    assert sparse_summary["median_T_persistent_observed_only"] == 20


def test_frozen_event_shape_labels() -> None:
    def payload(
        *,
        localization: float,
        scales: list[float],
        low_k: list[float] | None = None,
        components: list[int] | None = None,
        wrap_phase: int | None = None,
    ) -> dict[str, object]:
        phases = [4 * index for index in range(len(scales))]
        hit = phases[-1]
        low_k = low_k or [1.0] * len(scales)
        components = components or [100] * len(scales)
        topology = {
            "occupied_wrap_x": wrap_phase,
            "occupied_wrap_y": None,
            "empty_wrap_x": None,
            "empty_wrap_y": None,
        }
        return {
            "time_to_event": {
                "T_hit": hit,
                "observation_every": 4,
                "topology_first_times": topology,
            },
            "event_window": {
                "change_localization_fraction": localization,
            },
            "topological_series": [
                {
                    "phase": phase,
                    "finite_empty_scale": scale,
                    "low_k_power": spectrum,
                    "empty_finite_component_count": component_count,
                }
                for phase, scale, spectrum, component_count in zip(
                    phases, scales, low_k, components, strict=True
                )
            ],
        }

    flat = [1.0] * 11
    assert _mechanism_label(payload(localization=0.50, scales=flat)) == "A"
    assert _mechanism_label(
        payload(localization=0.125, scales=[float(index + 1) for index in range(11)])
    ) == "B"
    assert _mechanism_label(
        payload(
            localization=0.25,
            scales=flat,
            low_k=[1.0, 1.0, *([2.5] * 9)],
            components=[100, 100, 100, 95, 90, 85, 80, 75, 70, 60, 50],
        )
    ) == "C"
    assert _mechanism_label(
        payload(
            localization=0.50,
            scales=[1.0, *([2.0] * 10)],
            wrap_phase=40,
        )
    ) == "D"
    assert _mechanism_label(payload(localization=0.25, scales=flat)) == "E"


def test_tiny_complete_nucleation_report(tmp_path: Path, extension_root: Path) -> None:
    config_path = tmp_path / "tiny_v031.yaml"
    stage_common = {
        "sizes": [16],
        "densities": [0.26],
        "seeds": [5101],
        "steps": 12,
        "sample_every": 4,
        "topology_every": 4,
        "checkpoint_every": 4,
        "batch_size": 1,
        "rule_variants": ["winner"],
        "early_fit_start": 4,
        "early_fit_end": 12,
        "shells": [[1, 2], [2, 4]],
    }
    payload = {
        "schema": "zws-reversible-null-phase-study-v1",
        "name": "tiny-v031-report",
        "rule_path": str(extension_root / "rules" / "winner_d1fe99.json"),
        "device": "cpu",
        "criterion": {
            "min_peak_scale_ratio": 0.0,
            "min_final_scale_ratio": 0.0,
            "min_peak_largest_finite_empty_fraction": 0.0,
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
        },
        "resource": {"enabled": False},
        "stages": {
            "primary_homogeneous_1024": {
                **stage_common,
                "seeds": [5101, 5102],
                "arrangements": ["homogeneous"],
            },
            "area_scaling_512": {
                **stage_common,
                "arrangements": ["homogeneous"],
            },
            "seeded_basin_1024": {
                **stage_common,
                "arrangements": ["droplet", "slab"],
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = StudyConfig.load(config_path)
    run_dir = tmp_path / "run"
    run_study(config, run_dir)

    tasks = generate_tasks(config)
    frozen_tasks = []
    for task in tasks:
        result_path = run_dir / "tasks" / task.stage / f"{task.id}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        frozen_tasks.append(
            {**task.to_dict(), "initial_state_sha256": result["initial"]["state_sha256"]}
        )
    experiment = tmp_path / "experiment"
    atomic_write_json(
        experiment / "FROZEN_PLAN.json",
        {
            "schema": "zws-reversible-null-phase-v031-frozen-plan-v1",
            "task_count": len(tasks),
            "config_sha256": config.sha256(),
            "tasks": frozen_tasks,
        },
    )
    atomic_write_json(
        experiment / "SCIENTIFIC_DECISION.json",
        {
            "schema": "zws-reversible-null-phase-v031-scientific-decision-v1",
            "option": "A",
            "publication_answer": "The tiny synthetic run is not a publication result.",
            "further_v03_answer": "Freeze the completed mechanism study.",
            "scientific_claim": "This fixture tests decision-record plumbing only.",
            "rationale": "The explicit checksummed decision replaces automatic event-count inference.",
        },
    )
    report = build_nucleation_report(run_dir, experiment)
    assert report["audit_pass"] is True
    assert report["trajectories"] == 5
    assert (experiment / "NUCLEATION_REPORT.md").exists()
    assert (experiment / "SURVIVAL_SUMMARY.csv").exists()
    assert (experiment / "dashboard.html").exists()
    assert (experiment / "FIGURE_MANIFEST.json.sha256").exists()
    survival = pd.read_csv(experiment / "SURVIVAL_SUMMARY.csv")
    assert set(survival.endpoint) == {"T_hit", "T_persistent"}
    area = pd.read_csv(experiment / "AREA_SCALING.csv")
    assert set(area.N) == {1}
    seeded = pd.read_csv(experiment / "SEEDED_BASIN_RESULTS.csv")
    assert set(seeded.N) == {1}
    assert set(seeded.arrangement) == {"homogeneous", "droplet", "slab"}
    report_text = (experiment / "NUCLEATION_REPORT.md").read_text(encoding="utf-8")
    assert "median T_persistent" in report_text
    assert "by 32,768" in report_text
    assert "Descriptive hazard inspection" in report_text
    assert "Topology timing relative to first passage" in report_text
    assert "Matched area scaling" in report_text
    assert "Matched seeded-basin comparison" in report_text
    assert "common phase-262,144 horizon" in report_text
    assert "**A. Freeze the completed mechanism study.**" in report_text
    assert "The tiny synthetic run is not a publication result." in report_text
    dashboard_text = (experiment / "dashboard.html").read_text(encoding="utf-8")
    assert '"matched_area_summary"' in dashboard_text
    assert '"matched_seeded_summary"' in dashboard_text

    first_result = run_dir / "tasks" / tasks[0].stage / f"{tasks[0].id}.json"
    final_state = first_result.with_suffix("") / "final.npz"
    np.savez_compressed(final_state, state=np.zeros((16, 16), dtype=np.uint8))
    final_state.with_suffix(final_state.suffix + ".sha256").write_text(
        f"{sha256_file(final_state)}  {final_state.name}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="final state hash mismatch"):
        build_nucleation_report(run_dir, experiment)
