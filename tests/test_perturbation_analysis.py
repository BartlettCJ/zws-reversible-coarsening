from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zws_nullity_lab.atomic import atomic_write_json
from zws_nullity_lab.perturbation_analysis import (
    load_completed_baselines,
    load_completed_perturbations,
    paired_response_coordinates,
    summarize_paired_responses,
)


def _sample(
    tick: int,
    *,
    populations: list[int],
    scale: float,
    profile: list[float],
    recurrence: float | None,
) -> dict[str, object]:
    total = sum(populations)
    occupied = sum(populations[1:]) / total
    return {
        "tick": tick,
        "total_charge": populations[1] - populations[2],
        "state_populations": populations,
        "occupied_fraction": occupied,
        "activity_one_phase": None if tick == 8 else 0.1,
        "reaction_density_one_phase": None if tick == 8 else 0.02,
        "entropy_normalized": 0.4,
        "boundary_density": 0.1,
        "empty_domains": {
            "finite_area_weighted_linear_scale": scale,
            "component_count": 4,
            "largest_area_fraction_of_empty": 0.7,
        },
        "structure_factor": {
            "peak_radius": 2,
            "characteristic_wavelength_cells": scale,
            "peak_to_radial_median": 5.0,
            "radial_profile": [
                {
                    "radius_min": index + 1,
                    "radius_max_exclusive": index + 2,
                    "mean_power": value,
                }
                for index, value in enumerate(profile)
            ],
        },
        "exact_state_recurrence": {"4": recurrence} if recurrence is not None else {},
        "coarse_occupancy_correlation_block64": (
            {"4": recurrence - 0.1} if recurrence is not None else {}
        ),
    }


def _baseline() -> dict[str, object]:
    return {
        "schema": "zwslab-followup-trajectory-v1",
        "status": "complete",
        "run_id": "screen__winner__seed00",
        "condition_id": "winner",
        "seed_index": 0,
        "initial_seed": 123,
        "rule_table_sha256_uint8": "rule",
        "initial_state_sha256_uint8": "initial",
        "invariant_violation_counts": {"Q": 0},
        "samples": [
            _sample(
                8,
                populations=[80, 8, 8, 4],
                scale=10.0,
                profile=[1.0, 3.0],
                recurrence=0.5,
            ),
            _sample(
                16,
                populations=[80, 8, 8, 4],
                scale=10.0,
                profile=[1.0, 3.0],
                recurrence=0.5,
            ),
        ],
    }


def _perturbation() -> dict[str, object]:
    return {
        "schema": "zwslab-perturbation-response-trajectory-v1",
        "status": "complete",
        "run_id": "response__winner__t00008__q_balanced_creation__seed00",
        "baseline_run_id": "screen__winner__seed00",
        "condition_id": "winner",
        "conservation": "Q_only",
        "seed_index": 0,
        "initial_seed": 123,
        "selection_seed": 456,
        "intervention_tick": 8,
        "intervention_kind": "q_balanced_creation",
        "evaluation_ticks": [8, 16],
        "rule_table_sha256_uint8": "rule",
        "initial_state_sha256_uint8": "initial",
        "invariant_violation_counts": {"Q": 0},
        "perturbation_audit": {
            "requested_changed_sites": 4,
            "observed_changed_sites": 4,
            "charge_before": 0,
            "charge_after": 0,
            "mass_before": 20,
            "mass_after": 24,
            "charge_preserved": True,
        },
        "samples": [
            _sample(
                8,
                populations=[76, 10, 10, 4],
                scale=12.0,
                profile=[2.0, 2.0],
                recurrence=None,
            ),
            _sample(
                16,
                populations=[79, 8, 8, 5],
                scale=10.5,
                profile=[1.0, 3.0],
                recurrence=0.55,
            ),
        ],
    }


def test_paired_response_coordinates_retain_separate_physical_deltas() -> None:
    rows = paired_response_coordinates(
        [_perturbation()], [_baseline()], recurrence_lags=[4]
    )
    assert len(rows) == 2
    immediate, final = rows
    assert immediate["response_horizon_phases"] == 0
    assert immediate["occupancy_mass_delta"] == 4
    assert immediate["occupancy_fraction_delta"] == pytest.approx(0.04)
    assert immediate["state_fraction_positive_delta"] == pytest.approx(0.02)
    assert immediate["finite_domain_scale_relative_delta"] == pytest.approx(0.2)
    assert immediate["structure_spectrum_total_variation_distance"] == pytest.approx(0.25)
    assert immediate["exact_recurrence_lag4_delta"] is None
    assert final["finite_domain_scale_relative_delta"] == pytest.approx(0.05)
    assert final["exact_recurrence_lag4_delta"] == pytest.approx(0.05)


def test_response_summary_is_coordinate_wise_not_scalar() -> None:
    coordinates = paired_response_coordinates(
        [_perturbation()], [_baseline()], recurrence_lags=[4]
    )
    summaries = summarize_paired_responses(coordinates)
    assert len(summaries) == 2
    assert summaries[0]["paired_seed_count"] == 1
    assert summaries[0]["occupancy_fraction_delta_median"] == pytest.approx(0.04)
    assert all("resilience_score" not in key for row in summaries for key in row)


def test_paired_response_rejects_changed_initial_state() -> None:
    perturbation = _perturbation()
    perturbation["initial_state_sha256_uint8"] = "changed"
    with pytest.raises(ValueError, match="paired initial state digest differs"):
        paired_response_coordinates([perturbation], [_baseline()], recurrence_lags=[4])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completed_input_loaders_verify_completion_sets_and_hashes(tmp_path: Path) -> None:
    baseline_digest = "baseline-manifest"
    baseline_manifest = {
        "conditions": [
            {"condition_id": "winner", "rule_table_sha256_uint8": "rule"}
        ],
        "planned_trajectories": [
            {
                "run_id": "screen__winner__seed00",
                "condition_id": "winner",
                "seed_index": 0,
                "initial_seed": 123,
            }
        ],
    }
    baseline_root = tmp_path / "baseline"
    baseline_path = baseline_root / "trajectories" / "screen__winner__seed00.json"
    baseline_path.parent.mkdir(parents=True)
    atomic_write_json(baseline_path, _baseline())
    atomic_write_json(
        baseline_root / "campaign_summary.json",
        {
            "schema": "zwslab-followup-campaign-summary-v1",
            "status": "complete",
            "manifest_sha256": baseline_digest,
            "completed_conditions": ["winner"],
            "planned_condition_count": 1,
            "planned_trajectory_count": 1,
        },
    )
    condition_dir = baseline_root / "conditions" / "winner"
    condition_dir.mkdir(parents=True)
    atomic_write_json(
        condition_dir / "completion.json",
        {
            "status": "complete",
            "condition_id": "winner",
            "manifest_sha256": baseline_digest,
            "trajectories": [
                {"run_id": "screen__winner__seed00", "sha256": _digest(baseline_path)}
            ],
        },
    )
    baseline_record = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_record["manifest_sha256"] = baseline_digest
    atomic_write_json(baseline_path, baseline_record)
    completion = json.loads(
        (condition_dir / "completion.json").read_text(encoding="utf-8")
    )
    completion["trajectories"][0]["sha256"] = _digest(baseline_path)
    atomic_write_json(condition_dir / "completion.json", completion)

    perturbation_digest = "perturbation-manifest"
    perturbation_manifest = {
        "conditions": [
            {"condition_id": "winner", "rule_table_sha256_uint8": "rule"}
        ],
        "planned_branches": [
            {
                "run_id": "response__winner__t00008__q_balanced_creation__seed00",
                "baseline_run_id": "screen__winner__seed00",
                "condition_id": "winner",
                "seed_index": 0,
                "initial_seed": 123,
                "intervention_tick": 8,
                "intervention_kind": "q_balanced_creation",
                "selection_seed": 456,
                "evaluation_ticks": [8, 16],
            }
        ],
    }
    perturbation_root = tmp_path / "perturbation"
    perturbation_path = (
        perturbation_root
        / "trajectories"
        / "response__winner__t00008__q_balanced_creation__seed00.json"
    )
    perturbation_path.parent.mkdir(parents=True)
    perturbation_record = _perturbation()
    perturbation_record["manifest_sha256"] = perturbation_digest
    atomic_write_json(perturbation_path, perturbation_record)
    atomic_write_json(
        perturbation_root / "campaign_summary.json",
        {
            "schema": "zwslab-perturbation-campaign-summary-v1",
            "status": "complete",
            "manifest_sha256": perturbation_digest,
            "completed_groups": ["winner__t00008"],
            "planned_group_count": 1,
            "planned_branch_count": 1,
        },
    )
    group_dir = perturbation_root / "groups" / "winner__t00008"
    group_dir.mkdir(parents=True)
    atomic_write_json(
        group_dir / "completion.json",
        {
            "status": "complete",
            "group_id": "winner__t00008",
            "manifest_sha256": perturbation_digest,
            "trajectories": [
                {
                    "run_id": perturbation_record["run_id"],
                    "sha256": _digest(perturbation_path),
                }
            ],
        },
    )

    baselines, baseline_integrity = load_completed_baselines(
        baseline_root, baseline_manifest, baseline_digest
    )
    perturbations, perturbation_integrity = load_completed_perturbations(
        perturbation_root, perturbation_manifest, perturbation_digest
    )
    assert len(baselines) == len(baseline_integrity) == 1
    assert len(perturbations) == len(perturbation_integrity) == 1
