from __future__ import annotations

import hashlib
import json

from zws_nullity_lab.confirmation_final import (
    paired_bootstrap_difference,
    trajectory_endpoints,
    write_overnight_report,
)


GATE = {
    "final_dispersion_ratio_64_min": 8.0,
    "final_empty_domain_linear_scale_ratio_min": 2.0,
    "charge_violation_count_must_equal": 0,
}


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_bootstrap_difference([5.0, 7.0, 9.0], [1.0, 2.0, 3.0], seed=42)
    second = paired_bootstrap_difference([5.0, 7.0, 9.0], [1.0, 2.0, 3.0], seed=42)
    assert first == second
    assert first["mean_paired_difference"] == 5.0
    assert first["paired_seed_count"] == 3


def test_trajectory_endpoints_keeps_gate_and_recurrence_separate() -> None:
    samples = []
    for tick, scale, recurrence in [
        (0, 2.0, None),
        (512, 3.0, 0.8),
        (1024, 4.1, 0.9),
        (1536, 5.0, 0.95),
        (2048, 6.0, 1.0),
    ]:
        samples.append(
            {
                "tick": tick,
                "total_charge": 0,
                "occupied_fraction": 0.2 + tick / 100_000,
                "activity_one_phase": 0.1,
                "reaction_density_one_phase": 0.01,
                "entropy_normalized": 0.3,
                "boundary_density": 0.2,
                "empty_domains": {"finite_area_weighted_linear_scale": scale},
                "coarse_density": {"64": {"dispersion_ratio_analytic": 12.0}},
                "structure_factor": {"characteristic_wavelength_cells": 32.0},
                "exact_state_recurrence": (
                    {} if recurrence is None else {"4": recurrence, "8": recurrence - 0.1}
                ),
                "coarse_occupancy_correlation_block64": (
                    {} if recurrence is None else {"4": recurrence, "8": recurrence}
                ),
            }
        )
    trajectory = {
        "stage_id": "stage7a",
        "condition_id": "winner",
        "seed_index": 0,
        "run_id": "stage7a__winner__seed00",
        "status": "complete",
        "charge_violation_count": 0,
        "samples": samples,
    }
    result = trajectory_endpoints(trajectory, GATE)
    assert result["counts_as_macroscopic_coarsening"]
    assert result["finite_empty_domain_scale_ratio"] == 3.0
    assert result["late_median_exact_recurrence_lag4"] == 0.95
    assert result["late_median_exact_recurrence_lag8"] == 0.85


def test_write_overnight_report_audits_complete_matrix(tmp_path) -> None:
    conditions_a = [
        "winner",
        "rotation",
        "identity",
        *[f"matched_{index:02d}" for index in range(8)],
    ]
    conditions_b = ["winner", "rotation", *[f"matched_{index:02d}" for index in range(4)]]
    condition_records = [
        {
            "condition_id": condition,
            "rule_table_sha256_uint8": hashlib.sha256(condition.encode()).hexdigest(),
            "reactive_inputs": 24 if condition.startswith("matched_") or condition == "winner" else 0,
        }
        for condition in conditions_a
    ]
    manifest = {
        "preregistration_commit": "frozen-test-commit",
        "confirmation_seeds": list(range(100, 108)),
        "conditions": condition_records,
        "configuration": {
            "continuation_gate": {
                **GATE,
                "winner_required_seed_count": 6,
                "rotation_equal_required_seed_count": 6,
                "rotation_equal_median_dispersion_fraction": 0.8,
            },
            "stage_a": {
                "condition_ids": conditions_a,
                "seed_indices": list(range(8)),
                "phases": 32768,
            },
            "stage_b": {
                "condition_ids": conditions_b,
                "initial_seed_indices": list(range(4)),
                "phases": 65536,
            },
        },
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    campaign = tmp_path / "campaign"
    trajectory_dir = campaign / "trajectories"
    trajectory_dir.mkdir(parents=True)
    condition_by_id = {item["condition_id"]: item for item in condition_records}
    for stage_id, condition_ids, seeds, phases in [
        ("stage7a", conditions_a, range(8), 32768),
        ("stage7b", conditions_b, range(4), 65536),
    ]:
        for condition in condition_ids:
            winner = condition == "winner"
            for seed_index in seeds:
                samples = []
                for tick in range(0, phases + 1, 512):
                    progress = tick / phases
                    scale = 2.0 + (4.0 if winner else 0.2) * progress
                    samples.append(
                        {
                            "tick": tick,
                            "total_charge": 0,
                            "occupied_fraction": 0.1 + 0.02 * progress,
                            "activity_one_phase": None if tick == 0 else 0.1,
                            "reaction_density_one_phase": None if tick == 0 else 0.01,
                            "entropy_normalized": 0.3,
                            "boundary_density": 0.2,
                            "empty_domains": {"finite_area_weighted_linear_scale": scale},
                            "coarse_density": {
                                "64": {"dispersion_ratio_analytic": 12.0 if winner else 4.0}
                            },
                            "structure_factor": {"characteristic_wavelength_cells": 32.0},
                            "exact_state_recurrence": (
                                {} if tick == 0 else {"4": 0.9, "8": 0.85}
                            ),
                            "coarse_occupancy_correlation_block64": (
                                {} if tick == 0 else {"4": 0.95, "8": 0.9}
                            ),
                        }
                    )
                run_id = f"{stage_id}__{condition}__seed{seed_index:02d}"
                trajectory = {
                    "schema": "test",
                    "stage_id": stage_id,
                    "condition_id": condition,
                    "seed_index": seed_index,
                    "initial_seed": manifest["confirmation_seeds"][seed_index],
                    "run_id": run_id,
                    "status": "complete",
                    "world": [2048, 2048] if stage_id == "stage7a" else [4096, 4096],
                    "phases": phases,
                    "rule_table_sha256_uint8": condition_by_id[condition][
                        "rule_table_sha256_uint8"
                    ],
                    "manifest_sha256": manifest_sha,
                    "initial_state_sha256_uint8": hashlib.sha256(
                        f"{stage_id}-{seed_index}".encode()
                    ).hexdigest(),
                    "final_state_sha256_uint8": hashlib.sha256(run_id.encode()).hexdigest(),
                    "charge_violation_count": 0,
                    "samples": samples,
                }
                (trajectory_dir / f"{run_id}.json").write_text(
                    json.dumps(trajectory), encoding="utf-8"
                )
    experiment = tmp_path / "experiment"
    result = write_overnight_report(campaign, manifest_path, experiment, tmp_path / "report")
    assert result["integrity"]["valid"]
    assert result["integrity"]["observed_trajectory_count"] == 112
    assert (experiment / "OVERNIGHT_REPORT.md").is_file()
    assert (experiment / "CONFIRMATION_TRAJECTORIES.csv").is_file()
