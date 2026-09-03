from __future__ import annotations

import json

from zws_nullity_lab.confirmation_analysis import (
    evaluate_stage_a_gate,
    evaluate_trajectory_gate,
    write_stage_b_report,
)


GATE = {
    "winner_required_seed_count": 6,
    "final_dispersion_ratio_64_min": 8.0,
    "final_empty_domain_linear_scale_ratio_min": 2.0,
    "charge_violation_count_must_equal": 0,
    "rotation_equal_required_seed_count": 6,
    "rotation_equal_median_dispersion_fraction": 0.8,
}


def trajectory(condition: str, seed: int, *, dispersion: float, scale_factor: float) -> dict:
    samples = []
    for tick, scale in [(0, 2.0), (512, 2.5), (1024, 3.0), (1536, 3.5), (2048, 2.0 * scale_factor)]:
        samples.append(
            {
                "tick": tick,
                "total_charge": 0,
                "occupied_fraction": 0.2,
                "empty_domains": {"finite_area_weighted_linear_scale": scale},
                "coarse_density": {"64": {"dispersion_ratio_analytic": dispersion}},
                "structure_factor": {"characteristic_wavelength_cells": 8.0},
                "exact_state_recurrence": {"4": 1.0, "8": 1.0},
            }
        )
    return {
        "status": "complete",
        "run_id": f"stage7a__{condition}__seed{seed:02d}",
        "condition_id": condition,
        "seed_index": seed,
        "charge_violation_count": 0,
        "samples": samples,
    }


def manifest() -> dict:
    return {
        "configuration": {
            "continuation_gate": GATE,
            "stage_a": {
                "seed_indices": list(range(8)),
                "condition_ids": [
                    "winner",
                    "rotation",
                    "identity",
                    *[f"matched_{index:02d}" for index in range(8)],
                ],
            },
            "stage_b": {
                "condition_ids": [
                    "winner",
                    "rotation",
                    *[f"matched_{index:02d}" for index in range(4)],
                ]
            },
        }
    }


def test_trajectory_gate_requires_all_four_criteria() -> None:
    passing = trajectory("winner", 0, dispersion=9.0, scale_factor=2.5)
    result = evaluate_trajectory_gate(passing, GATE)
    assert result["counts_as_macroscopic_coarsening"]
    failing = trajectory("winner", 0, dispersion=7.99, scale_factor=2.5)
    result = evaluate_trajectory_gate(failing, GATE)
    assert not result["counts_as_macroscopic_coarsening"]
    assert not result["criteria"]["final_dispersion"]


def test_stage_gate_proceeds_when_winner_reproduces_and_rotation_does_not() -> None:
    rows = []
    for seed in range(8):
        rows.append(trajectory("winner", seed, dispersion=12.0, scale_factor=2.5))
        rows.append(trajectory("rotation", seed, dispersion=4.0, scale_factor=1.2))
        rows.append(trajectory("identity", seed, dispersion=1.0, scale_factor=1.0))
        for index in range(8):
            rows.append(
                trajectory(f"matched_{index:02d}", seed, dispersion=6.0, scale_factor=1.5)
            )
    result = evaluate_stage_a_gate(rows, manifest())
    assert result["proceed_to_stage7b"]
    assert result["classification"] == "A"


def test_stage_gate_stops_when_rotation_is_equal() -> None:
    rows = []
    for seed in range(8):
        rows.append(trajectory("winner", seed, dispersion=12.0, scale_factor=2.5))
        rows.append(trajectory("rotation", seed, dispersion=10.0, scale_factor=2.5))
        rows.append(trajectory("identity", seed, dispersion=1.0, scale_factor=1.0))
        for index in range(8):
            rows.append(
                trajectory(f"matched_{index:02d}", seed, dispersion=6.0, scale_factor=1.5)
            )
    result = evaluate_stage_a_gate(rows, manifest())
    assert not result["proceed_to_stage7b"]
    assert result["rotation_explains_equally_well"]
    assert result["classification"] == "C"


def test_stage_b_report_accounts_for_all_preregistered_rows(tmp_path) -> None:
    root = tmp_path / "campaign"
    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True)
    for condition in ["winner", "rotation", *[f"matched_{index:02d}" for index in range(4)]]:
        for seed in range(4):
            row = trajectory(condition, seed, dispersion=12.0, scale_factor=2.5)
            row["run_id"] = f"stage7b__{condition}__seed{seed:02d}"
            (trajectory_dir / f"{row['run_id']}.json").write_text(
                json.dumps(row), encoding="utf-8"
            )
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    result = write_stage_b_report(root, manifest_path, tmp_path / "report")
    assert result["summary"]["observed_trajectory_count"] == 24
    assert result["summary"]["failed_or_incomplete_trajectory_count"] == 0
    assert (tmp_path / "report" / "STAGE7B_REPORT.md").is_file()
