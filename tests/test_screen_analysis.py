from __future__ import annotations

import pytest

from zws_nullity_lab.screen_analysis import (
    double_conservation_classification,
    prevalence_classification,
    trajectory_coordinates,
    wilson_interval,
)


def _sample(tick: int, scale: float, occupied: float = 0.2) -> dict[str, object]:
    return {
        "tick": tick,
        "total_charge": 0,
        "state_populations": [80, 8, 8, 4],
        "occupied_fraction": occupied,
        "activity_one_phase": None if tick == 0 else 0.1,
        "reaction_density_one_phase": None if tick == 0 else 0.02,
        "boundary_density": 0.1,
        "empty_domains": {
            "finite_area_weighted_linear_scale": scale,
            "component_count": 4,
            "largest_area_fraction_of_empty": 0.8,
        },
        "structure_factor": {
            "characteristic_wavelength_cells": scale,
            "peak_to_radial_median": 5.0,
        },
        "coarse_density": {"64": {"dispersion_ratio_analytic": 3.0}},
        "exact_state_recurrence": {"4": 0.5, "8": 0.4},
        "coarse_occupancy_correlation_block64": {"4": 0.3, "8": 0.2},
    }


def _trajectory(
    condition_id: str,
    seed_index: int,
    *,
    peak_ratio: float,
    final_ratio: float,
) -> dict[str, object]:
    return {
        "run_id": f"run__{condition_id}__{seed_index}",
        "condition_id": condition_id,
        "conservation": "Q_and_M" if condition_id.startswith("qm_") else "Q_only",
        "seed_index": seed_index,
        "initial_seed": 1000 + seed_index,
        "required_invariants": ["Q", "M"] if condition_id.startswith("qm_") else ["Q"],
        "invariant_violation_counts": {"Q": 0, "M": 0},
        "samples": [
            _sample(0, 1.0),
            _sample(8, peak_ratio),
            _sample(16, final_ratio),
        ],
    }


def test_trajectory_coordinates_apply_frozen_strong_domain_rule() -> None:
    row = trajectory_coordinates(
        _trajectory("qm_sibling", 0, peak_ratio=9.0, final_ratio=2.5)
    )
    assert row["peak_tick"] == 8
    assert row["peak_scale_ratio"] == 9.0
    assert row["final_scale_ratio"] == 2.5
    assert row["strong_domain_organization"] is True
    assert row["final_positive_fraction"] == pytest.approx(0.08)


def test_double_conservation_disappears_classification_is_literal() -> None:
    rows = []
    for seed_index in range(8):
        rows.append(
            trajectory_coordinates(
                _trajectory("winner", seed_index, peak_ratio=10.0, final_ratio=4.0)
            )
        )
        rows.append(
            trajectory_coordinates(
                _trajectory("qm_sibling", seed_index, peak_ratio=1.5, final_ratio=1.1)
            )
        )
    result = double_conservation_classification(rows)
    assert result["formal_inference"]["classification"] == "disappears"
    assert result["measurement"]["sibling_strong_domain_count"] == 0


def test_prevalence_classification_uses_three_of_four_and_frozen_bins() -> None:
    rows = []
    conditions = []
    for rule_index in range(256):
        condition_id = f"prevalence_{rule_index:03d}"
        conditions.append({"condition_id": condition_id, "family_role": "sample"})
        passing_seeds = 3 if rule_index < 13 else 0
        for seed_index in range(4):
            passes = seed_index < passing_seeds
            rows.append(
                trajectory_coordinates(
                    _trajectory(
                        condition_id,
                        seed_index,
                        peak_ratio=9.0 if passes else 1.5,
                        final_ratio=2.5 if passes else 1.1,
                    )
                )
            )
    result = prevalence_classification(rows, conditions)
    assert result["measurement"]["winner_like_rule_count"] == 13
    assert result["measurement"]["winner_like_fraction"] == pytest.approx(13 / 256)
    assert result["formal_inference"]["prevalence_label"] == "moderately_unusual"


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(13, 256)
    assert lower < 13 / 256 < upper
