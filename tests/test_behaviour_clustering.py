from __future__ import annotations

import pytest

from zws_nullity_lab.behaviour_clustering import (
    RAW_FEATURES,
    build_behaviour_features,
    stable_cluster_analysis,
)


def _trajectory_coordinate(condition_id: str, seed: int, offset: float) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "initial_seed": seed,
        "peak_scale_ratio": 10.0 + offset,
        "final_scale_ratio": 3.0 + offset / 10.0,
        "peak_tick": 12000 + int(offset * 10),
        "occupied_fraction_change": 0.05 + offset / 1000.0,
        "final_dispersion_block64": 4.0 + offset / 10.0,
        "mean_reaction_density": 0.02 + offset / 10000.0,
        "final_boundary_density": 0.1 + offset / 1000.0,
        "final_positive_fraction": 0.04 + offset / 10000.0,
        "final_negative_fraction": 0.04 + offset / 10000.0,
        "final_bound_fraction": 0.03 + offset / 10000.0,
        "final_structure_peak_to_median": 6.0 + offset / 10.0,
        "final_coarse_recurrence_lag128": 0.8 - offset / 1000.0,
        "stasis_flag": False,
        "collapse_flag": False,
        "noise_like_flag": False,
    }


def test_behaviour_features_reduce_paired_seeds_without_scalar_fitness() -> None:
    coordinates = []
    for condition_id, base in (("winner", 0.0), ("prevalence_000", 1.0)):
        for seed in range(4):
            coordinates.append(_trajectory_coordinate(condition_id, seed, base + seed / 10.0))
    conditions = [
        {"condition_id": "winner", "family_role": "anchor"},
        {"condition_id": "prevalence_000", "family_role": "sample"},
    ]
    rows = build_behaviour_features(coordinates, conditions, phases=32768)
    assert len(rows) == 2
    assert set(RAW_FEATURES) <= set(rows[0])
    assert "fitness" not in rows[0]
    winner = next(row for row in rows if row["condition_id"] == "winner")
    assert winner["peak_phase_fraction"] == pytest.approx(12001.5 / 32768)


def test_cluster_stability_can_identify_a_reproducible_separated_winner_group() -> None:
    rows = []
    for index in range(5):
        value = -10.0 + index * 0.02
        rows.append(
            {
                "condition_id": "winner" if index == 0 else f"near_winner_{index}",
                "family_role": "anchor" if index == 0 else "sample",
                **{feature: value + feature_index * 0.001 for feature_index, feature in enumerate(RAW_FEATURES)},
            }
        )
    for index in range(56):
        group = index % 7
        value = group * 1.5 + (index // 7) * 0.03
        rows.append(
            {
                "condition_id": f"background_{index:03d}",
                "family_role": "sample",
                **{feature: value + feature_index * 0.01 for feature_index, feature in enumerate(RAW_FEATURES)},
            }
        )
    result = stable_cluster_analysis(rows, seed=2026082502)
    assert result["formal_inference"]["distinct_cluster"] is True
    assert len(result["passing_cluster_counts"]) >= 4
    assert len(result["pca_projection"]["coordinates"]) == len(rows)
    assert result["retained_features"]


def test_cluster_analysis_is_bitwise_deterministic_for_frozen_seed() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "condition_id": "winner" if index == 0 else f"rule_{index:02d}",
                "family_role": "anchor" if index == 0 else "sample",
                **{
                    feature: float((index + 1) * (feature_index + 1))
                    for feature_index, feature in enumerate(RAW_FEATURES)
                },
            }
        )
    first = stable_cluster_analysis(rows, seed=12345)
    second = stable_cluster_analysis(rows, seed=12345)
    assert first == second
