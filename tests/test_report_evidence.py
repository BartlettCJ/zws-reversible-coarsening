from __future__ import annotations

import pytest

from zws_nullity_lab.report_evidence import (
    compact_spatial_evidence,
    question_evidence_map,
    selected_perturbation_summaries,
    spatial_robustness_evidence,
)


def test_question_evidence_map_retains_all_fifteen_frozen_questions() -> None:
    values = question_evidence_map()
    assert [row["question_number"] for row in values] == list(range(1, 16))
    assert all(row["required_evidence_groups"] for row in values)
    assert "second exact local conservation law" in values[8]["question"]


def test_spatial_robustness_evidence_preserves_threshold_range_and_sign() -> None:
    rows = spatial_robustness_evidence(
        [
            {
                "stage_id": "stage7b",
                "role_group": "peak",
                "field": "occupied_fraction",
                "contrast_count": "108",
                "median_dense_minus_sparse": "0.25",
                "q10_dense_minus_sparse": "0.20",
                "q90_dense_minus_sparse": "0.30",
                "sign_robustness": "same_sign_all_definitions",
            }
        ]
    )
    assert rows[0]["contrast_count"] == 108
    assert rows[0]["median_dense_minus_sparse"] == pytest.approx(0.25)
    assert rows[0]["sign_robustness"] == "same_sign_all_definitions"


def test_spatial_robustness_rejects_duplicate_coordinate() -> None:
    row = {
        "stage_id": "stage7b",
        "role_group": "peak",
        "field": "occupied_fraction",
        "contrast_count": 1,
        "median_dense_minus_sparse": 0.1,
        "q10_dense_minus_sparse": 0.1,
        "q90_dense_minus_sparse": 0.1,
        "sign_robustness": "same_sign_all_definitions",
    }
    with pytest.raises(ValueError, match="duplicate spatial robustness coordinate"):
        spatial_robustness_evidence([row, row])


def test_perturbation_evidence_is_coordinate_wise_and_rejects_scalar_score() -> None:
    row = {
        "condition_id": "winner",
        "intervention_tick": "8192",
        "intervention_kind": "qm_state_swaps",
        "evaluation_tick": "8448",
        "response_horizon_phases": "256",
        "paired_seed_count": "8",
        "occupancy_fraction_delta_median": "0.01",
        "occupancy_fraction_delta_q25": "0.005",
        "occupancy_fraction_delta_q75": "0.015",
        "unrelated_field": "ignored",
    }
    selected = selected_perturbation_summaries([row])
    assert selected[0]["occupancy_fraction_delta_median"] == "0.01"
    assert "unrelated_field" not in selected[0]
    with pytest.raises(ValueError, match="scalar resilience score is prohibited"):
        selected_perturbation_summaries([{**row, "resilience_score": "0.9"}])


def test_compact_spatial_evidence_uses_medians_without_copying_raw_rows() -> None:
    compositions = [
        {
            "stage_id": "stage7b",
            "role_group": "peak",
            "analysis_family": "density_robustness",
            "environment": "dense_interior",
            "positive_fraction": value,
            "negative_fraction": value,
            "bound_neutral_fraction": 0.1,
            "charged_fraction": 2 * value,
            "occupied_fraction": 2 * value + 0.1,
            "normalized_state_entropy": 0.5,
            "signed_charge_per_site": 0.0,
        }
        for value in (0.2, 0.4)
    ]
    dynamics = [
        {
            "stage_id": "stage7b",
            "role_group": "peak",
            "environment": "dense_interior",
            "estimable": "True",
            "reaction_cell_fraction": 0.02,
            "empty_eulerian_net_count_change": -4,
            "positive_eulerian_net_count_change": 1,
            "negative_eulerian_net_count_change": 1,
            "bound_eulerian_net_count_change": 2,
        }
    ]
    local = [
        {
            "stage_id": "stage7b",
            "role_group": "peak",
            "environment": "dense_interior",
            "block_count": 4,
            "mean_signed_charge": 0.0,
            "mean_absolute_signed_charge": 1.0,
            "signed_charge_standard_deviation": 1.5,
            "median_occupancy_mass": 8,
            "median_normalized_state_entropy": 0.5,
        }
    ]
    compact = compact_spatial_evidence(compositions, dynamics, local)
    summary = compact["environment_composition_medians"][0]
    assert summary["row_count"] == 2
    assert summary["median_positive_fraction"] == pytest.approx(0.3)
