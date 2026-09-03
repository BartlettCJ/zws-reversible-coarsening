from __future__ import annotations

from pathlib import Path
import runpy

import pytest
import torch

from zws_nullity_lab.density_threshold import (
    assert_nested_occupied_support,
    classify_density_response,
    density_arm_id,
    exact_density_inventory,
    exact_inventory_homogeneous,
    finite_size_density_targets,
    fit_logistic_onset,
    logistic_fit_gate,
    predicted_probability,
    refinement_plan,
)
from zws_nullity_lab.density_threshold_campaign import verify_density_stage_inputs
from zws_nullity_lab.homogeneous_ic import tensor_digest


def _summary(target: float, fraction: float) -> dict:
    inventory = exact_density_inventory(1024, 1024, target)
    return {
        "target_occupancy": target,
        "realized_occupancy": inventory["realized_occupancy"],
        "strong_domain_pass_fraction": fraction,
    }


def test_exact_inventory_preserves_composition_and_known_32_percent_endpoint() -> None:
    inventory = exact_density_inventory(1024, 1024, 0.32)
    assert inventory["state_counts"] == [713032, 125829, 125829, 83886]
    assert inventory["occupancy_mass"] == 335544
    assert inventory["total_charge"] == 0
    assert inventory["positive_count"] * 2 == 3 * inventory["bound_count"]


def test_common_permutation_produces_deterministic_nested_homogeneous_support() -> None:
    states = []
    for target in (0.08, 0.16, 0.32):
        inventory = exact_density_inventory(64, 64, target)
        state = exact_inventory_homogeneous(
            64,
            64,
            state_counts=inventory["state_counts"],
            arrangement_seed=12345,
        )
        repeated = exact_inventory_homogeneous(
            64,
            64,
            state_counts=inventory["state_counts"],
            arrangement_seed=12345,
        )
        assert torch.equal(state, repeated)
        assert torch.bincount(state.long().flatten(), minlength=4).tolist() == inventory[
            "state_counts"
        ]
        states.append(state)
    assert_nested_occupied_support(states)


@pytest.mark.parametrize(
    ("fractions", "expected"),
    [
        ([0.0, 0.0, 0.0, 0.0], "A_NO_TRANSITION_IN_TESTED_RANGE"),
        ([0.0, 0.25, 0.75, 1.0], "D_VERY_SHARP_FINITE_SIZE_TRANSITION_REGION"),
        ([0.0, 0.25, 0.5, 0.75], "C_SHARP_FINITE_SIZE_TRANSITION_REGION"),
        ([0.0, 0.25, 0.5, 0.75], "B_GRADUAL_CROSSOVER"),
        ([0.0, 1.0, 0.25, 1.0], "E_NON_MONOTONIC_MULTIPLE_REGIMES"),
    ],
)
def test_frozen_response_classification(fractions: list[float], expected: str) -> None:
    if expected == "D_VERY_SHARP_FINITE_SIZE_TRANSITION_REGION":
        targets = (0.16, 0.18, 0.20, 0.22)
    elif expected == "C_SHARP_FINITE_SIZE_TRANSITION_REGION":
        targets = (0.10, 0.12, 0.16, 0.18)
    elif expected == "B_GRADUAL_CROSSOVER":
        targets = (0.08, 0.12, 0.18, 0.24)
    else:
        targets = (0.08, 0.12, 0.16, 0.20)
    result = classify_density_response(
        [_summary(target, fraction) for target, fraction in zip(targets, fractions, strict=True)]
    )
    assert result["classification"] == expected


def test_refinement_grid_follows_frozen_spacing_and_excludes_broad_points() -> None:
    summaries = [
        _summary(0.16, 0.0),
        _summary(0.18, 0.25),
        _summary(0.20, 0.75),
        _summary(0.22, 1.0),
    ]
    plan = refinement_plan(summaries)
    assert plan["run_refinement"] is True
    assert plan["spacing"] == 0.005
    assert plan["new_target_occupancies"] == [0.185, 0.19, 0.195]
    assert plan["seed_count_per_density"] == 16


def test_logistic_gate_and_blocked_bootstrap_are_deterministic() -> None:
    densities = (0.12, 0.16, 0.18, 0.20, 0.22, 0.24, 0.28)
    probabilities = (0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0)
    summaries = [_summary(density, probability) for density, probability in zip(densities, probabilities)]
    assert logistic_fit_gate(summaries)["passed"] is True
    rows = []
    for density, probability in zip(densities, probabilities, strict=True):
        inventory = exact_density_inventory(1024, 1024, density)
        pass_count = int(round(probability * 12))
        for seed_index in range(12):
            rows.append(
                {
                    "stage_id": "broad_1024",
                    "seed_index": seed_index,
                    "realized_occupancy": inventory["realized_occupancy"],
                    "strong_domain_pass": seed_index < pass_count,
                }
            )
    first = fit_logistic_onset(rows, bootstrap_replicates=20, bootstrap_seed=77)
    second = fit_logistic_onset(rows, bootstrap_replicates=20, bootstrap_seed=77)
    assert first == second
    assert first["bootstrap"]["intervals_valid"] is True
    assert 0.18 < first["point_estimate"]["occupancy_p50"] < 0.23
    assert predicted_probability(0.16, first) < predicted_probability(0.24, first)


def test_finite_size_targets_freeze_fitted_quartiles_to_arm_precision() -> None:
    fit = {
        "point_estimate": {
            "occupancy_p25": 0.183141,
            "occupancy_p50": 0.197859,
            "occupancy_p75": 0.212578,
        }
    }
    assert finite_size_density_targets(fit) == [0.1831, 0.1979, 0.2126]


def test_finite_size_analyzer_reports_independent_pass_counts() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "analyze_finite_size.py"
    )
    summarize = runpy.run_path(str(script))["summarize"]
    rows = []
    for seed_index, passed in enumerate((False, True, True, True)):
        rows.append(
            {
                "stage_id": "finite_size_2048",
                "world_size": 2048,
                "arm_id": "density_p2000",
                "target_occupancy": 0.2,
                "realized_occupancy": 0.2,
                "strong_finite_domain_pass": passed,
                "systematic_low_k_growth_pass": seed_index >= 2,
                "exact_q_at_all_samples": True,
                "final_simple_regular_spectral_signature": False,
                "peak_domain_scale_ratio": 10.0 + seed_index,
                "final_domain_scale_ratio": 5.0 + seed_index,
                "low_k_peak_mean_power_ratio": 8.0 + seed_index,
                "final_occupied_fraction": 0.3,
                "mean_reaction_density": 0.01,
            }
        )
    summary = summarize(rows)
    assert len(summary) == 1
    assert summary[0]["trajectory_count"] == 4
    assert summary[0]["strong_domain_pass_count"] == 3
    assert summary[0]["strong_domain_pass_fraction"] == 0.75
    assert summary[0]["low_k_growth_pass_count"] == 2
    assert summary[0]["exact_q_pass_count"] == 4


def test_refinement_analyzer_keeps_strong_and_low_k_classifications_separate() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "analyze_refinement.py"
    )
    summarize = runpy.run_path(str(script))["summarize"]
    rows = []
    for seed_index in range(4):
        rows.append(
            {
                "stage_id": "refinement_1024",
                "arm_id": "density_p1950",
                "target_occupancy": 0.195,
                "realized_occupancy": 0.195,
                "strong_finite_domain_pass": seed_index < 1,
                "systematic_low_k_growth_pass": seed_index < 3,
                "exact_q_at_all_samples": True,
                "final_simple_regular_spectral_signature": False,
                "peak_domain_scale_ratio": 2.0 + seed_index,
                "final_domain_scale_ratio": 1.0 + seed_index,
                "low_k_peak_mean_power_ratio": 3.0 + seed_index,
                "final_occupied_fraction": 0.25,
                "mean_reaction_density": 0.01,
                "final_entropy_normalized": 0.7,
            }
        )
    summary = summarize(rows)[0]
    assert summary["strong_domain_pass_count"] == 1
    assert summary["strong_domain_pass_fraction"] == 0.25
    assert summary["low_k_growth_pass_count"] == 3
    assert summary["low_k_growth_pass_fraction"] == 0.75


def test_morphology_selector_keeps_six_distinct_rows_for_a_narrow_fit() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "analyze_final_1024.py"
    )
    representative_summaries = runpy.run_path(str(script))["representative_summaries"]
    rows = [
        {
            "stage_id": "broad_1024",
            "arm_id": density_arm_id(target),
            "target_occupancy": target,
            "realized_occupancy": target,
            "strong_domain_pass_fraction": 0.0,
        }
        for target in (0.0646, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
    ]
    fit = {
        "point_estimate": {
            "occupancy_p25": 0.1991,
            "occupancy_p50": 0.2000,
            "occupancy_p75": 0.2009,
        }
    }
    selected = representative_summaries(rows, fit)
    densities = [float(row["realized_occupancy"]) for row in selected]
    assert len(densities) == 6
    assert densities == sorted(set(densities))
    assert densities[-1] == 0.32


@pytest.mark.parametrize("successes", (0, 12))
def test_broad_wilson_interval_contains_endpoint_proportion(successes: int) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "analyze_broad.py"
    )
    wilson_interval = runpy.run_path(str(script))["wilson_interval"]
    proportion = successes / 12
    lower, upper = wilson_interval(successes, 12)
    assert 0.0 <= lower <= proportion <= upper <= 1.0


def test_finite_size_runner_accepts_json_sorted_stage_keys() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "run_finite_size.py"
    )
    validate_frozen_stages = runpy.run_path(str(script))["validate_frozen_stages"]
    validate_frozen_stages(
        {
            "stages": {
                "finite_size_1024": {},
                "finite_size_2048": {},
                "finite_size_512": {},
            }
        }
    )
    with pytest.raises(RuntimeError, match="frozen stage set"):
        validate_frozen_stages(
            {"stages": {"finite_size_512": {}, "finite_size_1024": {}}}
        )


def test_finite_size_fit_support_flags_extrapolative_bootstrap() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "homogeneous_density_threshold_20260826"
        / "analyze_finite_size.py"
    )
    support = runpy.run_path(str(script))["fit_support_diagnostics"]
    summaries = [
        {"realized_occupancy": 0.27, "strong_domain_pass_fraction": 0.25},
        {"realized_occupancy": 0.28, "strong_domain_pass_fraction": 0.5},
        {"realized_occupancy": 0.29, "strong_domain_pass_fraction": 1.0},
    ]
    fit = {
        "point_estimate": {
            "occupancy_p25": 0.27,
            "occupancy_p50": 0.30,
            "occupancy_p75": 0.31,
        },
        "bootstrap": {
            "percentile_intervals": {
                "occupancy_p25": {"lower_95": -10.0, "upper_95": 0.28},
                "occupancy_p50": {"lower_95": 0.27, "upper_95": 100.0},
                "occupancy_p75": {"lower_95": 0.28, "upper_95": 200.0},
                "width_p25_to_p75": {"lower_95": 0.01, "upper_95": 210.0},
            }
        },
    }
    diagnostics = support(fit, summaries)
    assert diagnostics["empirical_25_to_75_bracket"]["width"] == pytest.approx(0.02)
    assert not diagnostics["point_p50_within_sampled_range"]
    assert not diagnostics["point_width_fully_supported_by_sampled_range"]
    assert not diagnostics["bootstrap_intervals_finite_and_within_unit_interval"]
    assert not diagnostics["bootstrap_p50_interval_within_sampled_range"]


def test_density_campaign_preflight_regenerates_hashes_and_nested_support() -> None:
    height = width = 64
    targets = (0.08, 0.16)
    rows = []
    arms = []
    for target in targets:
        arm = density_arm_id(target)
        arms.append(arm)
        inventory = exact_density_inventory(height, width, target)
        for seed_index, arrangement_seed in enumerate((101, 202)):
            state = exact_inventory_homogeneous(
                height,
                width,
                state_counts=inventory["state_counts"],
                arrangement_seed=arrangement_seed,
            )
            rows.append(
                {
                    "run_id": f"test__{arm}__seed{seed_index:02d}",
                    "stage_id": "test_stage",
                    "arm_id": arm,
                    "seed_index": seed_index,
                    "inventory_seed": seed_index + 1,
                    "arrangement_seed": arrangement_seed,
                    "diagnostic_seed": seed_index + 301,
                    "state_counts": inventory["state_counts"],
                    "initial_state_sha256_uint8": tensor_digest(state),
                }
            )
    stage = {
        "stage_id": "test_stage",
        "world": [height, width],
        "initial_condition": {"construction": "uniform_exact_inventory_nested"},
        "arm_ids": arms,
        "inventory_seeds": [1, 2],
        "planned_runs": rows,
    }
    preflight = verify_density_stage_inputs(stage)
    assert preflight["run_count"] == 4
    assert preflight["all_regenerated_hashes_exact"] is True
    assert preflight["all_nested_supports_exact"] is True
