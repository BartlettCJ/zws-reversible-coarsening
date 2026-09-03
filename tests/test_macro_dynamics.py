from __future__ import annotations

import math

import numpy as np

from zws_nullity_lab.macro_dynamics import (
    analyze_macro_trajectory,
    extract_macro_series,
    fit_post_peak_models,
    lagged_correlation,
    spectral_summary,
)


def _sample(tick: int, scale: float, occupancy: float) -> dict[str, object]:
    total = 1000
    occupied = round(total * occupancy)
    positive = occupied // 3
    negative = positive
    bound = occupied - positive - negative
    return {
        "tick": tick,
        "occupied_fraction": occupancy,
        "reaction_density_one_phase": occupancy / 10,
        "boundary_density": occupancy / 2,
        "state_populations": [total - occupied, positive, negative, bound],
        "empty_domains": {
            "finite_area_weighted_linear_scale": scale,
            "total_perimeter": 100 + tick,
            "component_count": 20,
            "finite_component_count": 19,
            "largest_area_fraction_of_empty": 0.8,
        },
        "structure_factor": {"characteristic_wavelength_cells": 2 * scale},
        "coarse_density": {"64": {"observed_variance": occupancy / 4}},
    }


def test_extract_macro_series_keeps_components_separate() -> None:
    samples = [_sample(0, 2.0, 0.1), _sample(10, 4.0, 0.2)]
    series = extract_macro_series(samples)
    assert series["finite_domain_scale"] == [2.0, 4.0]
    assert series["occupied_fraction"] == [0.1, 0.2]
    assert series["reaction_density"] == [0.01, 0.02]
    assert series["state_fraction_positive"][0] == 33 / 1000
    assert series["coarse_density_variance_64"] == [0.025, 0.05]


def test_lagged_correlation_reports_full_axis_and_missing_values() -> None:
    first = [0.0, 1.0, 2.0, 3.0, None, 5.0]
    second = [0.0, 2.0, 4.0, 6.0, None, 10.0]
    result = lagged_correlation(first, second, minimum_overlap=4)
    zero = next(record for record in result if record["lag_samples"] == 0)
    assert zero["correlation"] == 1.0
    assert len(result) == 2 * len(first) - 1


def test_spectral_summary_recovers_a_slow_synthetic_period() -> None:
    ticks = list(range(64))
    values = [math.sin(2 * math.pi * tick / 8) for tick in ticks]
    result = spectral_summary(ticks, values, minimum_period_samples=4)
    assert result["estimable"] is True
    assert abs(result["candidate_period_samples"] - 8.0) < 1e-9
    assert result["autocorrelation_at_nearest_period"] is not None
    assert result["autocorrelation_at_nearest_period"] > 0.99


def test_post_peak_model_comparison_is_estimable() -> None:
    ticks = list(range(30))
    values = [2.0 + 10.0 * math.exp(-tick / 5.0) for tick in ticks]
    result = fit_post_peak_models(ticks, values)
    assert result["estimable"] is True
    names = {model["model"] for model in result["models"]}
    assert "constant_plateau" in names
    assert "linear_return" in names
    assert "exponential_relaxation" in names


def test_analyze_macro_trajectory_reports_peak_relaxation_and_correlations() -> None:
    scales = list(np.concatenate([np.linspace(2, 20, 12), np.linspace(19, 8, 20)]))
    samples = [_sample(index * 16, float(scale), 0.1 + index / 1000) for index, scale in enumerate(scales)]
    result = analyze_macro_trajectory(
        {
            "run_id": "synthetic",
            "stage_id": "test",
            "condition_id": "winner",
            "seed_index": 0,
            "initial_seed": 1,
            "samples": samples,
        }
    )
    assert result["peak_scale_ratio"] == 10.0
    assert result["final_scale_ratio"] == 4.0
    assert result["final_to_peak_ratio"] == 0.4
    assert result["peak_tick"] == 11 * 16
    assert result["post_peak_models"]["estimable"] is True
