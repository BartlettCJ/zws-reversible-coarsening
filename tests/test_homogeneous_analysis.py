from __future__ import annotations

from zws_nullity_lab.homogeneous_analysis import (
    classify_stage1,
    reduce_trajectory,
    stage2_gate,
)
from zws_nullity_lab.homogeneous_ic import (
    ARM_GRADIENT,
    ARM_HOMOGENEOUS,
    ARM_LOCALIZED,
)


FINITE_RULE = {
    "initial_scale_floor_cells": 1.0,
    "peak_scale_ratio_min": 4.0,
    "final_scale_ratio_min": 2.0,
    "peak_finite_largest_area_absolute_min": 64,
    "peak_finite_largest_area_ratio_min": 16.0,
}
LOW_K_RULE = {
    "band": "k01_04",
    "peak_mean_power_ratio_min": 4.0,
    "final_mean_power_ratio_min": 2.0,
    "peak_spectral_fraction_ratio_min": 2.0,
    "numerical_floor": 1e-30,
}
REGULAR_RULE = {
    "strongest_mode_pair_fraction_min": 0.15,
    "radial_peak_shell_fraction_min": 0.30,
}


def _sample(tick: int, scale: float | None, area: int, low_power: float) -> dict:
    return {
        "tick": tick,
        "total_charge": 0,
        "state_populations": [3600, 160, 160, 176],
        "occupied_fraction": 496 / 4096,
        "activity_one_phase": None if tick == 0 else 0.02,
        "reaction_density_one_phase": None if tick == 0 else 0.01,
        "entropy_normalized": 0.4,
        "boundary_density": 0.12,
        "empty_domains": {
            "component_count": 3,
            "finite_area_weighted_linear_scale": scale,
            "finite_largest_area": area,
            "largest_area_fraction_of_empty": 0.9,
            "area_quantiles": {"min": 1.0, "median": 5.0, "max": 100.0},
            "area_log2_histogram": [1, 2, 3],
        },
        "low_wavenumber": {
            "bands": {
                "k01_04": {
                    "mean_power": low_power,
                    "fraction_nonzero_power": 0.01 * low_power,
                }
            },
            "strongest_mode_pair_fraction_upper_bound": 0.2,
            "radial_peak_shell_fraction": 0.35,
            "directional_mode_pair_power": {"kx1_ky0": 2.0, "kx0_ky1": 1.0},
            "spatial_autocorrelation": {str(lag): 0.1 for lag in (1, 2, 4, 8, 16, 32)},
        },
        "structure_factor": {
            "peak_radius": 2,
            "peak_power": 5.0,
            "peak_to_radial_median": 4.0,
            "characteristic_wavelength_cells": 32.0,
            "radial_profile": [
                {
                    "radius_min": 1,
                    "radius_max_exclusive": 2,
                    "radius_geometric_mean": 1.0,
                    "mean_power": 5.0,
                }
            ],
        },
        "occupancy_interface_edge_density": 0.1,
        "coarse_density": {
            str(block): {"observed_variance": 0.01} for block in (16, 32, 64)
        },
        "exact_state_recurrence": {str(lag): 0.5 for lag in (4, 8, 16, 32)},
    }


def test_trajectory_rule_uses_one_cell_floor_and_connected_area_gate() -> None:
    trajectory = {
        "run_id": "stage1__B__seed00",
        "stage_id": "stage1",
        "arm_id": ARM_HOMOGENEOUS,
        "seed_index": 0,
        "inventory_seed": 7,
        "initial_state_sha256_uint8": "initial",
        "initial_inventory": {"occupancy_mass": 496},
        "phases": 16,
        "samples": [
            _sample(0, None, 0, 1.0),
            _sample(8, 5.0, 100, 5.0),
            _sample(16, 2.5, 80, 2.5),
        ],
    }
    result = reduce_trajectory(
        trajectory,
        finite_rule=FINITE_RULE,
        low_k_rule=LOW_K_RULE,
        regular_rule=REGULAR_RULE,
    )
    assert result["initial_scale_denominator"] == 1.0
    assert result["peak_domain_scale_ratio"] == 5.0
    assert result["final_domain_scale_ratio"] == 2.5
    assert result["strong_finite_domain_pass"] is True
    assert result["systematic_low_k_growth_pass"] is True
    assert result["final_simple_regular_spectral_signature"] is True
    assert result["final_empty_domain_component_count"] == 3
    assert result["final_coarse_density_variance_block16"] == 0.01
    assert result["final_structure_factor_characteristic_wavelength_cells"] == 32.0
    assert result["final_recurrence_lag32"] == 0.5


def _summary(arm_id: str, passes: int, peak: float, final: float, low: int) -> dict:
    return {
        "arm_id": arm_id,
        "strong_finite_domain_pass_count": passes,
        "systematic_low_k_growth_pass_count": low,
        "final_simple_regular_spectral_signature_count": 0,
        "median_peak_domain_scale_ratio": peak,
        "median_final_domain_scale_ratio": final,
    }


def test_stage1_classification_and_stage2_gate_follow_frozen_priority() -> None:
    summaries = [
        _summary(ARM_LOCALIZED, 16, 20.0, 10.0, 16),
        _summary(ARM_HOMOGENEOUS, 12, 9.0, 4.5, 12),
        _summary(ARM_GRADIENT, 16, 15.0, 8.0, 16),
    ]
    assert classify_stage1(summaries) == "A_STRONG_SPONTANEOUS_ORGANISATION"
    passed, reasons = stage2_gate(summaries)
    assert passed is True
    assert reasons


def test_localization_dependent_class_requires_positive_reference() -> None:
    summaries = [
        _summary(ARM_LOCALIZED, 15, 20.0, 10.0, 16),
        _summary(ARM_HOMOGENEOUS, 0, 1.5, 1.0, 0),
        _summary(ARM_GRADIENT, 2, 3.0, 1.5, 2),
    ]
    assert classify_stage1(summaries) == "D_LOCALIZATION_DEPENDENT"
