from __future__ import annotations

import numpy as np
import torch

from zws_nullity_lab.homogeneous_ic import (
    ARM_GRADIENT,
    ARM_HIGH_DENSITY,
    ARM_HOMOGENEOUS,
    ARM_LOCALIZED,
    assert_matched_inventories,
    construct_initial_state,
    deterministic_seed,
    gradient_weights,
    low_k_growth_ratio,
    low_wavenumber_diagnostics,
    simple_regular_spectral_signature,
    state_inventory,
    tensor_digest,
)


def test_frozen_seed_derivation_matches_preregistration() -> None:
    assert deterministic_seed("stage1:seed:0") == 2_019_049_721
    assert deterministic_seed("arrangement:B_homogeneous:0") == 2_045_295_831


def test_matched_arms_change_arrangement_only() -> None:
    common = {
        "height": 64,
        "width": 64,
        "inventory_seed": 1234,
        "localized_fraction": 0.5,
        "positive_fraction": 0.12,
        "bound_fraction": 0.08,
        "modulation_amplitude": 0.25,
    }
    states = {
        ARM_LOCALIZED: construct_initial_state(
            ARM_LOCALIZED, arrangement_seed=1, **common
        ),
        ARM_HOMOGENEOUS: construct_initial_state(
            ARM_HOMOGENEOUS, arrangement_seed=2, **common
        ),
        ARM_GRADIENT: construct_initial_state(ARM_GRADIENT, arrangement_seed=3, **common),
    }
    audit = assert_matched_inventories(states)
    assert audit["matched"] is True
    assert audit["total_charge"] == 0
    assert len({tensor_digest(state) for state in states.values()}) == 3
    assert torch.count_nonzero(states[ARM_LOCALIZED][:16]).item() == 0
    assert torch.count_nonzero(states[ARM_HOMOGENEOUS][:16]).item() > 0


def test_weak_gradient_is_smooth_bounded_and_measurable() -> None:
    weights = gradient_weights(64, 64, modulation_amplitude=0.25)
    assert np.isclose(weights.mean(), 1.0)
    assert weights.min() >= 0.75
    assert weights.max() <= 1.25
    assert np.max(np.abs(np.diff(weights, axis=0))) < 0.013
    assert np.max(np.abs(np.diff(weights, axis=1))) < 0.013


def test_exploratory_arm_has_patch_internal_global_density() -> None:
    state = construct_initial_state(
        ARM_HIGH_DENSITY,
        height=64,
        width=64,
        inventory_seed=10,
        arrangement_seed=11,
    )
    inventory = state_inventory(state)
    assert inventory["positive_count"] == int(64 * 64 * 0.12)
    assert inventory["negative_count"] == int(64 * 64 * 0.12)
    assert inventory["bound_count"] == int(64 * 64 * 0.08)
    assert inventory["total_charge"] == 0


def test_low_wavenumber_diagnostics_detect_large_scale_stripe() -> None:
    generator = torch.Generator().manual_seed(99)
    random = (torch.rand((64, 64), generator=generator) < 0.5).to(torch.uint8)
    x = torch.arange(64)
    stripe = (torch.cos(2.0 * torch.pi * (x + 0.5) / 64) > 0).to(torch.uint8)
    stripe = stripe.repeat(64, 1)
    random_result, stripe_result = low_wavenumber_diagnostics(
        torch.stack((random, stripe))
    )
    assert (
        stripe_result["bands"]["k01_04"]["mean_power"]
        > 10 * random_result["bands"]["k01_04"]["mean_power"]
    )
    assert stripe_result["spatial_autocorrelation"]["1"] > 0.9
    assert simple_regular_spectral_signature(stripe_result) is True


def test_low_k_growth_ratios_are_coordinate_wise() -> None:
    initial = {
        "bands": {
            "k01_04": {"mean_power": 2.0, "fraction_nonzero_power": 0.01}
        }
    }
    current = {
        "bands": {
            "k01_04": {"mean_power": 8.0, "fraction_nonzero_power": 0.03}
        }
    }
    ratio = low_k_growth_ratio(initial, current)
    assert ratio == {"mean_power_ratio": 4.0, "spectral_fraction_ratio": 3.0}
