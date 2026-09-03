from __future__ import annotations

import math

import numpy as np
import torch

from zws_nullity_lab.confirmation_diagnostics import (
    coarse_density_statistics,
    radial_structure_factor,
    sample_diagnostics,
    toroidal_empty_domain_stats,
)
from zws_nullity_lab.rules import ReversibleRule


def test_toroidal_domain_merges_opposite_edges() -> None:
    empty = np.zeros((8, 8), dtype=np.bool_)
    empty[3, 0] = True
    empty[3, -1] = True
    stats = toroidal_empty_domain_stats(empty)
    assert stats["component_count"] == 1
    assert stats["largest_area"] == 2
    assert stats["total_perimeter"] == 6


def test_toroidal_domain_keeps_diagonal_cells_separate() -> None:
    empty = np.zeros((8, 8), dtype=np.bool_)
    empty[2, 2] = True
    empty[3, 3] = True
    stats = toroidal_empty_domain_stats(empty)
    assert stats["component_count"] == 2
    assert stats["finite_component_count"] == 1
    assert stats["finite_empty_area"] == 1
    assert stats["finite_largest_area"] == 1
    assert stats["finite_area_weighted_linear_scale"] == 1.0


def test_coarse_density_detects_clustering_against_fixed_density_shuffle() -> None:
    occupancy = torch.zeros((64, 64), dtype=torch.bool)
    occupancy[:32] = True
    stats = coarse_density_statistics(
        occupancy,
        blocks=[8, 16],
        shuffle_base_seed=123,
        tick=512,
    )
    assert stats["8"]["dispersion_ratio_analytic"] > 50
    assert stats["16"]["dispersion_ratio_seeded_shuffle"] > 10
    repeated = coarse_density_statistics(
        occupancy,
        blocks=[8, 16],
        shuffle_base_seed=123,
        tick=512,
    )
    assert stats == repeated


def test_structure_factor_recovers_stripe_wavelength() -> None:
    x = torch.arange(64)
    stripe = ((x // 4) % 2).to(torch.bool).repeat(64, 1)
    result = radial_structure_factor(stripe.unsqueeze(0))[0]
    assert result["peak_radius"] == 8
    assert math.isclose(result["characteristic_wavelength_cells"], 8.0)


def test_sample_diagnostics_charge_and_recurrence() -> None:
    state = torch.zeros((1, 64, 64), dtype=torch.uint8)
    state[0, 20, 20] = 1
    state[0, 21, 20] = 2
    lag = state.clone()
    result = sample_diagnostics(
        state,
        tick=128,
        rule=ReversibleRule.identity(),
        lag_states={1: lag, 4: lag},
        coarse_blocks=[8, 16, 32, 64],
        shuffle_base_seeds=[456],
        previous_occupied_fractions=[2 / (64 * 64)],
        previous_tick=64,
    )[0]
    assert result["total_charge"] == 0
    assert result["exact_state_recurrence"]["4"] == 1.0
    assert result["activity_one_phase"] == 0.0
    assert result["reaction_density_one_phase"] == 0.0
    assert result["occupancy_growth_per_phase"] == 0.0
