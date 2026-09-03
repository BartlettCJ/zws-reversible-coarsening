from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from zws_reversible_null_phase.history_erasure import (
    SourceCalibration,
    apply_global_shuffle,
    apply_tile_permutation,
    block_entropy_hierarchy,
    global_permutation,
    intervention_audit,
    multiscale_summary,
    permutation_digest,
    residual_record,
    spatial_mutual_information,
    spectral_summary,
    strong_domain_prefix,
    time_to_event,
    translate_state,
)
from zws_reversible_null_phase.history_stats import (
    clustered_bootstrap_rmst_delay,
    fixed_time_fraction,
    kaplan_meier,
    restricted_mean,
)
from zws_reversible_null_phase.simulator import step_rule


def patterned_state(size: int = 32) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return ((xx // 3 + yy // 5) % 4).astype(np.uint8)


def test_global_shuffle_exact_counts_q_m_phase_and_determinism() -> None:
    source = patterned_state()
    first_order = global_permutation(1234, source.size)
    second_order = global_permutation(1234, source.size)
    assert np.array_equal(first_order, second_order)
    assert permutation_digest(first_order) == hashlib.sha256(first_order.astype("<i8").tobytes()).hexdigest()
    altered = apply_global_shuffle(source, first_order)
    audit = intervention_audit(source, altered, source_phase=18, altered_phase=18)
    assert audit["valid"]
    assert audit["histogram_preserved"] and audit["Q_preserved"] and audit["M_preserved"]
    assert not np.array_equal(source, altered)


@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_tile_permutation_preserves_complete_tiles_and_inventory(tile_size: int) -> None:
    source = patterned_state()
    grid = source.shape[0] // tile_size
    order = np.arange(grid * grid, dtype=np.int64)[::-1]
    altered = apply_tile_permutation(
        source,
        tile_size=tile_size,
        permutation=order,
        temporary_even_offset_yx=(2, 6),
    )
    assert intervention_audit(source, altered, source_phase=7, altered_phase=7)["valid"]
    restored = apply_tile_permutation(
        altered,
        tile_size=tile_size,
        permutation=np.argsort(order),
        temporary_even_offset_yx=(2, 6),
    )
    assert np.array_equal(restored, source)


def test_odd_tile_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        apply_tile_permutation(
            patterned_state(),
            tile_size=8,
            permutation=np.arange(16),
            temporary_even_offset_yx=(1, 2),
        )


def test_even_translation_is_exact_dynamical_conjugacy(winner) -> None:
    rule = winner
    original = torch.from_numpy(patterned_state())
    offset = (6, 10)
    translated = torch.from_numpy(translate_state(original, offset))
    for phase in range(12):
        original = step_rule(original, rule, phase)
        translated = step_rule(translated, rule, phase)
        recovered = translate_state(translated, (-offset[0], -offset[1]))
        assert np.array_equal(recovered, original.numpy())


def test_reference_calibration_is_inherited_without_branch_reset() -> None:
    calibration = SourceCalibration(initial_finite_empty_scale=4.0, initial_charge=0)
    series = [
        {"phase": 0, "finite_empty_scale": 4.0, "largest_finite_empty_fraction": 0.002, "charge": 0},
        {"phase": 10, "finite_empty_scale": 16.0, "largest_finite_empty_fraction": 0.002, "charge": 0},
        {"phase": 20, "finite_empty_scale": 8.0, "largest_finite_empty_fraction": 0.002, "charge": 0},
    ]
    result = strong_domain_prefix(series, calibration)
    assert result["strong_domain"]
    branch_only_reset = SourceCalibration(initial_finite_empty_scale=16.0, initial_charge=0)
    assert not strong_domain_prefix(series[1:], branch_only_reset)["strong_domain"]


def test_censoring_is_not_converted_to_horizon_event() -> None:
    event = time_to_event([0, 8, 16, 24], [False, True, True, True], horizon=24)
    assert event["T_hit"] == 8 and event["T_persistent"] == 8
    censored = time_to_event([0, 8, 16, 24], [False, False, False, False], horizon=24)
    assert censored["T_hit"] is None and censored["right_censored_T_hit"]
    residual = residual_record(branch_phase=100, absolute_hit=None, absolute_persistent=None, branch_horizon=64)
    assert residual["R_hit"] is None and residual["censor_elapsed"] == 64


def test_multiscale_entropy_mi_and_anisotropy_are_finite_and_deterministic() -> None:
    source = patterned_state(512)
    first, first_power = multiscale_summary(source)
    second, second_power = multiscale_summary(source)
    assert first == second
    assert np.array_equal(first_power, second_power)
    assert set(first["block_entropy"]) == {"2", "4", "8", "16", "32", "64", "128", "256"}
    assert all(np.isfinite(list(first["block_entropy"].values())))
    assert all(np.isfinite(row["mean"]) for row in first["mutual_information"].values())
    assert np.isfinite(first["anisotropy"])


def test_independent_shuffle_reduces_neighbour_mi_for_pattern() -> None:
    source = patterned_state(256)
    altered = apply_global_shuffle(source, global_permutation(99, source.size))
    before = spatial_mutual_information(source, offsets=(1,))["1"]["mean"]
    after = spatial_mutual_information(altered, offsets=(1,))["1"]["mean"]
    assert after < before


def test_block_entropy_known_limits() -> None:
    uniform = np.zeros((256, 256), dtype=np.uint8)
    checker = (np.indices((256, 256)).sum(axis=0) % 2).astype(np.uint8)
    assert block_entropy_hierarchy(uniform)["2"] == 0.0
    assert block_entropy_hierarchy(checker)["2"] == pytest.approx(0.5)


def test_anisotropy_calculation_detects_stripes() -> None:
    stripes = np.zeros((256, 256), dtype=np.uint8)
    stripes[:, ::8] = 1
    summary, _ = spectral_summary(stripes)
    assert summary["anisotropy"] > 0.5


def test_anisotropy_of_seeded_random_field_is_not_forced_to_one() -> None:
    rng = np.random.default_rng(42)
    random_state = rng.integers(0, 4, size=(256, 256), dtype=np.uint8)
    summary, _ = spectral_summary(random_state)
    assert 0.0 <= summary["anisotropy"] < 0.5


def test_clustered_bootstrap_is_deterministic_and_km_rmst_exact() -> None:
    rows = [
        {"cluster": "a", "untouched_time": 10, "untouched_observed": True, "altered_time": 20, "altered_observed": True},
        {"cluster": "b", "untouched_time": 20, "untouched_observed": True, "altered_time": 40, "altered_observed": False},
        {"cluster": "c", "untouched_time": 30, "untouched_observed": False, "altered_time": 30, "altered_observed": True},
    ]
    first = clustered_bootstrap_rmst_delay(rows, horizon=40, replicates=200, seed=7)
    second = clustered_bootstrap_rmst_delay(rows, horizon=40, replicates=200, seed=7)
    assert first == second
    km = kaplan_meier([10, 20], [True, False], horizon=30)
    assert restricted_mean(km, 30) == pytest.approx(20.0)
    assert fixed_time_fraction([10, 20], [True, False], 15) == 0.5
