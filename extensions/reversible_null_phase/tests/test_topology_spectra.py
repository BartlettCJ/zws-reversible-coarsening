from __future__ import annotations

import numpy as np
import torch

from zws_reversible_null_phase.spectra import fit_log_growth_rate, radial_structure_factor
from zws_reversible_null_phase.topology import torus_components


def test_torus_wrap_detection() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[10, :] = True
    summary = torus_components(mask)
    assert summary.wrapping_x
    assert not summary.wrapping_y
    assert summary.wrapping_any

    dot = np.zeros((32, 32), dtype=bool)
    dot[4:8, 5:9] = True
    finite = torus_components(dot)
    assert not finite.wrapping_any
    assert finite.finite_component_count == 1
    assert finite.characteristic_finite_scale > 0


def test_seam_crossing_finite_component_is_not_a_winding_component() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[7, 0] = True
    mask[7, -1] = True
    summary = torus_components(mask)
    assert summary.component_count == 1
    assert summary.finite_component_count == 1
    assert not summary.wrapping_any
    assert summary.finite_areas.tolist() == [2]


def test_torus_component_metrics_are_translation_invariant() -> None:
    rng = np.random.default_rng(8320)
    mask = rng.random((64, 64)) < 0.61
    reference = torus_components(mask)
    for shift in ((2, 4), (18, 26), (62, 38)):
        translated = torus_components(np.roll(mask, shift, axis=(0, 1)))
        assert translated.component_count == reference.component_count
        assert translated.finite_component_count == reference.finite_component_count
        assert translated.wrapping_x == reference.wrapping_x
        assert translated.wrapping_y == reference.wrapping_y
        assert np.array_equal(np.sort(translated.areas), np.sort(reference.areas))
        assert np.array_equal(
            np.sort(translated.finite_areas), np.sort(reference.finite_areas)
        )
        assert translated.largest_finite_fraction == reference.largest_finite_fraction
        assert (
            translated.characteristic_finite_scale
            == reference.characteristic_finite_scale
        )


def test_low_k_structure_factor_detects_broad_stripe() -> None:
    state = torch.zeros((64, 64), dtype=torch.uint8)
    state[:, :32] = 1
    summary = radial_structure_factor(state, low_k_max=4)
    assert summary.low_k_fraction > 0.5
    assert summary.characteristic_wavelength >= 32


def test_log_growth_fit() -> None:
    t = np.arange(10, dtype=float)
    p = np.exp(0.25 * t + 1.0)
    rate, r2 = fit_log_growth_rate(t, p)
    assert abs(rate - 0.25) < 1e-10
    assert r2 > 0.999999
