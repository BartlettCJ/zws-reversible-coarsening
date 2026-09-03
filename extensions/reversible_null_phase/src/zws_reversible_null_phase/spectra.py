from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class SpectrumSummary:
    low_k_fraction: float
    low_k_power: float
    total_power: float
    dominant_k: float
    characteristic_wavelength: float
    radial_k: np.ndarray
    radial_power: np.ndarray


def occupancy_field(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 2:
        raise ValueError("state must be [H,W]")
    return (state != 0).to(dtype=torch.float32)


def radial_structure_factor(
    state: torch.Tensor,
    *,
    low_k_max: float = 8.0,
    radial_bins: int | None = None,
) -> SpectrumSummary:
    """Return the mean-subtracted occupancy structure factor.

    Wavenumbers are integer Fourier-shell radii.  `low_k_max=8` therefore
    measures the first several nonzero shells without counting the DC density.
    """
    field = occupancy_field(state)
    h, w = field.shape
    centered = field - field.mean()
    fft = torch.fft.rfft2(centered)
    power = (fft.real.square() + fft.imag.square()).detach().cpu().numpy()
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    power[0, 0] = 0.0
    total = float(power.sum())
    low_mask = (kr > 0) & (kr <= float(low_k_max))
    low_power = float(power[low_mask].sum())
    low_fraction = low_power / total if total > 0 else 0.0

    max_radius = int(math.ceil(float(kr.max())))
    bins = radial_bins or max_radius
    edges = np.linspace(0.0, float(max_radius), bins + 1)
    indices = np.clip(np.digitize(kr.reshape(-1), edges) - 1, 0, bins - 1)
    sums = np.bincount(indices, weights=power.reshape(-1), minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    radial = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if radial.size > 1:
        search = radial.copy()
        search[0] = 0.0
        index = int(np.argmax(search))
        dominant_k = float(centers[index]) if search[index] > 0 else 0.0
    else:
        dominant_k = 0.0
    min_dimension = float(min(h, w))
    wavelength = min_dimension / dominant_k if dominant_k > 0 else float("inf")
    return SpectrumSummary(
        low_k_fraction=low_fraction,
        low_k_power=low_power,
        total_power=total,
        dominant_k=dominant_k,
        characteristic_wavelength=wavelength,
        radial_k=centers,
        radial_power=radial,
    )


def shell_power(state: torch.Tensor, shells: tuple[tuple[float, float], ...]) -> np.ndarray:
    field = occupancy_field(state)
    h, w = field.shape
    centered = field - field.mean()
    fft = torch.fft.rfft2(centered)
    power = (fft.real.square() + fft.imag.square()).detach().cpu().numpy()
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    power[0, 0] = 0.0
    values = []
    for low, high in shells:
        mask = (kr >= low) & (kr < high)
        values.append(float(power[mask].mean()) if np.any(mask) else 0.0)
    return np.asarray(values, dtype=np.float64)


def fit_log_growth_rate(times: np.ndarray, powers: np.ndarray, *, floor: float = 1e-30) -> tuple[float, float]:
    """Fit log(power)=intercept+rate*time; return rate and R^2."""
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(powers, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1 or t.size != p.size or t.size < 3:
        raise ValueError("times and powers must be equal 1D arrays with at least three samples")
    y = np.log(np.maximum(p, floor))
    x = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(beta[1]), r2
