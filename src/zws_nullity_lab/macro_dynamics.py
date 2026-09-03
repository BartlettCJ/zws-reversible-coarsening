from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def finite_domain_scale(sample: dict[str, Any]) -> float | None:
    value = sample.get("empty_domains", {}).get("finite_area_weighted_linear_scale")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def extract_macro_series(samples: list[dict[str, Any]]) -> dict[str, list[float | None]]:
    """Extract declared macroscopic observables without combining them into a score."""

    def nested(sample: dict[str, Any], *keys: str) -> float | None:
        value: Any = sample
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    output: dict[str, list[float | None]] = {
        "tick": [float(sample["tick"]) for sample in samples],
        "finite_domain_scale": [finite_domain_scale(sample) for sample in samples],
        "occupied_fraction": [nested(sample, "occupied_fraction") for sample in samples],
        "reaction_density": [nested(sample, "reaction_density_one_phase") for sample in samples],
        "boundary_density": [nested(sample, "boundary_density") for sample in samples],
        "interface_length": [nested(sample, "empty_domains", "total_perimeter") for sample in samples],
        "domain_count": [nested(sample, "empty_domains", "component_count") for sample in samples],
        "finite_domain_count": [
            nested(sample, "empty_domains", "finite_component_count") for sample in samples
        ],
        "largest_domain_fraction": [
            nested(sample, "empty_domains", "largest_area_fraction_of_empty")
            for sample in samples
        ],
        "structure_wavelength": [
            nested(sample, "structure_factor", "characteristic_wavelength_cells")
            for sample in samples
        ],
    }
    for state_index, state_name in enumerate(("empty", "positive", "negative", "bound")):
        fractions: list[float | None] = []
        for sample in samples:
            populations = sample.get("state_populations")
            if not isinstance(populations, list) or len(populations) != 4:
                fractions.append(None)
                continue
            total = sum(int(value) for value in populations)
            fractions.append(int(populations[state_index]) / total if total else None)
        output[f"state_fraction_{state_name}"] = fractions
    coarse_values: list[float | None] = []
    for sample in samples:
        coarse = sample.get("coarse_density", {})
        record = coarse.get("64") or coarse.get(64) if isinstance(coarse, dict) else None
        coarse_values.append(
            float(record["observed_variance"])
            if isinstance(record, dict) and record.get("observed_variance") is not None
            else None
        )
    output["coarse_density_variance_64"] = coarse_values
    return output


def _paired_finite(
    first: list[float | None], second: list[float | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = [
        index for index, (a, b) in enumerate(zip(first, second, strict=True)) if a is not None and b is not None
    ]
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray([first[index] for index in indices], dtype=np.float64),
        np.asarray([second[index] for index in indices], dtype=np.float64),
    )


def lagged_correlation(
    first: list[float | None],
    second: list[float | None],
    *,
    detrend: bool = False,
    minimum_overlap: int = 4,
) -> list[dict[str, float | int | None]]:
    """Return the full normalized cross-correlation over sample-index lags."""
    if len(first) != len(second):
        raise ValueError("series lengths must match")
    n = len(first)
    output: list[dict[str, float | int | None]] = []
    for lag in range(-(n - 1), n):
        if lag < 0:
            first_slice = first[-lag:]
            second_slice = second[: n + lag]
        elif lag > 0:
            first_slice = first[: n - lag]
            second_slice = second[lag:]
        else:
            first_slice = first
            second_slice = second
        indices, x, y = _paired_finite(first_slice, second_slice)
        correlation: float | None = None
        if len(indices) >= minimum_overlap:
            if detrend:
                coordinate = indices.astype(np.float64)
                x = x - np.polyval(np.polyfit(coordinate, x, 1), coordinate)
                y = y - np.polyval(np.polyfit(coordinate, y, 1), coordinate)
            x = x - x.mean()
            y = y - y.mean()
            denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
            if denominator > 0:
                correlation = float(np.dot(x, y) / denominator)
        output.append({"lag_samples": lag, "overlap": len(indices), "correlation": correlation})
    return output


def spectral_summary(
    ticks: list[float | None],
    values: list[float | None],
    *,
    minimum_period_samples: int = 4,
) -> dict[str, Any]:
    """Search a regular sampled series for slow nonzero-frequency components."""
    indices, x, y = _paired_finite(ticks, values)
    if len(indices) < max(8, 2 * minimum_period_samples):
        return {"estimable": False, "reason": "too_few_finite_samples"}
    tick_steps = np.diff(x)
    interval = float(np.median(tick_steps))
    if interval <= 0 or not np.allclose(tick_steps, interval, rtol=1e-6, atol=1e-9):
        return {"estimable": False, "reason": "irregular_sampling"}
    coordinate = np.arange(len(y), dtype=np.float64)
    detrended = y - np.polyval(np.polyfit(coordinate, y, 1), coordinate)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0)
    valid = frequencies > 0
    periods_samples = np.full_like(frequencies, np.inf, dtype=np.float64)
    periods_samples[valid] = 1.0 / frequencies[valid]
    valid &= periods_samples >= minimum_period_samples
    if not valid.any() or float(power[valid].sum()) <= 0:
        return {"estimable": False, "reason": "no_slow_spectral_power"}
    candidates = np.flatnonzero(valid)
    peak_index = int(candidates[np.argmax(power[candidates])])
    period_samples = float(periods_samples[peak_index])
    lag = int(round(period_samples))
    autocorrelation = None
    if minimum_period_samples <= lag < len(detrended) - 1:
        first = detrended[:-lag]
        second = detrended[lag:]
        denominator = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
        if denominator > 0:
            autocorrelation = float(np.dot(first, second) / denominator)
    return {
        "estimable": True,
        "sample_interval_phases": interval,
        "peak_frequency_cycles_per_sample": float(frequencies[peak_index]),
        "candidate_period_samples": period_samples,
        "candidate_period_phases": period_samples * interval,
        "peak_power_fraction_of_slow_band": float(power[peak_index] / power[valid].sum()),
        "autocorrelation_at_nearest_period": autocorrelation,
        "edge_period_warning": period_samples >= len(detrended) / 2,
    }


def _aicc(squared_error: float, observations: int, parameters: int) -> float | None:
    if observations <= parameters + 1:
        return None
    mse = max(squared_error / observations, np.finfo(np.float64).tiny)
    aic = observations * math.log(mse) + 2 * parameters
    return aic + 2 * parameters * (parameters + 1) / (observations - parameters - 1)


def fit_post_peak_models(
    ticks: list[float | None], values: list[float | None]
) -> dict[str, Any]:
    """Compare frozen plateau/relaxation models on the post-peak finite series."""
    indices, x, y = _paired_finite(ticks, values)
    if len(indices) < 8:
        return {"estimable": False, "reason": "too_few_finite_samples"}
    peak_position = int(np.argmax(y))
    x = x[peak_position:]
    y = y[peak_position:]
    if len(y) < 8:
        return {"estimable": False, "reason": "too_few_post_peak_samples"}
    x = x - x[0]
    span = max(float(x[-1]), 1.0)
    scale = max(float(np.ptp(y)), abs(float(y[0])), 1.0)

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        curve_fit = None

    models: list[dict[str, Any]] = []

    def record(name: str, parameters: list[float], predicted: np.ndarray) -> None:
        residual = y - predicted
        squared_error = float(np.dot(residual, residual))
        models.append(
            {
                "model": name,
                "parameters": parameters,
                "parameter_count": len(parameters),
                "squared_error": squared_error,
                "aicc": _aicc(squared_error, len(y), len(parameters)),
            }
        )

    plateau = float(y.mean())
    record("constant_plateau", [plateau], np.full_like(y, plateau))
    linear = np.polyfit(x, y, 1)
    record("linear_return", [float(linear[1]), float(linear[0])], np.polyval(linear, x))

    if curve_fit is not None:
        specifications: list[
            tuple[
                str,
                Callable[..., np.ndarray],
                list[float],
                tuple[list[float], list[float]],
            ]
        ] = [
            (
                "exponential_relaxation",
                lambda t, c, a, tau: c + a * np.exp(-t / tau),
                [float(y[-1]), float(y[0] - y[-1]), span / 3],
                ([0.0, -10 * scale, max(span / 10_000, 1e-9)], [10 * scale, 10 * scale, 100 * span]),
            ),
            (
                "power_law_relaxation",
                lambda t, c, a, tau, exponent: c + a * (1.0 + t / tau) ** (-exponent),
                [float(y[-1]), float(y[0] - y[-1]), span / 3, 1.0],
                (
                    [0.0, -10 * scale, max(span / 10_000, 1e-9), 0.01],
                    [10 * scale, 10 * scale, 100 * span, 10.0],
                ),
            ),
            (
                "damped_sinusoid",
                lambda t, c, a, b, tau, omega: c
                + np.exp(-t / tau) * (a * np.cos(omega * t) + b * np.sin(omega * t)),
                [float(y[-1]), float(y[0] - y[-1]), 0.0, span, 2 * math.pi / span],
                (
                    [0.0, -10 * scale, -10 * scale, max(span / 10_000, 1e-9), 2 * math.pi / (4 * span)],
                    [10 * scale, 10 * scale, 10 * scale, 100 * span, math.pi / max(float(np.median(np.diff(x))), 1.0)],
                ),
            ),
        ]
        for name, function, initial, bounds in specifications:
            try:
                parameters, _ = curve_fit(
                    function,
                    x,
                    y,
                    p0=initial,
                    bounds=bounds,
                    maxfev=50_000,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            record(name, [float(value) for value in parameters], function(x, *parameters))

    finite_models = [model for model in models if model["aicc"] is not None]
    if not finite_models:
        return {"estimable": False, "reason": "no_model_with_finite_aicc", "models": models}
    finite_models.sort(key=lambda model: float(model["aicc"]))
    best = finite_models[0]
    runner_up = finite_models[1] if len(finite_models) > 1 else None
    delta = float(runner_up["aicc"] - best["aicc"]) if runner_up else None
    return {
        "estimable": True,
        "post_peak_sample_count": len(y),
        "peak_tick": float(x[0] + ticks[indices[peak_position]]),
        "models": models,
        "best_model": best["model"],
        "delta_aicc_to_runner_up": delta,
        "preferred_under_preregistered_delta4": delta is not None and delta >= 4.0,
    }


def analyze_macro_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered macro diagnostics to one retained trajectory record."""
    series = extract_macro_series(record["samples"])
    scales = series["finite_domain_scale"]
    finite_indices = [index for index, value in enumerate(scales) if value is not None]
    if not finite_indices:
        raise ValueError("trajectory has no estimable finite-domain scale")
    peak_index = max(finite_indices, key=lambda index: float(scales[index]))
    initial = float(scales[finite_indices[0]])
    peak = float(scales[peak_index])
    final = float(scales[finite_indices[-1]])
    correlations = {}
    for name, first, second in (
        ("occupancy_vs_domain_scale", series["occupied_fraction"], scales),
        ("reaction_vs_domain_scale", series["reaction_density"], scales),
        ("interface_vs_occupancy", series["interface_length"], series["occupied_fraction"]),
    ):
        correlations[name] = {
            "raw": lagged_correlation(first, second),
            "linearly_detrended": lagged_correlation(first, second, detrend=True),
        }
    return {
        "run_id": record.get("run_id"),
        "stage_id": record.get("stage_id"),
        "condition_id": record.get("condition_id"),
        "seed_index": record.get("seed_index"),
        "initial_seed": record.get("initial_seed"),
        "initial_scale": initial,
        "peak_scale": peak,
        "peak_tick": int(float(series["tick"][peak_index])),
        "final_scale": final,
        "peak_scale_ratio": peak / initial,
        "final_scale_ratio": final / initial,
        "final_to_peak_ratio": final / peak,
        "post_peak_models": fit_post_peak_models(series["tick"], scales),
        "domain_scale_spectrum": spectral_summary(series["tick"], scales),
        "occupancy_spectrum": spectral_summary(series["tick"], series["occupied_fraction"]),
        "cross_correlations": correlations,
        "series": series,
    }
