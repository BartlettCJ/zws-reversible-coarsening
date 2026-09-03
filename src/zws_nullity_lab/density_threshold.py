from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
import torch


BROAD_NAMESPACE = "ZWSLab-density-threshold-broad-v1:2026082601"
FIT_NAMESPACE = "ZWSLab-density-threshold-fit-v1:2026082601"
BROAD_TARGETS = (
    0.0646,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.32,
)
BROAD_INVENTORY_SEEDS = (
    2019049721,
    1098257983,
    1091929321,
    522236404,
    481960949,
    805102133,
    1866549897,
    1150730812,
    1073990928,
    1048271580,
    1253183600,
    2006309794,
)


def deterministic_seed(label: str, *, namespace: str = BROAD_NAMESPACE) -> int:
    payload = f"{namespace}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 31) - 1)


def density_arm_id(target_occupancy: float) -> str:
    scaled = int(math.floor(float(target_occupancy) * 10000.0 + 0.5))
    if scaled <= 0 or scaled >= 10000:
        raise ValueError("density target must round strictly inside (0,1)")
    return f"density_p{scaled:04d}"


def exact_density_inventory(height: int, width: int, target_occupancy: float) -> dict[str, Any]:
    if height <= 0 or width <= 0:
        raise ValueError("world dimensions must be positive")
    target = float(target_occupancy)
    if not 0.0 < target < 1.0:
        raise ValueError("target occupancy must be in (0,1)")
    population = height * width
    unit_count = int(math.floor(target * population / 8.0 + 0.5))
    occupancy_mass = 8 * unit_count
    if unit_count <= 0 or occupancy_mass >= population:
        raise ValueError("rounded inventory must contain both occupied and empty sites")
    positive = 3 * unit_count
    negative = 3 * unit_count
    bound = 2 * unit_count
    empty = population - occupancy_mass
    return {
        "target_occupancy": target,
        "rounding_rule": "u=floor(target*N/8+0.5); counts=(N-8u,3u,3u,2u)",
        "composition_unit_count": unit_count,
        "state_counts": [empty, positive, negative, bound],
        "empty_count": empty,
        "positive_count": positive,
        "negative_count": negative,
        "bound_count": bound,
        "occupancy_mass": occupancy_mass,
        "realized_occupancy": occupancy_mass / population,
        "occupancy_rounding_error": occupancy_mass / population - target,
        "total_charge": 0,
    }


def exact_inventory_homogeneous(
    height: int,
    width: int,
    *,
    state_counts: Iterable[int],
    arrangement_seed: int,
) -> torch.Tensor:
    counts = [int(value) for value in state_counts]
    population = height * width
    if len(counts) != 4 or any(value < 0 for value in counts):
        raise ValueError("state_counts must contain four nonnegative integers")
    if sum(counts) != population:
        raise ValueError("state_counts do not sum to the world population")
    if counts[1] != counts[2]:
        raise ValueError("exact homogeneous inventory must have positive == negative")
    flat = torch.zeros(population, dtype=torch.uint8)
    positive_end = counts[1]
    negative_end = positive_end + counts[2]
    bound_end = negative_end + counts[3]
    flat[:positive_end] = 1
    flat[positive_end:negative_end] = 2
    flat[negative_end:bound_end] = 3
    generator = torch.Generator(device="cpu").manual_seed(int(arrangement_seed))
    permutation = torch.randperm(population, generator=generator)
    return flat[permutation].reshape(height, width)


def assert_nested_occupied_support(states: Iterable[torch.Tensor]) -> None:
    previous: torch.Tensor | None = None
    previous_mass = -1
    for state in states:
        occupied = state.detach().cpu() != 0
        mass = int(occupied.sum().item())
        if mass < previous_mass:
            raise AssertionError("states are not ordered by nondecreasing occupancy")
        if previous is not None and bool((previous & ~occupied).any().item()):
            raise AssertionError("occupied support is not nested across density")
        previous = occupied
        previous_mass = mass


def _ordered_summaries(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(summaries, key=lambda row: float(row["realized_occupancy"]))
    if not rows:
        raise ValueError("at least one density summary is required")
    densities = [float(row["realized_occupancy"]) for row in rows]
    if len(densities) != len(set(densities)):
        raise ValueError("density summaries contain duplicate realized occupancies")
    for row in rows:
        fraction = float(row["strong_domain_pass_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("pass fractions must lie in [0,1]")
    return rows


def severe_reversal(summaries: Iterable[dict[str, Any]]) -> bool:
    rows = _ordered_summaries(summaries)
    fractions = [float(row["strong_domain_pass_fraction"]) for row in rows]
    for earlier in range(len(fractions)):
        for later in range(earlier + 1, len(fractions)):
            if fractions[earlier] - fractions[later] >= 0.50:
                return True
            if fractions[earlier] >= 0.75 and fractions[later] <= 0.50:
                return True
    return False


def primary_transition_bracket(
    summaries: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    rows = _ordered_summaries(summaries)
    candidates = []
    for lower_index, lower in enumerate(rows):
        if float(lower["strong_domain_pass_fraction"]) > 0.25:
            continue
        for upper_index in range(lower_index + 1, len(rows)):
            upper = rows[upper_index]
            if float(upper["strong_domain_pass_fraction"]) < 0.75:
                continue
            width = float(upper["realized_occupancy"]) - float(lower["realized_occupancy"])
            candidates.append((width, lower_index, upper_index, lower, upper))
    if not candidates:
        return None
    width, lower_index, upper_index, lower, upper = min(
        candidates, key=lambda value: (value[0], value[1], value[2])
    )
    return {
        "lower_index": lower_index,
        "upper_index": upper_index,
        "lower_target_occupancy": float(lower["target_occupancy"]),
        "upper_target_occupancy": float(upper["target_occupancy"]),
        "lower_realized_occupancy": float(lower["realized_occupancy"]),
        "upper_realized_occupancy": float(upper["realized_occupancy"]),
        "lower_pass_fraction": float(lower["strong_domain_pass_fraction"]),
        "upper_pass_fraction": float(upper["strong_domain_pass_fraction"]),
        "width": width,
    }


def classify_density_response(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = _ordered_summaries(summaries)
    fractions = [float(row["strong_domain_pass_fraction"]) for row in rows]
    reversal = severe_reversal(rows)
    bracket = primary_transition_bracket(rows)
    if reversal:
        classification = "E_NON_MONOTONIC_MULTIPLE_REGIMES"
    elif max(fractions) - min(fractions) <= 0.25 or max(fractions) <= 0.25:
        classification = "A_NO_TRANSITION_IN_TESTED_RANGE"
    elif bracket is not None and float(bracket["width"]) <= 0.03:
        classification = "D_VERY_SHARP_FINITE_SIZE_TRANSITION_REGION"
    elif bracket is not None and float(bracket["width"]) <= 0.06:
        classification = "C_SHARP_FINITE_SIZE_TRANSITION_REGION"
    elif bracket is not None:
        classification = "B_GRADUAL_CROSSOVER"
    else:
        classification = "F_MIXED_INCONCLUSIVE"
    return {
        "classification": classification,
        "severe_reversal": reversal,
        "primary_25_to_75_bracket": bracket,
        "minimum_pass_fraction": min(fractions),
        "maximum_pass_fraction": max(fractions),
    }


def refinement_plan(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = _ordered_summaries(summaries)
    bracket = primary_transition_bracket(rows)
    selection_basis = "primary_25_to_75_bracket"
    if bracket is None:
        increases = [
            (
                float(rows[index + 1]["strong_domain_pass_fraction"])
                - float(rows[index]["strong_domain_pass_fraction"]),
                index,
            )
            for index in range(len(rows) - 1)
        ]
        rise, index = max(increases, key=lambda value: (value[0], -value[1]))
        if rise < 0.25:
            return {
                "run_refinement": False,
                "selection_basis": "no_eligible_rise",
                "largest_adjacent_increase": rise,
                "new_target_occupancies": [],
            }
        lower = rows[index]
        upper = rows[index + 1]
        bracket = {
            "lower_target_occupancy": float(lower["target_occupancy"]),
            "upper_target_occupancy": float(upper["target_occupancy"]),
            "lower_realized_occupancy": float(lower["realized_occupancy"]),
            "upper_realized_occupancy": float(upper["realized_occupancy"]),
            "lower_pass_fraction": float(lower["strong_domain_pass_fraction"]),
            "upper_pass_fraction": float(upper["strong_domain_pass_fraction"]),
            "width": float(upper["realized_occupancy"])
            - float(lower["realized_occupancy"]),
        }
        selection_basis = "largest_adjacent_increase"
    width = float(bracket["width"])
    spacing = 0.005 if width <= 0.03 else 0.01
    lower_target = float(bracket["lower_target_occupancy"])
    upper_target = float(bracket["upper_target_occupancy"])
    broad_targets = {round(float(row["target_occupancy"]), 6) for row in rows}
    targets = []
    step = 1
    while True:
        value = round(lower_target + step * spacing, 6)
        if value >= upper_target - 1e-9:
            break
        if value not in broad_targets:
            targets.append(value)
        step += 1
    return {
        "run_refinement": bool(targets),
        "selection_basis": selection_basis,
        "selected_interval": bracket,
        "spacing": spacing,
        "new_target_occupancies": targets,
        "seed_count_per_density": 16,
    }


def logistic_fit_gate(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = _ordered_summaries(summaries)
    densities = np.asarray([float(row["realized_occupancy"]) for row in rows])
    fractions = np.asarray([float(row["strong_domain_pass_fraction"]) for row in rows])
    has_pass_and_failure = bool(np.any(fractions > 0.0) and np.any(fractions < 1.0))
    reversal = severe_reversal(rows)
    rho_value = spearmanr(densities, fractions).statistic
    rho = float(rho_value) if np.isfinite(rho_value) else None
    passed = bool(has_pass_and_failure and not reversal and rho is not None and rho >= 0.8)
    return {
        "passed": passed,
        "has_pass_and_failure": has_pass_and_failure,
        "severe_reversal": reversal,
        "spearman_density_vs_pass_fraction": rho,
        "spearman_min": 0.8,
    }


def _fit_logistic_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    x = (np.asarray([float(row["realized_occupancy"]) for row in rows]) - 0.20) / 0.10
    y = np.asarray([float(bool(row["strong_domain_pass"])) for row in rows])

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = parameters
        eta = intercept + slope * x
        nll = float(np.logaddexp(0.0, eta).sum() - np.dot(y, eta))
        penalty = 0.5 * ((intercept / 10.0) ** 2 + (slope / 10.0) ** 2)
        return nll + penalty

    result = minimize(
        objective,
        x0=np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=((-100.0, 100.0), (1e-6, 100.0)),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"logistic MAP fit failed: {result.message}")
    intercept, slope = (float(value) for value in result.x)
    output = {
        "intercept_centered_scaled": intercept,
        "slope_centered_scaled": slope,
        "negative_log_posterior": float(result.fun),
    }
    for probability in (0.25, 0.50, 0.75):
        logit = math.log(probability / (1.0 - probability))
        output[f"occupancy_p{int(probability * 100):02d}"] = 0.20 + 0.10 * (
            (logit - intercept) / slope
        )
    output["width_p25_to_p75"] = output["occupancy_p75"] - output["occupancy_p25"]
    return output


def fit_logistic_onset(
    rows: Iterable[dict[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("logistic fit requires trajectory rows")
    point = _fit_logistic_map(records)
    seed = (
        deterministic_seed("blocked-bootstrap", namespace=FIT_NAMESPACE)
        if bootstrap_seed is None
        else int(bootstrap_seed)
    )
    rng = np.random.default_rng(seed)
    by_stage: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in records:
        by_stage.setdefault(str(row["stage_id"]), {}).setdefault(
            int(row["seed_index"]), []
        ).append(row)
    bootstrap_values: dict[str, list[float]] = {
        "occupancy_p25": [],
        "occupancy_p50": [],
        "occupancy_p75": [],
        "width_p25_to_p75": [],
    }
    successful = 0
    for _ in range(int(bootstrap_replicates)):
        sample: list[dict[str, Any]] = []
        for seeds in by_stage.values():
            seed_ids = sorted(seeds)
            selected = rng.choice(seed_ids, size=len(seed_ids), replace=True)
            for selected_seed in selected:
                sample.extend(seeds[int(selected_seed)])
        try:
            fitted = _fit_logistic_map(sample)
        except RuntimeError:
            continue
        if not all(np.isfinite(fitted[key]) for key in bootstrap_values):
            continue
        successful += 1
        for key in bootstrap_values:
            bootstrap_values[key].append(float(fitted[key]))
    minimum_success = math.ceil(0.9 * int(bootstrap_replicates))
    intervals_valid = successful >= minimum_success
    intervals = {}
    if intervals_valid:
        for key, values in bootstrap_values.items():
            lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
            intervals[key] = {"lower_95": float(lower), "upper_95": float(upper)}
    return {
        "schema": "zwslab-density-threshold-logistic-fit-v1",
        "model": "binomial logistic MAP; density centered 0.20 scaled 0.10",
        "weak_normal_prior_sd": 10.0,
        "positive_slope_constrained": True,
        "point_estimate": point,
        "bootstrap": {
            "method": "blocked nonparametric bootstrap by seed within stage",
            "seed": seed,
            "requested_replicates": int(bootstrap_replicates),
            "successful_replicates": successful,
            "minimum_successful_replicates": minimum_success,
            "intervals_valid": intervals_valid,
            "percentile_intervals": intervals,
        },
    }


def predicted_probability(density: float, fit: dict[str, Any]) -> float:
    point = fit["point_estimate"]
    x = (float(density) - 0.20) / 0.10
    eta = float(point["intercept_centered_scaled"]) + float(
        point["slope_centered_scaled"]
    ) * x
    return float(expit(eta))


def finite_size_density_targets(fit: dict[str, Any]) -> list[float]:
    """Return deterministic four-decimal p25/p50/p75 targets for the gated size check."""
    point = fit.get("point_estimate", {})
    targets = []
    for probability in (25, 50, 75):
        value = float(point[f"occupancy_p{probability}"])
        rounded = math.floor(value * 10000.0 + 0.5) / 10000.0
        if not 0.0 < rounded < 1.0:
            raise ValueError("fitted finite-size target must lie strictly inside (0,1)")
        targets.append(rounded)
    if targets != sorted(set(targets)):
        raise ValueError("fitted p25/p50/p75 targets collapse after four-decimal freezing")
    return targets
