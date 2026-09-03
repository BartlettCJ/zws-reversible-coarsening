from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Any, Iterable


def _number(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0.0:
        return None
    return value / baseline


def _state_fractions(sample: dict[str, Any]) -> list[float]:
    populations = [int(value) for value in sample["state_populations"]]
    total = sum(populations)
    if total <= 0:
        raise ValueError("state population is empty")
    return [value / total for value in populations]


def trajectory_coordinates(trajectory: dict[str, Any]) -> dict[str, Any]:
    samples = sorted(trajectory["samples"], key=lambda sample: int(sample["tick"]))
    if not samples or int(samples[0]["tick"]) != 0:
        raise ValueError(f"trajectory lacks tick-zero sample: {trajectory.get('run_id')}")
    scales = [
        _number(sample["empty_domains"]["finite_area_weighted_linear_scale"])
        for sample in samples
    ]
    finite = [(index, value) for index, value in enumerate(scales) if value is not None]
    if not finite:
        raise ValueError(f"trajectory has no finite-domain scale: {trajectory.get('run_id')}")
    peak_index, peak_scale = max(finite, key=lambda item: (item[1], -item[0]))
    initial_scale = scales[0]
    final_scale = scales[-1]
    if initial_scale is None or initial_scale <= 0.0 or final_scale is None:
        raise ValueError(
            f"trajectory lacks a usable initial/final finite-domain scale: "
            f"{trajectory.get('run_id')}"
        )
    peak_ratio = _ratio(peak_scale, initial_scale)
    final_ratio = _ratio(final_scale, initial_scale)
    violation_counts = {
        str(key): int(value)
        for key, value in trajectory.get("invariant_violation_counts", {}).items()
    }
    required_invariants = [str(value) for value in trajectory.get("required_invariants", [])]
    if "Q" not in required_invariants:
        raise ValueError(f"trajectory lacks declared Q invariant: {trajectory.get('run_id')}")
    missing_invariants = sorted(set(required_invariants) - set(violation_counts))
    if missing_invariants:
        raise ValueError(
            f"trajectory lacks required invariant audit counts: "
            f"{trajectory.get('run_id')} {missing_invariants}"
        )
    invariant_pass = not any(violation_counts.values())
    strong = (
        invariant_pass
        and peak_ratio is not None
        and final_ratio is not None
        and peak_ratio >= 8.0
        and final_ratio >= 2.0
    )
    initial_state = _state_fractions(samples[0])
    final_state = _state_fractions(samples[-1])
    reaction_values = [
        value
        for value in (_number(sample.get("reaction_density_one_phase")) for sample in samples)
        if value is not None
    ]
    activity_values = [
        value
        for value in (_number(sample.get("activity_one_phase")) for sample in samples)
        if value is not None
    ]
    scale_values = [value for value in scales if value is not None]
    scale_span_ratio = (
        (max(scale_values) - min(scale_values)) / initial_scale
        if initial_scale is not None and initial_scale > 0
        else None
    )
    final_structure = samples[-1]["structure_factor"]
    final_dispersion = _number(
        samples[-1]["coarse_density"].get("64", {}).get("dispersion_ratio_analytic")
    )
    final_peak_to_median = _number(final_structure.get("peak_to_radial_median"))
    stasis_flag = (
        scale_span_ratio is not None
        and scale_span_ratio <= 0.05
        and abs(float(samples[-1]["occupied_fraction"]) - float(samples[0]["occupied_fraction"]))
        <= 1e-6
    )
    collapse_flag = (
        float(samples[-1]["occupied_fraction"]) <= 0.001
        or float(samples[-1]["occupied_fraction"]) >= 0.999
        or max(final_state) >= 0.999
    )
    noise_like_flag = (
        final_ratio is not None
        and final_ratio < 2.0
        and (final_dispersion is None or final_dispersion < 2.0)
        and (final_peak_to_median is None or final_peak_to_median < 3.0)
        and not stasis_flag
        and not collapse_flag
    )
    row: dict[str, Any] = {
        "run_id": trajectory["run_id"],
        "condition_id": trajectory["condition_id"],
        "seed_index": int(trajectory["seed_index"]),
        "initial_seed": int(trajectory["initial_seed"]),
        "conservation": trajectory.get("conservation", "Q_only"),
        "invariant_pass": invariant_pass,
        "strong_domain_organization": strong,
        "initial_finite_domain_scale": initial_scale,
        "peak_finite_domain_scale": peak_scale,
        "peak_scale_ratio": peak_ratio,
        "peak_tick": int(samples[peak_index]["tick"]),
        "final_finite_domain_scale": final_scale,
        "final_scale_ratio": final_ratio,
        "final_to_peak_scale_ratio": _ratio(final_scale, peak_scale),
        "initial_occupied_fraction": float(samples[0]["occupied_fraction"]),
        "peak_occupied_fraction": float(samples[peak_index]["occupied_fraction"]),
        "final_occupied_fraction": float(samples[-1]["occupied_fraction"]),
        "occupied_fraction_change": (
            float(samples[-1]["occupied_fraction"])
            - float(samples[0]["occupied_fraction"])
        ),
        "mean_reaction_density": sum(reaction_values) / len(reaction_values)
        if reaction_values
        else None,
        "max_reaction_density": max(reaction_values) if reaction_values else None,
        "mean_activity": sum(activity_values) / len(activity_values)
        if activity_values
        else None,
        "final_boundary_density": _number(samples[-1].get("boundary_density")),
        "final_empty_domain_count": int(samples[-1]["empty_domains"]["component_count"]),
        "final_largest_empty_domain_fraction": _number(
            samples[-1]["empty_domains"]["largest_area_fraction_of_empty"]
        ),
        "final_structure_wavelength": _number(
            final_structure.get("characteristic_wavelength_cells")
        ),
        "final_structure_peak_to_median": final_peak_to_median,
        "final_dispersion_block64": final_dispersion,
        "stasis_flag": stasis_flag,
        "collapse_flag": collapse_flag,
        "noise_like_flag": noise_like_flag,
    }
    for code, label in enumerate(("empty", "positive", "negative", "bound")):
        row[f"initial_{label}_fraction"] = initial_state[code]
        row[f"final_{label}_fraction"] = final_state[code]
    for lag in (4, 8, 64, 128, 256, 512):
        row[f"final_exact_recurrence_lag{lag}"] = _number(
            samples[-1].get("exact_state_recurrence", {}).get(str(lag))
        )
        row[f"final_coarse_recurrence_lag{lag}"] = _number(
            samples[-1]
            .get("coarse_occupancy_correlation_block64", {})
            .get(str(lag))
        )
    return row


def condition_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    for condition_id, members in grouped.items():
        peak = [float(row["peak_scale_ratio"]) for row in members]
        final = [float(row["final_scale_ratio"]) for row in members]
        output.append(
            {
                "condition_id": condition_id,
                "trajectory_count": len(members),
                "invariant_pass_count": sum(bool(row["invariant_pass"]) for row in members),
                "strong_domain_count": sum(
                    bool(row["strong_domain_organization"]) for row in members
                ),
                "median_peak_scale_ratio": median(peak),
                "median_final_scale_ratio": median(final),
                "median_peak_tick": median(int(row["peak_tick"]) for row in members),
                "median_occupied_fraction_change": median(
                    float(row["occupied_fraction_change"]) for row in members
                ),
                "median_mean_reaction_density": median(
                    float(row["mean_reaction_density"])
                    for row in members
                    if row["mean_reaction_density"] is not None
                )
                if any(row["mean_reaction_density"] is not None for row in members)
                else None,
                "collapse_count": sum(bool(row["collapse_flag"]) for row in members),
                "stasis_count": sum(bool(row["stasis_flag"]) for row in members),
                "noise_like_count": sum(bool(row["noise_like_flag"]) for row in members),
            }
        )
    return sorted(output, key=lambda row: str(row["condition_id"]))


def double_conservation_classification(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    winner = grouped.get("winner", [])
    sibling = grouped.get("qm_sibling", [])
    if len(winner) != 8 or len(sibling) != 8:
        raise ValueError("double-conservation classification requires 8 winner and 8 sibling rows")
    if {int(row["initial_seed"]) for row in winner} != {
        int(row["initial_seed"]) for row in sibling
    }:
        raise ValueError("winner and sibling seed blocks are not paired")
    sibling_passes = sum(bool(row["strong_domain_organization"]) for row in sibling)
    sibling_peak = median(float(row["peak_scale_ratio"]) for row in sibling)
    sibling_final = median(float(row["final_scale_ratio"]) for row in sibling)
    winner_peak = median(float(row["peak_scale_ratio"]) for row in winner)
    winner_final = median(float(row["final_scale_ratio"]) for row in winner)
    comparable = (
        sibling_passes >= 6
        and sibling_peak >= 0.5 * winner_peak
        and sibling_final >= 0.5 * winner_final
    )
    if comparable:
        classification = "survives_comparably"
    elif sibling_passes <= 2 and sibling_peak < 2.0:
        classification = "disappears"
    elif sibling_passes >= 3:
        classification = "weakens_but_survives"
    else:
        classification = "mixed"
    return {
        "schema": "zwslab-double-conservation-classification-v1",
        "measurement": {
            "sibling_strong_domain_count": sibling_passes,
            "sibling_trajectory_count": len(sibling),
            "sibling_median_peak_scale_ratio": sibling_peak,
            "sibling_median_final_scale_ratio": sibling_final,
            "winner_median_peak_scale_ratio": winner_peak,
            "winner_median_final_scale_ratio": winner_final,
            "sibling_to_winner_peak_median_ratio": sibling_peak / winner_peak,
            "sibling_to_winner_final_median_ratio": sibling_final / winner_final,
        },
        "formal_inference": {
            "classification": classification,
            "survives_comparably_criterion_pass": comparable,
        },
        "potentially_richer": {
            "status": "not_classified_by_primary_rule",
            "note": (
                "This optional interpretation flag requires separate morphology-coordinate "
                "review and is never inferred from domain scale alone."
            ),
        },
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def prevalence_classification(
    rows: Iterable[dict[str, Any]], conditions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    roles = {
        str(condition["condition_id"]): str(condition.get("family_role", "unknown"))
        for condition in conditions
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    sample_ids = sorted(condition_id for condition_id, role in roles.items() if role == "sample")
    if len(sample_ids) != 256:
        raise ValueError(
            f"prevalence classification requires 256 sampled rules, observed {len(sample_ids)}"
        )
    per_rule: list[dict[str, Any]] = []
    for condition_id in sample_ids:
        members = grouped.get(condition_id, [])
        if len(members) != 4:
            raise ValueError(f"prevalence rule does not have four trajectories: {condition_id}")
        passing = sum(bool(row["strong_domain_organization"]) for row in members)
        per_rule.append(
            {
                "condition_id": condition_id,
                "passing_seed_count": passing,
                "winner_like": passing >= 3,
                "median_peak_scale_ratio": median(
                    float(row["peak_scale_ratio"]) for row in members
                ),
                "median_final_scale_ratio": median(
                    float(row["final_scale_ratio"]) for row in members
                ),
            }
        )
    successes = sum(bool(row["winner_like"]) for row in per_rule)
    fraction = successes / len(per_rule)
    lower, upper = wilson_interval(successes, len(per_rule))
    if fraction >= 0.20:
        label = "common"
    elif fraction >= 0.05:
        label = "moderately_unusual"
    else:
        label = "rare_tail"
    return {
        "schema": "zwslab-prevalence-classification-v1",
        "measurement": {
            "sampled_rule_count": len(per_rule),
            "winner_like_rule_count": successes,
            "winner_like_fraction": fraction,
            "wilson_95_interval": [lower, upper],
        },
        "formal_inference": {"prevalence_label": label},
        "distinct_cluster": {
            "status": "requires_separate_stability_analysis",
            "note": "The prevalence threshold cannot establish a distinct behavioural cluster.",
        },
        "per_rule": per_rule,
    }
