from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .atomic import sha256_file
from .homogeneous_campaign import TRAJECTORY_SCHEMA
from .homogeneous_ic import (
    ARM_GRADIENT,
    ARM_HIGH_DENSITY,
    ARM_HOMOGENEOUS,
    ARM_LOCALIZED,
    low_k_growth_ratio,
)


def _manifest_rows(manifest: dict[str, Any], stage_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["run_id"]): row for row in manifest["stages"][stage_id]["planned_runs"]
    }


def load_completed_stage(
    root: str | Path,
    manifest: dict[str, Any],
    manifest_digest: str,
    *,
    stage_id: str,
) -> list[dict[str, Any]]:
    result_root = Path(root).resolve()
    summary_path = result_root / "campaign_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != "zwslab-homogeneous-campaign-summary-v1"
        or summary.get("status") != "complete"
        or summary.get("stage_id") != stage_id
        or summary.get("manifest_sha256") != manifest_digest
    ):
        raise RuntimeError("homogeneous campaign is not complete for the frozen stage")

    expected = _manifest_rows(manifest, stage_id)
    trajectories: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for arm_id in manifest["stages"][stage_id]["arm_ids"]:
        completion_path = result_root / "arms" / arm_id / "completion.json"
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("status") != "complete"
            or completion.get("stage_id") != stage_id
            or completion.get("arm_id") != arm_id
            or completion.get("manifest_sha256") != manifest_digest
        ):
            raise RuntimeError(f"arm completion is not frozen and complete: {arm_id}")
        for entry in completion["trajectories"]:
            path = Path(str(entry["path"])).resolve()
            if not path.is_relative_to(result_root):
                raise RuntimeError(f"trajectory escaped result root: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"trajectory digest mismatch: {path}")
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            run_id = str(trajectory["run_id"])
            if trajectory.get("schema") != TRAJECTORY_SCHEMA:
                raise ValueError(f"unexpected trajectory schema: {path}")
            if run_id not in expected or run_id in observed_ids:
                raise RuntimeError(f"unexpected or duplicate run id: {run_id}")
            row = expected[run_id]
            if (
                trajectory.get("status") != "complete"
                or trajectory.get("manifest_sha256") != manifest_digest
                or trajectory.get("stage_id") != stage_id
                or trajectory.get("arm_id") != row["arm_id"]
                or trajectory.get("initial_state_sha256_uint8")
                != row["initial_state_sha256_uint8"]
                or trajectory.get("initial_inventory", {}).get("state_counts")
                != row["state_counts"]
            ):
                raise RuntimeError(f"trajectory differs from frozen input row: {run_id}")
            samples = list(trajectory["samples"])
            if not samples or any(int(sample["total_charge"]) != 0 for sample in samples):
                raise RuntimeError(f"trajectory lacks exact Q=0 samples: {run_id}")
            trajectories.append(trajectory)
            observed_ids.add(run_id)
    if observed_ids != set(expected):
        missing = sorted(set(expected) - observed_ids)
        raise RuntimeError(f"completed stage lacks planned trajectories: {missing}")
    return sorted(trajectories, key=lambda item: str(item["run_id"]))


def _value_or_zero(value: float | int | None) -> float:
    return 0.0 if value is None else float(value)


def reduce_trajectory(
    trajectory: dict[str, Any],
    *,
    finite_rule: dict[str, Any],
    low_k_rule: dict[str, Any],
    regular_rule: dict[str, Any],
) -> dict[str, Any]:
    samples = sorted(trajectory["samples"], key=lambda item: int(item["tick"]))
    if int(samples[0]["tick"]) != 0 or int(samples[-1]["tick"]) != int(
        trajectory["phases"]
    ):
        raise ValueError(f"trajectory sample range is incomplete: {trajectory['run_id']}")
    initial = samples[0]
    final = samples[-1]
    initial_scale_raw = initial["empty_domains"]["finite_area_weighted_linear_scale"]
    scale_floor = float(finite_rule["initial_scale_floor_cells"])
    initial_scale_denominator = max(_value_or_zero(initial_scale_raw), scale_floor)
    scale_ratios = [
        _value_or_zero(sample["empty_domains"]["finite_area_weighted_linear_scale"])
        / initial_scale_denominator
        for sample in samples
    ]
    peak_scale_index = max(range(len(samples)), key=lambda index: scale_ratios[index])
    peak_scale_sample = samples[peak_scale_index]
    peak_scale_ratio = scale_ratios[peak_scale_index]
    final_scale_ratio = scale_ratios[-1]

    initial_finite_largest = int(initial["empty_domains"].get("finite_largest_area", 0))
    finite_largest = [int(sample["empty_domains"].get("finite_largest_area", 0)) for sample in samples]
    peak_finite_largest = max(finite_largest)
    area_threshold = max(
        int(finite_rule["peak_finite_largest_area_absolute_min"]),
        int(
            float(finite_rule["peak_finite_largest_area_ratio_min"])
            * max(initial_finite_largest, 1)
        ),
    )
    exact_q = all(int(sample["total_charge"]) == 0 for sample in samples)
    strong_pass = bool(
        peak_scale_ratio >= float(finite_rule["peak_scale_ratio_min"])
        and final_scale_ratio >= float(finite_rule["final_scale_ratio_min"])
        and peak_finite_largest >= area_threshold
        and exact_q
    )

    low_k_coordinates = [
        low_k_growth_ratio(
            initial["low_wavenumber"],
            sample["low_wavenumber"],
            band=str(low_k_rule["band"]),
            floor=float(low_k_rule["numerical_floor"]),
        )
        for sample in samples
    ]
    low_k_peak_mean_index = max(
        range(len(samples)),
        key=lambda index: low_k_coordinates[index]["mean_power_ratio"],
    )
    low_k_peak_mean = low_k_coordinates[low_k_peak_mean_index]["mean_power_ratio"]
    low_k_final_mean = low_k_coordinates[-1]["mean_power_ratio"]
    low_k_peak_fraction = max(value["spectral_fraction_ratio"] for value in low_k_coordinates)
    low_k_pass = bool(
        low_k_peak_mean >= float(low_k_rule["peak_mean_power_ratio_min"])
        and low_k_final_mean >= float(low_k_rule["final_mean_power_ratio_min"])
        and low_k_peak_fraction >= float(low_k_rule["peak_spectral_fraction_ratio_min"])
    )

    final_spectral = final["low_wavenumber"]
    strongest_fraction = final_spectral["strongest_mode_pair_fraction_upper_bound"]
    radial_fraction = final_spectral["radial_peak_shell_fraction"]
    regular_signature = bool(
        strongest_fraction is not None
        and radial_fraction is not None
        and float(strongest_fraction)
        >= float(regular_rule["strongest_mode_pair_fraction_min"])
        and float(radial_fraction) >= float(regular_rule["radial_peak_shell_fraction_min"])
    )
    final_populations = [int(value) for value in final["state_populations"]]
    population = sum(final_populations)
    reactions = [
        float(sample["reaction_density_one_phase"])
        for sample in samples
        if sample["reaction_density_one_phase"] is not None
    ]
    peak_occupancy = max(float(sample["occupied_fraction"]) for sample in samples)
    dominant_modes = final_spectral["directional_mode_pair_power"]
    dominant_mode = max(dominant_modes, key=lambda name: float(dominant_modes[name]))
    return {
        "run_id": trajectory["run_id"],
        "stage_id": trajectory["stage_id"],
        "arm_id": trajectory["arm_id"],
        "seed_index": int(trajectory["seed_index"]),
        "inventory_seed": int(trajectory["inventory_seed"]),
        "initial_state_sha256_uint8": trajectory["initial_state_sha256_uint8"],
        "initial_occupancy_mass": int(trajectory["initial_inventory"]["occupancy_mass"]),
        "initial_occupied_fraction": float(initial["occupied_fraction"]),
        "final_occupied_fraction": float(final["occupied_fraction"]),
        "peak_occupied_fraction": peak_occupancy,
        "initial_finite_domain_scale": initial_scale_raw,
        "initial_scale_denominator": initial_scale_denominator,
        "peak_finite_domain_scale": peak_scale_sample["empty_domains"][
            "finite_area_weighted_linear_scale"
        ],
        "peak_domain_scale_tick": int(peak_scale_sample["tick"]),
        "peak_domain_scale_ratio": peak_scale_ratio,
        "final_finite_domain_scale": final["empty_domains"][
            "finite_area_weighted_linear_scale"
        ],
        "final_domain_scale_ratio": final_scale_ratio,
        "initial_finite_largest_area": initial_finite_largest,
        "peak_finite_largest_area": peak_finite_largest,
        "finite_largest_area_threshold": area_threshold,
        "strong_finite_domain_pass": strong_pass,
        "exact_q_at_all_samples": exact_q,
        "low_k_peak_mean_power_ratio": low_k_peak_mean,
        "low_k_peak_mean_power_tick": int(samples[low_k_peak_mean_index]["tick"]),
        "low_k_final_mean_power_ratio": low_k_final_mean,
        "low_k_peak_spectral_fraction_ratio": low_k_peak_fraction,
        "systematic_low_k_growth_pass": low_k_pass,
        "final_strongest_mode_pair_fraction_upper_bound": strongest_fraction,
        "final_radial_peak_shell_fraction": radial_fraction,
        "final_simple_regular_spectral_signature": regular_signature,
        "final_dominant_declared_low_mode": dominant_mode,
        "final_largest_empty_domain_fraction": final["empty_domains"][
            "largest_area_fraction_of_empty"
        ],
        "final_empty_domain_component_count": final["empty_domains"]["component_count"],
        "final_empty_domain_area_quantiles_json": json.dumps(
            final["empty_domains"]["area_quantiles"], separators=(",", ":")
        ),
        "final_empty_domain_area_log2_histogram_json": json.dumps(
            final["empty_domains"]["area_log2_histogram"], separators=(",", ":")
        ),
        "final_occupancy_interface_edge_density": final["occupancy_interface_edge_density"],
        "final_exact_state_boundary_density": final["boundary_density"],
        "final_activity_one_phase": final["activity_one_phase"],
        "final_reaction_density_one_phase": final["reaction_density_one_phase"],
        **{
            f"initial_coarse_density_variance_block{block}": initial["coarse_density"][
                str(block)
            ]["observed_variance"]
            for block in (16, 32, 64)
        },
        **{
            f"final_coarse_density_variance_block{block}": final["coarse_density"][
                str(block)
            ]["observed_variance"]
            for block in (16, 32, 64)
        },
        "mean_reaction_density": sum(reactions) / len(reactions) if reactions else None,
        "final_entropy_normalized": float(final["entropy_normalized"]),
        "final_empty_fraction": final_populations[0] / population,
        "final_positive_fraction": final_populations[1] / population,
        "final_negative_fraction": final_populations[2] / population,
        "final_bound_fraction": final_populations[3] / population,
        "final_structure_factor_peak_radius": final["structure_factor"]["peak_radius"],
        "final_structure_factor_peak_power": final["structure_factor"]["peak_power"],
        "final_structure_factor_peak_to_radial_median": final["structure_factor"][
            "peak_to_radial_median"
        ],
        "final_structure_factor_characteristic_wavelength_cells": final[
            "structure_factor"
        ]["characteristic_wavelength_cells"],
        "final_radial_structure_factor_json": json.dumps(
            final["structure_factor"]["radial_profile"], separators=(",", ":")
        ),
        **{
            f"final_recurrence_lag{lag}": final["exact_state_recurrence"].get(str(lag))
            for lag in (4, 8, 16, 32, 64, 128)
        },
        **{
            f"final_spatial_autocorrelation_lag{lag}": final_spectral[
                "spatial_autocorrelation"
            ].get(str(lag))
            for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        },
    }


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return median(values) if values else None


def summarize_arms(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm_id in (ARM_LOCALIZED, ARM_HOMOGENEOUS, ARM_GRADIENT, ARM_HIGH_DENSITY):
        rows = [row for row in coordinates if row["arm_id"] == arm_id]
        if not rows:
            continue
        output.append(
            {
                "arm_id": arm_id,
                "trajectory_count": len(rows),
                "strong_finite_domain_pass_count": sum(
                    bool(row["strong_finite_domain_pass"]) for row in rows
                ),
                "systematic_low_k_growth_pass_count": sum(
                    bool(row["systematic_low_k_growth_pass"]) for row in rows
                ),
                "final_simple_regular_spectral_signature_count": sum(
                    bool(row["final_simple_regular_spectral_signature"]) for row in rows
                ),
                "exact_q_pass_count": sum(bool(row["exact_q_at_all_samples"]) for row in rows),
                "median_initial_occupancy_mass": _median(rows, "initial_occupancy_mass"),
                "median_peak_domain_scale_ratio": _median(rows, "peak_domain_scale_ratio"),
                "median_final_domain_scale_ratio": _median(rows, "final_domain_scale_ratio"),
                "median_low_k_peak_mean_power_ratio": _median(
                    rows, "low_k_peak_mean_power_ratio"
                ),
                "median_low_k_final_mean_power_ratio": _median(
                    rows, "low_k_final_mean_power_ratio"
                ),
                "median_final_occupied_fraction": _median(rows, "final_occupied_fraction"),
                "median_final_entropy_normalized": _median(rows, "final_entropy_normalized"),
                "median_final_radial_peak_shell_fraction": _median(
                    rows, "final_radial_peak_shell_fraction"
                ),
                "median_final_strongest_mode_pair_fraction_upper_bound": _median(
                    rows, "final_strongest_mode_pair_fraction_upper_bound"
                ),
            }
        )
    return output


def classify_stage1(arm_summaries: list[dict[str, Any]]) -> str:
    arms = {row["arm_id"]: row for row in arm_summaries}
    a = arms[ARM_LOCALIZED]
    b = arms[ARM_HOMOGENEOUS]
    c = arms[ARM_GRADIENT]
    pass_a = int(a["strong_finite_domain_pass_count"])
    pass_b = int(b["strong_finite_domain_pass_count"])
    pass_c = int(c["strong_finite_domain_pass_count"])
    low_b = int(b["systematic_low_k_growth_pass_count"])
    peak_b = float(b["median_peak_domain_scale_ratio"])
    final_b = float(b["median_final_domain_scale_ratio"])
    peak_c = float(c["median_peak_domain_scale_ratio"])
    if pass_b >= 12 and peak_b >= 8.0 and final_b >= 4.0 and low_b >= 12:
        return "A_STRONG_SPONTANEOUS_ORGANISATION"
    if pass_b >= 8 or (low_b >= 12 and peak_b >= 2.0):
        return "B_WEAK_CONDITIONAL_INSTABILITY"
    if pass_b < 8 and pass_c >= 12 and peak_c >= 2.0 * peak_b:
        return "C_SEEDED_AMPLIFICATION"
    if pass_a >= 12 and pass_b <= 4 and pass_c <= 4 and peak_b < 2.0 and peak_c < 4.0:
        return "D_LOCALIZATION_DEPENDENT"
    return "E_MIXED_INCONCLUSIVE"


def stage2_gate(arm_summaries: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    b = next(row for row in arm_summaries if row["arm_id"] == ARM_HOMOGENEOUS)
    reasons = []
    if int(b["strong_finite_domain_pass_count"]) >= 8:
        reasons.append("B strong finite-domain pass count >= 8")
    if (
        int(b["systematic_low_k_growth_pass_count"]) >= 12
        and float(b["median_peak_domain_scale_ratio"]) >= 2.0
    ):
        reasons.append("B low-k pass count >= 12 and median peak scale ratio >= 2")
    if int(b["final_simple_regular_spectral_signature_count"]) >= 12:
        reasons.append("B simple-regular spectral-signature count >= 12")
    return bool(reasons), reasons


def paired_comparisons(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in coordinates:
        by_seed.setdefault(int(row["seed_index"]), {})[str(row["arm_id"])] = row
    output = []
    for seed_index, records in sorted(by_seed.items()):
        if not set((ARM_LOCALIZED, ARM_HOMOGENEOUS, ARM_GRADIENT)).issubset(records):
            raise ValueError(f"paired coordinate rows missing for seed {seed_index}")
        a = records[ARM_LOCALIZED]
        b = records[ARM_HOMOGENEOUS]
        c = records[ARM_GRADIENT]
        output.append(
            {
                "seed_index": seed_index,
                "inventory_seed": a["inventory_seed"],
                "matched_initial_occupancy_mass": a["initial_occupancy_mass"],
                "A_peak_scale_ratio": a["peak_domain_scale_ratio"],
                "B_peak_scale_ratio": b["peak_domain_scale_ratio"],
                "C_peak_scale_ratio": c["peak_domain_scale_ratio"],
                "B_minus_A_peak_scale_ratio": (
                    b["peak_domain_scale_ratio"] - a["peak_domain_scale_ratio"]
                ),
                "C_minus_A_peak_scale_ratio": (
                    c["peak_domain_scale_ratio"] - a["peak_domain_scale_ratio"]
                ),
                "A_final_scale_ratio": a["final_domain_scale_ratio"],
                "B_final_scale_ratio": b["final_domain_scale_ratio"],
                "C_final_scale_ratio": c["final_domain_scale_ratio"],
                "A_strong_pass": a["strong_finite_domain_pass"],
                "B_strong_pass": b["strong_finite_domain_pass"],
                "C_strong_pass": c["strong_finite_domain_pass"],
                "B_low_k_pass": b["systematic_low_k_growth_pass"],
                "C_low_k_pass": c["systematic_low_k_growth_pass"],
            }
        )
    return output
