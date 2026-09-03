from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from zws_nullity_lab.atomic import atomic_write_json, sha256_file
from zws_nullity_lab.confirmation_diagnostics import toroidal_empty_domain_stats
from zws_nullity_lab.density_threshold import (
    classify_density_response,
    fit_logistic_onset,
    logistic_fit_gate,
    predicted_probability,
)
from zws_nullity_lab.density_threshold_campaign import load_and_verify_density_manifest
from zws_nullity_lab.domain_mechanism import density_environment_masks, state_composition
from zws_nullity_lab.homogeneous_analysis import load_completed_stage, reduce_trajectory


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
BROAD_MANIFEST = EXPERIMENT_DIR / "BROAD_MANIFEST.json"
REFINEMENT_MANIFEST = EXPERIMENT_DIR / "REFINEMENT_MANIFEST.json"
BROAD_DECISION = EXPERIMENT_DIR / "artifacts" / "broad" / "BROAD_DECISION.json"
RESULT_PARENT = ROOT / "results" / "homogeneous_density_threshold_20260826"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts" / "final_1024"
FIGURE_DIR = EXPERIMENT_DIR / "figures" / "final_1024"
DENSITY_RESULTS = EXPERIMENT_DIR / "DENSITY_RESULTS.csv"
DENSITY_SUMMARY = EXPERIMENT_DIR / "DENSITY_SUMMARY.csv"
FIT_PATH = EXPERIMENT_DIR / "DENSITY_THRESHOLD_FIT.json"
FIGURE_MANIFEST = EXPERIMENT_DIR / "FIGURE_MANIFEST.json"
STATE_CMAP = ListedColormap(["#090909", "#ff6b35", "#2ec4d6", "#9b5de5"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return median(values) if values else None


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return (
        max(0.0, min(proportion, float(center - half))),
        min(1.0, max(proportion, float(center + half))),
    )


def summarize_densities(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    density_values = sorted({float(row["realized_occupancy"]) for row in coordinates})
    for density in density_values:
        rows = [row for row in coordinates if float(row["realized_occupancy"]) == density]
        count = len(rows)
        strong = sum(bool(row["strong_finite_domain_pass"]) for row in rows)
        low_k = sum(bool(row["systematic_low_k_growth_pass"]) for row in rows)
        lower, upper = wilson_interval(strong, count)
        output.append(
            {
                "stage_id": rows[0]["stage_id"],
                "arm_id": rows[0]["arm_id"],
                "target_occupancy": rows[0]["target_occupancy"],
                "realized_occupancy": density,
                "occupancy_mass": rows[0]["initial_occupancy_mass"],
                "trajectory_count": count,
                "strong_domain_pass_count": strong,
                "strong_domain_pass_fraction": strong / count,
                "strong_domain_wilson95_lower": lower,
                "strong_domain_wilson95_upper": upper,
                "low_k_growth_pass_count": low_k,
                "low_k_growth_pass_fraction": low_k / count,
                "simple_regular_signature_count": sum(
                    bool(row["final_simple_regular_spectral_signature"]) for row in rows
                ),
                "simple_regular_signature_fraction": sum(
                    bool(row["final_simple_regular_spectral_signature"]) for row in rows
                )
                / count,
                "exact_q_pass_count": sum(bool(row["exact_q_at_all_samples"]) for row in rows),
                "median_peak_domain_scale_ratio": _median(rows, "peak_domain_scale_ratio"),
                "median_final_domain_scale_ratio": _median(rows, "final_domain_scale_ratio"),
                "median_low_k_peak_mean_power_ratio": _median(
                    rows, "low_k_peak_mean_power_ratio"
                ),
                "median_low_k_final_mean_power_ratio": _median(
                    rows, "low_k_final_mean_power_ratio"
                ),
                "median_initial_occupied_fraction": _median(rows, "initial_occupied_fraction"),
                "median_peak_occupied_fraction": _median(rows, "peak_occupied_fraction"),
                "median_final_occupied_fraction": _median(rows, "final_occupied_fraction"),
                "median_final_minus_initial_occupied_fraction": median(
                    float(row["final_occupied_fraction"])
                    - float(row["initial_occupied_fraction"])
                    for row in rows
                ),
                "median_mean_reaction_density": _median(rows, "mean_reaction_density"),
                "median_final_entropy_normalized": _median(rows, "final_entropy_normalized"),
                "median_final_largest_empty_domain_fraction": _median(
                    rows, "final_largest_empty_domain_fraction"
                ),
                "median_final_empty_domain_component_count": _median(
                    rows, "final_empty_domain_component_count"
                ),
                "median_final_interface_density": _median(
                    rows, "final_occupancy_interface_edge_density"
                ),
                "median_final_exact_state_boundary_density": _median(
                    rows, "final_exact_state_boundary_density"
                ),
                "median_final_structure_factor_characteristic_wavelength_cells": _median(
                    rows, "final_structure_factor_characteristic_wavelength_cells"
                ),
                "median_final_structure_factor_peak_to_radial_median": _median(
                    rows, "final_structure_factor_peak_to_radial_median"
                ),
                "median_final_coarse_density_variance_block16": _median(
                    rows, "final_coarse_density_variance_block16"
                ),
                "median_final_coarse_density_variance_block32": _median(
                    rows, "final_coarse_density_variance_block32"
                ),
                "median_final_coarse_density_variance_block64": _median(
                    rows, "final_coarse_density_variance_block64"
                ),
                **{
                    f"median_final_recurrence_lag{lag}": _median(
                        rows, f"final_recurrence_lag{lag}"
                    )
                    for lag in (4, 8, 16, 32, 64, 128)
                },
            }
        )
    return output


def verified_snapshot(path: Path, manifest_digest: str) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if sha256_file(path) != declared:
        raise RuntimeError(f"snapshot digest mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "zwslab-homogeneous-state-snapshot-v1"
        or payload.get("manifest_sha256") != manifest_digest
    ):
        raise RuntimeError(f"snapshot belongs to another campaign: {path}")
    return payload


def load_sources() -> list[dict[str, Any]]:
    broad_decision = json.loads(BROAD_DECISION.read_text(encoding="utf-8"))
    sources = [
        {
            "manifest_path": BROAD_MANIFEST,
            "stage_id": "broad_1024",
            "result_root": RESULT_PARENT / "broad_1024",
        }
    ]
    refinement_required = bool(
        broad_decision.get("refinement_plan", {}).get("run_refinement")
    )
    if refinement_required:
        if not REFINEMENT_MANIFEST.is_file():
            raise RuntimeError("broad decision requires the frozen refinement stage")
        sources.append(
            {
                "manifest_path": REFINEMENT_MANIFEST,
                "stage_id": "refinement_1024",
                "result_root": RESULT_PARENT / "refinement_1024",
            }
        )
    return sources


def load_all_trajectories() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coordinates = []
    sources = load_sources()
    for source in sources:
        manifest, digest, _ = load_and_verify_density_manifest(source["manifest_path"])
        stage_id = str(source["stage_id"])
        trajectories = load_completed_stage(
            source["result_root"], manifest, digest, stage_id=stage_id
        )
        frozen_rows = {
            str(row["run_id"]): row for row in manifest["stages"][stage_id]["planned_runs"]
        }
        for trajectory in trajectories:
            row = reduce_trajectory(
                trajectory,
                finite_rule=manifest["finite_domain_rule"],
                low_k_rule=manifest["low_k_growth_rule"],
                regular_rule=manifest["simple_regular_spectral_signature"],
            )
            frozen = frozen_rows[str(row["run_id"])]
            row.update(
                {
                    "target_occupancy": frozen["target_occupancy"],
                    "realized_occupancy": frozen["realized_occupancy"],
                    "occupancy_rounding_error": frozen["occupancy_rounding_error"],
                    "source_manifest_sha256": digest,
                }
            )
            coordinates.append(row)
        source.update({"manifest": manifest, "manifest_sha256": digest})
    return coordinates, sources


def spatial_and_cluster_rows(
    coordinates: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coordinate_map = {str(row["run_id"]): row for row in coordinates}
    spatial_rows = []
    cluster_rows = []
    for source in sources:
        manifest = source["manifest"]
        digest = str(source["manifest_sha256"])
        stage_id = str(source["stage_id"])
        stage = manifest["stages"][stage_id]
        for arm_id in stage["arm_ids"]:
            arm_rows = sorted(
                (row for row in stage["planned_runs"] if row["arm_id"] == arm_id),
                key=lambda row: int(row["seed_index"]),
            )
            snapshot_dir = source["result_root"] / "arms" / arm_id / "snapshots"
            batches = {
                "initial": verified_snapshot(snapshot_dir / "initial.pt", digest)["state"],
                "peak": verified_snapshot(snapshot_dir / "peak.pt", digest)["state"],
                "final": verified_snapshot(snapshot_dir / "final.pt", digest)["state"],
            }
            for role, batch in batches.items():
                for seed_index, state_cpu in enumerate(batch):
                    run_id = str(arm_rows[seed_index]["run_id"])
                    coordinate = coordinate_map[run_id]
                    state = state_cpu.to(device=device)
                    if role in {"peak", "final"}:
                        masks = density_environment_masks(
                            state,
                            width=32,
                            lower_quantile=0.2,
                            upper_quantile=0.8,
                            phase_threshold="quantile_midpoint",
                            interface_band_radius=1,
                        )
                        for environment in ("dense_interior", "interface", "sparse_interior"):
                            composition = state_composition(state, masks[environment])
                            spatial_rows.append(
                                {
                                    "run_id": run_id,
                                    "stage_id": stage_id,
                                    "arm_id": arm_id,
                                    "seed_index": seed_index,
                                    "target_occupancy": coordinate["target_occupancy"],
                                    "realized_occupancy": coordinate["realized_occupancy"],
                                    "role": role,
                                    "environment": environment,
                                    "genuine_strong_domain": coordinate[
                                        "strong_finite_domain_pass"
                                    ],
                                    "site_count": composition["site_count"],
                                    "occupied_fraction": composition["occupied_fraction"],
                                    "positive_fraction": composition["positive_fraction"],
                                    "negative_fraction": composition["negative_fraction"],
                                    "bound_neutral_fraction": composition[
                                        "bound_neutral_fraction"
                                    ],
                                    "signed_charge_per_site": (
                                        composition["total_signed_charge"]
                                        / composition["site_count"]
                                        if composition["site_count"]
                                        else None
                                    ),
                                    "normalized_state_entropy": composition[
                                        "normalized_state_entropy"
                                    ],
                                    "density_lower_threshold": masks["metadata"][
                                        "lower_threshold"
                                    ],
                                    "density_upper_threshold": masks["metadata"][
                                        "upper_threshold"
                                    ],
                                }
                            )
                    occupied_mask = (state_cpu.numpy() != 0)
                    occupied_stats = toroidal_empty_domain_stats(occupied_mask)
                    empty_stats = toroidal_empty_domain_stats(~occupied_mask)
                    population = state_cpu.numel()
                    cluster_rows.append(
                        {
                            "run_id": run_id,
                            "stage_id": stage_id,
                            "arm_id": arm_id,
                            "seed_index": seed_index,
                            "target_occupancy": coordinate["target_occupancy"],
                            "realized_occupancy": coordinate["realized_occupancy"],
                            "role": role,
                            "exploratory_no_winding_test": True,
                            "occupied_component_count": occupied_stats["component_count"],
                            "largest_occupied_cluster_fraction_of_occupied": occupied_stats[
                                "largest_area_fraction_of_empty"
                            ],
                            "largest_occupied_cluster_fraction_of_world": occupied_stats[
                                "largest_area"
                            ]
                            / population,
                            "occupied_cluster_log2_histogram_json": json.dumps(
                                occupied_stats["area_log2_histogram"], separators=(",", ":")
                            ),
                            "empty_component_count": empty_stats["component_count"],
                            "largest_empty_cluster_fraction_of_empty": empty_stats[
                                "largest_area_fraction_of_empty"
                            ],
                            "largest_empty_cluster_fraction_of_world": empty_stats["largest_area"]
                            / population,
                            "empty_cluster_log2_histogram_json": json.dumps(
                                empty_stats["area_log2_histogram"], separators=(",", ":")
                            ),
                        }
                    )
    return spatial_rows, cluster_rows


def augment_summaries(
    summaries: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> None:
    for summary in summaries:
        density = float(summary["realized_occupancy"])
        dense = [
            row
            for row in spatial_rows
            if float(row["realized_occupancy"]) == density
            and row["role"] == "final"
            and row["environment"] == "dense_interior"
            and bool(row["genuine_strong_domain"])
        ]
        final_clusters = [
            row
            for row in cluster_rows
            if float(row["realized_occupancy"]) == density and row["role"] == "final"
        ]
        summary["strong_domain_dense_entropy_count"] = len(dense)
        summary["median_final_dense_interior_entropy_strong_only"] = _median(
            dense, "normalized_state_entropy"
        )
        summary["median_final_dense_interior_occupancy_strong_only"] = _median(
            dense, "occupied_fraction"
        )
        summary["median_final_largest_occupied_cluster_fraction_of_world"] = _median(
            final_clusters, "largest_occupied_cluster_fraction_of_world"
        )
        summary["median_final_largest_occupied_cluster_fraction_of_occupied"] = _median(
            final_clusters, "largest_occupied_cluster_fraction_of_occupied"
        )
        summary["median_final_largest_empty_cluster_fraction_of_world_exploratory"] = _median(
            final_clusters, "largest_empty_cluster_fraction_of_world"
        )


def pass_probability_plot(summaries: list[dict[str, Any]], fit: dict[str, Any] | None) -> Path:
    density = np.asarray([float(row["realized_occupancy"]) for row in summaries])
    strong = np.asarray([float(row["strong_domain_pass_fraction"]) for row in summaries])
    low_k = np.asarray([float(row["low_k_growth_pass_fraction"]) for row in summaries])
    lower = np.asarray([float(row["strong_domain_wilson95_lower"]) for row in summaries])
    upper = np.asarray([float(row["strong_domain_wilson95_upper"]) for row in summaries])
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.errorbar(
        density,
        strong,
        yerr=np.vstack([strong - lower, upper - strong]),
        color="#1b9e77",
        marker="o",
        capsize=3,
        label="strong-domain fraction (Wilson 95%)",
    )
    axis.plot(density, low_k, color="#d95f02", marker="s", label="low-k-growth fraction")
    if fit is not None:
        grid = np.linspace(float(density.min()), float(density.max()), 400)
        axis.plot(
            grid,
            [predicted_probability(value, fit) for value in grid],
            color="#4c78a8",
            linewidth=2.0,
            label="descriptive logistic MAP",
        )
        point = fit["point_estimate"]
        for probability in (25, 50, 75):
            axis.axvline(
                float(point[f"occupancy_p{probability}"]),
                color="0.6",
                linestyle=":" if probability != 50 else "--",
                linewidth=0.9,
            )
    axis.axhline(0.25, color="0.75", linestyle=":", linewidth=0.9)
    axis.axhline(0.75, color="0.75", linestyle=":", linewidth=0.9)
    axis.set(xlabel="realized initial occupancy", ylabel="trajectory pass fraction", ylim=(-0.04, 1.04))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "density_pass_probability_and_fit.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def continuous_plot(summaries: list[dict[str, Any]]) -> Path:
    density = np.asarray([float(row["realized_occupancy"]) for row in summaries])
    fields = [
        ("median_peak_domain_scale_ratio", "peak L ratio", True),
        ("median_final_domain_scale_ratio", "final L ratio", True),
        ("median_low_k_peak_mean_power_ratio", "peak low-k ratio", True),
        ("median_final_minus_initial_occupied_fraction", "final - initial occupancy", False),
        ("median_mean_reaction_density", "mean reaction density", False),
        ("median_final_dense_interior_entropy_strong_only", "dense entropy (passes only)", False),
        ("median_final_largest_empty_domain_fraction", "largest empty / empty", False),
        ("median_final_interface_density", "interface density", False),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), sharex=True)
    for axis, (field, label, log_scale) in zip(axes.flat, fields, strict=True):
        values = np.asarray(
            [float(row[field]) if row.get(field) is not None else np.nan for row in summaries]
        )
        axis.plot(density, values, color="#4c78a8", marker="o")
        if log_scale and np.all(values[np.isfinite(values)] > 0):
            axis.set_yscale("log")
        axis.set_title(label, fontsize=10)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("realized initial occupancy")
    figure.tight_layout()
    path = FIGURE_DIR / "density_continuous_observables.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def cluster_plot(summaries: list[dict[str, Any]]) -> Path:
    density = [float(row["realized_occupancy"]) for row in summaries]
    occupied = [
        float(row["median_final_largest_occupied_cluster_fraction_of_world"])
        for row in summaries
    ]
    empty = [
        float(row["median_final_largest_empty_cluster_fraction_of_world_exploratory"])
        for row in summaries
    ]
    figure, axis = plt.subplots(figsize=(9.0, 5.3))
    axis.plot(density, occupied, marker="o", label="largest occupied cluster / world")
    axis.plot(density, empty, marker="s", label="largest empty cluster / world")
    axis.set(
        xlabel="realized initial occupancy",
        ylabel="median final cluster fraction of world",
        title="Exploratory four-neighbour cluster geometry (no winding test)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "density_exploratory_cluster_geometry.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def representative_summaries(
    summaries: list[dict[str, Any]], fit: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows = sorted(summaries, key=lambda row: float(row["realized_occupancy"]))
    if len(rows) < 6:
        raise ValueError("representative morphology requires at least six densities")
    densities = np.asarray([float(row["realized_occupancy"]) for row in rows])
    if fit is not None:
        point = fit["point_estimate"]
        targets = [
            float(point["occupancy_p25"]),
            float(point["occupancy_p50"]),
            float(point["occupancy_p75"]),
        ]
    else:
        transition = classify_density_response(rows)
        bracket = transition["primary_25_to_75_bracket"]
        if bracket is None:
            targets = [float(densities[len(densities) // 2])] * 3
        else:
            lower = float(bracket["lower_realized_occupancy"])
            upper = float(bracket["upper_realized_occupancy"])
            targets = [lower, 0.5 * (lower + upper), upper]
    eligible = range(1, len(rows) - 2)
    transition_indices = min(
        combinations(eligible, 3),
        key=lambda indices: sum(
            abs(float(densities[index]) - target)
            for index, target in zip(indices, targets, strict=True)
        ),
    )
    lower, _, upper = transition_indices
    below = max(0, lower - 2)
    above = min(len(rows) - 2, upper + 2)
    indices = [below, *transition_indices, above, len(rows) - 1]
    if indices != sorted(set(indices)):
        raise RuntimeError("representative morphology density selection is not distinct")
    return [rows[index] for index in indices]


def morphology_grid(
    summaries: list[dict[str, Any]], sources: list[dict[str, Any]], fit: dict[str, Any] | None
) -> tuple[Path, list[float]]:
    selected = representative_summaries(summaries, fit)
    source_map = {str(source["stage_id"]): source for source in sources}
    roles = ("initial", "early", "peak", "final")
    figure, axes = plt.subplots(len(selected), 4, figsize=(13.5, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, summary in enumerate(selected):
        source = source_map[str(summary["stage_id"])]
        arm_id = str(summary["arm_id"])
        digest = str(source["manifest_sha256"])
        snapshot_dir = source["result_root"] / "arms" / arm_id / "snapshots"
        batches = {
            "initial": verified_snapshot(snapshot_dir / "initial.pt", digest)["state"],
            "early": verified_snapshot(snapshot_dir / "early_t004096.pt", digest)["state"],
            "peak": verified_snapshot(snapshot_dir / "peak.pt", digest)["state"],
            "final": verified_snapshot(snapshot_dir / "final.pt", digest)["state"],
        }
        for column_index, role in enumerate(roles):
            axis = axes[row_index, column_index]
            axis.imshow(
                batches[role][0].numpy(),
                cmap=STATE_CMAP,
                vmin=0,
                vmax=3,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(role)
            if column_index == 0:
                axis.set_ylabel(
                    f"p={float(summary['realized_occupancy']):.4f}\n"
                    f"pass={float(summary['strong_domain_pass_fraction']):.2f}",
                    fontsize=9,
                )
    figure.suptitle("Common-scale density morphology — frozen seed 00")
    figure.tight_layout()
    path = FIGURE_DIR / "representative_density_morphology_grid.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path, [float(row["realized_occupancy"]) for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the complete final 1024 density map.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    coordinates, sources = load_all_trajectories()
    summaries = summarize_densities(coordinates)
    fit_gate = logistic_fit_gate(summaries)
    fit_rows = [
        {**row, "strong_domain_pass": row["strong_finite_domain_pass"]}
        for row in coordinates
    ]
    fit = (
        fit_logistic_onset(fit_rows, bootstrap_replicates=args.bootstrap_replicates)
        if fit_gate["passed"]
        else None
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    spatial_rows, cluster_rows = spatial_and_cluster_rows(
        coordinates, sources, device=device
    )
    augment_summaries(summaries, spatial_rows, cluster_rows)
    required_coordinate_fields = [
        "realized_occupancy",
        "final_occupied_fraction",
        "final_positive_fraction",
        "final_negative_fraction",
        "final_bound_fraction",
        "peak_finite_domain_scale",
        "peak_domain_scale_ratio",
        "final_domain_scale_ratio",
        "final_empty_domain_area_log2_histogram_json",
        "final_occupancy_interface_edge_density",
        "mean_reaction_density",
        "final_coarse_density_variance_block16",
        "final_coarse_density_variance_block32",
        "final_coarse_density_variance_block64",
        "final_radial_structure_factor_json",
        "low_k_peak_mean_power_ratio",
        "final_entropy_normalized",
        *[f"final_recurrence_lag{lag}" for lag in (4, 8, 16, 32, 64, 128)],
        *[
            f"final_spatial_autocorrelation_lag{lag}"
            for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        ],
    ]
    expected_trajectories = sum(
        int(source["manifest"]["stages"][source["stage_id"]]["planned_run_count"])
        for source in sources
    )
    final_integrity = {
        "schema": "zwslab-density-threshold-final-result-integrity-v1",
        "source_manifests": [
            {"stage_id": source["stage_id"], "sha256": source["manifest_sha256"]}
            for source in sources
        ],
        "expected_trajectory_count": expected_trajectories,
        "observed_trajectory_count": len(coordinates),
        "failures": 0,
        "all_run_ids_unique": len({str(row["run_id"]) for row in coordinates})
        == len(coordinates),
        "all_initial_hashes_match_frozen_manifests": True,
        "all_trajectory_file_hashes_match_completions": True,
        "all_exact_q_samples_zero": all(
            bool(row["exact_q_at_all_samples"]) for row in coordinates
        ),
        "required_compact_trajectory_fields": required_coordinate_fields,
        "all_required_compact_trajectory_fields_present": all(
            all(field in row for field in required_coordinate_fields) for row in coordinates
        ),
        "spatial_composition_row_count": len(spatial_rows),
        "expected_spatial_composition_row_count": expected_trajectories * 6,
        "exploratory_cluster_row_count": len(cluster_rows),
        "expected_exploratory_cluster_row_count": expected_trajectories * 3,
        "verification_method": (
            "load_completed_stage verified complete campaign/arm records, trajectory SHA-256, "
            "frozen initial hashes/inventories, schemas, run identities, and every Q sample"
        ),
    }
    if not (
        len(coordinates) == expected_trajectories
        and final_integrity["all_run_ids_unique"]
        and final_integrity["all_exact_q_samples_zero"]
        and final_integrity["all_required_compact_trajectory_fields_present"]
        and len(spatial_rows) == expected_trajectories * 6
        and len(cluster_rows) == expected_trajectories * 3
    ):
        raise RuntimeError("final 1024 result integrity audit failed")
    atomic_write_json(ARTIFACT_DIR / "FINAL_RESULT_INTEGRITY.json", final_integrity)
    transition = classify_density_response(summaries)
    size_gate = bool(
        fit is not None
        and transition["classification"]
        in {
            "C_SHARP_FINITE_SIZE_TRANSITION_REGION",
            "D_VERY_SHARP_FINITE_SIZE_TRANSITION_REGION",
        }
        and float(fit["point_estimate"]["width_p25_to_p75"]) <= 0.06
    )
    write_csv(DENSITY_RESULTS, coordinates)
    write_csv(DENSITY_SUMMARY, summaries)
    write_csv(ARTIFACT_DIR / "SPATIAL_COMPOSITIONS.csv", spatial_rows)
    write_csv(ARTIFACT_DIR / "EXPLORATORY_CLUSTER_GEOMETRY.csv", cluster_rows)
    if fit is not None:
        atomic_write_json(FIT_PATH, fit)
    elif FIT_PATH.exists():
        raise RuntimeError("stale density fit exists although the frozen fit gate failed")
    figure_paths = [
        pass_probability_plot(summaries, fit),
        continuous_plot(summaries),
        cluster_plot(summaries),
    ]
    morphology_path, morphology_densities = morphology_grid(summaries, sources, fit)
    figure_paths.append(morphology_path)
    figure_manifest = {
        "schema": "zwslab-density-threshold-figure-manifest-v1",
        "figures": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for path in figure_paths
        ],
    }
    atomic_write_json(FIGURE_MANIFEST, figure_manifest)
    decision = {
        "schema": "zwslab-density-threshold-final-1024-decision-v1",
        "source_manifests": [
            {
                "stage_id": source["stage_id"],
                "manifest_sha256": source["manifest_sha256"],
                "trajectory_count": source["manifest"]["stages"][source["stage_id"]][
                    "planned_run_count"
                ],
            }
            for source in sources
        ],
        "trajectory_count": len(coordinates),
        "density_count": len(summaries),
        "transition": transition,
        "logistic_fit_gate": fit_gate,
        "logistic_fit_applicable": fit is not None,
        "logistic_point_estimate": fit["point_estimate"] if fit is not None else None,
        "representative_morphology_densities": morphology_densities,
        "optional_finite_size_gate_passed": size_gate,
        "interpretation_boundary": (
            "finite-size response class follows frozen pass fractions; logistic fit is "
            "descriptive; cluster geometry is exploratory and has no winding test"
        ),
    }
    atomic_write_json(ARTIFACT_DIR / "FINAL_1024_DECISION.json", decision)
    artifacts = [
        DENSITY_RESULTS,
        DENSITY_SUMMARY,
        ARTIFACT_DIR / "SPATIAL_COMPOSITIONS.csv",
        ARTIFACT_DIR / "EXPLORATORY_CLUSTER_GEOMETRY.csv",
        ARTIFACT_DIR / "FINAL_RESULT_INTEGRITY.json",
        ARTIFACT_DIR / "FINAL_1024_DECISION.json",
        FIGURE_MANIFEST,
        *figure_paths,
    ]
    if fit is not None:
        artifacts.append(FIT_PATH)
    atomic_write_json(
        ARTIFACT_DIR / "FINAL_ANALYSIS_ARTIFACTS.json",
        {
            "schema": "zwslab-density-threshold-final-analysis-artifacts-v1",
            "artifacts": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
