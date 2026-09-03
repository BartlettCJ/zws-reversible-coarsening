from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from zws_nullity_lab.atomic import atomic_write_json, sha256_file
from zws_nullity_lab.density_threshold import (
    classify_density_response,
    logistic_fit_gate,
    refinement_plan,
)
from zws_nullity_lab.density_threshold_campaign import load_and_verify_density_manifest
from zws_nullity_lab.homogeneous_analysis import load_completed_stage, reduce_trajectory


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXPERIMENT_DIR / "BROAD_MANIFEST.json"
RESULT_ROOT = ROOT / "results" / "homogeneous_density_threshold_20260826" / "broad_1024"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts" / "broad"
FIGURE_DIR = EXPERIMENT_DIR / "figures" / "broad"


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
    if trials <= 0:
        raise ValueError("Wilson interval requires trials > 0")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    # Floating-point roundoff at the all-failure/all-success endpoints can put
    # the nominal interval a few ulps outside the observed proportion.  Keep
    # the interval closed around the estimate so plotting error lengths are
    # always non-negative.
    return (
        max(0.0, min(proportion, float(center - half))),
        min(1.0, max(proportion, float(center + half))),
    )


def summarize_densities(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    arm_ids = sorted(
        {str(row["arm_id"]) for row in coordinates},
        key=lambda arm: next(
            float(row["realized_occupancy"])
            for row in coordinates
            if str(row["arm_id"]) == arm
        ),
    )
    for arm_id in arm_ids:
        rows = [row for row in coordinates if row["arm_id"] == arm_id]
        trajectory_count = len(rows)
        strong_count = sum(bool(row["strong_finite_domain_pass"]) for row in rows)
        low_k_count = sum(bool(row["systematic_low_k_growth_pass"]) for row in rows)
        regular_count = sum(bool(row["final_simple_regular_spectral_signature"]) for row in rows)
        lower, upper = wilson_interval(strong_count, trajectory_count)
        output.append(
            {
                "stage_id": rows[0]["stage_id"],
                "arm_id": arm_id,
                "target_occupancy": rows[0]["target_occupancy"],
                "realized_occupancy": rows[0]["realized_occupancy"],
                "occupancy_mass": rows[0]["initial_occupancy_mass"],
                "trajectory_count": trajectory_count,
                "strong_domain_pass_count": strong_count,
                "strong_domain_pass_fraction": strong_count / trajectory_count,
                "strong_domain_wilson95_lower": lower,
                "strong_domain_wilson95_upper": upper,
                "low_k_growth_pass_count": low_k_count,
                "low_k_growth_pass_fraction": low_k_count / trajectory_count,
                "simple_regular_signature_count": regular_count,
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
                "median_final_interface_density": _median(
                    rows, "final_occupancy_interface_edge_density"
                ),
                "median_final_coarse_density_variance_block64": _median(
                    rows, "final_coarse_density_variance_block64"
                ),
            }
        )
    return output


def pass_probability_plot(summaries: list[dict[str, Any]]) -> Path:
    density = np.asarray([float(row["realized_occupancy"]) for row in summaries])
    strong = np.asarray([float(row["strong_domain_pass_fraction"]) for row in summaries])
    low_k = np.asarray([float(row["low_k_growth_pass_fraction"]) for row in summaries])
    lower = np.asarray([float(row["strong_domain_wilson95_lower"]) for row in summaries])
    upper = np.asarray([float(row["strong_domain_wilson95_upper"]) for row in summaries])
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
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
    axis.axhline(0.25, color="0.6", linestyle=":", linewidth=1.0)
    axis.axhline(0.75, color="0.6", linestyle=":", linewidth=1.0)
    axis.set(xlabel="realized initial occupancy", ylabel="trajectory pass fraction", ylim=(-0.04, 1.04))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "broad_pass_probability.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def continuous_metrics_plot(summaries: list[dict[str, Any]]) -> Path:
    density = np.asarray([float(row["realized_occupancy"]) for row in summaries])
    fields = [
        ("median_peak_domain_scale_ratio", "median peak L ratio", "log"),
        ("median_final_domain_scale_ratio", "median final L ratio", "log"),
        ("median_low_k_peak_mean_power_ratio", "median peak low-k ratio", "log"),
        (
            "median_final_minus_initial_occupied_fraction",
            "median final - initial occupancy",
            "linear",
        ),
        ("median_mean_reaction_density", "median mean reaction density", "linear"),
        ("median_final_entropy_normalized", "median final exact-state entropy", "linear"),
        (
            "median_final_largest_empty_domain_fraction",
            "median largest empty-domain fraction",
            "linear",
        ),
        ("median_final_interface_density", "median final interface density", "linear"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.0), sharex=True)
    for axis, (field, label, scale) in zip(axes.flat, fields, strict=True):
        values = np.asarray([float(row[field]) for row in summaries])
        axis.plot(density, values, color="#4c78a8", marker="o")
        axis.set_title(label, fontsize=10)
        if scale == "log" and np.all(values > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("realized initial occupancy")
    figure.tight_layout()
    path = FIGURE_DIR / "broad_continuous_metrics.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def blocked_seed_plot(coordinates: list[dict[str, Any]]) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), sharex=True)
    for seed_index in sorted({int(row["seed_index"]) for row in coordinates}):
        rows = sorted(
            (row for row in coordinates if int(row["seed_index"]) == seed_index),
            key=lambda row: float(row["realized_occupancy"]),
        )
        density = [float(row["realized_occupancy"]) for row in rows]
        axes[0].plot(
            density,
            [int(bool(row["strong_finite_domain_pass"])) for row in rows],
            alpha=0.55,
            linewidth=1.0,
        )
        axes[1].plot(
            density,
            [float(row["peak_domain_scale_ratio"]) for row in rows],
            alpha=0.55,
            linewidth=1.0,
        )
        axes[2].plot(
            density,
            [float(row["low_k_peak_mean_power_ratio"]) for row in rows],
            alpha=0.55,
            linewidth=1.0,
        )
    axes[0].set_ylabel("strong-domain pass")
    axes[1].set_ylabel("peak L ratio")
    axes[2].set_ylabel("peak low-k ratio")
    axes[2].set_xlabel("realized initial occupancy")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Broad-grid blocked seed trajectories")
    figure.tight_layout()
    path = FIGURE_DIR / "broad_blocked_seed_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze only a complete frozen broad grid.")
    parser.parse_args()
    manifest, manifest_digest, _ = load_and_verify_density_manifest(MANIFEST_PATH)
    trajectories = load_completed_stage(
        RESULT_ROOT,
        manifest,
        manifest_digest,
        stage_id="broad_1024",
    )
    manifest_rows = {
        str(row["run_id"]): row
        for row in manifest["stages"]["broad_1024"]["planned_runs"]
    }
    coordinates = []
    for trajectory in trajectories:
        row = reduce_trajectory(
            trajectory,
            finite_rule=manifest["finite_domain_rule"],
            low_k_rule=manifest["low_k_growth_rule"],
            regular_rule=manifest["simple_regular_spectral_signature"],
        )
        frozen = manifest_rows[str(row["run_id"])]
        row.update(
            {
                "target_occupancy": frozen["target_occupancy"],
                "realized_occupancy": frozen["realized_occupancy"],
                "occupancy_rounding_error": frozen["occupancy_rounding_error"],
            }
        )
        coordinates.append(row)
    summaries = summarize_densities(coordinates)
    classification = classify_density_response(summaries)
    refinement = refinement_plan(summaries)
    fit_gate = logistic_fit_gate(summaries)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACT_DIR / "DENSITY_RESULTS.csv", coordinates)
    write_csv(ARTIFACT_DIR / "DENSITY_SUMMARY.csv", summaries)
    decision = {
        "schema": "zwslab-density-threshold-broad-decision-v1",
        "manifest_sha256": manifest_digest,
        "stage_id": "broad_1024",
        "trajectory_count": len(coordinates),
        "density_count": len(summaries),
        "classification": classification,
        "refinement_plan": refinement,
        "logistic_fit_gate": fit_gate,
        "partial_outcome_policy": "analysis loaded only after complete 168-run campaign verification",
    }
    atomic_write_json(ARTIFACT_DIR / "BROAD_DECISION.json", decision)
    input_integrity = {
        "schema": "zwslab-density-threshold-broad-result-integrity-v1",
        "manifest_sha256": manifest_digest,
        "trajectory_count": len(trajectories),
        "all_initial_hashes_match_frozen_manifest": True,
        "all_q_samples_exact_zero": all(row["exact_q_at_all_samples"] for row in coordinates),
        "trajectory_sources": [
            {
                "run_id": trajectory["run_id"],
                "initial_state_sha256_uint8": trajectory["initial_state_sha256_uint8"],
                "final_state_sha256_uint8": trajectory["final_state_sha256_uint8"],
            }
            for trajectory in trajectories
        ],
    }
    atomic_write_json(ARTIFACT_DIR / "RESULT_INTEGRITY.json", input_integrity)
    figure_paths = [
        pass_probability_plot(summaries),
        continuous_metrics_plot(summaries),
        blocked_seed_plot(coordinates),
    ]
    artifacts = [
        ARTIFACT_DIR / "DENSITY_RESULTS.csv",
        ARTIFACT_DIR / "DENSITY_SUMMARY.csv",
        ARTIFACT_DIR / "BROAD_DECISION.json",
        ARTIFACT_DIR / "RESULT_INTEGRITY.json",
        *figure_paths,
    ]
    artifact_manifest = {
        "schema": "zwslab-density-threshold-broad-analysis-artifacts-v1",
        "manifest_sha256": manifest_digest,
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for path in artifacts
        ],
    }
    atomic_write_json(ARTIFACT_DIR / "BROAD_ANALYSIS_ARTIFACTS.json", artifact_manifest)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
