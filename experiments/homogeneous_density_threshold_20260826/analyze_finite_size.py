from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from zws_nullity_lab.atomic import atomic_write_json, sha256_file
from zws_nullity_lab.density_threshold import (
    deterministic_seed,
    fit_logistic_onset,
    logistic_fit_gate,
)
from zws_nullity_lab.density_threshold_campaign import load_and_verify_density_manifest
from zws_nullity_lab.homogeneous_analysis import load_completed_stage, reduce_trajectory


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST = EXPERIMENT_DIR / "FINITE_SIZE_MANIFEST.json"
RESULT_PARENT = ROOT / "results" / "homogeneous_density_threshold_20260826"
CAMPAIGN_SUMMARY = RESULT_PARENT / "finite_size_campaign_summary.json"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts" / "finite_size"
FIGURE_DIR = EXPERIMENT_DIR / "figures" / "finite_size"
RESULTS_PATH = EXPERIMENT_DIR / "FINITE_SIZE_RESULTS.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "FINITE_SIZE_SUMMARY.csv"
COMPARISON_PATH = EXPERIMENT_DIR / "FINITE_SIZE_COMPARISON.json"
FIGURE_MANIFEST = EXPERIMENT_DIR / "FINITE_SIZE_FIGURE_MANIFEST.json"
STAGES = ("finite_size_512", "finite_size_1024", "finite_size_2048")
BOOTSTRAP_NAMESPACE = "ZWSLab-density-threshold-finite-size-analysis-v1:2026082601"


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


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1.0 + z * z / trials
    center = (fraction + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(
        fraction * (1.0 - fraction) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return (
        max(0.0, min(fraction, float(center - half))),
        min(1.0, max(fraction, float(center + half))),
    )


def verify_complete_campaign(manifest_digest: str) -> None:
    payload = json.loads(CAMPAIGN_SUMMARY.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "zwslab-density-threshold-finite-size-campaign-summary-v1"
        or payload.get("status") != "complete"
        or payload.get("manifest_sha256") != manifest_digest
        or int(payload.get("completed_stage_count", -1)) != len(STAGES)
    ):
        raise RuntimeError("finite-size analysis requires all frozen stages to be complete")


def load_coordinates() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    manifest, digest, _ = load_and_verify_density_manifest(MANIFEST)
    verify_complete_campaign(digest)
    coordinates = []
    for stage_id in STAGES:
        stage = manifest["stages"][stage_id]
        world = int(stage["world"][0])
        trajectories = load_completed_stage(
            RESULT_PARENT / stage_id,
            manifest,
            digest,
            stage_id=stage_id,
        )
        frozen = {str(row["run_id"]): row for row in stage["planned_runs"]}
        for trajectory in trajectories:
            row = reduce_trajectory(
                trajectory,
                finite_rule=manifest["finite_domain_rule"],
                low_k_rule=manifest["low_k_growth_rule"],
                regular_rule=manifest["simple_regular_spectral_signature"],
            )
            planned = frozen[str(row["run_id"])]
            row.update(
                {
                    "world_size": world,
                    "target_occupancy": planned["target_occupancy"],
                    "realized_occupancy": planned["realized_occupancy"],
                    "occupancy_rounding_error": planned["occupancy_rounding_error"],
                    "source_manifest_sha256": digest,
                }
            )
            coordinates.append(row)
    expected = sum(int(stage["planned_run_count"]) for stage in manifest["stages"].values())
    if len(coordinates) != expected:
        raise RuntimeError("finite-size trajectory load is incomplete")
    return coordinates, manifest, digest


def summarize(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (int(row["world_size"]), float(row["realized_occupancy"]))
            for row in coordinates
        }
    )
    for world, density in keys:
        rows = [
            row
            for row in coordinates
            if int(row["world_size"]) == world
            and float(row["realized_occupancy"]) == density
        ]
        count = len(rows)
        strong = sum(bool(row["strong_finite_domain_pass"]) for row in rows)
        low_k = sum(bool(row["systematic_low_k_growth_pass"]) for row in rows)
        lower, upper = wilson_interval(strong, count)
        output.append(
            {
                "stage_id": rows[0]["stage_id"],
                "world_size": world,
                "arm_id": rows[0]["arm_id"],
                "target_occupancy": rows[0]["target_occupancy"],
                "realized_occupancy": density,
                "trajectory_count": count,
                "strong_domain_pass_count": strong,
                "strong_domain_pass_fraction": strong / count,
                "strong_domain_wilson95_lower": lower,
                "strong_domain_wilson95_upper": upper,
                "low_k_growth_pass_count": low_k,
                "low_k_growth_pass_fraction": low_k / count,
                "exact_q_pass_count": sum(bool(row["exact_q_at_all_samples"]) for row in rows),
                "simple_regular_signature_count": sum(
                    bool(row["final_simple_regular_spectral_signature"]) for row in rows
                ),
                "median_peak_domain_scale_ratio": _median(rows, "peak_domain_scale_ratio"),
                "median_final_domain_scale_ratio": _median(rows, "final_domain_scale_ratio"),
                "median_low_k_peak_mean_power_ratio": _median(
                    rows, "low_k_peak_mean_power_ratio"
                ),
                "median_final_occupied_fraction": _median(rows, "final_occupied_fraction"),
                "median_mean_reaction_density": _median(rows, "mean_reaction_density"),
            }
        )
    return output


def fit_worlds(
    coordinates: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    output = {}
    for world in (512, 1024, 2048):
        world_summaries = [row for row in summaries if int(row["world_size"]) == world]
        gate = logistic_fit_gate(world_summaries)
        fit = None
        if gate["passed"]:
            fit_rows = [
                {**row, "strong_domain_pass": row["strong_finite_domain_pass"]}
                for row in coordinates
                if int(row["world_size"]) == world
            ]
            fit = fit_logistic_onset(
                fit_rows,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=deterministic_seed(
                    f"bootstrap:world{world}", namespace=BOOTSTRAP_NAMESPACE
                ),
            )
        output[str(world)] = {"gate": gate, "fit": fit}
    return output


def empirical_25_to_75_bracket(
    summaries: list[dict[str, Any]],
) -> dict[str, float] | None:
    ordered = sorted(summaries, key=lambda row: float(row["realized_occupancy"]))
    candidates = []
    for lower_index, lower in enumerate(ordered):
        if float(lower["strong_domain_pass_fraction"]) > 0.25:
            continue
        for upper in ordered[lower_index + 1 :]:
            if float(upper["strong_domain_pass_fraction"]) < 0.75:
                continue
            lower_density = float(lower["realized_occupancy"])
            upper_density = float(upper["realized_occupancy"])
            candidates.append(
                {
                    "lower_realized_occupancy": lower_density,
                    "upper_realized_occupancy": upper_density,
                    "lower_pass_fraction": float(lower["strong_domain_pass_fraction"]),
                    "upper_pass_fraction": float(upper["strong_domain_pass_fraction"]),
                    "width": upper_density - lower_density,
                }
            )
    return min(candidates, key=lambda row: (row["width"], row["lower_realized_occupancy"])) if candidates else None


def fit_support_diagnostics(
    fit: dict[str, Any] | None,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    sampled = [float(row["realized_occupancy"]) for row in summaries]
    lower = min(sampled)
    upper = max(sampled)
    output: dict[str, Any] = {
        "sampled_density_min": lower,
        "sampled_density_max": upper,
        "empirical_25_to_75_bracket": empirical_25_to_75_bracket(summaries),
        "fit_available": fit is not None,
    }
    if fit is None:
        return output
    point = fit["point_estimate"]
    p25 = float(point["occupancy_p25"])
    p50 = float(point["occupancy_p50"])
    p75 = float(point["occupancy_p75"])
    intervals = fit["bootstrap"]["percentile_intervals"]
    interval_bounds = [
        float(bound)
        for interval in intervals.values()
        for bound in (interval["lower_95"], interval["upper_95"])
    ]
    output.update(
        {
            "point_p25_within_sampled_range": lower <= p25 <= upper,
            "point_p50_within_sampled_range": lower <= p50 <= upper,
            "point_p75_within_sampled_range": lower <= p75 <= upper,
            "point_width_fully_supported_by_sampled_range": (
                lower <= p25 <= p75 <= upper
            ),
            "bootstrap_intervals_finite_and_within_unit_interval": all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in interval_bounds
            ),
            "bootstrap_p50_interval_within_sampled_range": (
                lower <= float(intervals["occupancy_p50"]["lower_95"])
                <= float(intervals["occupancy_p50"]["upper_95"])
                <= upper
            ),
            "interpretation": (
                "A converged fit may still be extrapolative; onset location and width are "
                "treated as constrained only where the corresponding point/interval lies "
                "inside the three sampled densities."
            ),
        }
    )
    return output


def plot_pass_fractions(summaries: list[dict[str, Any]]) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    for world, marker in ((512, "o"), (1024, "s"), (2048, "^")):
        rows = [row for row in summaries if int(row["world_size"]) == world]
        axis.plot(
            [float(row["realized_occupancy"]) for row in rows],
            [float(row["strong_domain_pass_fraction"]) for row in rows],
            marker=marker,
            label=f"{world} x {world}",
        )
    axis.axhline(0.25, color="0.75", linestyle=":", linewidth=0.9)
    axis.axhline(0.75, color="0.75", linestyle=":", linewidth=0.9)
    axis.set(
        xlabel="realized initial occupancy",
        ylabel="strong-domain pass fraction",
        ylim=(-0.04, 1.04),
        title="Gated finite-size onset-location check",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "finite_size_pass_probability.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_fit_comparison(fits: dict[str, Any], support: dict[str, Any]) -> Path:
    worlds = []
    p50 = []
    widths = []
    for world in (512, 1024, 2048):
        fit = fits[str(world)]["fit"]
        if fit is None:
            continue
        worlds.append(world)
        p50.append(float(fit["point_estimate"]["occupancy_p50"]))
        widths.append(float(fit["point_estimate"]["width_p25_to_p75"]))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for world, location, width in zip(worlds, p50, widths, strict=True):
        diagnostics = support[str(world)]
        axes[0].scatter(
            [world],
            [location],
            marker="o" if diagnostics["point_p50_within_sampled_range"] else "x",
            color="#1b9e77" if diagnostics["point_p50_within_sampled_range"] else "#777777",
            s=55,
        )
        axes[1].scatter(
            [world],
            [width],
            marker=(
                "o"
                if diagnostics["point_width_fully_supported_by_sampled_range"]
                else "x"
            ),
            color=(
                "#1b9e77"
                if diagnostics["point_width_fully_supported_by_sampled_range"]
                else "#777777"
            ),
            s=55,
        )
    axes[0].set(
        xlabel="world width",
        ylabel="fitted p50 occupancy",
        title="Point fits (x = outside sampled density range)",
    )
    axes[1].set(
        xlabel="world width",
        ylabel="fitted p25-to-p75 width",
        title="Point widths (x = extrapolative edge)",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xscale("log", base=2)
        axis.set_xticks((512, 1024, 2048), labels=("512", "1024", "2048"))
    figure.tight_layout()
    path = FIGURE_DIR / "finite_size_fit_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the complete finite-size check.")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    coordinates, manifest, digest = load_coordinates()
    summaries = summarize(coordinates)
    fits = fit_worlds(
        coordinates,
        summaries,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    point_estimates = {
        world: payload["fit"]["point_estimate"]
        for world, payload in fits.items()
        if payload["fit"] is not None
    }
    p50_values = [float(value["occupancy_p50"]) for value in point_estimates.values()]
    width_values = [float(value["width_p25_to_p75"]) for value in point_estimates.values()]
    support = {}
    in_range_p50 = []
    in_range_width = []
    for world in (512, 1024, 2048):
        rows = [row for row in summaries if int(row["world_size"]) == world]
        diagnostics = fit_support_diagnostics(fits[str(world)]["fit"], rows)
        support[str(world)] = diagnostics
        fit = fits[str(world)]["fit"]
        if fit is not None and diagnostics.get("point_p50_within_sampled_range"):
            in_range_p50.append(float(fit["point_estimate"]["occupancy_p50"]))
        if fit is not None and diagnostics.get("point_width_fully_supported_by_sampled_range"):
            in_range_width.append(float(fit["point_estimate"]["width_p25_to_p75"]))
    comparison = {
        "schema": "zwslab-density-threshold-finite-size-comparison-v1",
        "manifest_sha256": digest,
        "worlds": [512, 1024, 2048],
        "targets": manifest["selected_finite_size_targets"],
        "trajectory_count": len(coordinates),
        "failures": 0,
        "world_fits": fits,
        "world_fit_support_diagnostics": support,
        "fitted_p50_range_across_available_worlds": (
            max(p50_values) - min(p50_values) if len(p50_values) >= 2 else None
        ),
        "fitted_width_range_across_available_worlds": (
            max(width_values) - min(width_values) if len(width_values) >= 2 else None
        ),
        "point_p50_range_across_worlds_with_in_range_point_estimates": (
            max(in_range_p50) - min(in_range_p50) if len(in_range_p50) >= 2 else None
        ),
        "point_width_range_across_worlds_fully_supported_by_sampled_range": (
            max(in_range_width) - min(in_range_width) if len(in_range_width) >= 2 else None
        ),
        "fit_caution": (
            "All optimizer-success fits are retained, but extrapolative point estimates and "
            "bootstrap intervals are not treated as measured size trends. Raw pass counts and "
            "support diagnostics control interpretation."
        ),
        "interpretation_boundary": (
            "three strategically selected densities only; descriptive location/sharpening "
            "check, not finite-size scaling or a thermodynamic-limit inference"
        ),
    }
    write_csv(RESULTS_PATH, coordinates)
    write_csv(SUMMARY_PATH, summaries)
    atomic_write_json(COMPARISON_PATH, comparison)
    figure_paths = [plot_pass_fractions(summaries), plot_fit_comparison(fits, support)]
    atomic_write_json(
        FIGURE_MANIFEST,
        {
            "schema": "zwslab-density-threshold-finite-size-figure-manifest-v1",
            "figures": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
                for path in figure_paths
            ],
        },
    )
    artifacts = [RESULTS_PATH, SUMMARY_PATH, COMPARISON_PATH, FIGURE_MANIFEST, *figure_paths]
    atomic_write_json(
        ARTIFACT_DIR / "FINITE_SIZE_ANALYSIS_ARTIFACTS.json",
        {
            "schema": "zwslab-density-threshold-finite-size-analysis-artifacts-v1",
            "artifacts": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
