from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from zws_nullity_lab.atomic import atomic_write_json, atomic_write_text, sha256_file
from zws_nullity_lab.density_threshold_campaign import load_and_verify_density_manifest
from zws_nullity_lab.homogeneous_analysis import load_completed_stage, reduce_trajectory


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = EXPERIMENT_DIR / "REFINEMENT_MANIFEST.json"
ADDENDUM = EXPERIMENT_DIR / "STAGE2_ADDENDUM.md"
BROAD_DECISION = EXPERIMENT_DIR / "artifacts" / "broad" / "BROAD_DECISION.json"
RESULT_ROOT = ROOT / "results" / "homogeneous_density_threshold_20260826" / "refinement_1024"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts" / "refinement"
RESULTS_PATH = ARTIFACT_DIR / "STAGE2_REFINEMENT_RESULTS.csv"
SUMMARY_PATH = ARTIFACT_DIR / "STAGE2_REFINEMENT_SUMMARY.csv"
DECISION_PATH = ARTIFACT_DIR / "STAGE2_REFINEMENT_DECISION.json"
REPORT_PATH = EXPERIMENT_DIR / "STAGE2_REFINEMENT_REPORT.md"


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


def summarize(coordinates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        rows = [row for row in coordinates if str(row["arm_id"]) == arm_id]
        count = len(rows)
        strong = sum(bool(row["strong_finite_domain_pass"]) for row in rows)
        low_k = sum(bool(row["systematic_low_k_growth_pass"]) for row in rows)
        output.append(
            {
                "stage_id": "refinement_1024",
                "arm_id": arm_id,
                "target_occupancy": rows[0]["target_occupancy"],
                "realized_occupancy": rows[0]["realized_occupancy"],
                "trajectory_count": count,
                "strong_domain_pass_count": strong,
                "strong_domain_pass_fraction": strong / count,
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
                "median_final_entropy_normalized": _median(rows, "final_entropy_normalized"),
            }
        )
    return output


def completion_runtime_seconds(manifest: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for arm_id in manifest["stages"]["refinement_1024"]["arm_ids"]:
        path = RESULT_ROOT / "arms" / arm_id / "completion.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"refinement arm is not complete: {arm_id}")
        rows.append(
            {
                "arm_id": arm_id,
                "elapsed_seconds_this_process": float(payload["elapsed_seconds_this_process"]),
                "completed_utc": payload["completed_utc"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
    return sum(row["elapsed_seconds_this_process"] for row in rows), rows


def report_text(
    *,
    manifest_digest: str,
    decision: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> str:
    plan = decision["broad_refinement_plan"]
    bracket = plan["selected_interval"]
    table = [
        "| Target occupancy | Realized occupancy | Strong domain | Low-k growth | "
        "Median peak L ratio | Median final L ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            f"| {float(row['target_occupancy']):.4f} | "
            f"{float(row['realized_occupancy']):.9f} | "
            f"{int(row['strong_domain_pass_count'])}/{int(row['trajectory_count'])} | "
            f"{int(row['low_k_growth_pass_count'])}/{int(row['trajectory_count'])} | "
            f"{float(row['median_peak_domain_scale_ratio']):.3f} | "
            f"{float(row['median_final_domain_scale_ratio']):.3f} |"
        )
    return "\n".join(
        [
            "# ZWSLab v0.1.4 density-threshold Stage-2 refinement report",
            "",
            "## Frozen selection",
            "",
            "The full 168-trajectory broad grid completed before this refinement was selected. "
            f"The frozen rule selected the {plan['selection_basis']} interval from "
            f"{float(bracket['lower_realized_occupancy']):.9f} to "
            f"{float(bracket['upper_realized_occupancy']):.9f}, then inserted targets at "
            f"spacing {float(plan['spacing']):.3f}. No density was selected from an individual "
            "seed outcome.",
            "",
            f"Refinement manifest SHA-256: `{manifest_digest}`.",
            f"Stage-2 addendum SHA-256: `{decision['stage2_addendum_sha256']}`.",
            "",
            "## Results",
            "",
            *table,
            "",
            f"Completed trajectories: {decision['trajectory_count']}/"
            f"{decision['trajectory_count']}; failures: {decision['failures']}. Exact Q=0 "
            f"held in {decision['exact_q_pass_count']}/{decision['trajectory_count']} runs.",
            "",
            "These measurements are the preregistered refinement stage. The combined broad-plus-"
            "refinement classification, logistic onset estimate, continuous curves, and morphology "
            "are reported in `DENSITY_THRESHOLD_REPORT.md`; this document does not independently "
            "redefine the transition class.",
            "",
            "## Integrity and runtime",
            "",
            f"The sum of per-arm recorded process intervals was "
            f"{float(decision['recorded_process_seconds']):.1f} seconds. All trajectory and snapshot "
            "hashes were verified by the campaign loader before analysis. Large raw trajectories "
            "remain outside git.",
            "",
        ]
    )


def main() -> None:
    manifest, digest, _ = load_and_verify_density_manifest(MANIFEST_PATH)
    broad_decision = json.loads(BROAD_DECISION.read_text(encoding="utf-8"))
    if not broad_decision.get("refinement_plan", {}).get("run_refinement"):
        raise RuntimeError("broad decision did not authorize refinement")
    trajectories = load_completed_stage(
        RESULT_ROOT,
        manifest,
        digest,
        stage_id="refinement_1024",
    )
    frozen = {
        str(row["run_id"]): row
        for row in manifest["stages"]["refinement_1024"]["planned_runs"]
    }
    coordinates = []
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
                "target_occupancy": planned["target_occupancy"],
                "realized_occupancy": planned["realized_occupancy"],
                "occupancy_rounding_error": planned["occupancy_rounding_error"],
                "source_manifest_sha256": digest,
            }
        )
        coordinates.append(row)
    summaries = summarize(coordinates)
    runtime, arm_completions = completion_runtime_seconds(manifest)
    decision = {
        "schema": "zwslab-density-threshold-stage2-refinement-decision-v1",
        "manifest_sha256": digest,
        "broad_decision_sha256": sha256_file(BROAD_DECISION),
        "stage2_addendum_sha256": sha256_file(ADDENDUM),
        "broad_refinement_plan": broad_decision["refinement_plan"],
        "trajectory_count": len(coordinates),
        "density_count": len(summaries),
        "failures": 0,
        "exact_q_pass_count": sum(bool(row["exact_q_at_all_samples"]) for row in coordinates),
        "recorded_process_seconds": runtime,
        "arm_completions": arm_completions,
        "interpretation_boundary": (
            "stage-specific evidence only; final class and fit use the combined 1024 map"
        ),
    }
    write_csv(RESULTS_PATH, coordinates)
    write_csv(SUMMARY_PATH, summaries)
    atomic_write_json(DECISION_PATH, decision)
    atomic_write_text(
        REPORT_PATH,
        report_text(manifest_digest=digest, decision=decision, summaries=summaries),
    )
    artifacts = [RESULTS_PATH, SUMMARY_PATH, DECISION_PATH, REPORT_PATH]
    atomic_write_json(
        ARTIFACT_DIR / "STAGE2_REFINEMENT_ARTIFACTS.json",
        {
            "schema": "zwslab-density-threshold-stage2-refinement-artifacts-v1",
            "artifacts": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
                for path in artifacts
            ],
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
