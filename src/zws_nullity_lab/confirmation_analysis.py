from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .atomic import atomic_write_json, atomic_write_text, sha256_file


def _finite_domain_scale(sample: dict[str, Any]) -> float | None:
    value = sample["empty_domains"].get("finite_area_weighted_linear_scale")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _dispersion_64(sample: dict[str, Any]) -> float | None:
    value = sample["coarse_density"]["64"].get("dispersion_ratio_analytic")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _late_half_slope(samples: list[dict[str, Any]]) -> float | None:
    final_tick = int(samples[-1]["tick"])
    pairs = [
        (int(sample["tick"]), _finite_domain_scale(sample))
        for sample in samples
        if int(sample["tick"]) >= final_tick / 2
    ]
    valid = [(tick, value) for tick, value in pairs if value is not None]
    if len(valid) < 3:
        return None
    ticks = np.asarray([tick for tick, _ in valid], dtype=np.float64)
    values = np.asarray([value for _, value in valid], dtype=np.float64)
    return float(np.polyfit(ticks, values, 1)[0])


def evaluate_trajectory_gate(
    trajectory: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    samples = list(trajectory.get("samples", []))
    if trajectory.get("status") != "complete" or not samples:
        return {
            "run_id": trajectory.get("run_id"),
            "seed_index": trajectory.get("seed_index"),
            "counts_as_macroscopic_coarsening": False,
            "reason": "trajectory missing or incomplete",
        }
    initial_scale = _finite_domain_scale(samples[0])
    final_scale = _finite_domain_scale(samples[-1])
    scale_ratio = (
        final_scale / initial_scale
        if initial_scale is not None and final_scale is not None and initial_scale > 0
        else None
    )
    final_dispersion = _dispersion_64(samples[-1])
    slope = _late_half_slope(samples)
    charge_violations = int(trajectory.get("charge_violation_count", -1))
    criteria = {
        "final_dispersion": (
            final_dispersion is not None
            and final_dispersion >= float(gate["final_dispersion_ratio_64_min"])
        ),
        "finite_empty_domain_scale_ratio": (
            scale_ratio is not None
            and scale_ratio >= float(gate["final_empty_domain_linear_scale_ratio_min"])
        ),
        "late_half_scale_slope": slope is not None and slope > 0.0,
        "exact_charge": charge_violations == int(gate["charge_violation_count_must_equal"]),
    }
    return {
        "run_id": trajectory["run_id"],
        "seed_index": trajectory["seed_index"],
        "counts_as_macroscopic_coarsening": all(criteria.values()),
        "criteria": criteria,
        "final_dispersion_ratio_64": final_dispersion,
        "initial_finite_empty_domain_scale": initial_scale,
        "final_finite_empty_domain_scale": final_scale,
        "finite_empty_domain_scale_ratio": scale_ratio,
        "late_half_finite_empty_domain_scale_slope": slope,
        "charge_violation_count": charge_violations,
    }


def _load_trajectories(root: Path, stage_id: str) -> list[dict[str, Any]]:
    paths = sorted((root / "trajectories").glob(f"{stage_id}__*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _median_final_dispersion(evaluations: list[dict[str, Any]]) -> float | None:
    values = [
        float(item["final_dispersion_ratio_64"])
        for item in evaluations
        if item.get("final_dispersion_ratio_64") is not None
    ]
    return median(values) if values else None


def evaluate_stage_a_gate(
    trajectories: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    gate = manifest["configuration"]["continuation_gate"]
    planned = int(manifest["configuration"]["stage_a"]["seed_indices"][-1]) + 1
    planned *= len(manifest["configuration"]["stage_a"]["condition_ids"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory["condition_id"])].append(trajectory)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["seed_index"]))

    evaluations = {
        condition_id: [evaluate_trajectory_gate(item, gate) for item in values]
        for condition_id, values in grouped.items()
    }
    winner = evaluations.get("winner", [])
    rotation = evaluations.get("rotation", [])
    winner_count = sum(item["counts_as_macroscopic_coarsening"] for item in winner)
    rotation_count = sum(item["counts_as_macroscopic_coarsening"] for item in rotation)
    winner_median = _median_final_dispersion(winner)
    rotation_median = _median_final_dispersion(rotation)
    rotation_dispersion_equal = (
        winner_median is not None
        and rotation_median is not None
        and rotation_median
        >= float(gate["rotation_equal_median_dispersion_fraction"]) * winner_median
    )
    rotation_equal = (
        rotation_count >= int(gate["rotation_equal_required_seed_count"])
        and rotation_dispersion_equal
    )
    winner_reproducible = winner_count >= int(gate["winner_required_seed_count"])
    proceed = winner_reproducible and not rotation_equal

    matched_summaries = []
    for condition_id in sorted(key for key in evaluations if key.startswith("matched_")):
        items = evaluations[condition_id]
        matched_summaries.append(
            {
                "condition_id": condition_id,
                "macroscopic_seed_count": sum(
                    item["counts_as_macroscopic_coarsening"] for item in items
                ),
                "median_final_dispersion_ratio_64": _median_final_dispersion(items),
            }
        )

    matched_reproducible_count = sum(
        item["macroscopic_seed_count"] >= int(gate["winner_required_seed_count"])
        for item in matched_summaries
    )
    matched_medians = [
        float(item["median_final_dispersion_ratio_64"])
        for item in matched_summaries
        if item["median_final_dispersion_ratio_64"] is not None
    ]
    failed_count = sum(item.get("status") != "complete" for item in trajectories)
    failed_count += max(0, planned - len(trajectories))
    if failed_count:
        classification = "E"
        classification_reason = "One or more planned Stage 7A trajectories failed or are missing."
    elif not winner_reproducible:
        classification = "D"
        classification_reason = "The discovery winner did not reproduce the preregistered criterion."
    elif rotation_equal:
        classification = "C"
        classification_reason = "Pure rotation met the preregistered equal-explanation rule."
    elif matched_reproducible_count >= 4:
        classification = "B"
        classification_reason = (
            "At least half of the independently matched reactive-rotation controls reproduced "
            "the same macroscopic criterion."
        )
    elif matched_medians and winner_median is not None and winner_median > max(matched_medians):
        classification = "A"
        classification_reason = (
            "The winner passed the gate, rotation did not explain it equally, and its median final "
            "dispersion exceeded every matched-control median."
        )
    else:
        classification = "E"
        classification_reason = (
            "The winner passed the continuation gate, but its position within the matched-rule "
            "distribution was mixed rather than clearly exceptional or generic."
        )

    return {
        "schema": "zwslab-stage7a-continuation-gate-v1",
        "planned_trajectory_count": planned,
        "observed_trajectory_count": len(trajectories),
        "failed_or_incomplete_trajectory_count": failed_count,
        "winner_macroscopic_seed_count": winner_count,
        "winner_required_seed_count": int(gate["winner_required_seed_count"]),
        "winner_reproducible": winner_reproducible,
        "rotation_macroscopic_seed_count": rotation_count,
        "winner_median_final_dispersion_ratio_64": winner_median,
        "rotation_median_final_dispersion_ratio_64": rotation_median,
        "rotation_dispersion_equal": rotation_dispersion_equal,
        "rotation_explains_equally_well": rotation_equal,
        "proceed_to_stage7b": proceed and failed_count == 0,
        "classification": classification,
        "classification_reason": classification_reason,
        "matched_controls_reproducible_count": matched_reproducible_count,
        "matched_controls": matched_summaries,
        "trajectory_gate_evaluations": evaluations,
        "finite_domain_scale_definition": (
            "sqrt(sum(area_i^2)/sum(area_i)) across exact four-neighbour toroidal empty "
            "components after excluding the largest background component"
        ),
    }


def _condition_color(condition_id: str) -> tuple[float, float, float, float] | str:
    if condition_id == "winner":
        return "#ef5250"
    if condition_id == "rotation":
        return "#4895ef"
    if condition_id == "identity":
        return "#808080"
    index = int(condition_id.split("_")[-1])
    return plt.get_cmap("viridis")((index + 1) / 9)


def _series_value(sample: dict[str, Any], metric: str) -> float | None:
    if metric == "occupied_fraction":
        return sample.get("occupied_fraction")
    if metric == "finite_domain_scale":
        return _finite_domain_scale(sample)
    if metric == "dispersion_64":
        return _dispersion_64(sample)
    if metric == "structure_wavelength":
        return sample["structure_factor"].get("characteristic_wavelength_cells")
    if metric == "recurrence_4":
        return sample["exact_state_recurrence"].get("4")
    if metric == "recurrence_8":
        return sample["exact_state_recurrence"].get("8")
    if metric == "total_charge":
        return sample.get("total_charge")
    raise KeyError(metric)


def plot_stage_curves(
    trajectories: list[dict[str, Any]], output: str | Path, *, stage_label: str = "Stage 7A"
) -> list[dict[str, str]]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        if trajectory.get("status") == "complete":
            grouped[str(trajectory["condition_id"])].append(trajectory)
    specifications = [
        ("occupied_fraction", "Occupied fraction", "occupancy.png"),
        ("finite_domain_scale", "Finite empty-domain scale (cells)", "domain_scale.png"),
        ("dispersion_64", "64×64 density dispersion / shuffle expectation", "dispersion64.png"),
        ("structure_wavelength", "Structure-factor wavelength (cells)", "structure_wavelength.png"),
        ("recurrence_4", "Exact-state recurrence, lag 4", "recurrence_lag4.png"),
        ("recurrence_8", "Exact-state recurrence, lag 8", "recurrence_lag8.png"),
        ("total_charge", "Exact total charge", "charge.png"),
    ]
    outputs = []
    for metric, ylabel, filename in specifications:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for condition_id in sorted(grouped, key=lambda value: (value.startswith("matched_"), value)):
            rows = grouped[condition_id]
            ticks = np.asarray([sample["tick"] for sample in rows[0]["samples"]])
            values = np.asarray(
                [
                    [
                        np.nan if _series_value(sample, metric) is None else _series_value(sample, metric)
                        for sample in trajectory["samples"]
                    ]
                    for trajectory in rows
                ],
                dtype=np.float64,
            )
            available = np.any(np.isfinite(values), axis=0)
            if not np.any(available):
                continue
            plot_ticks = ticks[available]
            available_values = values[:, available]
            center = np.nanmedian(available_values, axis=0)
            lower = np.nanquantile(available_values, 0.25, axis=0)
            upper = np.nanquantile(available_values, 0.75, axis=0)
            color = _condition_color(condition_id)
            width = 2.4 if condition_id in {"winner", "rotation"} else 1.0
            alpha = 1.0 if condition_id in {"winner", "rotation", "identity"} else 0.55
            ax.plot(
                plot_ticks,
                center,
                label=condition_id,
                color=color,
                linewidth=width,
                alpha=alpha,
            )
            if condition_id in {"winner", "rotation"}:
                ax.fill_between(plot_ticks, lower, upper, color=color, alpha=0.15)
        ax.set_xlabel("Alternating Margolus phases")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{stage_label} paired-seed curves — {ylabel}")
        ax.grid(alpha=0.2)
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append({"metric": metric, "path": str(path), "sha256": sha256_file(path)})
    return outputs


def write_stage_a_report(
    root: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    campaign_root = Path(root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    trajectories = _load_trajectories(campaign_root, "stage7a")
    gate = evaluate_stage_a_gate(trajectories, manifest)
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = plot_stage_curves(trajectories, output_dir / "plots")
    gate_path = output_dir / "STAGE7A_GATE.json"
    atomic_write_json(gate_path, gate)
    atomic_write_text(
        gate_path.with_suffix(gate_path.suffix + ".sha256"),
        f"{sha256_file(gate_path)}  {gate_path.name}\n",
    )
    decision = "PROCEED" if gate["proceed_to_stage7b"] else "DO NOT PROCEED"
    matched_rows = "\n".join(
        f"| {item['condition_id']} | {item['macroscopic_seed_count']}/8 | "
        f"{item['median_final_dispersion_ratio_64']} |"
        for item in gate["matched_controls"]
    )
    report = f"""# ZWSLab Stage 7A Confirmation Report

## Outcome

**Frozen continuation decision: {decision} to Stage 7B.**

- Winner qualifying seeds: **{gate['winner_macroscopic_seed_count']}/8** (required 6/8).
- Pure-rotation qualifying seeds: **{gate['rotation_macroscopic_seed_count']}/8**.
- Pure rotation explains the result equally well: **{gate['rotation_explains_equally_well']}**.
- Outcome classification: **{gate['classification']} — {gate['classification_reason']}**
- Failed or incomplete planned trajectories: **{gate['failed_or_incomplete_trajectory_count']}**.

## What was tested

This is the preregistered confirmation of whether frozen rule `d1fe99...` reproducibly produces
macroscopic domain formation/coarsening from the localized balanced initial-condition family,
compared with pure rotation, identity, and eight independently matched 24-reactive-input controls.
All 88 planned Stage 7A trajectories used the eight seed values frozen before outcomes.

The finite empty-domain scale is `{gate['finite_domain_scale_definition']}`. The largest empty
component is reported but excluded from that scale so the initial exterior of a localized seed is
not relabelled as a newly formed finite domain.

## Matched-control distribution

| Condition | Qualifying seeds | Median final dispersion ratio (64×64) |
|---|---:|---:|
{matched_rows}

## Interpretation boundary

This experiment can support or weaken a claim about reversible null-conserving spatial domain
formation. It does **not** test life, organisms, replication, heredity, intelligence,
consciousness, FOA, primitive occurrence, ETNE, cosmology, or a fundamental process measure.
High recurrence can be inherited from the rotational scaffold; occupancy can grow while exact
signed charge remains zero because empty and bound states both carry zero charge.

## Artifacts

- Gate record: `{gate_path.name}` (`{sha256_file(gate_path)}`)
- Manifest: `{manifest_file}` (`{sha256_file(manifest_file)}`)
- Complete trajectory records read: `{len(trajectories)}`
- Plots: `{len(plots)}` paired-seed curve figures with SHA-256 values in `REPORT_INDEX.json`.
"""
    report_path = output_dir / "STAGE7A_REPORT.md"
    atomic_write_text(report_path, report)
    index = {
        "schema": "zwslab-stage7a-report-index-v1",
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "manifest": {"path": str(manifest_file), "sha256": sha256_file(manifest_file)},
        "plots": plots,
        "trajectory_count": len(trajectories),
        "trajectory_sha256": {
            path.name: sha256_file(path)
            for path in sorted((campaign_root / "trajectories").glob("stage7a__*.json"))
        },
    }
    atomic_write_json(output_dir / "REPORT_INDEX.json", index)
    return {"gate": gate, "report": str(report_path), "index": index}


def write_stage_b_report(
    root: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    campaign_root = Path(root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    trajectories = _load_trajectories(campaign_root, "stage7b")
    gate = manifest["configuration"]["continuation_gate"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory["condition_id"])].append(trajectory)
    summaries = []
    evaluations: dict[str, list[dict[str, Any]]] = {}
    for condition_id in manifest["configuration"]["stage_b"]["condition_ids"]:
        rows = sorted(grouped.get(condition_id, []), key=lambda item: int(item["seed_index"]))
        items = [evaluate_trajectory_gate(row, gate) for row in rows]
        evaluations[condition_id] = items
        dispersion_values = [
            float(item["final_dispersion_ratio_64"])
            for item in items
            if item.get("final_dispersion_ratio_64") is not None
        ]
        scale_ratios = [
            float(item["finite_empty_domain_scale_ratio"])
            for item in items
            if item.get("finite_empty_domain_scale_ratio") is not None
        ]
        slopes = [
            float(item["late_half_finite_empty_domain_scale_slope"])
            for item in items
            if item.get("late_half_finite_empty_domain_scale_slope") is not None
        ]
        summaries.append(
            {
                "condition_id": condition_id,
                "complete_trajectory_count": sum(
                    row.get("status") == "complete" for row in rows
                ),
                "macroscopic_seed_count": sum(
                    item["counts_as_macroscopic_coarsening"] for item in items
                ),
                "median_final_dispersion_ratio_64": (
                    median(dispersion_values) if dispersion_values else None
                ),
                "median_finite_empty_domain_scale_ratio": (
                    median(scale_ratios) if scale_ratios else None
                ),
                "median_late_half_scale_slope": median(slopes) if slopes else None,
                "charge_violation_count": sum(
                    int(row.get("charge_violation_count", -1)) for row in rows
                ),
            }
        )
    planned_count = 4 * len(manifest["configuration"]["stage_b"]["condition_ids"])
    failed_count = sum(row.get("status") != "complete" for row in trajectories)
    failed_count += max(0, planned_count - len(trajectories))
    result = {
        "schema": "zwslab-stage7b-large-scale-summary-v1",
        "planned_trajectory_count": planned_count,
        "observed_trajectory_count": len(trajectories),
        "failed_or_incomplete_trajectory_count": failed_count,
        "conditions": summaries,
        "trajectory_gate_evaluations": evaluations,
        "interpretation_rule": (
            "Stage 7B is a gated scale-up, not a second opportunity to tune the Stage 7A gate. "
            "The same per-trajectory criteria are reported descriptively for the first four "
            "preregistered seeds and six frozen conditions."
        ),
    }
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = plot_stage_curves(trajectories, output_dir / "plots_stage7b", stage_label="Stage 7B")
    summary_path = output_dir / "STAGE7B_SUMMARY.json"
    atomic_write_json(summary_path, result)
    atomic_write_text(
        summary_path.with_suffix(summary_path.suffix + ".sha256"),
        f"{sha256_file(summary_path)}  {summary_path.name}\n",
    )
    table = "\n".join(
        f"| {item['condition_id']} | {item['macroscopic_seed_count']}/4 | "
        f"{item['median_final_dispersion_ratio_64']} | "
        f"{item['median_finite_empty_domain_scale_ratio']} | "
        f"{item['charge_violation_count']} |"
        for item in summaries
    )
    report = f"""# ZWSLab Stage 7B Large-Scale Confirmation

## Scope

Stage 7B was authorized by the frozen Stage 7A gate. It uses 4,096×4,096 worlds for 65,536
alternating Margolus phases, the first four preregistered seeds, and the winner, pure rotation,
and first four matched controls in manifest order. It does not retune the gate or choose controls
from their Stage 7A appearance.

## Results

| Condition | Same-criterion seeds | Median final dispersion | Median finite-domain scale ratio | Charge violations |
|---|---:|---:|---:|---:|
{table}

- Failed or incomplete planned trajectories: **{failed_count}**.
- Complete trajectory records read: **{len(trajectories)}**.

## Interpretation boundary

These results address reversible null-conserving spatial domain formation and its scale dependence.
They do not establish life, replication, heredity, intelligence, consciousness, FOA, primitive
occurrence, ETNE, cosmology, or a fundamental process measure. A matched rule with high coarse
dispersion but no finite-domain growth is not counted as the same phenomenon.
"""
    report_path = output_dir / "STAGE7B_REPORT.md"
    atomic_write_text(report_path, report)
    index = {
        "schema": "zwslab-stage7b-report-index-v1",
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "manifest": {"path": str(manifest_file), "sha256": sha256_file(manifest_file)},
        "plots": plots,
        "trajectory_count": len(trajectories),
        "trajectory_sha256": {
            path.name: sha256_file(path)
            for path in sorted((campaign_root / "trajectories").glob("stage7b__*.json"))
        },
    }
    atomic_write_json(output_dir / "STAGE7B_REPORT_INDEX.json", index)
    return {"summary": result, "report": str(report_path), "index": index}
