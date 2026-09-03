from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .confirmation_analysis import evaluate_stage_a_gate, evaluate_trajectory_gate


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _median_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return median(valid) if valid else None


def _late_median(samples: list[dict[str, Any]], key: str, lag: str) -> float | None:
    final_tick = int(samples[-1]["tick"])
    values = [
        _finite(sample[key].get(lag))
        for sample in samples
        if int(sample["tick"]) >= final_tick / 2
    ]
    return _median_or_none(values)


def trajectory_endpoints(
    trajectory: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    samples = list(trajectory.get("samples", []))
    row: dict[str, Any] = {
        "stage_id": trajectory.get("stage_id"),
        "condition_id": trajectory.get("condition_id"),
        "seed_index": trajectory.get("seed_index"),
        "initial_seed": trajectory.get("initial_seed"),
        "run_id": trajectory.get("run_id"),
        "status": trajectory.get("status"),
        "world": trajectory.get("world"),
        "phases": trajectory.get("phases"),
        "rule_table_sha256_uint8": trajectory.get("rule_table_sha256_uint8"),
        "initial_state_sha256_uint8": trajectory.get("initial_state_sha256_uint8"),
        "final_state_sha256_uint8": trajectory.get("final_state_sha256_uint8"),
        "charge_violation_count": trajectory.get("charge_violation_count"),
        "sample_count": len(samples),
    }
    if trajectory.get("status") != "complete" or not samples:
        row["counts_as_macroscopic_coarsening"] = False
        return row
    evaluation = evaluate_trajectory_gate(trajectory, gate)
    first, final = samples[0], samples[-1]
    finite_scales = [
        (int(sample["tick"]), _finite(sample["empty_domains"].get("finite_area_weighted_linear_scale")))
        for sample in samples
    ]
    finite_scales = [(tick, value) for tick, value in finite_scales if value is not None]
    peak_tick, peak_scale = max(finite_scales, key=lambda item: item[1])
    initial_scale = _finite(first["empty_domains"].get("finite_area_weighted_linear_scale"))
    row.update(
        {
            "counts_as_macroscopic_coarsening": evaluation[
                "counts_as_macroscopic_coarsening"
            ],
            "initial_occupied_fraction": _finite(first.get("occupied_fraction")),
            "final_occupied_fraction": _finite(final.get("occupied_fraction")),
            "occupied_fraction_change": _finite(final.get("occupied_fraction"))
            - _finite(first.get("occupied_fraction")),
            "final_activity": _finite(final.get("activity_one_phase")),
            "final_reaction_density": _finite(final.get("reaction_density_one_phase")),
            "final_entropy": _finite(final.get("entropy_normalized")),
            "final_boundary_density": _finite(final.get("boundary_density")),
            "final_dispersion_ratio_64": evaluation["final_dispersion_ratio_64"],
            "initial_finite_empty_domain_scale": evaluation[
                "initial_finite_empty_domain_scale"
            ],
            "final_finite_empty_domain_scale": evaluation["final_finite_empty_domain_scale"],
            "finite_empty_domain_scale_ratio": evaluation[
                "finite_empty_domain_scale_ratio"
            ],
            "peak_finite_empty_domain_scale": peak_scale,
            "peak_finite_empty_domain_scale_tick": peak_tick,
            "peak_finite_empty_domain_scale_ratio": (
                peak_scale / initial_scale if initial_scale is not None and initial_scale > 0 else None
            ),
            "late_half_finite_empty_domain_scale_slope": evaluation[
                "late_half_finite_empty_domain_scale_slope"
            ],
            "final_structure_wavelength": _finite(
                final["structure_factor"].get("characteristic_wavelength_cells")
            ),
            "late_median_exact_recurrence_lag4": _late_median(
                samples, "exact_state_recurrence", "4"
            ),
            "late_median_exact_recurrence_lag8": _late_median(
                samples, "exact_state_recurrence", "8"
            ),
            "late_median_coarse_correlation_lag4": _late_median(
                samples, "coarse_occupancy_correlation_block64", "4"
            ),
            "late_median_coarse_correlation_lag8": _late_median(
                samples, "coarse_occupancy_correlation_block64", "8"
            ),
            "final_total_charge": int(final["total_charge"]),
        }
    )
    return row


def paired_bootstrap_difference(
    winner: list[float], comparison: list[float], *, seed: int, draws: int = 20_000
) -> dict[str, Any]:
    if len(winner) != len(comparison) or not winner:
        raise ValueError("paired samples must have the same non-zero length")
    differences = np.asarray(winner, dtype=np.float64) - np.asarray(
        comparison, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    standard_deviation = float(differences.std(ddof=1)) if len(differences) > 1 else None
    return {
        "paired_seed_count": len(differences),
        "mean_paired_difference": float(differences.mean()),
        "median_paired_difference": float(np.median(differences)),
        "paired_difference_standard_deviation": standard_deviation,
        "paired_standardized_mean_difference_dz": (
            float(differences.mean() / standard_deviation)
            if standard_deviation is not None and standard_deviation > 0
            else None
        ),
        "percentile_bootstrap_95ci_mean_difference": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "bootstrap_draws": draws,
    }


def _bootstrap_seed(stage: str, comparison: str, metric: str) -> int:
    digest = hashlib.sha256(f"{stage}|{comparison}|{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_trajectories(campaign_root: Path, stage_id: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((campaign_root / "trajectories").glob(f"{stage_id}__*.json"))
    ]


def validate_confirmation(
    trajectories: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    expected_manifest = sha256_file(Path(manifest["_manifest_path"]))
    conditions = {item["condition_id"]: item for item in manifest["conditions"]}
    expected_runs: set[tuple[str, str, int]] = set()
    stage_specs = {
        "stage7a": (
            manifest["configuration"]["stage_a"]["condition_ids"],
            manifest["configuration"]["stage_a"]["seed_indices"],
            manifest["configuration"]["stage_a"]["phases"],
            65,
        ),
        "stage7b": (
            manifest["configuration"]["stage_b"]["condition_ids"],
            manifest["configuration"]["stage_b"]["initial_seed_indices"],
            manifest["configuration"]["stage_b"]["phases"],
            129,
        ),
    }
    for stage_id, (condition_ids, seed_indices, _phases, _samples) in stage_specs.items():
        expected_runs.update(
            (stage_id, condition_id, int(seed_index))
            for condition_id in condition_ids
            for seed_index in seed_indices
        )
    observed_runs: set[tuple[str, str, int]] = set()
    initial_hashes: dict[tuple[str, int], set[str]] = {}
    for trajectory in trajectories:
        stage_id = str(trajectory.get("stage_id"))
        condition_id = str(trajectory.get("condition_id"))
        seed_index = int(trajectory.get("seed_index", -1))
        key = (stage_id, condition_id, seed_index)
        if key in observed_runs:
            errors.append(f"duplicate trajectory: {key}")
        observed_runs.add(key)
        if key not in expected_runs:
            errors.append(f"unplanned trajectory: {key}")
            continue
        condition_ids, seed_indices, expected_phases, expected_samples = stage_specs[stage_id]
        del condition_ids, seed_indices
        checks = {
            "status": trajectory.get("status") == "complete",
            "phases": int(trajectory.get("phases", -1)) == int(expected_phases),
            "final_tick": bool(trajectory.get("samples"))
            and int(trajectory["samples"][-1]["tick"]) == int(expected_phases),
            "sample_count": len(trajectory.get("samples", [])) == expected_samples,
            "charge": int(trajectory.get("charge_violation_count", -1)) == 0,
            "manifest": trajectory.get("manifest_sha256") == expected_manifest,
            "rule": trajectory.get("rule_table_sha256_uint8")
            == conditions[condition_id]["rule_table_sha256_uint8"],
            "seed": int(trajectory.get("initial_seed", -1))
            == int(manifest["confirmation_seeds"][seed_index]),
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{trajectory.get('run_id')}: failed {name} check")
        initial_hashes.setdefault((stage_id, seed_index), set()).add(
            str(trajectory.get("initial_state_sha256_uint8"))
        )
    missing = sorted(expected_runs - observed_runs)
    errors.extend(f"missing trajectory: {item}" for item in missing)
    for key, hashes in initial_hashes.items():
        if len(hashes) != 1:
            errors.append(f"paired initial-state hashes differ for {key}")
    return {
        "schema": "zwslab-confirmation-integrity-v1",
        "valid": not errors,
        "expected_trajectory_count": len(expected_runs),
        "observed_trajectory_count": len(trajectories),
        "error_count": len(errors),
        "errors": errors,
        "paired_initial_state_hash_groups": len(initial_hashes),
        "manifest_sha256": expected_manifest,
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "final_occupied_fraction",
        "occupied_fraction_change",
        "final_dispersion_ratio_64",
        "finite_empty_domain_scale_ratio",
        "peak_finite_empty_domain_scale_ratio",
        "late_half_finite_empty_domain_scale_slope",
        "final_structure_wavelength",
        "late_median_exact_recurrence_lag4",
        "late_median_exact_recurrence_lag8",
    ]
    return {
        "condition_id": rows[0]["condition_id"],
        "complete_trajectory_count": sum(row["status"] == "complete" for row in rows),
        "macroscopic_seed_count": sum(
            bool(row.get("counts_as_macroscopic_coarsening")) for row in rows
        ),
        "charge_violation_count": sum(int(row.get("charge_violation_count", -1)) for row in rows),
        "medians": {
            metric: _median_or_none([_finite(row.get(metric)) for row in rows])
            for metric in metrics
        },
    }


def _effect_rows(
    endpoints: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    metrics = [
        "final_occupied_fraction",
        "occupied_fraction_change",
        "final_dispersion_ratio_64",
        "finite_empty_domain_scale_ratio",
        "late_half_finite_empty_domain_scale_slope",
        "late_median_exact_recurrence_lag4",
        "late_median_exact_recurrence_lag8",
    ]
    output: list[dict[str, Any]] = []
    for stage_id in ["stage7a", "stage7b"]:
        stage_rows = [row for row in endpoints if row["stage_id"] == stage_id]
        winner_by_seed = {
            int(row["seed_index"]): row
            for row in stage_rows
            if row["condition_id"] == "winner"
        }
        stage_key = "stage_a" if stage_id == "stage7a" else "stage_b"
        for comparison in manifest["configuration"][stage_key]["condition_ids"]:
            if comparison in {"winner", "identity"}:
                continue
            comparison_by_seed = {
                int(row["seed_index"]): row
                for row in stage_rows
                if row["condition_id"] == comparison
            }
            for metric in metrics:
                seeds = sorted(set(winner_by_seed) & set(comparison_by_seed))
                pairs = [
                    (winner_by_seed[seed].get(metric), comparison_by_seed[seed].get(metric))
                    for seed in seeds
                ]
                valid = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
                if not valid:
                    continue
                result = paired_bootstrap_difference(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    seed=_bootstrap_seed(stage_id, comparison, metric),
                )
                output.append(
                    {
                        "stage_id": stage_id,
                        "contrast": f"winner-minus-{comparison}",
                        "comparison_condition": comparison,
                        "metric": metric,
                        **result,
                    }
                )
    return output


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def write_overnight_report(
    campaign: str | Path,
    manifest_path: str | Path,
    experiment_output: str | Path,
    report_output: str | Path,
) -> dict[str, Any]:
    campaign_root = Path(campaign).resolve()
    manifest_file = Path(manifest_path).resolve()
    experiment_dir = Path(experiment_output).resolve()
    report_dir = Path(report_output).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_file)
    trajectories = _load_trajectories(campaign_root, "stage7a") + _load_trajectories(
        campaign_root, "stage7b"
    )
    gate = manifest["configuration"]["continuation_gate"]
    endpoints = [trajectory_endpoints(trajectory, gate) for trajectory in trajectories]
    integrity = validate_confirmation(trajectories, manifest)
    stage_a_gate = evaluate_stage_a_gate(
        [row for row in trajectories if row.get("stage_id") == "stage7a"], manifest
    )
    condition_summaries: dict[str, list[dict[str, Any]]] = {}
    for stage_id in ["stage7a", "stage7b"]:
        stage_rows = [row for row in endpoints if row["stage_id"] == stage_id]
        condition_ids = sorted({str(row["condition_id"]) for row in stage_rows})
        condition_summaries[stage_id] = [
            _condition_summary([row for row in stage_rows if row["condition_id"] == condition])
            for condition in condition_ids
        ]
    effects = _effect_rows(endpoints, manifest)
    stage_b = {item["condition_id"]: item for item in condition_summaries["stage7b"]}
    winner_b = stage_b.get("winner", {"macroscopic_seed_count": 0, "medians": {}})
    rotation_b = stage_b.get("rotation", {"macroscopic_seed_count": 0, "medians": {}})
    winner_dispersion = winner_b["medians"].get("final_dispersion_ratio_64")
    rotation_dispersion = rotation_b["medians"].get("final_dispersion_ratio_64")
    rotation_equal = (
        rotation_b["macroscopic_seed_count"] >= 3
        and winner_dispersion is not None
        and rotation_dispersion is not None
        and rotation_dispersion >= 0.8 * winner_dispersion
    )
    matched_b = [value for key, value in stage_b.items() if key.startswith("matched_")]
    failed_count = sum(row.get("status") != "complete" for row in trajectories)
    if not integrity["valid"] or failed_count:
        classification = "E"
        classification_reason = "The confirmatory matrix is incomplete or failed integrity checks."
    elif winner_b["macroscopic_seed_count"] < 3:
        classification = "D"
        classification_reason = (
            "The winner did not reproduce the full sustained-coarsening criterion in at least "
            "three of four Stage 7B seeds."
        )
    elif rotation_equal:
        classification = "C"
        classification_reason = "Pure rotation reproduced the same large-scale criterion and dispersion."
    elif stage_a_gate["classification"] == "B" and sum(
        item["macroscopic_seed_count"] >= 3 for item in matched_b
    ) >= 2:
        classification = "B"
        classification_reason = "The effect reproduced across multiple matched same-family rules."
    elif stage_a_gate["classification"] == "A" and winner_dispersion is not None and all(
        item["medians"].get("final_dispersion_ratio_64") is None
        or winner_dispersion > item["medians"]["final_dispersion_ratio_64"]
        for item in matched_b
    ):
        classification = "A"
        classification_reason = "Winner-specific strength reproduced at both registered scales."
    else:
        classification = "E"
        classification_reason = (
            "The winner reproduced, but its matched-family position or diagnostic pattern was mixed."
        )

    winner_a = next(
        item for item in condition_summaries["stage7a"] if item["condition_id"] == "winner"
    )
    rotation_a = next(
        item for item in condition_summaries["stage7a"] if item["condition_id"] == "rotation"
    )
    matched_a = [
        item
        for item in condition_summaries["stage7a"]
        if item["condition_id"].startswith("matched_")
    ]
    stage_a_dispersion_rank = 1 + sum(
        item["medians"].get("final_dispersion_ratio_64") is not None
        and winner_a["medians"]["final_dispersion_ratio_64"]
        < item["medians"]["final_dispersion_ratio_64"]
        for item in matched_a
    )
    stage_b_dispersion_rank = 1 + sum(
        item["medians"].get("final_dispersion_ratio_64") is not None
        and winner_b["medians"]["final_dispersion_ratio_64"]
        < item["medians"]["final_dispersion_ratio_64"]
        for item in matched_b
    )
    winner_recurrence = [
        winner_b["medians"].get("late_median_exact_recurrence_lag4"),
        winner_b["medians"].get("late_median_exact_recurrence_lag8"),
    ]
    rotation_recurrence = [
        rotation_b["medians"].get("late_median_exact_recurrence_lag4"),
        rotation_b["medians"].get("late_median_exact_recurrence_lag8"),
    ]
    oscillator_inherited = all(
        winner is not None
        and rotation is not None
        and (winner == 0 or rotation >= 0.8 * winner)
        for winner, rotation in zip(winner_recurrence, rotation_recurrence, strict=True)
    )
    occupancy_change = winner_b["medians"].get("occupied_fraction_change")
    winner_b_rows = [
        row
        for row in endpoints
        if row["stage_id"] == "stage7b" and row["condition_id"] == "winner"
    ]
    large_domain_seed_count = sum(
        row.get("final_dispersion_ratio_64") is not None
        and row["final_dispersion_ratio_64"] >= float(gate["final_dispersion_ratio_64_min"])
        and row.get("finite_empty_domain_scale_ratio") is not None
        and row["finite_empty_domain_scale_ratio"]
        >= float(gate["final_empty_domain_linear_scale_ratio_min"])
        for row in winner_b_rows
    )
    peak_scale_ratio = winner_b["medians"].get("peak_finite_empty_domain_scale_ratio")
    final_scale_ratio = winner_b["medians"].get("finite_empty_domain_scale_ratio")
    questions = {
        "1_large_domains": (
            f"Yes: all {large_domain_seed_count}/4 Stage 7B winner seeds retained both large "
            f"dispersion and at least twofold finite-domain growth; the median final scale ratio "
            f"was {_format_number(final_scale_ratio)}. The stricter sustained-coarsening count was "
            f"{winner_b['macroscopic_seed_count']}/4 because the late slope was negative."
        ),
        "2_pure_rotation": (
            f"Pure rotation counts were {rotation_a['macroscopic_seed_count']}/8 and "
            f"{rotation_b['macroscopic_seed_count']}/4; equal-explanation status at 4096² was "
            f"{rotation_equal}."
        ),
        "3_unusual_vs_matched": (
            f"Winner final-dispersion rank was {stage_a_dispersion_rank}/9 against the eight "
            f"Stage 7A matched controls and {stage_b_dispersion_rank}/5 against the four Stage 7B "
            f"controls, because matched_00 had still greater coarse dispersion. However, the "
            f"winner's median final finite-domain scale ratio was {_format_number(final_scale_ratio)} "
            "while every Stage 7B matched-control median was below 1. The winner was unusual in "
            f"domain growth but not uniquely largest in coarse dispersion. Strict classification "
            f"is {classification}: {classification_reason}"
        ),
        "4_rotation_oscillator": (
            f"{'Yes, largely' if oscillator_inherited else 'Not clearly'}: Stage 7B median late "
            f"lag-4/8 recurrence was {_format_number(winner_recurrence[0])}/"
            f"{_format_number(winner_recurrence[1])} for the winner and "
            f"{_format_number(rotation_recurrence[0])}/"
            f"{_format_number(rotation_recurrence[1])} for pure rotation. Recurrence shared with "
            "rotation is scaffold-inherited, not an independent signature of organization."
        ),
        "5_genuine_coarsening": (
            f"Early-to-mid-time growth was reproducible and strong (median peak scale ratio "
            f"{_format_number(peak_scale_ratio)}), but it was not sustained: the scale relaxed to "
            f"a median {_format_number(final_scale_ratio)} times initial and every Stage 7B winner "
            "late-half slope was negative. This supports domain formation followed by saturation/"
            "relaxation, not unbounded or sustained late-time coarsening."
        ),
        "6_reversible_occupancy_creation": (
            f"The Stage 7B winner median occupied-fraction change was "
            f"{_format_number(occupancy_change)} while every charge audit stayed exact, and the "
            "discovery run reversed byte-for-byte. The observations are consistent with reversible "
            "balanced occupancy creation, but this campaign does not identify a unique causal fraction."
        ),
        "7_beyond_phase_separation": (
            "No. There is no lineage, heredity, reproduction, adaptive fitness, or organism test."
        ),
    }
    summary = {
        "schema": "zwslab-overnight-confirmation-summary-v1",
        "classification": classification,
        "classification_reason": classification_reason,
        "classification_rubric_status": (
            "descriptive conservative rubric frozen after Stage 7A and before inspecting Stage 7B outcomes"
        ),
        "manifest_sha256": sha256_file(manifest_file),
        "winner_rule_table_sha256_uint8": manifest["conditions"][0][
            "rule_table_sha256_uint8"
        ],
        "confirmation_seeds": manifest["confirmation_seeds"],
        "rules": [
            {
                "condition_id": item["condition_id"],
                "rule_table_sha256_uint8": item["rule_table_sha256_uint8"],
                "reactive_inputs": item["reactive_inputs"],
            }
            for item in manifest["conditions"]
        ],
        "integrity": integrity,
        "stage_a_gate": stage_a_gate,
        "condition_summaries": condition_summaries,
        "paired_effects": effects,
        "questions": questions,
        "failed_runs": [row["run_id"] for row in endpoints if row["status"] != "complete"],
        "interpretation_boundary": (
            "This confirms or weakens reversible null-conserving spatial phase-separation claims only."
        ),
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_dir / "CONFIRMATION_SUMMARY.json"
    integrity_path = experiment_dir / "CONFIRMATION_INTEGRITY.json"
    trajectory_csv = experiment_dir / "CONFIRMATION_TRAJECTORIES.csv"
    effects_csv = experiment_dir / "CONFIRMATION_EFFECTS.csv"
    atomic_write_json(summary_path, summary)
    atomic_write_json(integrity_path, integrity)
    atomic_write_text(trajectory_csv, _csv_text(endpoints))
    atomic_write_text(effects_csv, _csv_text(effects))
    table_rows = []
    for stage_id in ["stage7a", "stage7b"]:
        denominator = 8 if stage_id == "stage7a" else 4
        for item in condition_summaries[stage_id]:
            table_rows.append(
                f"| {stage_id} | {item['condition_id']} | "
                f"{item['macroscopic_seed_count']}/{denominator} | "
                f"{_format_number(item['medians']['final_dispersion_ratio_64'])} | "
                f"{_format_number(item['medians']['finite_empty_domain_scale_ratio'])} | "
                f"{_format_number(item['medians']['late_median_exact_recurrence_lag4'])} | "
                f"{_format_number(item['medians']['late_median_exact_recurrence_lag8'])} | "
                f"{item['charge_violation_count']} |"
            )
    question_lines = "\n\n".join(
        f"### {index}. {title}\n\n{questions[key]}"
        for index, (key, title) in enumerate(
            [
                ("1_large_domains", "Does the winner reproducibly form large domains?"),
                ("2_pure_rotation", "Does pure rotation do the same thing?"),
                ("3_unusual_vs_matched", "Is the winner unusual among matched rules?"),
                ("4_rotation_oscillator", "Is the 4/8-tick oscillator scaffold-inherited?"),
                ("5_genuine_coarsening", "Does domain length grow with time?"),
                ("6_reversible_occupancy_creation", "Can reversible occupancy creation explain it?"),
                ("7_beyond_phase_separation", "Is there evidence beyond phase separation?"),
            ],
            start=1,
        )
    )
    report = f"""# ZWSLab v0.1.4 Overnight Independent Confirmation

## Bottom line

**Classification {classification}: {classification_reason}**

This strict classification does not mean that the large domains vanished: all four 4,096² winner
seeds produced strong finite-domain growth. Their characteristic scale peaked near the midpoint
and then relaxed, so they failed the registered requirement for a positive late-half slope.

The original `observe_4096` trajectory remains discovery evidence. All rows below are
confirmatory trajectories from seeds and rules frozen in commit `{manifest['preregistration_commit']}`
before outcomes. The discovery rule is unchanged at
`{manifest['conditions'][0]['rule_table_sha256_uint8']}`.

## Discovery data (not confirmatory)

The original 4,096² × 65,536-phase run showed localized sparse/dense morphology, increasing
occupancy, and strong lag-4/8 recurrence. Its exact inverse recovered the initial 16,777,216-cell
state byte-for-byte after all 65,536 phases, with zero mismatches at every retained checkpoint
boundary. These observations motivated, but are not counted in, the matrix below.

## Confirmatory matrix

| Stage | Condition | Coarsening criterion | Median final dispersion | Median finite-scale ratio | Late recurrence 4 | Late recurrence 8 | Charge violations |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

All {integrity['observed_trajectory_count']} / {integrity['expected_trajectory_count']} planned
trajectory records passed the frozen seed, rule, sample-count, endpoint, manifest, paired-initial-
state, and exact-charge checks: **{integrity['valid']}**. Failed runs: **{failed_count}**.

Paired winner-minus-control effect sizes and deterministic 20,000-draw percentile-bootstrap 95%
intervals are in `CONFIRMATION_EFFECTS.csv`. With only 8 or 4 paired seeds, these intervals are
descriptive and should not be treated as high-powered population estimates.

## Explicit scientific questions

{question_lines}

## Interpretation limits

The campaign tests reversible null-conserving spatial domain formation. It does not test or
establish life, organisms, replication, heredity, intelligence, consciousness, open-ended
evolution, FOA, primitive occurrence, ETNE, cosmology, or a fundamental process measure.
Coarse-density dispersion alone is insufficient: a rule can make a static localized density
contrast without growth of finite empty-domain length.

## Machine-readable artifacts

- `CONFIRMATION_SUMMARY.json`: complete summaries, seeds, hashes, gate evaluations, effects, answers.
- `CONFIRMATION_TRAJECTORIES.csv`: one row per confirmatory trajectory.
- `CONFIRMATION_EFFECTS.csv`: paired contrasts and uncertainty intervals.
- `CONFIRMATION_INTEGRITY.json`: frozen-matrix audit.
- `reports/phase_separation_confirmation_20260825`: paired curves and Stage 7A/7B reports.
"""
    report_path = experiment_dir / "OVERNIGHT_REPORT.md"
    atomic_write_text(report_path, report)
    artifacts = [summary_path, integrity_path, trajectory_csv, effects_csv, report_path]
    index = {
        "schema": "zwslab-overnight-report-index-v1",
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        ],
    }
    index_path = report_dir / "OVERNIGHT_REPORT_INDEX.json"
    atomic_write_json(index_path, index)
    return {
        "classification": classification,
        "report": str(report_path),
        "summary": str(summary_path),
        "integrity": integrity,
        "index": str(index_path),
    }
