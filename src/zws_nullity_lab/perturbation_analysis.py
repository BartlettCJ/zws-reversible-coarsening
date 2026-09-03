from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .atomic import sha256_file


STATE_LABELS = ("empty", "positive", "negative", "bound")


def _number(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sample_index(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    samples = {int(sample["tick"]): sample for sample in record["samples"]}
    if len(samples) != len(record["samples"]):
        raise ValueError(f"trajectory has duplicate sample ticks: {record.get('run_id')}")
    return samples


def _state_fractions(sample: dict[str, Any]) -> list[float]:
    populations = [int(value) for value in sample["state_populations"]]
    if len(populations) != 4 or any(value < 0 for value in populations):
        raise ValueError("state populations must be four non-negative counts")
    total = sum(populations)
    if total <= 0:
        raise ValueError("state population is empty")
    return [value / total for value in populations]


def _profile_distance(
    perturbed: dict[str, Any], baseline: dict[str, Any]
) -> float | None:
    first = perturbed.get("radial_profile", [])
    second = baseline.get("radial_profile", [])
    first_bins = [
        (int(row["radius_min"]), int(row["radius_max_exclusive"])) for row in first
    ]
    second_bins = [
        (int(row["radius_min"]), int(row["radius_max_exclusive"])) for row in second
    ]
    if first_bins != second_bins:
        raise ValueError("paired structure-factor radial bins differ")
    first_power = [max(0.0, float(row["mean_power"])) for row in first]
    second_power = [max(0.0, float(row["mean_power"])) for row in second]
    first_total = sum(first_power)
    second_total = sum(second_power)
    if first_total <= 0.0 or second_total <= 0.0:
        return None
    first_norm = [value / first_total for value in first_power]
    second_norm = [value / second_total for value in second_power]
    return 0.5 * sum(
        abs(first_value - second_value)
        for first_value, second_value in zip(first_norm, second_norm, strict=True)
    )


def _add_pair(
    row: dict[str, Any], name: str, perturbed: object, baseline: object
) -> None:
    perturbed_value = _number(perturbed)
    baseline_value = _number(baseline)
    row[f"{name}_perturbed"] = perturbed_value
    row[f"{name}_baseline"] = baseline_value
    row[f"{name}_delta"] = (
        perturbed_value - baseline_value
        if perturbed_value is not None and baseline_value is not None
        else None
    )
    row[f"{name}_relative_delta"] = (
        (perturbed_value - baseline_value) / baseline_value
        if perturbed_value is not None
        and baseline_value is not None
        and baseline_value > 0.0
        else None
    )


def paired_response_coordinates(
    perturbations: Iterable[dict[str, Any]],
    baselines: Iterable[dict[str, Any]],
    *,
    recurrence_lags: Iterable[int],
) -> list[dict[str, Any]]:
    baseline_records = list(baselines)
    baseline_by_id = {str(record["run_id"]): record for record in baseline_records}
    if len(baseline_by_id) != len(baseline_records):
        raise ValueError("baseline run identifiers are not unique")
    lags = sorted({int(value) for value in recurrence_lags})
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for trajectory in perturbations:
        run_id = str(trajectory["run_id"])
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate perturbation run identifier: {run_id}")
        seen_run_ids.add(run_id)
        baseline_id = str(trajectory["baseline_run_id"])
        if baseline_id not in baseline_by_id:
            raise ValueError(f"paired baseline is missing: {baseline_id}")
        baseline = baseline_by_id[baseline_id]
        if trajectory.get("status") != "complete" or baseline.get("status") != "complete":
            raise ValueError(f"paired trajectory is not complete: {run_id}")
        for key in ("condition_id", "seed_index", "initial_seed"):
            if trajectory.get(key) != baseline.get(key):
                raise ValueError(f"paired identity differs for {run_id}: {key}")
        if (
            trajectory.get("rule_table_sha256_uint8")
            != baseline.get("rule_table_sha256_uint8")
        ):
            raise ValueError(f"paired rule digest differs for {run_id}")
        if (
            trajectory.get("initial_state_sha256_uint8")
            != baseline.get("initial_state_sha256_uint8")
        ):
            raise ValueError(f"paired initial state digest differs for {run_id}")
        if any(int(value) for value in trajectory["invariant_violation_counts"].values()):
            raise ValueError(f"perturbation trajectory failed an exact invariant: {run_id}")
        if any(int(value) for value in baseline["invariant_violation_counts"].values()):
            raise ValueError(f"baseline trajectory failed an exact invariant: {baseline_id}")
        audit = trajectory["perturbation_audit"]
        if not bool(audit["charge_preserved"]):
            raise ValueError(f"perturbation failed exact Q audit: {run_id}")
        if int(audit["observed_changed_sites"]) != int(audit["requested_changed_sites"]):
            raise ValueError(f"perturbation amplitude differs from request: {run_id}")

        perturbed_samples = _sample_index(trajectory)
        baseline_samples = _sample_index(baseline)
        evaluation_ticks = [int(value) for value in trajectory["evaluation_ticks"]]
        if evaluation_ticks != sorted(set(evaluation_ticks)):
            raise ValueError(f"perturbation evaluation ticks are not unique and sorted: {run_id}")
        missing_perturbed = sorted(set(evaluation_ticks) - set(perturbed_samples))
        missing_baseline = sorted(set(evaluation_ticks) - set(baseline_samples))
        if missing_perturbed or missing_baseline:
            raise ValueError(
                f"paired evaluation samples are missing for {run_id}: "
                f"perturbed={missing_perturbed}, baseline={missing_baseline}"
            )
        intervention_tick = int(trajectory["intervention_tick"])
        if not evaluation_ticks or evaluation_ticks[0] != intervention_tick:
            raise ValueError(f"evaluation does not begin at intervention: {run_id}")
        for tick in evaluation_ticks:
            perturbed = perturbed_samples[tick]
            control = baseline_samples[tick]
            perturbed_fractions = _state_fractions(perturbed)
            baseline_fractions = _state_fractions(control)
            result: dict[str, Any] = {
                "run_id": run_id,
                "baseline_run_id": baseline_id,
                "condition_id": trajectory["condition_id"],
                "conservation": trajectory["conservation"],
                "seed_index": int(trajectory["seed_index"]),
                "initial_seed": int(trajectory["initial_seed"]),
                "intervention_kind": trajectory["intervention_kind"],
                "intervention_tick": intervention_tick,
                "evaluation_tick": tick,
                "response_horizon_phases": tick - intervention_tick,
                "selection_seed": int(trajectory["selection_seed"]),
                "requested_changed_sites": int(audit["requested_changed_sites"]),
                "observed_changed_sites": int(audit["observed_changed_sites"]),
                "perturbation_application_mass_delta": (
                    int(audit["mass_after"]) - int(audit["mass_before"])
                ),
                "perturbation_application_charge_delta": (
                    int(audit["charge_after"]) - int(audit["charge_before"])
                ),
            }
            _add_pair(
                result,
                "finite_domain_scale",
                perturbed["empty_domains"].get("finite_area_weighted_linear_scale"),
                control["empty_domains"].get("finite_area_weighted_linear_scale"),
            )
            _add_pair(
                result,
                "empty_domain_count",
                perturbed["empty_domains"].get("component_count"),
                control["empty_domains"].get("component_count"),
            )
            _add_pair(
                result,
                "largest_empty_domain_fraction",
                perturbed["empty_domains"].get("largest_area_fraction_of_empty"),
                control["empty_domains"].get("largest_area_fraction_of_empty"),
            )
            perturbed_structure = perturbed["structure_factor"]
            baseline_structure = control["structure_factor"]
            for field, label in (
                ("characteristic_wavelength_cells", "structure_wavelength"),
                ("peak_radius", "structure_peak_radius"),
                ("peak_to_radial_median", "structure_peak_to_median"),
            ):
                _add_pair(
                    result,
                    label,
                    perturbed_structure.get(field),
                    baseline_structure.get(field),
                )
            result["structure_spectrum_total_variation_distance"] = _profile_distance(
                perturbed_structure, baseline_structure
            )
            _add_pair(
                result,
                "occupancy_fraction",
                perturbed.get("occupied_fraction"),
                control.get("occupied_fraction"),
            )
            _add_pair(
                result,
                "occupancy_mass",
                sum(int(value) for value in perturbed["state_populations"][1:]),
                sum(int(value) for value in control["state_populations"][1:]),
            )
            _add_pair(
                result,
                "total_charge",
                perturbed.get("total_charge"),
                control.get("total_charge"),
            )
            for index, label in enumerate(STATE_LABELS):
                _add_pair(
                    result,
                    f"state_fraction_{label}",
                    perturbed_fractions[index],
                    baseline_fractions[index],
                )
            for field, label in (
                ("boundary_density", "state_boundary_density"),
                ("reaction_density_one_phase", "reaction_density"),
                ("activity_one_phase", "activity"),
                ("entropy_normalized", "state_entropy_normalized"),
            ):
                _add_pair(result, label, perturbed.get(field), control.get(field))
            for lag in lags:
                _add_pair(
                    result,
                    f"exact_recurrence_lag{lag}",
                    perturbed.get("exact_state_recurrence", {}).get(str(lag)),
                    control.get("exact_state_recurrence", {}).get(str(lag)),
                )
                _add_pair(
                    result,
                    f"coarse_recurrence_lag{lag}",
                    perturbed.get("coarse_occupancy_correlation_block64", {}).get(str(lag)),
                    control.get("coarse_occupancy_correlation_block64", {}).get(str(lag)),
                )
            rows.append(result)
    return sorted(
        rows,
        key=lambda row: (
            str(row["condition_id"]),
            int(row["intervention_tick"]),
            str(row["intervention_kind"]),
            int(row["seed_index"]),
            int(row["evaluation_tick"]),
        ),
    )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_paired_responses(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    members = list(rows)
    if not members:
        raise ValueError("cannot summarize an empty perturbation response")
    coordinate_fields = sorted(
        {
            key
            for row in members
            for key in row
            if key.endswith("_delta")
            or key.endswith("_relative_delta")
            or key.endswith("_distance")
        }
    )
    grouped: dict[tuple[str, int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        key = (
            str(row["condition_id"]),
            int(row["intervention_tick"]),
            str(row["intervention_kind"]),
            int(row["evaluation_tick"]),
            int(row["response_horizon_phases"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        condition_id, intervention_tick, kind, evaluation_tick, horizon = key
        summary: dict[str, Any] = {
            "condition_id": condition_id,
            "intervention_tick": intervention_tick,
            "intervention_kind": kind,
            "evaluation_tick": evaluation_tick,
            "response_horizon_phases": horizon,
            "paired_seed_count": len(group),
        }
        for field in coordinate_fields:
            values = [
                float(value)
                for row in group
                if (value := _number(row.get(field))) is not None
            ]
            summary[f"{field}_n"] = len(values)
            summary[f"{field}_median"] = median(values) if values else None
            summary[f"{field}_q25"] = _quantile(values, 0.25) if values else None
            summary[f"{field}_q75"] = _quantile(values, 0.75) if values else None
            summary[f"{field}_median_absolute"] = (
                median(abs(value) for value in values) if values else None
            )
        output.append(summary)
    return output


def _read_complete_summary(root: Path, digest: str, schema: str) -> dict[str, Any]:
    path = root / "campaign_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"campaign summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != schema:
        raise ValueError(f"unexpected campaign summary schema: {path}")
    if summary.get("status") != "complete" or summary.get("manifest_sha256") != digest:
        raise RuntimeError(f"campaign is not complete under the frozen manifest: {path}")
    return summary


def load_completed_baselines(
    root: str | Path,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results_root = Path(root).resolve()
    summary = _read_complete_summary(
        results_root, manifest_digest, "zwslab-followup-campaign-summary-v1"
    )
    expected = {str(row["run_id"]): row for row in manifest["planned_trajectories"]}
    conditions = {
        str(row["condition_id"]): row for row in manifest["conditions"]
    }
    condition_ids = [str(row["condition_id"]) for row in manifest["conditions"]]
    if int(summary.get("planned_condition_count", -1)) != len(condition_ids) or int(
        summary.get("planned_trajectory_count", -1)
    ) != len(expected):
        raise RuntimeError("baseline campaign planned counts differ from manifest")
    if set(summary.get("completed_conditions", [])) != set(condition_ids):
        raise RuntimeError("baseline campaign completed-condition set differs from manifest")
    entries: dict[str, dict[str, Any]] = {}
    for condition_id in condition_ids:
        path = results_root / "conditions" / condition_id / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        if (
            completion.get("status") != "complete"
            or completion.get("manifest_sha256") != manifest_digest
            or completion.get("condition_id") != condition_id
        ):
            raise RuntimeError(f"baseline condition is not complete: {condition_id}")
        for entry in completion["trajectories"]:
            run_id = str(entry["run_id"])
            if run_id in entries:
                raise RuntimeError(f"duplicate baseline completion entry: {run_id}")
            entries[run_id] = entry
    if set(entries) != set(expected):
        raise RuntimeError("baseline completed trajectory set differs from frozen manifest")
    records: list[dict[str, Any]] = []
    integrity: list[dict[str, str]] = []
    for run_id, planned in expected.items():
        path = results_root / "trajectories" / f"{run_id}.json"
        digest = sha256_file(path)
        if digest != entries[run_id]["sha256"]:
            raise RuntimeError(f"baseline trajectory digest mismatch: {run_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("manifest_sha256") != manifest_digest or record.get("status") != "complete":
            raise RuntimeError(f"baseline trajectory is not complete under manifest: {run_id}")
        for field in ("run_id", "condition_id", "seed_index", "initial_seed"):
            if record.get(field) != planned.get(field):
                raise RuntimeError(f"baseline trajectory identity mismatch: {run_id} {field}")
        expected_rule_digest = conditions[str(planned["condition_id"])][
            "rule_table_sha256_uint8"
        ]
        if record.get("rule_table_sha256_uint8") != expected_rule_digest:
            raise RuntimeError(f"baseline trajectory rule digest mismatch: {run_id}")
        records.append(record)
        integrity.append({"run_id": run_id, "sha256": digest})
    return records, integrity


def load_completed_perturbations(
    root: str | Path,
    manifest: dict[str, Any],
    manifest_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results_root = Path(root).resolve()
    summary = _read_complete_summary(
        results_root, manifest_digest, "zwslab-perturbation-campaign-summary-v1"
    )
    expected = {str(row["run_id"]): row for row in manifest["planned_branches"]}
    conditions = {
        str(row["condition_id"]): row for row in manifest["conditions"]
    }
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["planned_branches"]:
        groups[(str(row["condition_id"]), int(row["intervention_tick"]))].append(row)
    expected_group_ids = {
        f"{condition_id}__t{tick:05d}" for condition_id, tick in groups
    }
    if int(summary.get("planned_group_count", -1)) != len(expected_group_ids) or int(
        summary.get("planned_branch_count", -1)
    ) != len(expected):
        raise RuntimeError("perturbation campaign planned counts differ from manifest")
    if set(summary.get("completed_groups", [])) != expected_group_ids:
        raise RuntimeError("perturbation completed-group set differs from manifest")
    entries: dict[str, dict[str, Any]] = {}
    for group_id in sorted(expected_group_ids):
        path = results_root / "groups" / group_id / "completion.json"
        completion = json.loads(path.read_text(encoding="utf-8"))
        if (
            completion.get("status") != "complete"
            or completion.get("manifest_sha256") != manifest_digest
            or completion.get("group_id") != group_id
        ):
            raise RuntimeError(f"perturbation group is not complete: {group_id}")
        for entry in completion["trajectories"]:
            run_id = str(entry["run_id"])
            if run_id in entries:
                raise RuntimeError(f"duplicate perturbation completion entry: {run_id}")
            entries[run_id] = entry
    if set(entries) != set(expected):
        raise RuntimeError("perturbation completed trajectory set differs from frozen manifest")
    records: list[dict[str, Any]] = []
    integrity: list[dict[str, str]] = []
    for run_id, planned in expected.items():
        path = results_root / "trajectories" / f"{run_id}.json"
        digest = sha256_file(path)
        if digest != entries[run_id]["sha256"]:
            raise RuntimeError(f"perturbation trajectory digest mismatch: {run_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("manifest_sha256") != manifest_digest or record.get("status") != "complete":
            raise RuntimeError(f"perturbation trajectory is not complete under manifest: {run_id}")
        for field in (
            "run_id",
            "condition_id",
            "seed_index",
            "initial_seed",
            "baseline_run_id",
            "intervention_tick",
            "intervention_kind",
            "selection_seed",
            "evaluation_ticks",
        ):
            if record.get(field) != planned.get(field):
                raise RuntimeError(f"perturbation trajectory identity mismatch: {run_id} {field}")
        expected_rule_digest = conditions[str(planned["condition_id"])][
            "rule_table_sha256_uint8"
        ]
        if record.get("rule_table_sha256_uint8") != expected_rule_digest:
            raise RuntimeError(f"perturbation trajectory rule digest mismatch: {run_id}")
        records.append(record)
        integrity.append({"run_id": run_id, "sha256": digest})
    return records, integrity
