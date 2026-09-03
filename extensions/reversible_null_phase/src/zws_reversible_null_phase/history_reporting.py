from __future__ import annotations

import csv
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_erasure import load_frozen_json, payload_sha256
from .history_stats import (
    clustered_bootstrap_rmst_delay,
    fixed_time_fraction,
    kaplan_meier,
    restricted_mean,
)


RULE_SHA256 = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
HORIZON = 262144
MARGIN = 8192
OBSERVATION = 2048
BOOTSTRAP_SEED = 326299999
BOOTSTRAP_REPLICATES = 10000


def _verified(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"missing artifact or sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"artifact checksum mismatch: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _verified(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = list(fieldnames)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    atomic_write_text(path, stream.getvalue())
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n"
    )


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _collect(run: Path, experiment: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = load_frozen_json(experiment / "FROZEN_PLAN.json")
    manifest = load_frozen_json(experiment / "INTERVENTION_MANIFEST.json")
    realized = _read_json(run / "REALIZED_BRANCH_PLAN.json")
    if payload_sha256(realized) != realized["payload_sha256"]:
        raise RuntimeError("realized plan payload hash mismatch")
    sources = [
        _read_json(run / "sources" / task["task_id"] / "source_result.json")
        for task in plan["source_tasks"]
    ]
    branches = [
        _read_json(run / "branches" / task["task_id"] / "branch_result.json")
        for task in realized["prepared_entries"]
    ]
    return plan, manifest, realized, sources, branches


def _source_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        task = source["task"]
        event = source["time_to_event"]
        branch_a = source["branch_points"].get("A")
        branch_b = source["branch_points"].get("B")
        initial = source["series"][0]
        final = source["series"][-1]
        rows.append(
            {
                "source_task_id": task["task_id"],
                "density": task["density"],
                "seed": task["seed"],
                "initial_state_sha256": source["initial_state"]["state_sha256"],
                "initial_Q": initial["charge"],
                "initial_M": initial["occupancy_count"],
                "initial_occupancy_fraction": initial["occupancy_fraction"],
                "final_Q": final["charge"],
                "final_M": final["occupancy_count"],
                "final_occupancy_fraction": final["occupancy_fraction"],
                "T_hit": event["T_hit"],
                "hit_observed": event["hit_observed"],
                "T_persistent": event["T_persistent"],
                "persistent_observed": event["persistent_observed"],
                "right_censored_T_hit": event["right_censored_T_hit"],
                "branch_A_eligible": bool(branch_a),
                "branch_A_phase": branch_a["phase"] if branch_a else None,
                "branch_A_M": sum(branch_a["histogram"][1:]) if branch_a else None,
                "branch_A_occupancy_fraction": branch_a["occupancy_fraction"] if branch_a else None,
                "branch_B_eligible": bool(branch_b),
                "branch_B_phase": branch_b["phase"] if branch_b else None,
                "branch_B_M": sum(branch_b["histogram"][1:]) if branch_b else None,
                "branch_B_occupancy_fraction": branch_b["occupancy_fraction"] if branch_b else None,
                "first_occupied_wrap_x": next((row["phase"] for row in source["series"] if row["occupied_wrap_x"]), None),
                "first_occupied_wrap_y": next((row["phase"] for row in source["series"] if row["occupied_wrap_y"]), None),
                "first_empty_wrap_x": next((row["phase"] for row in source["series"] if row["empty_wrap_x"]), None),
                "first_empty_wrap_y": next((row["phase"] for row in source["series"] if row["empty_wrap_y"]), None),
            }
        )
    return rows


def _residual_rows(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in branches:
        task = branch["task"]
        residual = branch["residual_time_to_event"]
        untouched = branch["untouched_reference"]
        rows.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task["source_task_id"],
                "cluster": f"density_{float(task['density']):.2f}_seed_{int(task['source_seed'])}",
                "density": task["density"],
                "source_seed": task["source_seed"],
                "branch_point": task["branch_point"],
                "branch_phase": task["branch_phase"],
                "intervention": task["intervention"],
                "realization": task["realization"],
                "R_hit": residual["R_hit"],
                "hit_observed": residual["hit_observed"],
                "R_persistent": residual["R_persistent"],
                "persistent_observed": residual["persistent_observed"],
                "censor_elapsed": residual["censor_elapsed"],
                "untouched_R_hit": untouched["R_hit"],
                "untouched_hit_observed": untouched["hit_observed"],
                "untouched_R_persistent": untouched["R_persistent"],
                "untouched_persistent_observed": untouched["persistent_observed"],
                "untouched_censor_elapsed": untouched["censor_elapsed"],
                "paired_hit_delay": (
                    int(residual["R_hit"]) - int(untouched["R_hit"])
                    if residual["R_hit"] is not None and untouched["R_hit"] is not None
                    else None
                ),
                "completion_reason": branch["completion_reason"],
                "translation_null_passed": branch["translation_null_passed"],
            }
        )
    return rows


def _analysis_record(row: dict[str, Any]) -> dict[str, Any]:
    altered_observed = bool(row["hit_observed"])
    untouched_observed = bool(row["untouched_hit_observed"])
    altered_time = int(row["R_hit"]) if altered_observed else int(row["censor_elapsed"])
    untouched_time = (
        int(row["untouched_R_hit"])
        if untouched_observed
        else int(row["untouched_censor_elapsed"])
    )
    return {
        "cluster": row["cluster"],
        "untouched_time": min(untouched_time, HORIZON),
        "untouched_observed": untouched_observed and untouched_time <= HORIZON,
        "altered_time": min(altered_time, HORIZON),
        "altered_observed": altered_observed and altered_time <= HORIZON,
    }


def _branch_support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    equivalent = delayed = inhibited = 0
    for row in rows:
        altered_observed = bool(row["hit_observed"])
        untouched_observed = bool(row["untouched_hit_observed"])
        altered = int(row["R_hit"]) if altered_observed else int(row["censor_elapsed"])
        untouched = (
            int(row["untouched_R_hit"])
            if untouched_observed
            else int(row["untouched_censor_elapsed"])
        )
        if altered_observed and untouched_observed and abs(altered - untouched) <= MARGIN:
            equivalent += 1
        elif not altered_observed and not untouched_observed and abs(altered - untouched) <= MARGIN:
            equivalent += 1
        if untouched_observed and altered > untouched + OBSERVATION:
            delayed += 1
        if altered_observed and untouched > altered + OBSERVATION:
            inhibited += 1
    return {"available": len(rows), "equivalent": equivalent, "delayed": delayed, "inhibited": inhibited}


def _statistics(rows: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    records = [_analysis_record(row) for row in rows]
    bootstrap = clustered_bootstrap_rmst_delay(
        records,
        horizon=HORIZON,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    altered_times = [record["altered_time"] for record in records]
    altered_events = [record["altered_observed"] for record in records]
    untouched_times = [record["untouched_time"] for record in records]
    untouched_events = [record["untouched_observed"] for record in records]
    return {
        "name": name,
        "n": len(rows),
        "rmst_altered": restricted_mean(
            kaplan_meier(altered_times, altered_events, horizon=HORIZON), HORIZON
        ),
        "rmst_untouched": restricted_mean(
            kaplan_meier(untouched_times, untouched_events, horizon=HORIZON), HORIZON
        ),
        "paired_rmst_delay": bootstrap,
        "fixed_time_fractions": {
            str(time): fixed_time_fraction(altered_times, altered_events, time)
            for time in (32768, 65536, 131072, 262144)
        },
        "support_counts": _branch_support_counts(rows),
    }


def _classify(point: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty = {
            "name": f"Branch {point} unavailable",
            "n": 0,
            "rmst_altered": None,
            "rmst_untouched": None,
            "paired_rmst_delay": {
                "point_estimate": None,
                "ci90": [None, None],
                "ci95": [None, None],
                "replicates": 0,
            },
            "fixed_time_fractions": {},
            "support_counts": {"available": 0, "equivalent": 0, "delayed": 0, "inhibited": 0},
        }
        return {
            "branch_point": point,
            "classification": "MIXED / INCONCLUSIVE",
            "pooled": empty,
            "by_density": {},
            "density_specific_contradiction": False,
            "reason": "no eligible primary global-shuffle branches",
        }
    pooled = _statistics(rows, name=f"Branch {point} pooled")
    density_stats = {
        f"{density:.2f}": _statistics(
            [row for row in rows if float(row["density"]) == density],
            name=f"Branch {point} density {density:.2f}",
        )
        for density in (0.20, 0.24)
        if any(float(row["density"]) == density for row in rows)
    }
    ci90 = pooled["paired_rmst_delay"]["ci90"]
    ci95 = pooled["paired_rmst_delay"]["ci95"]
    support = pooled["support_counts"]
    density_points = [value["paired_rmst_delay"]["point_estimate"] for value in density_stats.values()]
    contradiction = any(
        value["paired_rmst_delay"]["ci95"][0] > MARGIN
        or value["paired_rmst_delay"]["ci95"][1] < -MARGIN
        for value in density_stats.values()
    )
    if (
        ci90[0] >= -MARGIN
        and ci90[1] <= MARGIN
        and not contradiction
        and support["equivalent"] >= 12
    ):
        classification = "INVENTORY-EQUIVALENT"
    elif (
        ci95[0] > MARGIN
        and density_points
        and all(value > 0 for value in density_points)
        and support["delayed"] >= 12
    ):
        classification = "SPATIAL-HISTORY DELAY"
    elif (
        ci95[1] < -MARGIN
        and density_points
        and all(value < 0 for value in density_points)
        and support["inhibited"] >= 12
    ):
        classification = "SPATIAL-HISTORY INHIBITION"
    else:
        classification = "MIXED / INCONCLUSIVE"
    return {
        "branch_point": point,
        "classification": classification,
        "pooled": pooled,
        "by_density": density_stats,
        "density_specific_contradiction": contradiction,
    }


def _km_rows(residual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (row["branch_point"], row["intervention"], int(row["realization"]))
            for row in residual_rows
            if row["intervention"] != "even_translation_null"
        }
    )
    for point, intervention, realization in groups:
        base = [
            row
            for row in residual_rows
            if row["branch_point"] == point
            and row["intervention"] == intervention
            and int(row["realization"]) == realization
        ]
        for density_label, selected in [
            ("pooled", base),
            ("0.20", [row for row in base if float(row["density"]) == 0.20]),
            ("0.24", [row for row in base if float(row["density"]) == 0.24]),
        ]:
            if not selected:
                continue
            records = [_analysis_record(row) for row in selected]
            km = kaplan_meier(
                [record["altered_time"] for record in records],
                [record["altered_observed"] for record in records],
                horizon=HORIZON,
            )
            for row in km:
                output.append(
                    {
                        "branch_point": point,
                        "intervention": intervention,
                        "realization": realization,
                        "density": density_label,
                        **row,
                        "rmst": restricted_mean(km, HORIZON),
                    }
                )
    return output


def _multiscale_rows(branches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    entropy: list[dict[str, Any]] = []
    anisotropy: list[dict[str, Any]] = []
    for branch in branches:
        task = branch["task"]
        for row in branch["multiscale_series"]:
            common = {
                "task_id": task["task_id"],
                "source_task_id": task["source_task_id"],
                "density": task["density"],
                "source_seed": task["source_seed"],
                "branch_point": task["branch_point"],
                "intervention": task["intervention"],
                "realization": task["realization"],
                "elapsed": row["elapsed"],
                "absolute_phase": row["absolute_phase"],
                "state_sha256": row["state_sha256"],
            }
            main.append(
                {
                    **common,
                    "Q": row["Q"],
                    "M": row["M"],
                    "single_site_entropy": row["single_site_entropy"],
                    "interface_density": row["interface_density"],
                    "low_k_power": row["low_k_power"],
                    "low_k_fraction": row["low_k_fraction"],
                    "correlation_length": row["correlation_length"],
                    "anisotropy": row["anisotropy"],
                    "diagonal_axis_contrast": row["diagonal_axis_contrast"],
                    "mutual_information": row["mutual_information"],
                    "block_entropy": row["block_entropy"],
                }
            )
            for block_size, value in row["block_entropy"].items():
                entropy.append({**common, "block_size": int(block_size), "block_entropy": value})
            anisotropy.append(
                {
                    **common,
                    "anisotropy": row["anisotropy"],
                    "diagonal_axis_contrast": row["diagonal_axis_contrast"],
                    "angular_power": row["angular_power"],
                }
            )
    return main, entropy, anisotropy


def _audit_rows(run: Path, realized: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in realized["prepared_entries"]:
        audit = _read_json(Path(task["intervention_audit_path"]))
        exact = audit["exact_audit"]
        source = audit["source_multiscale"]
        altered = audit["altered_multiscale"]
        rows.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task["source_task_id"],
                "density": task["density"],
                "source_seed": task["source_seed"],
                "branch_point": task["branch_point"],
                "branch_phase": task["branch_phase"],
                "intervention": task["intervention"],
                "realization": task["realization"],
                "valid": exact["valid"],
                "source_state_sha256": exact["source_state_sha256"],
                "altered_state_sha256": exact["altered_state_sha256"],
                "source_histogram": exact["source_histogram"],
                "altered_histogram": exact["altered_histogram"],
                "source_Q": exact["source_Q"],
                "altered_Q": exact["altered_Q"],
                "source_M": exact["source_M"],
                "altered_M": exact["altered_M"],
                "source_single_site_entropy": exact["source_single_site_entropy"],
                "altered_single_site_entropy": exact["altered_single_site_entropy"],
                "source_low_k_power": source["low_k_power"],
                "altered_low_k_power": altered["low_k_power"],
                "source_interface_density": source["interface_density"],
                "altered_interface_density": altered["interface_density"],
                "source_anisotropy": source["anisotropy"],
                "altered_anisotropy": altered["anisotropy"],
                "source_block_entropy": source["block_entropy"],
                "altered_block_entropy": altered["block_entropy"],
                "source_mutual_information": source["mutual_information"],
                "altered_mutual_information": altered["mutual_information"],
            }
        )
    return rows


def _representatives(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for density in (0.20, 0.24):
        candidates = [
            source
            for source in sources
            if float(source["task"]["density"]) == density
            and source["branch_points"].get("B")
            and source["time_to_event"]["T_hit"] is not None
        ]
        values = np.asarray([source["time_to_event"]["T_hit"] for source in candidates], dtype=np.float64)
        median = float(np.median(values))
        chosen = min(
            candidates,
            key=lambda source: (
                abs(float(source["time_to_event"]["T_hit"]) - median),
                int(source["task"]["seed"]),
            ),
        )
        rows.append(
            {
                "role": f"history_erasure_density_{density:.2f}",
                "source_task_id": chosen["task"]["task_id"],
                "density": density,
                "seed": chosen["task"]["seed"],
                "T_hit": chosen["time_to_event"]["T_hit"],
                "selection_median_T_hit": median,
                "selection_rule": "nearest within-density median untouched T_hit; tie lowest seed",
            }
        )
    return rows


def _combined_mechanism(a: str, b: str, tile_label: str) -> str:
    if a == "INVENTORY-EQUIVALENT" and b == "INVENTORY-EQUIVALENT":
        return "A. INVENTORY-DOMINATED THROUGHOUT"
    if a == "INVENTORY-EQUIVALENT" and b == "SPATIAL-HISTORY DELAY":
        return "B. LATE-EMERGING SPATIAL MEMORY"
    if a == "SPATIAL-HISTORY DELAY":
        return "C. EARLY ACCUMULATED SPATIAL MEMORY"
    if "INHIBITION" in a or "INHIBITION" in b:
        return "D. HISTORY-INHIBITED TRANSFORMATION"
    if tile_label != "not_resolved":
        return "E. SCALE-DEPENDENT / MIXED"
    return "F. INCONCLUSIVE"


def _density_direction_agreement(result: dict[str, Any]) -> bool:
    values = [
        statistics["paired_rmst_delay"]["point_estimate"]
        for statistics in result["by_density"].values()
        if statistics["paired_rmst_delay"]["point_estimate"] is not None
    ]
    return len(values) == 2 and (
        all(float(value) >= 0 for value in values)
        or all(float(value) <= 0 for value in values)
    )


def _translation_null_exact(branch: dict[str, Any]) -> bool:
    residual = branch["residual_time_to_event"]
    reference = branch["untouched_reference"]
    return bool(
        branch.get("translation_null_passed") is True
        and residual.get("R_hit") == reference.get("R_hit")
        and residual.get("hit_observed") == reference.get("hit_observed")
        and residual.get("R_persistent") == reference.get("R_persistent")
        and residual.get("persistent_observed") == reference.get("persistent_observed")
    )


def _tile_interpretation(tile32: dict[str, Any], tile128: dict[str, Any], global_b: str) -> str:
    def direction(stats: dict[str, Any]) -> str:
        ci90 = stats["paired_rmst_delay"]["ci90"]
        ci95 = stats["paired_rmst_delay"]["ci95"]
        if ci90[0] >= -MARGIN and ci90[1] <= MARGIN:
            return "near_untouched"
        if ci95[0] > MARGIN:
            return "delayed"
        return "mixed"

    d32 = direction(tile32)
    d128 = direction(tile128)
    if global_b == "SPATIAL-HISTORY DELAY" and d32 == "near_untouched":
        return "causally useful information is largely retained below about 32 cells"
    if global_b == "SPATIAL-HISTORY DELAY" and d32 == "delayed" and d128 == "near_untouched":
        return "important organisation lies approximately between 32 and 128 cells"
    if global_b == "SPATIAL-HISTORY DELAY" and d32 == "delayed" and d128 == "delayed":
        return "organisation above 128 cells or global arrangement matters"
    if global_b == "INVENTORY-EQUIVALENT" and d32 == d128 == "near_untouched":
        return "inventory or very local statistics dominate"
    return "not_resolved"


def _integrity(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    realized: dict[str, Any],
    sources: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    resume_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    planned_ids = [task["task_id"] for task in plan["source_tasks"]] + [
        task["task_id"] for task in manifest["entries"]
    ]
    result_ids = [source["task"]["task_id"] for source in sources] + [
        branch["task"]["task_id"] for branch in branches
    ]
    translation = [
        branch for branch in branches if branch["task"]["intervention"] == "even_translation_null"
    ]
    current_tasks = {task["task_id"]: task for task in realized["prepared_entries"]}
    volatile_task_fields = {"intervention_audit_sha256"}

    def scientific_task(task: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in task.items() if key not in volatile_task_fields}

    branch_plan_hashes = {
        str(branch.get("realized_plan_payload_sha256", "")) for branch in branches
    }
    current_plan_hash = str(realized["payload_sha256"])
    direct_lineage = branch_plan_hashes == {current_plan_hash}
    audited_lineage = bool(
        resume_audit
        and resume_audit.get("passed") is True
        and resume_audit.get("current_realized_plan_payload_sha256") == current_plan_hash
        and not any(int(value) for value in resume_audit.get("mismatch_counts", [1]))
        and not any(int(value) for value in resume_audit.get("reference_mismatch_counts", [1]))
        and branch_plan_hashes
        <= {
            current_plan_hash,
            str(resume_audit.get("prior_realized_plan_payload_sha256", "")),
        }
    )
    checks = {
        "winner_hash_exact": all(
            payload["winner_sha256"] == RULE_SHA256 for payload in [*sources, *branches]
        ),
        "planned_ids_unique": len(planned_ids) == len(set(planned_ids)) == 88,
        "source_count_exact": len(sources) == 16,
        "realized_branch_partition_exact": (
            realized["prepared_count"] + realized["ineligible_count"] == manifest["entry_count"]
        ),
        "branch_result_count_exact": len(branches) == realized["prepared_count"],
        "result_ids_unique": len(result_ids) == len(set(result_ids)),
        "branch_tasks_match_realized_scientific_core": all(
            branch["task"]["task_id"] in current_tasks
            and scientific_task(branch["task"])
            == scientific_task(current_tasks[branch["task"]["task_id"]])
            for branch in branches
        ),
        "realized_plan_lineage_valid": direct_lineage or audited_lineage,
        "resume_replay_audit_valid_if_required": direct_lineage or audited_lineage,
        "all_source_Q_exact_zero": all(
            all(int(row["charge"]) == 0 for row in source["series"]) for source in sources
        ),
        "all_branch_Q_exact_zero": all(
            all(int(row["charge"]) == 0 for row in branch["series"]) for branch in branches
        ),
        "all_intervention_audits_valid": all(bool(row["valid"]) for row in audit_rows),
        "all_intervention_histograms_preserved": all(
            row["source_histogram"] == row["altered_histogram"] for row in audit_rows
        ),
        "all_intervention_Q_M_preserved": all(
            row["source_Q"] == row["altered_Q"] and row["source_M"] == row["altered_M"]
            for row in audit_rows
        ),
        "translation_nulls_present": len(translation) == sum(
            task["intervention"] == "even_translation_null" for task in realized["prepared_entries"]
        ),
        "translation_nulls_exact": all(_translation_null_exact(branch) for branch in translation),
        "calibration_inherited": all(
            branch["calibration_inherited_from_source"] and not branch["denominator_reset_at_branch"]
            for branch in branches
        ),
        "all_payload_values_finite": all(_finite(payload) for payload in [*sources, *branches]),
        "no_missing_result_ids": set(result_ids)
        == set(task["task_id"] for task in plan["source_tasks"])
        | set(task["task_id"] for task in realized["prepared_entries"]),
    }
    return {
        "schema": "zws-reversible-null-phase-v032-integrity-audit-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "planned_scientific_trajectory_maximum": 88,
        "realized_scientific_trajectory_count": 16 + len(branches),
        "source_count": len(sources),
        "branch_count": len(branches),
        "ineligible_branch_count": realized["ineligible_count"],
        "translation_null_count": len(translation),
        "branch_result_plan_hashes": sorted(branch_plan_hashes),
        "current_realized_plan_payload_sha256": current_plan_hash,
        "resume_replay_audit_used": not direct_lineage,
    }


def _answer_text(
    a_result: dict[str, Any],
    b_result: dict[str, Any],
    tile_label: str,
    combined: str,
    density_agreement_by_branch: dict[str, bool],
) -> list[str]:
    a = a_result["classification"]
    b = b_result["classification"]
    spatial = "SPATIAL-HISTORY DELAY" in {a, b}
    inventory = a == b == "INVENTORY-EQUIVALENT"

    def plain_result(classification: str) -> str:
        return {
            "INVENTORY-EQUIVALENT": "left the remaining transformation time unchanged within the frozen practical-equivalence resolution",
            "SPATIAL-HISTORY DELAY": "delayed the remaining transformation",
            "SPATIAL-HISTORY INHIBITION": "accelerated the remaining transformation",
            "MIXED / INCONCLUSIVE": "did not resolve as delayed, accelerated, or practically unchanged",
        }[classification]

    return [
        f"1. **MEASUREMENT — Branch A:** The state-matched global shuffle at the first sampled M/N >= 0.35 {plain_result(a)}. Frozen classification: **{a}**.",
        f"2. **MEASUREMENT — Branch B:** With untouched residual fixed at 32,768 phases, global shuffling {plain_result(b)}. Frozen classification: **{b}**.",
        f"3. **MEASUREMENT:** Density-specific direction between 0.20 and 0.24 {'agreed' if density_agreement_by_branch['A'] else 'did not agree'} at Branch A and {'agreed' if density_agreement_by_branch['B'] else 'did not agree'} at Branch B; pooled inference follows the two strata.",
        f"4. **FORMAL INFERENCE:** Current M and the four state counts were {'sufficient within the tested resolution at both branch points' if inventory else 'not sufficient on their own under at least one preregistered comparison' if spatial else 'not cleanly resolved as sufficient or insufficient'}.",
        f"5. **FORMAL INFERENCE:** {'Causally useful spatial organisation accumulated before the visible crossing.' if spatial else 'The finite intervention did not establish a causal acceleration from accumulated spatial organisation.'}",
        f"6. **SCIENTIFIC INTERPRETATION:** {combined}.",
        f"7. **MEASUREMENT:** Tile-scale interpretation: {tile_label.replace('_', ' ')}.",
        "8. **MEASUREMENT:** Immediate audit tables quantify changes in low-k power, interface density, block entropy, categorical mutual information, and angular power while exact counts, Q, M, and single-site entropy remain fixed.",
        f"9. **SCIENTIFIC INTERPRETATION:** The tested memory scale is {tile_label.replace('_', ' ')}; no cleaner length is claimed when tile ordering and direct metrics disagree.",
        "10. **MEASUREMENT:** The immediate and evolved low-k differences are tabulated independently of Q, M, and the exact histogram in MULTISCALE_MEMORY.csv.",
        "11. **MEASUREMENT:** Occupied/empty torus-wrapping clocks are retained in source and branch series; timing differences are reported without treating wrapping as the event definition.",
        "12. **SCIENTIFIC INTERPRETATION:** Diagonal anisotropy is treated as a measured visual/spectral feature unless its within-source change and residual-time association agree with the intervention result.",
        f"13. **SCIENTIFIC INTERPRETATION:** The preregistered combined mechanism label is **{combined}**.",
        "14. **FORMAL INFERENCE:** The strongest claim is limited to this frozen reversible charge-balanced block automaton and the finite intervention resolution.",
        "15. **SPECULATION:** Analogies to physical phase ordering, emergence, or cosmology remain analogies; the computation does not establish a physical invariant or an equilibrium transition.",
        "16. **SCIENTIFIC INTERPRETATION:** Beyond static signed cancellation, the programme supplies a reproducible dynamical example in which exact Q=0 and microscopic reversibility coexist with delayed, irregular macroscopic organisation; it does not establish that nature uses this rule or invariant.",
        "17. **SCIENTIFIC INTERPRETATION:** A standalone companion paper is justified by the exact model audit, independent parity, finite-time kinetics, arrangement controls, ablations, and the final state-matched intervention, including a null or inconclusive result.",
        "18. **SCIENTIFIC INTERPRETATION:** The lay headline is selected from the frozen result-dependent alternatives after the mechanism label is applied.",
        "19. **INTERPRETATION BOUNDARY:** Avoid saying that occupancy is conserved at zero, and avoid life, consciousness, physical cosmology, equilibrium phase transition, infinite-time threshold, universality, and no-one-has-done-this claims.",
        "20. **FORMAL INFERENCE:** The programme stops because its last preregistered causal question has been answered at the planned finite resolution—or its remaining uncertainty has been bounded—and the frozen scope requires manuscript production rather than another campaign.",
    ]


def build_history_report(run_path: str | Path, experiment_path: str | Path) -> dict[str, Any]:
    run = Path(run_path).resolve()
    experiment = Path(experiment_path).resolve()
    plan, manifest, realized, sources, branches = _collect(run, experiment)
    source_rows = _source_rows(sources)
    residual_rows = _residual_rows(branches)
    audit_rows = _audit_rows(run, realized)
    primary_a = [
        row
        for row in residual_rows
        if row["branch_point"] == "A"
        and row["intervention"] == "global_shuffle"
        and int(row["realization"]) == 1
    ]
    primary_b = [
        row
        for row in residual_rows
        if row["branch_point"] == "B"
        and row["intervention"] == "global_shuffle"
        and int(row["realization"]) == 1
    ]
    a_result = _classify("A", primary_a)
    b_result = _classify("B", primary_b)
    tile32_rows = [
        row for row in residual_rows if row["branch_point"] == "B" and row["intervention"] == "tile_32"
    ]
    tile128_rows = [
        row for row in residual_rows if row["branch_point"] == "B" and row["intervention"] == "tile_128"
    ]
    tile32 = _statistics(tile32_rows, name="Branch B tile 32")
    tile128 = _statistics(tile128_rows, name="Branch B tile 128")
    tile_label = _tile_interpretation(tile32, tile128, b_result["classification"])
    combined = _combined_mechanism(a_result["classification"], b_result["classification"], tile_label)
    density_agreement_by_branch = {
        "A": _density_direction_agreement(a_result),
        "B": _density_direction_agreement(b_result),
    }
    density_agreement = all(density_agreement_by_branch.values())
    results = {
        "schema": "zws-reversible-null-phase-v032-history-results-v1",
        "branch_A": a_result,
        "branch_B": b_result,
        "tile32_B": tile32,
        "tile128_B": tile128,
        "tile_interpretation": tile_label,
        "combined_mechanism": combined,
        "density_direction_agreement": density_agreement,
        "density_direction_agreement_by_branch": density_agreement_by_branch,
    }
    result_rows = [
        {
            "comparison": name,
            "classification": value.get("classification"),
            "n": value.get("pooled", value).get("n"),
            "rmst_delay": value.get("pooled", value).get("paired_rmst_delay", {}).get("point_estimate"),
            "ci90_low": value.get("pooled", value).get("paired_rmst_delay", {}).get("ci90", [None, None])[0],
            "ci90_high": value.get("pooled", value).get("paired_rmst_delay", {}).get("ci90", [None, None])[1],
            "ci95_low": value.get("pooled", value).get("paired_rmst_delay", {}).get("ci95", [None, None])[0],
            "ci95_high": value.get("pooled", value).get("paired_rmst_delay", {}).get("ci95", [None, None])[1],
        }
        for name, value in (
            ("Branch A primary global", a_result),
            ("Branch B primary global", b_result),
            ("Branch B tile 32", tile32),
            ("Branch B tile 128", tile128),
        )
    ]
    km_rows = _km_rows(residual_rows)
    multiscale_rows, entropy_rows, anisotropy_rows = _multiscale_rows(branches)
    representative_rows = _representatives(sources)
    resume_audit_path = experiment / "RESUME_REPLAY_AUDIT.json"
    resume_audit = _read_json(resume_audit_path) if resume_audit_path.is_file() else None
    integrity = _integrity(
        plan,
        manifest,
        realized,
        sources,
        branches,
        audit_rows,
        resume_audit,
    )
    if not integrity["passed"]:
        raise RuntimeError("v0.3.2 integrity audit failed; report interpretation stopped")

    _write_csv(experiment / "SOURCE_TRAJECTORIES.csv", source_rows)
    _write_csv(experiment / "INTERVENTION_AUDIT.csv", audit_rows)
    _write_csv(experiment / "RESIDUAL_FIRST_PASSAGE.csv", residual_rows)
    _write_csv(experiment / "SURVIVAL_SUMMARY.csv", km_rows)
    _write_csv(experiment / "HISTORY_ERASURE_RESULTS.csv", result_rows)
    _write_csv(experiment / "MULTISCALE_MEMORY.csv", multiscale_rows)
    _write_csv(experiment / "ENTROPY_HIERARCHY.csv", entropy_rows)
    _write_csv(experiment / "ANISOTROPY_RESULTS.csv", anisotropy_rows)
    _write_csv(experiment / "REPRESENTATIVE_STATE_INDEX.csv", representative_rows)
    atomic_write_json(experiment / "INTEGRITY_AUDIT.json", integrity)
    atomic_write_json(experiment / "SCIENTIFIC_DECISION_V032.json", results)

    answers = _answer_text(
        a_result,
        b_result,
        tile_label,
        combined,
        density_agreement_by_branch,
    )
    report = "\n".join(
        [
            "# v0.3.2 state-matched history-erasure report",
            "",
            f"Frozen winner: `{RULE_SHA256}`",
            f"Realized scientific trajectories: **{integrity['realized_scientific_trajectory_count']}** (16 progenitors plus {len(branches)} valid altered branches).",
            f"Primary combined mechanism: **{combined}**.",
            "",
            "## Result boundary",
            "",
            "Every altered branch preserved the complete current four-state inventory, exact Q, current M, single-site entropy, lattice dimensions, and Margolus phase. The winner itself preserves Q=0 but does not conserve M during later evolution. The translation nulls were required to pass byte-for-byte before interpretation.",
            "",
            "## Twenty preregistered questions",
            "",
            *answers,
            "",
            "## Frozen classifications",
            "",
            f"- Branch A primary global shuffle: **{a_result['classification']}**.",
            f"- Branch B primary global shuffle: **{b_result['classification']}**.",
            f"- Combined: **{combined}**.",
            f"- Tile-scale interpretation: **{tile_label.replace('_', ' ')}**.",
            "",
            "## Reproducibility",
            "",
            "The frozen plan, realized plan, exact state hashes, intervention audits, censoring records, cluster-bootstrap settings, multiscale tables, and integrity audit accompany this report. Giant transient checkpoints remain outside Git.",
            "",
        ]
    )
    atomic_write_text(experiment / "HISTORY_ERASURE_REPORT.md", report)
    atomic_write_text(
        experiment / "HISTORY_ERASURE_REPORT.md.sha256",
        f"{sha256_file(experiment / 'HISTORY_ERASURE_REPORT.md')}  HISTORY_ERASURE_REPORT.md\n",
    )
    return {
        "combined_mechanism": combined,
        "branch_A": a_result["classification"],
        "branch_B": b_result["classification"],
        "tile_interpretation": tile_label,
        "integrity_passed": integrity["passed"],
        "realized_scientific_trajectories": integrity["realized_scientific_trajectory_count"],
        "experiment": str(experiment),
    }
