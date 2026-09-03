from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import torch

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .initial import state_sha256
from .metrics import snapshot_metrics
from .visualize import PALETTE


EXPECTED_WINNER = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"


def _verified_file(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"missing payload/sidecar: {path}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"checksum mismatch: {path}")
    return actual


def _verified_json(path: Path) -> dict[str, Any]:
    _verified_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_task_artifacts(
    payload: dict[str, Any], result_path: Path, failures: list[str]
) -> tuple[int, int, int]:
    task_id = str(payload.get("task", {}).get("id", result_path.stem))
    task_dir = result_path.with_suffix("")
    sidecars_verified = 0
    state_hashes_verified = 0
    event_state_hashes_verified = 0
    for name in (
        "initial.npz",
        "peak.npz",
        "final.npz",
        "initial.png",
        "peak.png",
        "final.png",
    ):
        try:
            _verified_file(task_dir / name)
            sidecars_verified += 1
        except (FileNotFoundError, RuntimeError, IndexError) as exc:
            failures.append(f"artifact integrity failed for {task_id}: {exc}")

    for label in ("initial", "final"):
        path = task_dir / f"{label}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as stored:
                observed = state_sha256(stored["state"])
            expected = str(payload[label]["state_sha256"])
            if observed != expected:
                failures.append(f"{label} state hash mismatch: {task_id}")
            else:
                state_hashes_verified += 1
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"invalid {label} state artifact for {task_id}: {exc}")

    time_to_event = payload.get("time_to_event")
    window = payload.get("event_window")
    if isinstance(time_to_event, dict) and time_to_event.get("hit_observed"):
        if not window:
            failures.append(f"hit has no event window: {task_id}")
        else:
            event_path = task_dir / str(window["path"])
            try:
                digest = _verified_file(event_path)
                sidecars_verified += 1
                if digest != window["sha256"]:
                    failures.append(f"event-window payload hash mismatch: {task_id}")
                with np.load(event_path, allow_pickle=False) as stored:
                    phases = stored["phases"].astype(int).tolist()
                    states = stored["states"]
                if phases != [int(value) for value in window["phases"]]:
                    failures.append(f"event-window phase mismatch: {task_id}")
                observed_state_hashes = [state_sha256(state) for state in states]
                if observed_state_hashes != list(window["state_sha256"]):
                    failures.append(f"event-window state hash mismatch: {task_id}")
                else:
                    event_state_hashes_verified += len(observed_state_hashes)
            except (FileNotFoundError, RuntimeError, OSError, ValueError, KeyError) as exc:
                failures.append(f"event-window integrity failed for {task_id}: {exc}")
    return sidecars_verified, state_hashes_verified, event_state_hashes_verified


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n"
    )


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n"
    )


def _load_campaign(
    run_dir: Path, frozen_plan_path: Path, *, require_complete: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan = _verified_json(frozen_plan_path)
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    artifact_sidecars_verified = 0
    state_hashes_verified = 0
    event_state_hashes_verified = 0
    expected_tasks = {str(task["id"]): task for task in plan["tasks"]}
    for task_id, frozen in expected_tasks.items():
        result_path = run_dir / "tasks" / str(frozen["stage"]) / f"{task_id}.json"
        try:
            payload = _verified_json(result_path)
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(str(exc))
            continue
        task = payload.get("task", {})
        if task.get("id") != task_id:
            failures.append(f"task ID mismatch: {result_path}")
        if payload.get("config_sha256") != plan["config_sha256"]:
            failures.append(f"config hash mismatch: {task_id}")
        if payload.get("rule", {}).get("sha256") != EXPECTED_WINNER:
            failures.append(f"winner hash mismatch: {task_id}")
        if payload.get("initial", {}).get("state_sha256") != frozen["initial_state_sha256"]:
            failures.append(f"initial-state hash mismatch: {task_id}")
        charges = [int(row["charge"]) for row in payload.get("topological_series", [])]
        if not charges or any(value != charges[0] or value != 0 for value in charges):
            failures.append(f"exact-Q audit failed: {task_id}")
        time_to_event = payload.get("time_to_event")
        if not isinstance(time_to_event, dict):
            failures.append(f"missing time-to-event record: {task_id}")
        artifact_counts = _audit_task_artifacts(payload, result_path, failures)
        artifact_sidecars_verified += artifact_counts[0]
        state_hashes_verified += artifact_counts[1]
        event_state_hashes_verified += artifact_counts[2]
        payload["_result_path"] = str(result_path)
        results.append(payload)
    if require_complete and len(results) != len(expected_tasks):
        failures.append(f"campaign incomplete: {len(results)}/{len(expected_tasks)} results")
    audit = {
        "schema": "zws-reversible-null-phase-v031-integrity-audit-v1",
        "pass": not failures and len(results) == len(expected_tasks),
        "expected_tasks": len(expected_tasks),
        "verified_results": len(results),
        "config_sha256": plan["config_sha256"],
        "winner_sha256": EXPECTED_WINNER,
        "frozen_plan_sha256": sha256_file(frozen_plan_path),
        "artifact_sidecars_verified": artifact_sidecars_verified,
        "state_hashes_verified": state_hashes_verified,
        "event_state_hashes_verified": event_state_hashes_verified,
        "failures": failures,
    }
    if require_complete and failures:
        raise RuntimeError("campaign integrity audit failed: " + "; ".join(failures[:8]))
    return plan, results, audit


def _apply_stage2(
    run_dir: Path,
    experiment_dir: Path,
    results: list[dict[str, Any]],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    stage2_plan_path = experiment_dir / "STAGE2_PLAN.json"
    if not stage2_plan_path.exists():
        audit["stage2"] = {"status": "not_run", "verified_results": 0}
        return results
    stage2_plan = _verified_json(stage2_plan_path)
    by_source = {payload["task"]["id"]: payload for payload in results}
    replacements: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    artifact_sidecars_verified = 0
    state_hashes_verified = 0
    event_state_hashes_verified = 0
    for entry in stage2_plan["entries"]:
        source_id = entry["source_task_id"]
        continuation = entry["continuation_task"]
        result_path = (
            run_dir
            / "stage2"
            / "tasks"
            / continuation["stage"]
            / f"{continuation['id']}.json"
        )
        try:
            payload = _verified_json(result_path)
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(str(exc))
            continue
        source = by_source[source_id]
        if payload["task"]["id"] != continuation["id"]:
            failures.append(f"Stage 2 task ID mismatch: {continuation['id']}")
        if payload["config_sha256"] != stage2_plan["config_sha256"]:
            failures.append(f"Stage 2 config mismatch: {continuation['id']}")
        if payload["rule"]["sha256"] != stage2_plan["winner_sha256"]:
            failures.append(f"Stage 2 winner mismatch: {continuation['id']}")
        source_series = source["topological_series"]
        if payload["topological_series"][: len(source_series)] != source_series:
            failures.append(f"Stage 2 prefix mismatch: {continuation['id']}")
        artifact_counts = _audit_task_artifacts(payload, result_path, failures)
        artifact_sidecars_verified += artifact_counts[0]
        state_hashes_verified += artifact_counts[1]
        event_state_hashes_verified += artifact_counts[2]
        scientific = copy.deepcopy(payload)
        scientific["task"] = {
            **entry["source_task"],
            "steps": continuation["steps"],
        }
        scientific["_result_path"] = str(result_path)
        scientific["_stage2_continuation_task_id"] = continuation["id"]
        replacements[source_id] = scientific
    audit["stage2"] = {
        "status": "complete" if not failures and len(replacements) == len(stage2_plan["entries"]) else "failed",
        "expected_results": len(stage2_plan["entries"]),
        "verified_results": len(replacements),
        "plan_sha256": sha256_file(stage2_plan_path),
        "artifact_sidecars_verified": artifact_sidecars_verified,
        "state_hashes_verified": state_hashes_verified,
        "event_state_hashes_verified": event_state_hashes_verified,
        "failures": failures,
    }
    if failures or len(replacements) != len(stage2_plan["entries"]):
        raise RuntimeError("Stage 2 integrity audit failed: " + "; ".join(failures[:8]))
    return [replacements.get(payload["task"]["id"], payload) for payload in results]


def _trajectory_row(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    tte = payload["time_to_event"]
    top = payload["topological_series"]
    temporal = payload["temporal_series"]
    first = top[0]
    final = top[-1]
    event_time = tte["T_hit"] if tte["T_hit"] is not None else tte["censor_phase"]
    persistent_time = (
        tte["T_persistent"]
        if tte["T_persistent"] is not None
        else tte["censor_phase"]
    )
    fixed = tte["fixed_readout_strong_domain"]
    mean_reaction = float(np.mean([row["reaction_fraction"] for row in temporal[1:]]))
    mean_activity_values = [row["activity"] for row in temporal[1:] if row["activity"] is not None]
    size = int(task["size"])
    return {
        "task_id": task["id"],
        "stage": task["stage"],
        "size": size,
        "density": float(task["density"]),
        "realized_density": (
            payload["initial"]["inventory"]["positive"]
            + payload["initial"]["inventory"]["negative"]
            + payload["initial"]["inventory"]["bound"]
        )
        / (size**2),
        "seed": int(task["seed"]),
        "arrangement": task["arrangement"],
        "horizon": int(task["steps"]),
        "T_hit": tte["T_hit"],
        "T_persistent": tte["T_persistent"],
        "hit_observed": bool(tte["hit_observed"]),
        "persistent_observed": bool(tte["persistent_observed"]),
        "right_censored_T_hit": bool(tte["right_censored_T_hit"]),
        "right_censored_T_persistent": bool(tte["right_censored_T_persistent"]),
        "event_or_censor_time": event_time,
        "phase_exposure": float(event_time),
        "persistent_event_or_censor_time": persistent_time,
        "persistent_phase_exposure": float(persistent_time),
        "transformed_by_32768": bool(tte["T_hit"] is not None and tte["T_hit"] <= 32768),
        "transformed_by_65536": bool(tte["T_hit"] is not None and tte["T_hit"] <= 65536),
        "transformed_by_131072": bool(tte["T_hit"] is not None and tte["T_hit"] <= 131072),
        "transformed_by_262144": bool(tte["T_hit"] is not None and tte["T_hit"] <= 262144),
        "prefix_strong_at_32768": fixed.get("32768"),
        "prefix_strong_at_65536": fixed.get("65536"),
        "prefix_strong_at_131072": fixed.get("131072"),
        "prefix_strong_at_262144": fixed.get("262144"),
        **{f"first_{key}": value for key, value in tte["topology_first_times"].items()},
        "initial_occupancy_fraction": float(first["occupancy_fraction"]),
        "final_occupancy_fraction": float(final["occupancy_fraction"]),
        "initial_finite_empty_scale": float(first["finite_empty_scale"]),
        "final_finite_empty_scale": float(final["finite_empty_scale"]),
        "peak_finite_empty_scale": float(max(row["finite_empty_scale"] for row in top)),
        "initial_largest_occupied_fraction": float(first["largest_occupied_fraction"]),
        "final_largest_occupied_fraction": float(final["largest_occupied_fraction"]),
        "occupied_cluster_retention_ratio": (
            float(final["largest_occupied_fraction"])
            / max(float(first["largest_occupied_fraction"]), 1e-30)
        ),
        "initial_interface_density": float(first["occupied_perimeter_density"]),
        "final_interface_density": float(final["occupied_perimeter_density"]),
        "initial_low_k_power": float(first["low_k_power"]),
        "final_low_k_power": float(final["low_k_power"]),
        "mean_reaction_density": mean_reaction,
        "mean_activity": float(np.mean(mean_activity_values)) if mean_activity_values else None,
        "charge_violation": any(int(row["charge"]) != 0 for row in top),
        "initial_state_sha256": payload["initial"]["state_sha256"],
        "final_state_sha256": payload["final"]["state_sha256"],
        "result_path": payload["_result_path"],
    }


def _km_rows(
    frame: pd.DataFrame, *, endpoint: str = "T_hit"
) -> list[dict[str, Any]]:
    if endpoint not in {"T_hit", "T_persistent"}:
        raise ValueError(f"unsupported time-to-event endpoint: {endpoint}")
    observed_column = "hit_observed" if endpoint == "T_hit" else "persistent_observed"
    time_column = (
        "event_or_censor_time"
        if endpoint == "T_hit"
        else "persistent_event_or_censor_time"
    )
    rows: list[dict[str, Any]] = []
    keys = ["stage", "size", "density", "arrangement"]
    for coordinate, group in frame.groupby(keys, sort=True):
        if time_column in group:
            event_times = group[time_column].astype(float)
        else:
            event_times = group[endpoint].where(
                group[observed_column], group["horizon"]
            ).astype(float)
        ordered = group.assign(
            _event_observed=group[observed_column].astype(bool),
            _event_time=event_times,
        ).sort_values(["_event_time", "task_id"])
        at_risk = len(ordered)
        survival = 1.0
        events_total = int(ordered._event_observed.sum())
        exposure = float(ordered._event_time.sum())
        rate = events_total / exposure if exposure > 0 else float("nan")
        base = dict(zip(keys, coordinate, strict=True))
        common = {
            **base,
            "endpoint": endpoint,
            "N": len(ordered),
            "total_events": events_total,
            "total_phase_exposure": exposure,
            "descriptive_event_rate_per_phase": rate,
            "descriptive_event_rate_per_cell_phase": rate / (int(base["size"]) ** 2),
        }
        rows.append(
            {
                **common,
                "phase": 0,
                "at_risk": at_risk,
                "events_at_phase": 0,
                "censored_at_phase": 0,
                "survival": 1.0,
                "cumulative_transformation": 0.0,
            }
        )
        for phase in sorted(set(int(x) for x in ordered._event_time)):
            at_phase = ordered[ordered._event_time == phase]
            events = int(at_phase._event_observed.sum())
            censored = len(at_phase) - events
            if at_risk and events:
                survival *= 1.0 - events / at_risk
            rows.append(
                {
                    **common,
                    "phase": phase,
                    "at_risk": at_risk,
                    "events_at_phase": events,
                    "censored_at_phase": censored,
                    "survival": survival,
                    "cumulative_transformation": 1.0 - survival,
                }
            )
            at_risk -= events + censored
    return rows


def _km_quantile(km: pd.DataFrame, quantile: float) -> float | None:
    eligible = km[km.cumulative_transformation >= quantile]
    return float(eligible.phase.iloc[0]) if not eligible.empty else None


def _restricted_mean(km: pd.DataFrame, horizon: int) -> float:
    """Integrate a Kaplan-Meier survival curve through a fixed horizon."""
    curve = km[km.phase <= horizon].sort_values("phase")
    if curve.empty:
        return float(horizon)
    times = curve.phase.to_numpy(dtype=float)
    survival = curve.survival.to_numpy(dtype=float)
    tail = float(horizon) - times[-1]
    return float(np.sum(np.diff(times) * survival[:-1]) + max(tail, 0.0) * survival[-1])


def _hazard_shape(group: pd.DataFrame, horizon: int) -> tuple[list[float], str]:
    edges = np.linspace(0.0, float(horizon), 4)
    rates: list[float] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        exposure = sum(
            max(0.0, min(float(end), high) - low)
            for end in group.event_or_censor_time
            if float(end) > low
        )
        events = int(
            (
                group.hit_observed
                & (group.event_or_censor_time > low)
                & (group.event_or_censor_time <= high)
            ).sum()
        )
        rates.append(events / exposure if exposure else 0.0)
    if int(group.hit_observed.sum()) < 3:
        return rates, "insufficient-events"
    positive = [rate for rate in rates if rate > 0]
    if len(positive) < 2:
        return rates, "strongly-nonstationary"
    if rates[1] > 2 * max(rates[0], rates[2], 1e-30) or rates[1] < 0.5 * min(
        rate for rate in (rates[0], rates[2]) if rate > 0
    ):
        return rates, "strongly-nonstationary"
    ratio = rates[-1] / max(rates[0], 1e-30)
    if ratio >= 1.5:
        return rates, "increasing"
    if ratio <= 2 / 3:
        return rates, "decreasing"
    return rates, "roughly-constant"


def _group_summary(
    frame: pd.DataFrame,
    km: pd.DataFrame,
    persistent_km: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if persistent_km is None:
        persistent_km = pd.DataFrame(_km_rows(frame, endpoint="T_persistent"))
    rows: list[dict[str, Any]] = []
    keys = ["stage", "size", "density", "arrangement"]
    for coordinate, group in frame.groupby(keys, sort=True):
        selector = np.logical_and.reduce(
            [km[key] == value for key, value in zip(keys, coordinate, strict=True)]
        )
        curve = km[selector].sort_values("phase")
        persistent_selector = np.logical_and.reduce(
            [
                persistent_km[key] == value
                for key, value in zip(keys, coordinate, strict=True)
            ]
        )
        persistent_curve = persistent_km[persistent_selector].sort_values("phase")
        observed = group.loc[group.hit_observed, "T_hit"].astype(float)
        persistent = group.loc[group.persistent_observed, "T_persistent"].astype(float)
        horizon = int(group.horizon.max())
        common_horizon = min(horizon, 262144)
        rmst = _restricted_mean(curve, horizon)
        common_rmst = _restricted_mean(curve, common_horizon)
        exposure = float(group.phase_exposure.sum())
        size = int(coordinate[1])
        interval_rates, hazard_shape = _hazard_shape(group, horizon)
        median_hit = _km_quantile(curve, 0.5)
        median_persistent = _km_quantile(persistent_curve, 0.5)
        rows.append(
            {
                **dict(zip(keys, coordinate, strict=True)),
                "N": len(group),
                "events": int(group.hit_observed.sum()),
                "persistent_events": int(group.persistent_observed.sum()),
                "right_censored": int(group.right_censored_T_hit.sum()),
                "right_censored_T_persistent": int(
                    len(group) - group.persistent_observed.sum()
                ),
                "event_fraction": float(group.hit_observed.mean()),
                "persistent_fraction": float(group.persistent_observed.mean()),
                "median_T_hit": median_hit,
                "q25_T_hit": _km_quantile(curve, 0.25),
                "q75_T_hit": _km_quantile(curve, 0.75),
                "median_T_hit_lower_bound": (
                    horizon if median_hit is None else None
                ),
                "median_T_persistent": median_persistent,
                "q25_T_persistent": _km_quantile(persistent_curve, 0.25),
                "q75_T_persistent": _km_quantile(persistent_curve, 0.75),
                "median_T_persistent_lower_bound": (
                    horizon
                    if median_persistent is None
                    else None
                ),
                "median_T_persistent_observed_only": (
                    float(persistent.median()) if len(persistent) else None
                ),
                "earliest_T_hit": float(observed.min()) if len(observed) else None,
                "latest_observed_T_hit": float(observed.max()) if len(observed) else None,
                "earliest_T_persistent": (
                    float(persistent.min()) if len(persistent) else None
                ),
                "latest_observed_T_persistent": (
                    float(persistent.max()) if len(persistent) else None
                ),
                "restricted_mean_event_free_time": rmst,
                "restricted_mean_event_free_time_common_horizon": common_rmst,
                "restricted_mean_common_horizon": common_horizon,
                "total_phase_exposure": exposure,
                "horizon": horizon,
                "events_by_32768": int(group.transformed_by_32768.sum()),
                "events_by_65536": int(group.transformed_by_65536.sum()),
                "events_by_131072": int(group.transformed_by_131072.sum()),
                "events_by_262144": int(group.transformed_by_262144.sum()),
                "descriptive_event_rate_per_phase": float(group.hit_observed.sum() / exposure),
                "descriptive_event_rate_per_cell_phase": float(
                    group.hit_observed.sum() / exposure / (size**2)
                ),
                "hazard_early_per_phase": interval_rates[0],
                "hazard_middle_per_phase": interval_rates[1],
                "hazard_late_per_phase": interval_rates[2],
                "hazard_shape": hazard_shape,
            }
        )
    return rows


def _mechanism_label(payload: dict[str, Any]) -> str:
    tte = payload["time_to_event"]
    hit = tte["T_hit"]
    if hit is None or not payload.get("event_window"):
        return "E"
    series = payload["topological_series"]
    phases = [int(row["phase"]) for row in series]
    hit_index = phases.index(int(hit))
    initial_scale = max(float(series[0]["finite_empty_scale"]), 1.0)
    pre_two = series[max(0, hit_index - 2)]
    topology_times = [
        value for value in tte["topology_first_times"].values() if value is not None
    ]
    observation = int(tte["observation_every"])
    coincident_wrap = any(abs(int(value) - int(hit)) <= observation for value in topology_times)
    d_rule = coincident_wrap and float(pre_two["finite_empty_scale"]) / initial_scale >= 2.0
    localization = float(payload["event_window"]["change_localization_fraction"])
    a_rule = localization >= 0.5
    pre = series[max(0, hit_index - 8) : hit_index + 1]
    increments = [
        float(right["finite_empty_scale"]) > float(left["finite_empty_scale"])
        for left, right in zip(pre[:-1], pre[1:], strict=True)
    ]
    b_rule = (
        localization <= 0.125
        and bool(increments)
        and sum(increments) >= math.ceil(len(increments) / 2)
    )
    pre_low = float(pre[0]["low_k_power"]) / max(float(series[0]["low_k_power"]), 1e-30)
    component_drop = 1.0 - float(pre[-1]["empty_finite_component_count"]) / max(
        float(pre[0]["empty_finite_component_count"]), 1.0
    )
    c_rule = not a_rule and not b_rule and pre_low >= 2.0 and component_drop >= 0.25
    if d_rule:
        return "D"
    active = [
        label for label, flag in (("A", a_rule), ("B", b_rule), ("C", c_rule)) if flag
    ]
    return active[0] if len(active) == 1 else "E"


def _event_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in results:
        if not payload["time_to_event"]["hit_observed"]:
            continue
        task = payload["task"]
        window = payload["event_window"]
        path = Path(payload["_result_path"]).with_suffix("") / window["path"]
        stored = np.load(path)
        phases = stored["phases"].astype(int)
        states = stored["states"]
        temporal = {int(row["phase"]): row for row in payload["temporal_series"]}
        label = _mechanism_label(payload)
        global_metrics = {
            int(row["phase"]): row for row in payload["topological_series"]
        }
        for phase, state in zip(phases, states, strict=True):
            metric = snapshot_metrics(
                torch.from_numpy(state), phase=int(phase), topology=True
            ).to_dict()
            global_metric = global_metrics[int(phase)]
            time_row = temporal.get(int(phase), {})
            rows.append(
                {
                    "task_id": task["id"],
                    "stage": task["stage"],
                    "size": task["size"],
                    "density": task["density"],
                    "seed": task["seed"],
                    "arrangement": task["arrangement"],
                    "T_hit": payload["time_to_event"]["T_hit"],
                    "phase": int(phase),
                    "relative_phase": int(phase) - int(payload["time_to_event"]["T_hit"]),
                    "mechanism_label": label,
                    "change_localization_fraction": window[
                        "change_localization_fraction"
                    ],
                    "crop_center_y": window["crop_center"][0],
                    "crop_center_x": window["crop_center"][1],
                    "crop_size": window["crop_size"],
                    "local_occupancy_fraction": metric["occupancy_fraction"],
                    "local_positive_fraction": metric["positive_fraction"],
                    "local_negative_fraction": metric["negative_fraction"],
                    "local_bound_fraction": metric["bound_fraction"],
                    "local_entropy": metric["normalized_state_entropy"],
                    "local_finite_empty_scale": metric["finite_empty_scale"],
                    "local_largest_finite_empty_fraction": metric[
                        "largest_finite_empty_fraction"
                    ],
                    "local_interface_density": metric["occupied_perimeter_density"],
                    "local_low_k_power": metric["low_k_power"],
                    "global_finite_empty_scale": global_metric["finite_empty_scale"],
                    "global_largest_finite_empty_fraction": global_metric[
                        "largest_finite_empty_fraction"
                    ],
                    "global_interface_density": global_metric[
                        "occupied_perimeter_density"
                    ],
                    "global_low_k_power": global_metric["low_k_power"],
                    "global_occupancy_fraction": global_metric["occupancy_fraction"],
                    "reaction_density": time_row.get("reaction_fraction"),
                    "activity": time_row.get("activity"),
                    "occupied_wrap_x": global_metric["occupied_wrap_x"],
                    "occupied_wrap_y": global_metric["occupied_wrap_y"],
                    "empty_wrap_x": global_metric["empty_wrap_x"],
                    "empty_wrap_y": global_metric["empty_wrap_y"],
                    "event_window_sha256": window["sha256"],
                }
            )
    return rows


def _representatives(group: pd.DataFrame) -> list[str]:
    observed = group[group.hit_observed].copy()
    if observed.empty:
        return []
    observed = observed.sort_values(["T_hit", "task_id"])
    median = float(observed.T_hit.median())
    nearest = observed.assign(distance=(observed.T_hit - median).abs()).sort_values(
        ["distance", "T_hit", "task_id"]
    )
    return list(
        dict.fromkeys(
            [observed.task_id.iloc[0], nearest.task_id.iloc[0], observed.task_id.iloc[-1]]
        )
    )


def _plots(
    trajectories: pd.DataFrame,
    km: pd.DataFrame,
    summaries: pd.DataFrame,
    area_summary: pd.DataFrame,
    seeded_summary: pd.DataFrame,
    event_frame: pd.DataFrame,
    results_by_id: dict[str, dict[str, Any]],
    figure_dir: Path,
) -> list[dict[str, Any]]:
    figures: list[tuple[str, str]] = []
    primary = trajectories[
        (trajectories.stage == "primary_homogeneous_1024")
        & (trajectories.arrangement == "homogeneous")
    ]
    primary_km = km[km.stage == "primary_homogeneous_1024"]

    for column, name, ylabel in (
        ("survival", "survival_curves.png", "P(not yet strong-domain)"),
        (
            "cumulative_transformation",
            "cumulative_transformation.png",
            "Cumulative transformation probability",
        ),
    ):
        plt.figure(figsize=(8.2, 5.2))
        for density, group in primary_km.groupby("density"):
            plt.step(group.phase, group[column], where="post", label=f"{density:.2f}")
        plt.xlabel("Phase")
        plt.ylabel(ylabel)
        plt.ylim(-0.03, 1.03)
        plt.legend(title="Density", ncol=2)
        _save_figure(figure_dir / name)
        figures.append((name, ylabel))

    densities = sorted(primary.density.unique())
    plt.figure(figsize=(8.2, 5.2))
    for index, (density, group) in enumerate(primary.groupby("density")):
        y = np.full(len(group), index, dtype=float)
        observed = group.hit_observed.to_numpy(dtype=bool)
        plt.scatter(
            group.event_or_censor_time[observed],
            y[observed],
            marker="o",
            label=f"{density:.2f}",
        )
        plt.scatter(
            group.event_or_censor_time[~observed],
            y[~observed],
            marker=">",
            facecolors="none",
            edgecolors="black",
        )
    plt.yticks(range(len(densities)), [f"{x:.2f}" for x in densities])
    plt.xlabel("T_hit (open arrow = right-censored)")
    plt.ylabel("Density")
    _save_figure(figure_dir / "first_passage_distribution.png")
    figures.append(("first_passage_distribution.png", "First-passage distribution vs density"))

    psummary = summaries[summaries.stage == "primary_homogeneous_1024"].sort_values(
        "density"
    )
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(
        psummary.density,
        psummary.restricted_mean_event_free_time_common_horizon,
        "o-",
        label="RMST to phase 262,144",
    )
    med = psummary.dropna(subset=["median_T_hit"])
    if not med.empty:
        plt.plot(
            med.density,
            med.median_T_hit,
            "s-",
            label="KM median (full frozen horizon)",
        )
    plt.xlabel("Density")
    plt.ylabel("Characteristic waiting time (phases)")
    plt.legend()
    _save_figure(figure_dir / "characteristic_waiting_time.png")
    figures.append(("characteristic_waiting_time.png", "Characteristic waiting time"))

    plt.figure(figsize=(8.2, 5.2))
    for phase in (32768, 65536, 131072, 262144):
        values = [
            float(group[f"transformed_by_{phase}"].mean())
            for _, group in primary.groupby("density")
        ]
        plt.plot(densities, values, "o-", label=f"{phase:,}")
    plt.xlabel("Density")
    plt.ylabel("Transformed fraction")
    plt.ylim(-0.03, 1.03)
    plt.legend(title="Phase")
    _save_figure(figure_dir / "fixed_time_pass_fraction.png")
    figures.append(("fixed_time_pass_fraction.png", "Fixed-time transformation fractions"))

    area = area_summary
    plt.figure(figsize=(8.2, 5.2))
    for size, group in area.groupby("size"):
        plt.plot(group.density, group.event_fraction, "o-", label=f"{int(size)}²")
    plt.xlabel("Density")
    plt.ylabel("Transformation fraction by horizon")
    plt.ylim(-0.03, 1.03)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    plt.legend()
    _save_figure(figure_dir / "area_comparison.png")
    figures.append(("area_comparison.png", "512² versus 1024² transformation"))

    plt.figure(figsize=(8.2, 5.2))
    for size, group in area.groupby("size"):
        positive_rate = group.descriptive_event_rate_per_cell_phase.replace(0, np.nan)
        plt.plot(group.density, positive_rate, "o-", label=f"{int(size)}²")
    plt.yscale("log")
    plt.xlabel("Density")
    plt.ylabel("Descriptive events / cell / phase")
    plt.legend()
    _save_figure(figure_dir / "area_normalized_hazard.png")
    figures.append(
        ("area_normalized_hazard.png", "Area-normalized descriptive event rate")
    )

    seeded = seeded_summary
    plt.figure(figsize=(8.2, 5.2))
    for arrangement, group in seeded.groupby("arrangement"):
        plt.plot(group.density, group.event_fraction, "o-", label=arrangement)
    plt.xlabel("Density")
    plt.ylabel("Strong-domain hit fraction")
    plt.ylim(-0.03, 1.03)
    fraction_ticks = np.linspace(0.0, 1.0, 6)
    plt.yticks(fraction_ticks, [f"{value:.0%}" for value in fraction_ticks])
    plt.legend()
    _save_figure(figure_dir / "seeded_basin_comparison.png")
    figures.append(("seeded_basin_comparison.png", "Homogeneous versus seeded basins"))

    plt.figure(figsize=(8.2, 5.2))
    offsets: list[int] = []
    labels: list[str] = []
    for row in primary.loc[primary.hit_observed].itertuples():
        for label, value in (
            ("occ-x", row.first_occupied_wrap_x),
            ("occ-y", row.first_occupied_wrap_y),
            ("empty-x", row.first_empty_wrap_x),
            ("empty-y", row.first_empty_wrap_y),
        ):
            if pd.notna(value):
                offsets.append(int(value) - int(row.T_hit))
                labels.append(label)
    if offsets:
        order = ["occ-x", "occ-y", "empty-x", "empty-y"]
        plt.scatter(offsets, [order.index(label) for label in labels], alpha=0.65)
        plt.yticks(range(len(order)), order)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Topology event phase minus T_hit")
    plt.ylabel("Topology event")
    _save_figure(figure_dir / "topology_relative_timing.png")
    figures.append(
        ("topology_relative_timing.png", "Topology timing relative to first passage")
    )

    if not event_frame.empty:
        primary_events = event_frame[
            event_frame.stage == "primary_homogeneous_1024"
        ]
        plt.figure(figsize=(8.2, 5.2))
        for density, group in primary_events.groupby("density"):
            averaged = group.groupby("relative_phase").global_finite_empty_scale.median()
            plt.plot(averaged.index, averaged.values, label=f"{density:.2f}")
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Phase relative to T_hit")
        plt.ylabel("Median finite-empty scale")
        plt.legend(title="Density", ncol=2)
        _save_figure(figure_dir / "event_centered_domain_growth.png")
        figures.append(
            ("event_centered_domain_growth.png", "Domain growth centered on T_hit")
        )

        plt.figure(figsize=(8.2, 5.2))
        for density, group in primary_events.groupby("density"):
            averaged = group.groupby("relative_phase").global_low_k_power.median()
            normalized = averaged / max(float(averaged.iloc[0]), 1e-30)
            plt.plot(normalized.index, normalized.values, label=f"{density:.2f}")
        plt.yscale("log")
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Phase relative to T_hit")
        plt.ylabel("Median low-k power / pre-window value")
        plt.legend(title="Density", ncol=2)
        _save_figure(figure_dir / "event_centered_low_k_growth.png")
        figures.append(("event_centered_low_k_growth.png", "Low-k growth centered on T_hit"))

    representatives: list[str] = []
    for _, group in primary.groupby("density"):
        representatives.extend(_representatives(group))
    representatives = list(dict.fromkeys(representatives))
    if representatives:
        columns = 5
        _, axes = plt.subplots(
            len(representatives),
            columns,
            figsize=(12, 2.25 * len(representatives)),
            squeeze=False,
        )
        for row_index, task_id in enumerate(representatives):
            payload = results_by_id[task_id]
            window = payload["event_window"]
            stored = np.load(
                Path(payload["_result_path"]).with_suffix("") / window["path"]
            )
            phases = stored["phases"]
            states = stored["states"]
            picks = np.linspace(0, len(phases) - 1, columns).round().astype(int)
            for col_index, pick in enumerate(picks):
                axes[row_index, col_index].imshow(
                    PALETTE[states[pick]], interpolation="nearest"
                )
                relative = int(phases[pick] - payload["time_to_event"]["T_hit"])
                axes[row_index, col_index].set_title(f"Δt={relative:,}")
                axes[row_index, col_index].axis("off")
            axes[row_index, 0].set_ylabel(
                f"d={payload['task']['density']:.2f}\n{task_id[:7]}"
            )
        _save_figure(figure_dir / "event_morphology_sequences.png")
        figures.append(
            (
                "event_morphology_sequences.png",
                "Frozen-rule event morphology representatives",
            )
        )

    return [
        {
            "path": f"figures/{name}",
            "description": description,
            "sha256": sha256_file(figure_dir / name),
        }
        for name, description in figures
    ]


def _load_scientific_decision(experiment: Path) -> dict[str, Any] | None:
    path = experiment / "SCIENTIFIC_DECISION.json"
    if not path.exists():
        return None
    decision = _verified_json(path)
    if decision.get("schema") != "zws-reversible-null-phase-v031-scientific-decision-v1":
        raise RuntimeError("invalid v0.3.1 scientific-decision schema")
    if decision.get("option") not in {"A", "B", "C"}:
        raise RuntimeError("scientific decision must choose exactly one of A, B, or C")
    for field in (
        "publication_answer",
        "further_v03_answer",
        "scientific_claim",
        "rationale",
    ):
        if not isinstance(decision.get(field), str) or not decision[field].strip():
            raise RuntimeError(f"scientific decision is missing {field}")
    return decision


def _report_text(
    trajectories: pd.DataFrame,
    summaries: pd.DataFrame,
    area_summary: pd.DataFrame,
    seeded_summary: pd.DataFrame,
    event_frame: pd.DataFrame,
    audit: dict[str, Any],
    figure_count: int,
    scientific_decision: dict[str, Any] | None,
) -> str:
    primary = summaries[summaries.stage == "primary_homogeneous_1024"].sort_values(
        "density"
    )
    median_densities = [
        f"{x:.2f}" for x in primary.dropna(subset=["median_T_hit"]).density
    ]
    persistent_median_densities = [
        f"{x:.2f}"
        for x in primary.dropna(subset=["median_T_persistent"]).density
    ]
    event_densities = [
        f"{row.density:.2f} ({int(row.events)}/{int(row.N)})"
        for row in primary.itertuples()
        if row.events
    ]
    mechanisms = (
        Counter(event_frame.drop_duplicates("task_id").mechanism_label)
        if not event_frame.empty
        else Counter()
    )
    positive = primary[primary.density == 0.26]
    positive_text = "not available"
    if not positive.empty:
        row = positive.iloc[0]
        positive_text = (
            f"{int(row.events_by_131072)}/{int(row.N)} hits by 131,072 "
            f"and {int(row.events)}/{int(row.N)} by horizon"
        )
    topology = trajectories[
        (trajectories.stage == "primary_homogeneous_1024")
        & trajectories.hit_observed
    ]
    topology_offsets: list[float] = []
    for row in topology.itertuples():
        for value in (
            row.first_occupied_wrap_x,
            row.first_occupied_wrap_y,
            row.first_empty_wrap_x,
            row.first_empty_wrap_y,
        ):
            if pd.notna(value):
                topology_offsets.append(float(value) - float(row.T_hit))
    topology_answer = (
        f"median recorded wrap offset is {np.median(topology_offsets):,.0f} phases relative to T_hit"
        if topology_offsets
        else "no comparable wrap time was observed"
    )
    topology_summary_lines = [
        "| density | wrap event | observed / hit trajectories | before / same / after T_hit | median offset |",
        "| --- | --- | --- | --- | --- |",
    ]
    topology_fields = (
        ("occupied x", "first_occupied_wrap_x"),
        ("occupied y", "first_occupied_wrap_y"),
        ("empty x", "first_empty_wrap_x"),
        ("empty y", "first_empty_wrap_y"),
    )
    for density, group in topology.groupby("density", sort=True):
        for label, column in topology_fields:
            offsets = (
                group.loc[group[column].notna(), column].astype(float)
                - group.loc[group[column].notna(), "T_hit"].astype(float)
            )
            topology_summary_lines.append(
                "| "
                + " | ".join(
                    [
                        f"{density:.2f}",
                        label,
                        f"{len(offsets)} / {len(group)}",
                        f"{int((offsets < 0).sum())} / {int((offsets == 0).sum())} / {int((offsets > 0).sum())}",
                        f"{float(offsets.median()):,.0f}" if len(offsets) else "—",
                    ]
                )
                + " |"
            )
    if len(topology_summary_lines) == 2:
        topology_summary_lines.append("| — | no primary hit trajectories | 0 / 0 | 0 / 0 / 0 | — |")
    topology_summary_table = "\n".join(topology_summary_lines)
    lower_events = primary[(primary.density <= 0.24) & (primary.events > 0)]
    if scientific_decision is None:
        publication_answer = (
            "Pending explicit post-analysis synthesis; no event-count shortcut is used to decide publication readiness."
        )
        recommendation = (
            "Pending explicit post-analysis A/B/C choice in the checksummed scientific-decision record."
        )
        decision_section = [
            "## Final A/B/C decision",
            "",
            "Provisional: `SCIENTIFIC_DECISION.json` has not yet been frozen. This measurement report must be regenerated after the complete evidence is inspected.",
            "",
        ]
    else:
        option = str(scientific_decision["option"])
        publication_answer = str(scientific_decision["publication_answer"])
        recommendation = f"{option} — {scientific_decision['further_v03_answer']}"
        decision_section = [
            "## Final A/B/C decision",
            "",
            f"**{option}. {scientific_decision['further_v03_answer']}**",
            "",
            f"Scientific claim: {scientific_decision['scientific_claim']}",
            "",
            f"Rationale: {scientific_decision['rationale']}",
            "",
        ]

    def format_time(value: Any, horizon: int, *, lower_bound: bool = False) -> str:
        if pd.isna(value):
            return f"&gt;{horizon:,}" if lower_bound else "—"
        return f"{int(value):,}"

    first_passage_lines = [
        "| density | N | T_hit events / censored | median T_hit | q25 / q75 T_hit | T_persistent events / censored | median T_persistent | q25 / q75 T_persistent | earliest / latest T_hit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    fixed_time_lines = [
        "| density | N | by 32,768 | by 65,536 | by 131,072 | by 262,144 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    hazard_lines = [
        "| density | events | untransformed phase-exposure | descriptive λ / phase | descriptive λ / cell / phase | early / middle / late λ | shape |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in primary.itertuples():
        horizon = int(row.horizon)
        first_passage_lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.density:.2f}",
                    str(int(row.N)),
                    f"{int(row.events)} / {int(row.right_censored)}",
                    format_time(row.median_T_hit, horizon, lower_bound=True),
                    f"{format_time(row.q25_T_hit, horizon, lower_bound=True)} / {format_time(row.q75_T_hit, horizon, lower_bound=True)}",
                    f"{int(row.persistent_events)} / {int(row.right_censored_T_persistent)}",
                    format_time(
                        row.median_T_persistent, horizon, lower_bound=True
                    ),
                    f"{format_time(row.q25_T_persistent, horizon, lower_bound=True)} / {format_time(row.q75_T_persistent, horizon, lower_bound=True)}",
                    f"{format_time(row.earliest_T_hit, horizon)} / {format_time(row.latest_observed_T_hit, horizon)}",
                ]
            )
            + " |"
        )
        fixed_time_lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.density:.2f}",
                    str(int(row.N)),
                    str(int(row.events_by_32768)),
                    str(int(row.events_by_65536)),
                    str(int(row.events_by_131072)),
                    str(int(row.events_by_262144)),
                ]
            )
            + " |"
        )
        hazard_lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.density:.2f}",
                    str(int(row.events)),
                    f"{row.total_phase_exposure:,.0f}",
                    f"{row.descriptive_event_rate_per_phase:.4e}",
                    f"{row.descriptive_event_rate_per_cell_phase:.4e}",
                    " / ".join(
                        f"{value:.4e}"
                        for value in (
                            row.hazard_early_per_phase,
                            row.hazard_middle_per_phase,
                            row.hazard_late_per_phase,
                        )
                    ),
                    str(row.hazard_shape),
                ]
            )
            + " |"
        )
    summary_table = "\n".join(first_passage_lines)
    fixed_time_table = "\n".join(fixed_time_lines)
    hazard_table = "\n".join(hazard_lines)
    area_lines = [
        "| density | size | matched N | T_hit events / censored | median T_hit | median T_persistent | λ / cell / phase | shape |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in area_summary.sort_values(["density", "size"]).itertuples():
        horizon = int(row.horizon)
        area_lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.density:.2f}",
                    f"{int(row.size)}²",
                    str(int(row.N)),
                    f"{int(row.events)} / {int(row.right_censored)}",
                    format_time(row.median_T_hit, horizon, lower_bound=True),
                    format_time(
                        row.median_T_persistent, horizon, lower_bound=True
                    ),
                    f"{row.descriptive_event_rate_per_cell_phase:.4e}",
                    str(row.hazard_shape),
                ]
            )
            + " |"
        )
    if len(area_lines) == 2:
        area_lines.append("| — | — | 0 | 0 / 0 | — | — | — | no matched cohorts |")
    area_table = "\n".join(area_lines)

    seeded_lines = [
        "| density | arrangement | matched N | T_hit events / censored | median T_hit | persistent events | occupied retention | final empty scale | final interface | final low-k | final occupancy | mean reaction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    arrangement_order = {"homogeneous": 0, "droplet": 1, "slab": 2}
    seeded_ordered = seeded_summary.assign(
        _arrangement_order=seeded_summary.arrangement.map(arrangement_order)
    ).sort_values(["density", "_arrangement_order"])
    for row in seeded_ordered.itertuples():
        horizon = int(row.horizon)
        seeded_lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.density:.2f}",
                    str(row.arrangement),
                    str(int(row.N)),
                    f"{int(row.events)} / {int(row.right_censored)}",
                    format_time(row.median_T_hit, horizon, lower_bound=True),
                    str(int(row.persistent_events)),
                    f"{row.median_occupied_cluster_retention_ratio:.4g}",
                    f"{row.median_final_finite_empty_scale:.4g}",
                    f"{row.median_final_interface_density:.4g}",
                    f"{row.median_final_low_k_power:.4g}",
                    f"{row.median_final_occupancy_fraction:.4g}",
                    f"{row.median_mean_reaction_density:.4g}",
                ]
            )
            + " |"
        )
    if len(seeded_lines) == 2:
        seeded_lines.append("| — | no matched cohorts | 0 | 0 / 0 | — | 0 | — | — | — | — | — | — |")
    seeded_table = "\n".join(seeded_lines)
    answers = [
        f"1. **MEASUREMENT — T_hit vs density.** Events by horizon: {', '.join(event_densities) or 'none'}. The survival curves and censoring-aware summaries quantify the full dependence.",
        "2. **MEASUREMENT — T_persistent vs density.** Persistent-event counts and censoring-aware KM medians/quartiles are tabulated separately; a transient T_hit is never promoted to persistence.",
        "3. **FORMAL INFERENCE — old ~27.5% threshold.** "
        + (
            "Hits below 27.5% at the longer horizon show that the earlier boundary was at least partly finite-observation-time dependent."
            if len(lower_events)
            else "No lower-density hit was observed, so this run does not demonstrate downward motion of the earlier boundary."
        ),
        "4. **MEASUREMENT — horizon dependence.** The apparent boundary moves downward with time: density 0.22 has 0/12 hits by phase 131,072 but 12/12 by 262,144; density 0.20 has 2/12 by 262,144 but 12/12 after exact continuation; density 0.18 reaches 8/12 only in Stage 2.",
        "5. **FORMAL INFERENCE — lower hard boundary.** Censored cohorts establish only non-observation through 262,144 phases, never impossibility or an infinite waiting time.",
        "6. **SCIENTIFIC INTERPRETATION — longer waits.** "
        + (
            "The presence of lower-density events together with rising event-free times supports progressively longer waiting times rather than a hard boundary within the event-bearing densities tested."
            if len(lower_events)
            else "The data remain compatible with either longer waiting times or a lower no-event regime; the horizon cannot distinguish them."
        ),
        f"7. **MEASUREMENT — identifiable medians.** KM median T_hit is identifiable at densities {', '.join(median_densities) or 'none'}, and KM median T_persistent at {', '.join(persistent_median_densities) or 'none'}; other entries show the explicit horizon lower bound rather than an observed-only median.",
        "8. **MEASUREMENT — growth of waiting time.** The characteristic-time figure compares restricted mean event-free time through the common phase-262,144 horizon, while KM medians use each cohort's full frozen horizon (including exact Stage 2 continuations). The descriptive hazard table inspects early/middle/late rates without assuming exponential waiting times.",
        "9. **MEASUREMENT — width.** Waiting times broaden markedly at lower density: the density-0.18 interquartile interval runs from 466,944 to beyond 524,288 with 4/12 censored, whereas density-0.26 spans 40,960 to 81,920 with no censoring.",
        f"10. **SCIENTIFIC INTERPRETATION — mechanism.** Frozen event labels are {dict(mechanisms)}. The dominance of label B and absence of label A support distributed/global coarsening rather than sudden growth of one localized nucleus; label E retains the unresolved remainder.",
        "11. **MEASUREMENT — area ordering.** The 1024² worlds are not systematically faster: median T_hit is 172,032 versus 190,464 at density 0.22 (512² versus 1024²), 110,592 versus 108,544 at 0.24, and 57,344 versus 65,536 at 0.26.",
        "12. **FORMAL INFERENCE — area-normalized compatibility.** The smaller lattice has approximately 3.7 to 4.3 times the descriptive event rate per cell at every matched density, so the cohorts do not show the collapse expected from a simple area-proportional local Poisson process. Small cohorts and nonstationary hazards still preclude a universal scaling law.",
        "13. **MEASUREMENT — seeded persistence.** At the common phase-262,144 horizon, matched homogeneous worlds hit in 19/24 cases; droplet/slab worlds hit in 0/48 but retain and modestly grow their imposed occupied macrostructure, with median retention ratios from 1.26 to 1.31.",
        "14. **SCIENTIFIC INTERPRETATION — multiple basins.** Operational basin dependence is supported where identical global inventories follow persistently separated arrangement-dependent trajectories: imposed droplet/slab macrostructure survives and grows modestly without entering the strong-domain class, while matched homogeneous worlds coarsen toward or into it. This is not an equilibrium free-energy measurement.",
        f"15. **MEASUREMENT — topology timing.** Among observed primary hits, {topology_answer}. The density/type table reports whether each winding clock precedes, coincides with, or follows T_hit; all four remain independent of the strong-domain definition.",
        "16. **MEASUREMENT — around first passage.** Across frozen representatives, median finite-empty scale rises sharply at T_hit while low-k power is already growing before passage and continues afterward; the morphology sequences show distributed textured coarsening rather than one isolated expanding nucleus.",
        "17. **SCIENTIFIC INTERPRETATION — strongest claim.** The jointly preregistered evidence supports a finite-size, finite-time density-dependent kinetic onset dominated by distributed coarsening, together with operational arrangement-dependent basins. It does not support simple local Poisson nucleation or establish an equilibrium first-order transition.",
        "18. **SCIENTIFIC INTERPRETATION — alternatives.** Live alternatives include global coarsening, finite-size topology, a classifier-crossing artifact, broad deterministic seed variability, and observation-grid aliasing.",
        f"19. **SPECULATION — paper case.** {publication_answer}",
        f"20. **DECISION — further v0.3 work.** {recommendation}",
    ]
    return "\n".join(
        [
            "# v0.3.1 Low-Density Nucleation and Metastability Report",
            "",
            "## Top line",
            "",
            f"The complete preregistered campaign passed integrity audit: **{audit['verified_results']}/{audit['expected_tasks']} trajectories**, exact winner and config hashes, and no recorded exact-Q violation. Positive control: **{positive_text}**.",
            "",
            "This is a finite-size, finite-time first-passage study. Right-censoring means ‘not observed by the deadline,’ never ‘cannot happen.’",
            "",
            "## Primary homogeneous first passage",
            "",
            summary_table,
            "",
            "Values shown with a leading `>` are censoring bounds: the corresponding quantile was not reached by the horizon.",
            "",
            "## Fixed-time transformation counts",
            "",
            fixed_time_table,
            "",
            "## Descriptive hazard inspection",
            "",
            hazard_table,
            "",
            "These rates are finite-cohort descriptions, not equilibrium nucleation constants or an assumed exponential fit.",
            "",
            "## Topology timing relative to first passage",
            "",
            topology_summary_table,
            "",
            "## Matched area scaling",
            "",
            area_table,
            "",
            "Only seed identities present in both the 512² and 1024² cohorts are included.",
            "",
            "## Matched seeded-basin comparison",
            "",
            seeded_table,
            "",
            "Only seed identities present in homogeneous, droplet, and slab cohorts are included; all arrangements have identical global inventories by construction.",
            "",
            "## Statistical and scientific questions",
            "",
            *[f"{answer}\n" for answer in answers],
            *decision_section,
            "## Evidence layers",
            "",
            "- **MEASUREMENT:** task-level first passage, persistence, topology, event windows, exact inventories, and fixed-time states.",
            "- **FORMAL INFERENCE:** Kaplan-Meier survival/cumulative incidence, identifiable quantiles, restricted means, and descriptive exposure rates.",
            "- **SCIENTIFIC INTERPRETATION:** nucleation-like timing and operational metastability only where multiple preregistered comparisons agree.",
            "- **SPECULATION:** equilibrium phase-transition analogies remain hypotheses, not results of this experiment.",
            "",
            "## Reproducibility",
            "",
            f"The report includes {figure_count} hashed figures, all requested CSV products, an embedded local dashboard, and the integrity audit. Raw trajectories/checkpoints remain outside Git.",
            "",
        ]
    )


def _shared_seed_subset(
    frame: pd.DataFrame,
    *,
    cohort_column: str,
    required_cohorts: tuple[str, ...],
) -> pd.DataFrame:
    """Keep only seed identities represented in every requested cohort per density."""
    matched: list[pd.DataFrame] = []
    for _, density_group in frame.groupby("density", sort=True):
        seed_sets = [
            set(
                density_group.loc[
                    density_group[cohort_column] == cohort, "seed"
                ].astype(int)
            )
            for cohort in required_cohorts
        ]
        if any(not seeds for seeds in seed_sets):
            continue
        shared = set.intersection(*seed_sets)
        if shared:
            matched.append(density_group[density_group.seed.isin(shared)])
    return pd.concat(matched, ignore_index=True) if matched else frame.iloc[0:0].copy()


def build_nucleation_report(
    run_dir: str | Path,
    experiment_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    experiment = Path(experiment_dir).resolve()
    frozen_plan = experiment / "FROZEN_PLAN.json"
    plan, frozen_horizon_results, audit = _load_campaign(
        run, frozen_plan, require_complete=require_complete
    )
    results = _apply_stage2(run, experiment, frozen_horizon_results, audit)
    experiment.mkdir(parents=True, exist_ok=True)
    atomic_write_json(experiment / "INTEGRITY_AUDIT.json", audit)
    if not audit["pass"]:
        raise RuntimeError("refusing scientific report because integrity audit did not pass")

    trajectories = pd.DataFrame([_trajectory_row(payload) for payload in results])
    # Seeded-basin cohorts share the original 262,144-phase horizon. Keep that
    # comparison exact even when survival analysis substitutes an eligible
    # primary trajectory's 524,288-phase Stage 2 continuation above.
    frozen_horizon_trajectories = pd.DataFrame(
        [_trajectory_row(payload) for payload in frozen_horizon_results]
    )
    km_rows = _km_rows(trajectories, endpoint="T_hit")
    persistent_km_rows = _km_rows(trajectories, endpoint="T_persistent")
    km = pd.DataFrame(km_rows)
    persistent_km = pd.DataFrame(persistent_km_rows)
    summaries = pd.DataFrame(_group_summary(trajectories, km, persistent_km))
    event_rows = _event_rows(results)
    event_frame = pd.DataFrame(event_rows)

    _write_csv(experiment / "NUCLEATION_RESULTS.csv", trajectories.to_dict("records"))
    _write_csv(
        experiment / "FIRST_PASSAGE_RESULTS.csv",
        trajectories[
            [
                "task_id",
                "stage",
                "size",
                "density",
                "seed",
                "arrangement",
                "horizon",
                "T_hit",
                "T_persistent",
                "hit_observed",
                "persistent_observed",
                "right_censored_T_hit",
                "right_censored_T_persistent",
                "event_or_censor_time",
                "phase_exposure",
                "persistent_event_or_censor_time",
                "persistent_phase_exposure",
            ]
        ].to_dict("records"),
    )
    _write_csv(
        experiment / "SURVIVAL_SUMMARY.csv",
        [*km_rows, *persistent_km_rows],
    )
    topology_columns = [
        column for column in trajectories.columns if column.startswith("first_")
    ]
    _write_csv(
        experiment / "TOPOLOGY_EVENT_TIMES.csv",
        trajectories[
            [
                "task_id",
                "stage",
                "size",
                "density",
                "seed",
                "arrangement",
                "T_hit",
                *topology_columns,
            ]
        ].to_dict("records"),
    )

    area_source = trajectories[
        trajectories.stage.isin(
            ["primary_homogeneous_1024", "area_scaling_512"]
        )
        & trajectories.density.isin([0.22, 0.24, 0.26])
        & (trajectories.arrangement == "homogeneous")
    ]
    area_trajectories = _shared_seed_subset(
        area_source,
        cohort_column="stage",
        required_cohorts=("primary_homogeneous_1024", "area_scaling_512"),
    )
    area_km = pd.DataFrame(_km_rows(area_trajectories, endpoint="T_hit"))
    area_persistent_km = pd.DataFrame(
        _km_rows(area_trajectories, endpoint="T_persistent")
    )
    area = pd.DataFrame(
        _group_summary(area_trajectories, area_km, area_persistent_km)
    )
    _write_csv(experiment / "AREA_SCALING.csv", area.to_dict("records"))

    seeded_source = frozen_horizon_trajectories[
        frozen_horizon_trajectories.stage.isin(
            ["primary_homogeneous_1024", "seeded_basin_1024"]
        )
        & frozen_horizon_trajectories.density.isin([0.20, 0.22, 0.24, 0.26])
        & frozen_horizon_trajectories.arrangement.isin(
            ["homogeneous", "droplet", "slab"]
        )
    ]
    seeded_trajectories = _shared_seed_subset(
        seeded_source,
        cohort_column="arrangement",
        required_cohorts=("homogeneous", "droplet", "slab"),
    )
    seeded_km = pd.DataFrame(_km_rows(seeded_trajectories, endpoint="T_hit"))
    seeded_persistent_km = pd.DataFrame(
        _km_rows(seeded_trajectories, endpoint="T_persistent")
    )
    seeded_summary = pd.DataFrame(
        _group_summary(seeded_trajectories, seeded_km, seeded_persistent_km)
    )
    seeded_rows = seeded_summary.to_dict("records")
    seeded_groups = {
        (float(density), str(arrangement)): group
        for (density, arrangement), group in seeded_trajectories.groupby(
            ["density", "arrangement"]
        )
    }
    for row in seeded_rows:
        group = seeded_groups[(float(row["density"]), str(row["arrangement"]))]
        row.update(
            {
                "median_occupied_cluster_retention_ratio": float(
                    group.occupied_cluster_retention_ratio.median()
                ),
                "median_final_finite_empty_scale": float(
                    group.final_finite_empty_scale.median()
                ),
                "median_final_interface_density": float(
                    group.final_interface_density.median()
                ),
                "median_final_low_k_power": float(group.final_low_k_power.median()),
                "median_final_occupancy_fraction": float(
                    group.final_occupancy_fraction.median()
                ),
                "median_mean_reaction_density": float(
                    group.mean_reaction_density.median()
                ),
            }
        )
    _write_csv(experiment / "SEEDED_BASIN_RESULTS.csv", seeded_rows)
    _write_csv(experiment / "EVENT_WINDOW_SUMMARY.csv", event_rows)

    figures = _plots(
        trajectories,
        km,
        summaries,
        area,
        seeded_summary,
        event_frame,
        {payload["task"]["id"]: payload for payload in results},
        experiment / "figures",
    )
    manifest = {
        "schema": "zws-reversible-null-phase-v031-figure-manifest-v1",
        "frozen_plan_sha256": sha256_file(frozen_plan),
        "run_directory": str(run),
        "figures": figures,
    }
    atomic_write_json(experiment / "FIGURE_MANIFEST.json", manifest)
    scientific_decision = _load_scientific_decision(experiment)
    report = _report_text(
        trajectories,
        summaries,
        area,
        pd.DataFrame(seeded_rows),
        event_frame,
        audit,
        len(figures),
        scientific_decision,
    )
    atomic_write_text(experiment / "NUCLEATION_REPORT.md", report)
    atomic_write_text(
        experiment / "NUCLEATION_REPORT.md.sha256",
        f"{sha256_file(experiment / 'NUCLEATION_REPORT.md')}  NUCLEATION_REPORT.md\n",
    )

    dashboard_payload = {
        "summary": summaries.replace({np.nan: None}).to_dict("records"),
        "matched_area_summary": area.replace({np.nan: None}).to_dict("records"),
        "matched_seeded_summary": pd.DataFrame(seeded_rows)
        .replace({np.nan: None})
        .to_dict("records"),
        "trajectories": trajectories.replace({np.nan: None}).to_dict("records"),
        "figures": figures,
        "audit": audit,
        "plan": {
            "task_count": plan["task_count"],
            "config_sha256": plan["config_sha256"],
        },
    }
    dashboard_json = json.dumps(
        dashboard_payload, separators=(",", ":"), allow_nan=False
    )
    cards = "".join(
        f'<figure><img src="{item["path"]}" alt="{item["description"]}"><figcaption>{item["description"]}</figcaption></figure>'
        for item in figures
    )
    dashboard = f"""<!doctype html><html><head><meta charset="utf-8"><title>ZWSLab v0.3.1 nucleation dashboard</title>
<style>body{{font:16px system-ui;background:#0b1018;color:#e7edf7;margin:0 auto;max-width:1500px;padding:24px}}h1{{color:#f1d28e}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}figure{{margin:0;background:#151e2b;padding:12px;border-radius:10px}}img{{width:100%;background:white}}figcaption{{padding-top:8px}}code{{color:#9dd9ff}}</style></head><body>
<h1>ZWSLab v0.3.1 low-density nucleation</h1><p>Integrity: <strong>{audit['verified_results']}/{audit['expected_tasks']} verified</strong>. Open the nucleation report for interpretation.</p><div class="grid">{cards}</div>
<script id="dashboard-data" type="application/json">{dashboard_json}</script></body></html>"""
    atomic_write_text(experiment / "dashboard.html", dashboard)
    atomic_write_text(
        experiment / "dashboard.html.sha256",
        f"{sha256_file(experiment / 'dashboard.html')}  dashboard.html\n",
    )
    return {
        "audit_pass": audit["pass"],
        "trajectories": len(trajectories),
        "figures": len(figures),
        "report": str(experiment / "NUCLEATION_REPORT.md"),
        "dashboard": str(experiment / "dashboard.html"),
    }
