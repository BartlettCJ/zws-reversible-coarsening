from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def kaplan_meier(
    times: Iterable[int | float],
    observed: Iterable[bool],
    *,
    horizon: int,
) -> list[dict[str, Any]]:
    durations = np.asarray(tuple(times), dtype=np.float64)
    events = np.asarray(tuple(observed), dtype=bool)
    if durations.shape != events.shape or durations.ndim != 1:
        raise ValueError("times and observed must be equal one-dimensional arrays")
    if np.any(durations < 0) or np.any(durations > horizon):
        raise ValueError("durations must lie within the analysis horizon")
    rows: list[dict[str, Any]] = [
        {"time": 0.0, "at_risk": int(durations.size), "events": 0, "censored": 0, "survival": 1.0}
    ]
    survival = 1.0
    for time in sorted(set(float(v) for v in durations)):
        at_risk = int(np.sum(durations >= time))
        event_count = int(np.sum((durations == time) & events))
        censored_count = int(np.sum((durations == time) & ~events))
        if event_count:
            survival *= 1.0 - event_count / at_risk
        rows.append(
            {
                "time": time,
                "at_risk": at_risk,
                "events": event_count,
                "censored": censored_count,
                "survival": float(survival),
            }
        )
    if not rows or rows[-1]["time"] < horizon:
        rows.append({"time": float(horizon), "at_risk": 0, "events": 0, "censored": 0, "survival": float(survival)})
    return rows


def restricted_mean(km_rows: list[dict[str, Any]], horizon: int) -> float:
    if not km_rows or float(km_rows[0]["time"]) != 0:
        raise ValueError("Kaplan-Meier rows must start at zero")
    area = 0.0
    previous_time = 0.0
    previous_survival = 1.0
    for row in km_rows[1:]:
        time = min(float(row["time"]), float(horizon))
        area += max(0.0, time - previous_time) * previous_survival
        previous_time = time
        previous_survival = float(row["survival"])
        if previous_time >= horizon:
            break
    if previous_time < horizon:
        area += (horizon - previous_time) * previous_survival
    return float(area)


def rmst(
    times: Iterable[int | float],
    observed: Iterable[bool],
    *,
    horizon: int,
) -> float:
    return restricted_mean(kaplan_meier(times, observed, horizon=horizon), horizon)


def paired_rmst_delay(records: list[dict[str, Any]], *, horizon: int) -> float:
    if not records:
        raise ValueError("paired RMST requires records")
    untouched_times = [min(float(row["untouched_time"]), horizon) for row in records]
    untouched_events = [bool(row["untouched_observed"]) and float(row["untouched_time"]) <= horizon for row in records]
    altered_times = [min(float(row["altered_time"]), horizon) for row in records]
    altered_events = [bool(row["altered_observed"]) and float(row["altered_time"]) <= horizon for row in records]
    return rmst(altered_times, altered_events, horizon=horizon) - rmst(
        untouched_times, untouched_events, horizon=horizon
    )


def clustered_bootstrap_rmst_delay(
    records: list[dict[str, Any]],
    *,
    horizon: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        clusters.setdefault(str(row["cluster"]), []).append(row)
    keys = sorted(clusters)
    if not keys:
        raise ValueError("bootstrap requires at least one source cluster")
    point = paired_rmst_delay(records, horizon=horizon)
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = generator.integers(0, len(keys), size=len(keys))
        sampled_rows: list[dict[str, Any]] = []
        for draw, index in enumerate(sample):
            for row in clusters[keys[int(index)]]:
                copy = dict(row)
                copy["cluster"] = f"bootstrap_{draw}"
                sampled_rows.append(copy)
        values[replicate] = paired_rmst_delay(sampled_rows, horizon=horizon)
    return {
        "point_estimate": float(point),
        "replicates": int(replicates),
        "seed": int(seed),
        "ci90": [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))],
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_mean": float(values.mean()),
    }


def fixed_time_fraction(
    times: Iterable[int | float],
    observed: Iterable[bool],
    fixed_time: int,
) -> float:
    durations = np.asarray(tuple(times), dtype=np.float64)
    events = np.asarray(tuple(observed), dtype=bool)
    if durations.shape != events.shape or durations.size == 0:
        raise ValueError("fixed-time fraction requires equal nonempty arrays")
    return float(np.mean(events & (durations <= fixed_time)))
