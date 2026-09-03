from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .atomic import atomic_write_bytes, atomic_write_text, sha256_file
from .initial import state_sha256
from .metrics import SnapshotMetrics, StrongDomainCriterion


TOPOLOGY_FIELDS = (
    "occupied_wrap_x",
    "occupied_wrap_y",
    "empty_wrap_x",
    "empty_wrap_y",
)


def first_passage_from_flags(
    phases: list[int],
    flags: list[bool],
    *,
    horizon: int,
    persistence_followups: int,
) -> dict[str, Any]:
    if len(phases) != len(flags):
        raise ValueError("phases and flags must have equal length")
    if phases != sorted(phases) or len(phases) != len(set(phases)):
        raise ValueError("phases must be strictly increasing")
    hit = next((phase for phase, flag in zip(phases, flags, strict=True) if flag), None)
    persistent = None
    width = persistence_followups + 1
    for start in range(0, len(flags) - width + 1):
        if all(flags[start : start + width]):
            persistent = phases[start]
            break
    hit_index = phases.index(hit) if hit is not None else None
    followup_complete = bool(
        hit_index is not None and hit_index + persistence_followups < len(phases)
    )
    return {
        "T_hit": hit,
        "hit_observed": hit is not None,
        "T_persistent": persistent,
        "persistent_observed": persistent is not None,
        "censor_phase": int(horizon),
        "right_censored_T_hit": hit is None,
        "right_censored_T_persistent": persistent is None,
        "hit_persistence_followup_complete": followup_complete,
    }


def topology_first_times(series: list[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in TOPOLOGY_FIELDS:
        result[field] = next(
            (int(row["phase"]) for row in series if bool(row[field])),
            None,
        )
    return result


def _as_numpy(state: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        return np.asarray(state.detach().cpu().numpy(), dtype=np.uint8)
    return np.asarray(state, dtype=np.uint8)


def _event_crop_geometry(
    before: torch.Tensor | np.ndarray,
    hit: torch.Tensor | np.ndarray,
    fraction: float,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    first = _as_numpy(before)
    last = _as_numpy(hit)
    if first.shape != last.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("event crop requires equal square states")
    size = first.shape[0]
    crop = max(1, min(size, int(round(size * fraction))))
    changed = ((first != 0) != (last != 0)).astype(np.int64)
    tiled = np.tile(changed, (2, 2))
    integral = np.pad(tiled, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    origins = (
        integral[crop : size + crop, crop : size + crop]
        - integral[:size, crop : size + crop]
        - integral[crop : size + crop, :size]
        + integral[:size, :size]
    )
    half = crop // 2
    center_scores = np.roll(origins, shift=(half, half), axis=(0, 1))
    center = tuple(int(x) for x in np.unravel_index(int(np.argmax(center_scores)), center_scores.shape))
    origin = ((center[0] - half) % size, (center[1] - half) % size)
    return center, origin, crop


def periodic_crop(
    state: torch.Tensor | np.ndarray,
    origin: tuple[int, int],
    crop_size: int,
) -> np.ndarray:
    arr = _as_numpy(state)
    ys = (np.arange(crop_size) + origin[0]) % arr.shape[0]
    xs = (np.arange(crop_size) + origin[1]) % arr.shape[1]
    return np.ascontiguousarray(arr[np.ix_(ys, xs)], dtype=np.uint8)


def new_tracker(
    initial_state: torch.Tensor,
    initial_metric: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prefix_phases": [int(initial_metric["phase"])],
        "prefix_flags": [False],
        "pre_hit_ring": [
            {"phase": int(initial_metric["phase"]), "state": initial_state.detach().cpu()}
        ],
        "event_window_phases": [],
        "event_window_states": [],
        "event_crop_center": None,
        "event_crop_origin": None,
        "event_crop_size": None,
        "event_change_total": None,
        "event_change_in_crop": None,
        "event_change_localization_fraction": None,
    }


def update_tracker(
    tracker: dict[str, Any],
    *,
    state: torch.Tensor,
    series: list[dict[str, Any]],
    criterion: StrongDomainCriterion,
    settings: dict[str, Any],
) -> None:
    phase = int(series[-1]["phase"])
    metrics = [SnapshotMetrics(**row) for row in series]
    flag = bool(criterion.classify(metrics)["strong_domain"]) if len(metrics) >= 2 else False
    tracker["prefix_phases"].append(phase)
    tracker["prefix_flags"].append(flag)
    passage = first_passage_from_flags(
        tracker["prefix_phases"],
        tracker["prefix_flags"],
        horizon=int(series[-1]["phase"]),
        persistence_followups=int(settings["persistence_followups"]),
    )
    t_hit = passage["T_hit"]
    before = int(settings["event_window_before"])
    after = int(settings["event_window_after"])

    if tracker["event_crop_origin"] is None:
        ring = tracker["pre_hit_ring"]
        ring.append({"phase": phase, "state": state.detach().cpu()})
        tracker["pre_hit_ring"] = [row for row in ring if phase - int(row["phase"]) <= before]
        if t_hit is not None:
            ring = tracker["pre_hit_ring"]
            center, origin, crop = _event_crop_geometry(
                ring[0]["state"], ring[-1]["state"], float(settings["event_crop_fraction"])
            )
            tracker["event_crop_center"] = list(center)
            tracker["event_crop_origin"] = list(origin)
            tracker["event_crop_size"] = crop
            before_occupied = _as_numpy(ring[0]["state"]) != 0
            hit_occupied = _as_numpy(ring[-1]["state"]) != 0
            changed = before_occupied != hit_occupied
            changed_crop = periodic_crop(changed.astype(np.uint8), origin, crop)
            total_changed = int(changed.sum())
            crop_changed = int(changed_crop.sum())
            tracker["event_change_total"] = total_changed
            tracker["event_change_in_crop"] = crop_changed
            tracker["event_change_localization_fraction"] = (
                float(crop_changed / total_changed) if total_changed else 0.0
            )
            tracker["event_window_phases"] = [int(row["phase"]) for row in ring]
            tracker["event_window_states"] = [
                periodic_crop(row["state"], origin, crop) for row in ring
            ]
            tracker["pre_hit_ring"] = []
    elif t_hit is not None and phase > int(t_hit) and phase <= int(t_hit) + after:
        origin = tuple(int(x) for x in tracker["event_crop_origin"])
        crop = int(tracker["event_crop_size"])
        tracker["event_window_phases"].append(phase)
        tracker["event_window_states"].append(periodic_crop(state, origin, crop))


def finalize_time_to_event(
    tracker: dict[str, Any],
    *,
    series: list[dict[str, Any]],
    settings: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    passage = first_passage_from_flags(
        [int(x) for x in tracker["prefix_phases"]],
        [bool(x) for x in tracker["prefix_flags"]],
        horizon=horizon,
        persistence_followups=int(settings["persistence_followups"]),
    )
    fixed = {
        str(int(phase)): (
            bool(tracker["prefix_flags"][tracker["prefix_phases"].index(int(phase))])
            if int(phase) in tracker["prefix_phases"]
            else None
        )
        for phase in settings["fixed_readout_phases"]
    }
    passage.update(
        {
            "observation_every": int(settings["observation_every"]),
            "prefix_phases": [int(x) for x in tracker["prefix_phases"]],
            "strong_domain_prefix_flags": [bool(x) for x in tracker["prefix_flags"]],
            "fixed_readout_strong_domain": fixed,
            "topology_first_times": topology_first_times(series),
        }
    )
    return passage


def save_event_window(path: Path, tracker: dict[str, Any]) -> dict[str, Any] | None:
    if not tracker["event_window_states"]:
        return None
    states = np.stack(tracker["event_window_states"], axis=0).astype(np.uint8, copy=False)
    phases = np.asarray(tracker["event_window_phases"], dtype=np.int64)
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        phases=phases,
        states=states,
        crop_center=np.asarray(tracker["event_crop_center"], dtype=np.int64),
        crop_origin=np.asarray(tracker["event_crop_origin"], dtype=np.int64),
        crop_size=np.asarray([tracker["event_crop_size"]], dtype=np.int64),
    )
    atomic_write_bytes(path, buffer.getvalue())
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return {
        "path": path.name,
        "sha256": digest,
        "phases": phases.tolist(),
        "crop_center": [int(x) for x in tracker["event_crop_center"]],
        "crop_origin": [int(x) for x in tracker["event_crop_origin"]],
        "crop_size": int(tracker["event_crop_size"]),
        "change_total": int(tracker["event_change_total"]),
        "change_in_crop": int(tracker["event_change_in_crop"]),
        "change_localization_fraction": float(tracker["event_change_localization_fraction"]),
        "state_sha256": [state_sha256(state) for state in states],
    }
