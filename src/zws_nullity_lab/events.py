from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from .atomic import atomic_write_json
from .margolus import _to_blocks, encode_blocks
from .rules import ReversibleRule
from .visualize import state_rgb


@dataclass(frozen=True)
class EventCandidate:
    event_id: int
    frame: int
    tick: int
    score: float
    reasons: tuple[str, ...]
    activity: float
    reaction_fraction: float
    entropy: float
    boundary_density: float
    recurrence: float
    occupied_fraction: float
    bound_fraction: float
    crop: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["crop"] = list(self.crop)
        return value


def _entropy(frame: np.ndarray) -> float:
    counts = np.bincount(frame.reshape(-1), minlength=4).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    valid = p > 0
    return float(-(p[valid] * np.log(p[valid])).sum() / math.log(4.0))


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-9, 1.4826 * mad)
    return (values - median) / scale


def _change_crop(before: np.ndarray, after: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    changed = np.argwhere(before != after)
    height, width = after.shape
    if changed.size == 0:
        occupied = np.argwhere(after != 0)
        changed = occupied if occupied.size else np.array([[height // 2, width // 2]])
    y0 = max(0, int(changed[:, 0].min()) - padding)
    y1 = min(height, int(changed[:, 0].max()) + padding + 1)
    x0 = max(0, int(changed[:, 1].min()) - padding)
    x1 = min(width, int(changed[:, 1].max()) + padding + 1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def scan_interesting_events(
    trajectory: torch.Tensor,
    rule: ReversibleRule,
    *,
    max_events: int = 20,
    min_separation: int = 12,
    stride: int = 1,
    padding: int = 10,
) -> tuple[list[EventCandidate], dict[str, list[float]]]:
    """Rank frames for human review without claiming they are organisms.

    The ranking combines local reaction density, activity/entropy transitions,
    spatial boundary density and short-lag recurrence. It is explicitly a
    bookmark generator, not a scientific discovery metric.
    """
    if trajectory.ndim == 4:
        states = trajectory[:, 0]
    elif trajectory.ndim == 3:
        states = trajectory
    else:
        raise ValueError("trajectory must be [T,H,W] or [T,B,H,W]")
    array = states.detach().cpu().to(torch.uint8).numpy()
    frames, height, width = array.shape
    if frames < 2:
        return [], {}
    max_events = max(0, int(max_events))
    activity = np.zeros(frames, dtype=np.float64)
    reaction = np.zeros(frames, dtype=np.float64)
    entropy = np.zeros(frames, dtype=np.float64)
    boundary = np.zeros(frames, dtype=np.float64)
    recurrence = np.zeros(frames, dtype=np.float64)
    occupied = np.zeros(frames, dtype=np.float64)
    bound = np.zeros(frames, dtype=np.float64)
    reaction_mask = rule.reaction_mask().cpu()
    state_tensor = states.detach().cpu()
    for t in range(frames):
        frame = array[t]
        entropy[t] = _entropy(frame)
        occupied[t] = float(np.mean(frame != 0))
        bound[t] = float(np.mean(frame == 3))
        boundary[t] = 0.5 * (
            float(np.mean(frame != np.roll(frame, -1, axis=0)))
            + float(np.mean(frame != np.roll(frame, -1, axis=1)))
        )
        if t > 0:
            activity[t] = float(np.mean(frame != array[t - 1]))
            blocks, _ = _to_blocks(state_tensor[t - 1].unsqueeze(0), (t - 1) % 2)
            codes = encode_blocks(blocks).reshape(-1).long()
            reaction[t] = float(reaction_mask[codes].float().mean())
            matches: list[float] = []
            for lag in (2, 4, 8, 16, 32):
                if t >= lag:
                    matches.append(float(np.mean(frame == array[t - lag])))
            recurrence[t] = max(matches, default=0.0)

    delta_entropy = np.abs(np.diff(entropy, prepend=entropy[0]))
    delta_boundary = np.abs(np.diff(boundary, prepend=boundary[0]))
    features = {
        "activity": activity,
        "reaction": reaction,
        "entropy_transition": delta_entropy,
        "boundary_transition": delta_boundary,
        "recurrence": recurrence,
    }
    z = {name: np.clip(_robust_z(values), -3.0, 8.0) for name, values in features.items()}
    score = (
        0.28 * np.maximum(z["activity"], 0.0)
        + 0.25 * np.maximum(z["reaction"], 0.0)
        + 0.18 * np.maximum(z["entropy_transition"], 0.0)
        + 0.14 * np.maximum(z["boundary_transition"], 0.0)
        + 0.15 * np.maximum(z["recurrence"], 0.0)
    )
    # Exclude startup noise and respect requested scan stride.
    eligible = [t for t in range(max(2, stride), frames, max(1, stride))]
    order = sorted(eligible, key=lambda t: (float(score[t]), t), reverse=True)
    selected: list[int] = []
    if max_events == 0:
        order = []
    for t in order:
        if float(score[t]) <= 0.0:
            continue
        if all(abs(t - previous) >= min_separation for previous in selected):
            selected.append(t)
        if len(selected) >= max_events:
            break
    selected.sort()
    events: list[EventCandidate] = []
    for event_id, t in enumerate(selected):
        contributions = sorted(
            ((float(z[name][t]), name) for name in z), reverse=True
        )
        labels = {
            "activity": "activity spike",
            "reaction": "local reaction-density spike",
            "entropy_transition": "rapid entropy transition",
            "boundary_transition": "rapid spatial-boundary transition",
            "recurrence": "short-lag recurrence pulse",
        }
        reasons = tuple(labels[name] for value, name in contributions[:3] if value > 0.75)
        if not reasons:
            reasons = ("combined multi-metric anomaly",)
        crop = _change_crop(array[t - 1], array[t], padding)
        events.append(
            EventCandidate(
                event_id=event_id,
                frame=t,
                tick=t,
                score=float(score[t]),
                reasons=reasons,
                activity=float(activity[t]),
                reaction_fraction=float(reaction[t]),
                entropy=float(entropy[t]),
                boundary_density=float(boundary[t]),
                recurrence=float(recurrence[t]),
                occupied_fraction=float(occupied[t]),
                bound_fraction=float(bound[t]),
                crop=crop,
            )
        )
    series = {
        "activity": activity.tolist(),
        "reaction_fraction": reaction.tolist(),
        "entropy": entropy.tolist(),
        "boundary_density": boundary.tolist(),
        "recurrence": recurrence.tolist(),
        "occupied_fraction": occupied.tolist(),
        "bound_fraction": bound.tolist(),
        "review_score": score.tolist(),
    }
    return events, series


def save_event_captures(
    trajectory: torch.Tensor,
    events: list[EventCandidate],
    directory: str | Path,
    *,
    scale: int = 5,
    contact_sheet_columns: int = 4,
) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    states = trajectory[:, 0] if trajectory.ndim == 4 else trajectory
    cards: list[Image.Image] = []
    paths: list[dict[str, Any]] = []
    for event in events:
        frame = states[event.frame].detach().cpu().to(torch.uint8)
        stem = f"event_{event.event_id:03d}_t{event.tick:06d}"
        png_path = root / f"{stem}.png"
        crop_path = root / f"{stem}_crop.png"
        state_path = root / f"{stem}_state.npy"
        metadata_path = root / f"{stem}.json"
        full_rgb = state_rgb(frame, scale=scale)
        Image.fromarray(full_rgb, mode="RGB").save(png_path)
        x, y, width, height = event.crop
        crop = frame[y : y + height, x : x + width]
        Image.fromarray(state_rgb(crop, scale=max(scale, 8)), mode="RGB").save(crop_path)
        np.save(state_path, frame.numpy(), allow_pickle=False)
        atomic_write_json(metadata_path, event.to_dict())
        card = Image.fromarray(full_rgb, mode="RGB")
        draw = ImageDraw.Draw(card)
        label = f"t={event.tick} score={event.score:.2f}"
        draw.rectangle((0, 0, min(card.width, 260), 22), fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255))
        cards.append(card)
        paths.append(
            {
                "event_id": event.event_id,
                "frame": event.frame,
                "png": str(png_path),
                "crop_png": str(crop_path),
                "state_npy": str(state_path),
                "metadata_json": str(metadata_path),
            }
        )
    contact_path: Path | None = None
    if cards:
        columns = max(1, int(contact_sheet_columns))
        rows = math.ceil(len(cards) / columns)
        cell_width = max(card.width for card in cards)
        cell_height = max(card.height for card in cards)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (12, 14, 18))
        for index, card in enumerate(cards):
            sheet.paste(card, ((index % columns) * cell_width, (index // columns) * cell_height))
        contact_path = root / "event_contact_sheet.png"
        sheet.save(contact_path)
    atomic_write_json(root / "events.json", [event.to_dict() for event in events])
    return {
        "events_json": str(root / "events.json"),
        "contact_sheet": str(contact_path) if contact_path else None,
        "captures": paths,
    }
