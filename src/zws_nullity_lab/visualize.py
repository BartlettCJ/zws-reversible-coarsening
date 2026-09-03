from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import matplotlib

# ZWSLab writes plots to files and is routinely run unattended. Selecting the
# noninteractive backend before importing pyplot avoids a Tk dependency on
# headless or minimally provisioned Windows Python installations.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


# Empty, positive, negative, bound. Fixed semantic palette for state images.
_PALETTE = np.array(
    [
        [15, 18, 24],
        [239, 82, 80],
        [72, 149, 239],
        [236, 200, 95],
    ],
    dtype=np.uint8,
)


def state_rgb(state: torch.Tensor, scale: int = 4) -> np.ndarray:
    frame = state.detach().cpu().to(torch.int64).numpy()
    rgb = _PALETTE[frame]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def save_state_png(state: torch.Tensor, path: str | Path, scale: int = 5) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(state_rgb(state, scale=scale), mode="RGB").save(destination)
    return destination


def save_trajectory_gif(
    trajectory: torch.Tensor,
    path: str | Path,
    *,
    stride: int = 1,
    max_frames: int = 180,
    scale: int = 4,
    fps: int = 20,
) -> Path:
    if trajectory.ndim == 4:
        trajectory = trajectory[:, 0]
    indices = list(range(0, trajectory.shape[0], max(1, stride)))[:max_frames]
    frames = [state_rgb(trajectory[index], scale=scale) for index in indices]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(destination, frames, duration=1000 / fps, loop=0)
    return destination


def plot_search_history(records: Iterable[dict[str, float]], path: str | Path) -> Path:
    rows = list(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [int(row["generation"]) for row in rows]
    ax.plot(x, [float(row["best_quality"]) for row in rows], label="best quality")
    ax.plot(x, [float(row["mean_quality"]) for row in rows], label="mean quality")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Declared search heuristic")
    ax.set_title("Rule-search history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def plot_control_metrics(
    names: list[str], metrics: dict[str, list[float]], path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = ["activity", "reaction", "spatial_mi", "temporal_mi", "perturbation_distance"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, key in enumerate(keys):
        ax.bar(x + (index - 2) * width, metrics[key], width, label=key.replace("_", " "))
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("Normalized metric")
    ax.set_title("Selected rule and controls")
    ax.legend(ncols=3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
