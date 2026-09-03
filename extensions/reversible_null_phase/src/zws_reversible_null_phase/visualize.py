from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Empty charcoal, positive warm, negative cool, bound pale neutral.
PALETTE = np.asarray(
    [
        [8, 11, 16],
        [242, 108, 76],
        [67, 143, 255],
        [224, 214, 172],
    ],
    dtype=np.uint8,
)


def state_rgb(state: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy()
    else:
        arr = np.asarray(state)
    if arr.ndim != 2:
        raise ValueError("state must be 2D")
    return PALETTE[arr.astype(np.int64)]


def save_state_png(
    state: torch.Tensor | np.ndarray,
    path: str | Path,
    *,
    max_pixels: int = 2048,
) -> None:
    rgb = state_rgb(state)
    image = Image.fromarray(rgb, mode="RGB")
    if max(image.size) > max_pixels:
        scale = max_pixels / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            resample=Image.Resampling.NEAREST,
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def save_state_npz(state: torch.Tensor | np.ndarray, path: str | Path) -> None:
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy()
    else:
        arr = np.asarray(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, state=np.asarray(arr, dtype=np.uint8))
