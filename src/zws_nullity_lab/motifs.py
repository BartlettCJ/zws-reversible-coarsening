from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchRepetition:
    repeated_fraction: torch.Tensor
    distinct_ratio: torch.Tensor
    max_multiplicity: torch.Tensor


def toroidal_patch_codes(state: torch.Tensor, patch: int = 4) -> torch.Tensor:
    """Encode every toroidally rooted patch as an exact base-4 integer."""
    if state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3:
        raise ValueError("state must be [H,W] or [B,H,W]")
    if patch < 1 or patch > 5:
        raise ValueError("patch must be 1..5 so codes fit safely in int64")
    padded = F.pad(state.long(), (0, patch - 1, 0, patch - 1), mode="circular")
    windows = padded.unfold(1, patch, 1).unfold(2, patch, 1)
    powers = (4 ** torch.arange(patch * patch, device=state.device, dtype=torch.int64)).reshape(
        patch, patch
    )
    return (windows * powers).sum(dim=(-1, -2))


def patch_repetition(
    state: torch.Tensor, patch: int = 4, root_stride: int = 1
) -> PatchRepetition:
    """A cautious replication *proxy*, not a detector of causal reproduction.

    It reports how often nonempty local patches occur more than once at a
    frame. Repetition can arise from symmetry or periodic texture, so the
    experiment report never labels this alone as reproduction. Batch offsets
    make one vectorized ``unique`` operation sufficient on CPU or GPU.
    """
    codes = toroidal_patch_codes(state, patch=patch)
    if root_stride > 1:
        codes = codes[:, ::root_stride, ::root_stride]
    batch = codes.shape[0]
    flat = codes.reshape(batch, -1)
    code_space = 4 ** (patch * patch)
    offsets = torch.arange(batch, device=codes.device, dtype=torch.int64)[:, None] * code_space
    tagged = (flat + offsets).flatten()
    unique, inverse, counts = torch.unique(
        tagged, sorted=False, return_inverse=True, return_counts=True
    )
    instance_counts = counts[inverse].reshape_as(flat)
    nonempty = flat != 0
    occupied_count = nonempty.sum(dim=1).clamp_min(1)
    repeated = ((instance_counts >= 2) & nonempty).sum(dim=1).float() / occupied_count

    unique_batch = torch.div(unique, code_space, rounding_mode="floor")
    unique_code = unique % code_space
    unique_nonempty = unique_code != 0
    distinct_count = torch.zeros(batch, device=codes.device, dtype=torch.float32)
    distinct_count.scatter_add_(
        0, unique_batch[unique_nonempty], torch.ones_like(unique_batch[unique_nonempty], dtype=torch.float32)
    )
    distinct = distinct_count / occupied_count
    max_mult = torch.zeros(batch, device=codes.device, dtype=torch.int64)
    max_mult.scatter_reduce_(0, unique_batch, counts, reduce="amax", include_self=True)
    return PatchRepetition(repeated, distinct, max_mult.float())


def component_sizes(state: torch.Tensor, occupied_states: tuple[int, ...] = (1, 2, 3)) -> list[int]:
    """4-neighbour toroidal component sizes for one CPU frame."""
    if state.ndim != 2:
        raise ValueError("component_sizes expects one [H,W] frame")
    grid = state.detach().cpu()
    height, width = grid.shape
    occupied = torch.zeros_like(grid, dtype=torch.bool)
    for value in occupied_states:
        occupied |= grid == value
    seen = torch.zeros_like(occupied)
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not bool(occupied[y, x]) or bool(seen[y, x]):
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in (
                    ((cy - 1) % height, cx),
                    ((cy + 1) % height, cx),
                    (cy, (cx - 1) % width),
                    (cy, (cx + 1) % width),
                ):
                    if bool(occupied[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sorted(sizes, reverse=True)
