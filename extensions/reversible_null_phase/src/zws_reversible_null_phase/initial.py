from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import numpy as np
import torch

from .alphabet import DEFAULT_ALPHABET


Arrangement = Literal["homogeneous", "slab", "droplet", "localized_patch"]


@dataclass(frozen=True)
class Inventory:
    empty: int
    positive: int
    negative: int
    bound: int

    @property
    def total(self) -> int:
        return self.empty + self.positive + self.negative + self.bound

    @property
    def occupied(self) -> int:
        return self.positive + self.negative + self.bound

    @property
    def charge(self) -> int:
        return self.positive - self.negative

    @property
    def occupancy_fraction(self) -> float:
        return self.occupied / self.total

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.empty, self.positive, self.negative, self.bound)

    def assert_valid(self) -> None:
        if min(self.as_tuple()) < 0:
            raise ValueError("inventory counts must be nonnegative")
        if self.charge != 0:
            raise ValueError("inventory must have exact Q=0")


def inventory_for_density(height: int, width: int, density: float) -> Inventory:
    """Use the frozen 3/8:+, 3/8:-, 2/8:bound rounding rule.

        u = floor(density*N/8 + 0.5)
        counts = (N-8u, 3u, 3u, 2u)
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must lie in [0,1]")
    n = height * width
    u = int(math.floor(density * n / 8.0 + 0.5))
    inventory = Inventory(n - 8 * u, 3 * u, 3 * u, 2 * u)
    inventory.assert_valid()
    return inventory


def inventory_values(inventory: Inventory) -> np.ndarray:
    inventory.assert_valid()
    return np.concatenate(
        [
            np.zeros(inventory.empty, dtype=np.uint8),
            np.ones(inventory.positive, dtype=np.uint8),
            np.full(inventory.negative, 2, dtype=np.uint8),
            np.full(inventory.bound, 3, dtype=np.uint8),
        ]
    )


def state_sha256(state: torch.Tensor | np.ndarray) -> str:
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy()
    else:
        arr = np.asarray(state)
    contiguous = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
    header = f"{contiguous.shape}|uint8|".encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _random_region_mask(
    height: int,
    width: int,
    inventory: Inventory,
    arrangement: Arrangement,
    rng: np.random.Generator,
    concentration: float,
) -> np.ndarray:
    if arrangement == "homogeneous":
        return np.ones((height, width), dtype=bool)
    if not 0.0 < concentration <= 1.0:
        raise ValueError("concentration must be in (0,1]")
    target_area = min(height * width, max(inventory.occupied, math.ceil(inventory.occupied / concentration)))
    cy = int(rng.integers(0, height))
    cx = int(rng.integers(0, width))
    yy, xx = np.ogrid[:height, :width]

    if arrangement in {"slab", "localized_patch"}:
        # A slab makes hysteresis/nucleation tests easy to interpret.  The
        # localized patch is a square-like region retaining the discovery motif.
        if arrangement == "slab":
            slab_width = max(1, min(width, math.ceil(target_area / height)))
            xdist = (np.arange(width) - cx) % width
            mask = xdist < slab_width
            return np.broadcast_to(mask[None, :], (height, width)).copy()
        side = max(1, min(min(height, width), math.ceil(math.sqrt(target_area))))
        ydist = (np.arange(height) - cy) % height
        xdist = (np.arange(width) - cx) % width
        return (ydist[:, None] < side) & (xdist[None, :] < side)

    if arrangement == "droplet":
        dy = np.minimum((np.arange(height) - cy) % height, (cy - np.arange(height)) % height)
        dx = np.minimum((np.arange(width) - cx) % width, (cx - np.arange(width)) % width)
        radius = math.sqrt(target_area / math.pi)
        mask = dy[:, None] ** 2 + dx[None, :] ** 2 <= radius**2
        # Ensure enough sites under integer geometry.
        while int(mask.sum()) < inventory.occupied:
            radius += 0.5
            mask = dy[:, None] ** 2 + dx[None, :] ** 2 <= radius**2
        return mask

    raise ValueError(f"unknown arrangement {arrangement}")


def make_state(
    height: int,
    width: int,
    inventory: Inventory,
    *,
    seed: int,
    arrangement: Arrangement = "homogeneous",
    concentration: float = 0.65,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create an exact-inventory state whose only controlled change is arrangement."""
    if inventory.total != height * width:
        raise ValueError("inventory total does not match lattice size")
    inventory.assert_valid()
    rng = np.random.default_rng(int(seed))
    state = np.zeros((height, width), dtype=np.uint8)

    if arrangement == "homogeneous":
        values = inventory_values(inventory)
        rng.shuffle(values)
        state[:] = values.reshape(height, width)
    else:
        region = _random_region_mask(height, width, inventory, arrangement, rng, concentration)
        region_indices = np.flatnonzero(region.reshape(-1))
        if region_indices.size < inventory.occupied:
            raise RuntimeError("selected region is too small for occupied inventory")
        chosen = rng.choice(region_indices, size=inventory.occupied, replace=False)
        occupied_values = np.concatenate(
            [
                np.ones(inventory.positive, dtype=np.uint8),
                np.full(inventory.negative, 2, dtype=np.uint8),
                np.full(inventory.bound, 3, dtype=np.uint8),
            ]
        )
        rng.shuffle(occupied_values)
        flat = state.reshape(-1)
        flat[chosen] = occupied_values

    tensor = torch.from_numpy(state).to(device=device)
    counts = DEFAULT_ALPHABET.state_counts(tensor).cpu().tolist()
    if tuple(int(x) for x in counts) != inventory.as_tuple():
        raise RuntimeError(f"constructed inventory mismatch: {counts} != {inventory.as_tuple()}")
    if int(DEFAULT_ALPHABET.total_charge(tensor)) != 0:
        raise RuntimeError("constructed state does not have exact Q=0")
    return tensor


def matched_arrangements(
    height: int,
    width: int,
    density: float,
    *,
    seed: int,
    arrangements: tuple[Arrangement, ...] = ("homogeneous", "slab", "droplet"),
    concentration: float = 0.65,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    inventory = inventory_for_density(height, width, density)
    return {
        arrangement: make_state(
            height,
            width,
            inventory,
            seed=seed,
            arrangement=arrangement,
            concentration=concentration,
            device=device,
        )
        for arrangement in arrangements
    }
