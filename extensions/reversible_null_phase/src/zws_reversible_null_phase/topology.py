from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy import ndimage


_CROSS = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


@dataclass(frozen=True)
class ComponentSummary:
    component_count: int
    finite_component_count: int
    wrapping_x: bool
    wrapping_y: bool
    wrapping_any: bool
    largest_fraction: float
    largest_finite_fraction: float
    characteristic_finite_scale: float
    perimeter_density: float
    areas: np.ndarray
    finite_areas: np.ndarray


def _as_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    return np.asarray(arr, dtype=bool)


def torus_components(mask: torch.Tensor | np.ndarray) -> ComponentSummary:
    """Exact four-neighbour components and wrap detection on a 2D torus.

    Open-boundary components are joined across the two periodic seams with a
    weighted union-find.  The integer weights track which translated copy of
    an open component is being joined.  A non-zero displacement around a
    closed union cycle is therefore an exact winding witness.  This avoids the
    origin-dependent component splitting produced by the former 3x3-tiling
    implementation.
    """
    core = _as_mask(mask)
    h, w = core.shape
    labels, open_component_count = ndimage.label(core, structure=_CROSS)
    if open_component_count == 0:
        return ComponentSummary(
            component_count=0,
            finite_component_count=0,
            wrapping_x=False,
            wrapping_y=False,
            wrapping_any=False,
            largest_fraction=0.0,
            largest_finite_fraction=0.0,
            characteristic_finite_scale=0.0,
            perimeter_density=0.0,
            areas=np.empty(0, dtype=np.int64),
            finite_areas=np.empty(0, dtype=np.int64),
        )

    parent = np.arange(open_component_count + 1, dtype=np.int32)
    rank = np.zeros(open_component_count + 1, dtype=np.int8)
    # offset[label] = tile coordinate(label) - tile coordinate(parent[label]),
    # stored as (y, x) in units of whole lattice periods.
    offset = np.zeros((open_component_count + 1, 2), dtype=np.int64)
    winding = np.zeros((open_component_count + 1, 2), dtype=bool)

    def find(label: int) -> int:
        parent_label = int(parent[label])
        if parent_label == label:
            return label
        root = find(parent_label)
        offset[label] += offset[parent_label]
        parent[label] = root
        return root

    def union(first: int, second: int, delta: tuple[int, int]) -> None:
        """Enforce tile(second) - tile(first) == delta."""

        root_first = find(first)
        first_offset = offset[first].copy()
        root_second = find(second)
        second_offset = offset[second].copy()
        required = np.asarray(delta, dtype=np.int64)
        if root_first == root_second:
            residual = (second_offset - first_offset) - required
            winding[root_first] |= residual != 0
            return
        second_to_first = required + first_offset - second_offset
        combined_winding = winding[root_first] | winding[root_second]
        if rank[root_first] < rank[root_second]:
            parent[root_first] = root_second
            offset[root_first] = -second_to_first
            winding[root_second] = combined_winding
        else:
            parent[root_second] = root_first
            offset[root_second] = second_to_first
            winding[root_first] = combined_winding
            if rank[root_first] == rank[root_second]:
                rank[root_first] += 1

    for x in range(w):
        if core[0, x] and core[-1, x]:
            union(int(labels[0, x]), int(labels[-1, x]), (-1, 0))
    for y in range(h):
        if core[y, 0] and core[y, -1]:
            union(int(labels[y, 0]), int(labels[y, -1]), (0, -1))

    roots = np.zeros(open_component_count + 1, dtype=np.int32)
    for label in range(1, open_component_count + 1):
        roots[label] = find(label)
    active_roots = np.unique(roots[1:])
    open_areas = np.bincount(labels[core], minlength=open_component_count + 1)
    periodic_areas = np.bincount(
        roots[1:], weights=open_areas[1:], minlength=open_component_count + 1
    )
    areas = periodic_areas[active_roots].astype(np.int64, copy=False)
    root_winding = winding[active_roots]
    finite_mask = ~root_winding.any(axis=1)
    finite_areas = areas[finite_mask]
    total_sites = h * w
    largest = float(areas.max() / total_sites) if areas.size else 0.0
    largest_finite = float(finite_areas.max() / total_sites) if finite_areas.size else 0.0
    if finite_areas.size:
        area_weighted = float(np.square(finite_areas, dtype=np.float64).sum() / finite_areas.sum())
        finite_scale = math.sqrt(area_weighted / math.pi)
    else:
        finite_scale = 0.0

    horizontal_edges = np.count_nonzero(core != np.roll(core, -1, axis=1))
    vertical_edges = np.count_nonzero(core != np.roll(core, -1, axis=0))
    perimeter_density = float((horizontal_edges + vertical_edges) / (2.0 * total_sites))
    wrapping_x = bool(root_winding[:, 1].any())
    wrapping_y = bool(root_winding[:, 0].any())
    return ComponentSummary(
        component_count=int(active_roots.size),
        finite_component_count=int(finite_areas.size),
        wrapping_x=wrapping_x,
        wrapping_y=wrapping_y,
        wrapping_any=wrapping_x or wrapping_y,
        largest_fraction=largest,
        largest_finite_fraction=largest_finite,
        characteristic_finite_scale=finite_scale,
        perimeter_density=perimeter_density,
        areas=areas,
        finite_areas=finite_areas,
    )


def occupied_and_empty_topology(state: torch.Tensor | np.ndarray) -> dict[str, ComponentSummary]:
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy()
    else:
        arr = np.asarray(state)
    occupied = arr != 0
    return {
        "occupied": torus_components(occupied),
        "empty": torus_components(~occupied),
    }
