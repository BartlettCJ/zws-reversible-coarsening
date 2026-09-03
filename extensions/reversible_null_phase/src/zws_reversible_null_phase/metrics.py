from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch

from .alphabet import DEFAULT_ALPHABET
from .spectra import radial_structure_factor
from .topology import occupied_and_empty_topology


@dataclass(frozen=True)
class SnapshotMetrics:
    phase: int
    charge: int
    occupancy_count: int
    occupancy_fraction: float
    empty_fraction: float
    positive_fraction: float
    negative_fraction: float
    bound_fraction: float
    normalized_state_entropy: float
    activity_from_previous: float | None
    recurrence_from_lag: float | None
    low_k_fraction: float
    low_k_power: float
    dominant_k: float
    spectral_wavelength: float
    empty_component_count: int
    empty_finite_component_count: int
    empty_wrap_x: bool
    empty_wrap_y: bool
    largest_empty_fraction: float
    largest_finite_empty_fraction: float
    finite_empty_scale: float
    occupied_component_count: int
    occupied_wrap_x: bool
    occupied_wrap_y: bool
    largest_occupied_fraction: float
    occupied_perimeter_density: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_entropy_from_counts(counts: torch.Tensor) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.double() / total
    nonzero = p[p > 0]
    return float(-(nonzero * torch.log(nonzero)).sum() / math.log(4.0))


def snapshot_metrics(
    state: torch.Tensor,
    *,
    phase: int,
    previous: torch.Tensor | None = None,
    lag_state: torch.Tensor | None = None,
    low_k_max: float = 8.0,
    topology: bool = True,
) -> SnapshotMetrics:
    if state.ndim != 2:
        raise ValueError("snapshot state must be [H,W]")
    counts = DEFAULT_ALPHABET.state_counts(state)
    n = state.numel()
    fractions = counts.double() / n
    spectrum = radial_structure_factor(state, low_k_max=low_k_max)
    if topology:
        topo = occupied_and_empty_topology(state)
        empty = topo["empty"]
        occupied = topo["occupied"]
    else:
        # Fast placeholders for dense temporal sampling.  Milestone snapshots
        # are subsequently recomputed with exact torus topology.
        from .topology import ComponentSummary

        empty = ComponentSummary(0, 0, False, False, False, 0, 0, 0, 0, np.empty(0), np.empty(0))
        occupied = empty
    activity = (
        float((state != previous).float().mean()) if previous is not None else None
    )
    recurrence = (
        float((state == lag_state).float().mean()) if lag_state is not None else None
    )
    return SnapshotMetrics(
        phase=int(phase),
        charge=int(DEFAULT_ALPHABET.total_charge(state)),
        occupancy_count=int(counts[1:].sum()),
        occupancy_fraction=float(fractions[1:].sum()),
        empty_fraction=float(fractions[0]),
        positive_fraction=float(fractions[1]),
        negative_fraction=float(fractions[2]),
        bound_fraction=float(fractions[3]),
        normalized_state_entropy=normalized_entropy_from_counts(counts),
        activity_from_previous=activity,
        recurrence_from_lag=recurrence,
        low_k_fraction=spectrum.low_k_fraction,
        low_k_power=spectrum.low_k_power,
        dominant_k=spectrum.dominant_k,
        spectral_wavelength=spectrum.characteristic_wavelength,
        empty_component_count=empty.component_count,
        empty_finite_component_count=empty.finite_component_count,
        empty_wrap_x=empty.wrapping_x,
        empty_wrap_y=empty.wrapping_y,
        largest_empty_fraction=empty.largest_fraction,
        largest_finite_empty_fraction=empty.largest_finite_fraction,
        finite_empty_scale=empty.characteristic_finite_scale,
        occupied_component_count=occupied.component_count,
        occupied_wrap_x=occupied.wrapping_x,
        occupied_wrap_y=occupied.wrapping_y,
        largest_occupied_fraction=occupied.largest_fraction,
        occupied_perimeter_density=occupied.perimeter_density,
    )


@dataclass(frozen=True)
class StrongDomainCriterion:
    min_peak_scale_ratio: float = 4.0
    min_final_scale_ratio: float = 2.0
    min_peak_largest_finite_empty_fraction: float = 0.001
    initial_scale_floor: float = 1.0

    def classify(self, series: list[SnapshotMetrics]) -> dict[str, float | bool | int]:
        if len(series) < 2:
            raise ValueError("at least two snapshots are required")
        initial = max(float(series[0].finite_empty_scale), self.initial_scale_floor)
        ratios = np.asarray([m.finite_empty_scale / initial for m in series], dtype=np.float64)
        peak_index = int(np.argmax(ratios))
        peak_fraction = float(series[peak_index].largest_finite_empty_fraction)
        passed = bool(
            ratios[peak_index] >= self.min_peak_scale_ratio
            and ratios[-1] >= self.min_final_scale_ratio
            and peak_fraction >= self.min_peak_largest_finite_empty_fraction
            and all(m.charge == series[0].charge for m in series)
        )
        return {
            "strong_domain": passed,
            "initial_finite_empty_scale": initial,
            "peak_scale_ratio": float(ratios[peak_index]),
            "final_scale_ratio": float(ratios[-1]),
            "peak_phase": int(series[peak_index].phase),
            "peak_largest_finite_empty_fraction": peak_fraction,
        }
