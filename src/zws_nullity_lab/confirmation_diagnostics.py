from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch

from .alphabet import DEFAULT_ALPHABET
from .margolus import _to_blocks, encode_blocks
from .rules import ReversibleRule


def _derived_seed(base_seed: int, tick: int, block: int) -> int:
    payload = f"ZWSLab-shuffle-v1:{base_seed}:{tick}:{block}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def toroidal_empty_domain_stats(empty: np.ndarray) -> dict[str, Any]:
    """Return exact four-neighbour component statistics on a 2-D torus.

    The area-weighted finite-domain scale excludes the largest component. This
    prevents the initial empty background surrounding a localized seed from
    being mistaken for a newly formed finite empty domain. Raw largest-domain
    measurements are retained separately.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - exercised by environment checks
        raise RuntimeError("confirmation diagnostics require the 'science' extra") from exc

    mask = np.asarray(empty, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("empty-domain input must be a two-dimensional mask")
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, component_count_open = ndimage.label(mask, structure=structure)
    if component_count_open == 0:
        return {
            "component_count": 0,
            "empty_area": 0,
            "largest_area": 0,
            "largest_area_fraction_of_empty": None,
            "total_perimeter": 0,
            "largest_perimeter": 0,
            "area_quantiles": None,
            "area_log2_histogram": [],
            "area_weighted_linear_scale": None,
            "finite_component_count": 0,
            "finite_empty_area": 0,
            "finite_largest_area": 0,
            "finite_area_weighted_linear_scale": None,
        }

    parent = np.arange(component_count_open + 1, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(first: int, second: int) -> None:
        if first == 0 or second == 0:
            return
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in zip(labels[0], labels[-1], strict=True):
        if first and second:
            union(int(first), int(second))
    for first, second in zip(labels[:, 0], labels[:, -1], strict=True):
        if first and second:
            union(int(first), int(second))

    roots = np.arange(component_count_open + 1, dtype=np.int32)
    for label_value in range(1, component_count_open + 1):
        roots[label_value] = find(label_value)
    root_labels = roots[labels]
    occupied_roots = np.unique(root_labels[mask])
    occupied_roots = occupied_roots[occupied_roots != 0]
    areas_all = np.bincount(root_labels[mask], minlength=component_count_open + 1)
    areas = areas_all[occupied_roots].astype(np.int64, copy=False)

    neighbour_empty = (
        np.roll(mask, 1, axis=0).astype(np.uint8)
        + np.roll(mask, -1, axis=0).astype(np.uint8)
        + np.roll(mask, 1, axis=1).astype(np.uint8)
        + np.roll(mask, -1, axis=1).astype(np.uint8)
    )
    perimeter_weights = (4 - neighbour_empty)[mask]
    perimeters_all = np.bincount(
        root_labels[mask],
        weights=perimeter_weights,
        minlength=component_count_open + 1,
    )
    perimeters = perimeters_all[occupied_roots].astype(np.int64)

    largest_index = int(np.argmax(areas))
    largest_area = int(areas[largest_index])
    empty_area = int(areas.sum())
    area_weighted = math.sqrt(float(np.square(areas.astype(np.float64)).sum() / empty_area))
    finite_areas = np.delete(areas, largest_index)
    finite_empty_area = int(finite_areas.sum())
    finite_largest_area = int(finite_areas.max()) if finite_areas.size else 0
    finite_scale = None
    if finite_areas.size and finite_empty_area > 0:
        finite_scale = math.sqrt(
            float(
                np.square(finite_areas.astype(np.float64)).sum()
                / finite_areas.astype(np.float64).sum()
            )
        )

    quantiles = np.quantile(areas, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    log_bins = np.floor(np.log2(areas)).astype(np.int64)
    histogram = np.bincount(log_bins)
    return {
        "component_count": int(areas.size),
        "empty_area": empty_area,
        "largest_area": largest_area,
        "largest_area_fraction_of_empty": largest_area / empty_area,
        "total_perimeter": int(perimeters.sum()),
        "largest_perimeter": int(perimeters[largest_index]),
        "area_quantiles": {
            key: float(value)
            for key, value in zip(
                ("min", "q25", "median", "q75", "q90", "q99", "max"),
                quantiles,
                strict=True,
            )
        },
        "area_log2_histogram": histogram.astype(np.int64).tolist(),
        "area_weighted_linear_scale": area_weighted,
        "finite_component_count": int(finite_areas.size),
        "finite_empty_area": finite_empty_area,
        "finite_largest_area": finite_largest_area,
        "finite_area_weighted_linear_scale": finite_scale,
    }


def coarse_density_statistics(
    occupancy: torch.Tensor,
    *,
    blocks: list[int],
    shuffle_base_seed: int,
    tick: int,
) -> dict[str, dict[str, float | int | None]]:
    """Compare observed block-density variance with fixed-density shuffles."""
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    height, width = occupancy.shape
    population = height * width
    occupied_count = int(occupancy.sum().item())
    density = occupied_count / population
    result: dict[str, dict[str, float | int | None]] = {}
    work = occupancy.to(dtype=torch.float32)
    for block in blocks:
        if height % block or width % block:
            raise ValueError(f"block {block} does not divide {height}x{width}")
        coarse = work.reshape(height // block, block, width // block, block).mean(dim=(1, 3))
        observed_variance = float(coarse.var(unbiased=False).item())
        block_cells = block * block
        block_count = population // block_cells
        expected_variance = 0.0
        if population > 1:
            expected_variance = (
                density
                * (1.0 - density)
                * (population - block_cells)
                / (block_cells * (population - 1))
            )
        shuffle_seed = _derived_seed(shuffle_base_seed, tick, block)
        rng = np.random.default_rng(shuffle_seed)
        shuffled_counts = rng.multivariate_hypergeometric(
            np.full(block_count, block_cells, dtype=np.int64),
            occupied_count,
            method="marginals",
        )
        shuffled_variance = float(np.var(shuffled_counts / block_cells))
        result[str(block)] = {
            "block": block,
            "block_count": block_count,
            "observed_variance": observed_variance,
            "analytic_shuffle_expected_variance": expected_variance,
            "seeded_shuffle_variance": shuffled_variance,
            "dispersion_ratio_analytic": (
                observed_variance / expected_variance if expected_variance > 0.0 else None
            ),
            "dispersion_ratio_seeded_shuffle": (
                observed_variance / shuffled_variance if shuffled_variance > 0.0 else None
            ),
            "shuffle_seed": shuffle_seed,
        }
    return result


def radial_structure_factor(occupancy: torch.Tensor, max_output_bins: int = 128) -> list[dict[str, Any]]:
    """Compute a radial occupancy structure factor for each batch member."""
    if occupancy.ndim != 3:
        raise ValueError("occupancy must have shape [batch,height,width]")
    batch, height, width = occupancy.shape
    field = occupancy.to(dtype=torch.float32)
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(field, norm="ortho").abs().square()
    fy = torch.fft.fftfreq(height, device=field.device) * height
    fx = torch.fft.rfftfreq(width, device=field.device) * width
    radii = torch.sqrt(fy[:, None].square() + fx[None, :].square()).round().long()
    maximum_radius = min(height, width) // 2
    valid = (radii > 0) & (radii <= maximum_radius)
    radial_indices = radii[valid]
    counts = torch.bincount(radial_indices, minlength=maximum_radius + 1).to(torch.float64)
    output: list[dict[str, Any]] = []
    for batch_index in range(batch):
        sums = torch.bincount(
            radial_indices,
            weights=power[batch_index][valid].to(torch.float64),
            minlength=maximum_radius + 1,
        )
        profile = sums / counts.clamp_min(1.0)
        profile[0] = 0.0
        peak_radius = int(torch.argmax(profile).item()) if maximum_radius >= 1 else 0
        peak_power = float(profile[peak_radius].item()) if peak_radius else 0.0
        wavelength = min(height, width) / peak_radius if peak_radius else None

        edges = np.unique(
            np.rint(np.geomspace(1, maximum_radius + 1, min(max_output_bins, maximum_radius) + 1))
            .astype(np.int64)
            .clip(1, maximum_radius + 1)
        )
        if edges[-1] != maximum_radius + 1:
            edges = np.append(edges, maximum_radius + 1)
        profile_cpu = profile.detach().cpu().numpy()
        compact = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            values = profile_cpu[lower:upper]
            if values.size:
                compact.append(
                    {
                        "radius_min": int(lower),
                        "radius_max_exclusive": int(upper),
                        "radius_geometric_mean": float(math.sqrt(lower * max(lower, upper - 1))),
                        "mean_power": float(values.mean()),
                    }
                )
        positive_profile = profile[1:]
        median_power = float(torch.median(positive_profile).item()) if positive_profile.numel() else 0.0
        output.append(
            {
                "peak_radius": peak_radius or None,
                "peak_power": peak_power,
                "peak_to_radial_median": peak_power / median_power if median_power > 0 else None,
                "characteristic_wavelength_cells": wavelength,
                "radial_profile": compact,
            }
        )
    return output


def _coarse_occupancy(state: torch.Tensor, block: int) -> torch.Tensor:
    occupancy = (state != DEFAULT_ALPHABET.empty).to(torch.float32)
    batch, height, width = occupancy.shape
    return occupancy.reshape(batch, height // block, block, width // block, block).mean(
        dim=(2, 4)
    )


def _correlation(first: torch.Tensor, second: torch.Tensor) -> list[float | None]:
    first_flat = first.flatten(1).to(torch.float64)
    second_flat = second.flatten(1).to(torch.float64)
    first_centered = first_flat - first_flat.mean(dim=1, keepdim=True)
    second_centered = second_flat - second_flat.mean(dim=1, keepdim=True)
    numerator = (first_centered * second_centered).sum(dim=1)
    denominator = torch.sqrt(
        first_centered.square().sum(dim=1) * second_centered.square().sum(dim=1)
    )
    values = numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    return [float(value) if float(scale) > 0 else None for value, scale in zip(values, denominator)]


def sample_diagnostics(
    state: torch.Tensor,
    *,
    tick: int,
    rule: ReversibleRule,
    lag_states: dict[int, torch.Tensor],
    coarse_blocks: list[int],
    shuffle_base_seeds: list[int],
    previous_occupied_fractions: list[float | None],
    previous_tick: int | None,
) -> list[dict[str, Any]]:
    """Measure one batch without modifying it or advancing the automaton."""
    if state.ndim != 3:
        raise ValueError("state must have shape [batch,height,width]")
    batch, height, width = state.shape
    if len(shuffle_base_seeds) != batch:
        raise ValueError("one shuffle seed is required for every batch member")
    populations = torch.stack(
        [torch.bincount(item.long().flatten(), minlength=4) for item in state], dim=0
    )
    population = height * width
    occupied = populations[:, 1:].sum(dim=1)
    occupied_fractions = occupied.to(torch.float64) / population
    probabilities = populations.to(torch.float64) / population
    entropy_terms = torch.where(probabilities > 0, probabilities * torch.log(probabilities), 0.0)
    entropy = -entropy_terms.sum(dim=1) / math.log(4.0)
    boundary = 0.5 * (
        (state != torch.roll(state, -1, dims=-2)).to(torch.float64).mean(dim=(-2, -1))
        + (state != torch.roll(state, -1, dims=-1)).to(torch.float64).mean(dim=(-2, -1))
    )
    charges = populations[:, DEFAULT_ALPHABET.positive] - populations[:, DEFAULT_ALPHABET.negative]

    activity: list[float | None] = [None] * batch
    reaction: list[float | None] = [None] * batch
    if 1 in lag_states:
        previous = lag_states[1]
        activity = [
            float(value)
            for value in (state != previous).to(torch.float64).mean(dim=(-2, -1)).tolist()
        ]
        blocks, _ = _to_blocks(previous, tick - 1)
        codes = encode_blocks(blocks)
        reaction_mask = rule.reaction_mask().to(state.device)
        reaction = [
            float(value)
            for value in reaction_mask[codes.long()].to(torch.float64).mean(dim=(-2, -1)).tolist()
        ]

    recurrence: dict[int, list[float | None]] = {}
    coarse_recurrence: dict[int, list[float | None]] = {}
    current_coarse = _coarse_occupancy(state, 64)
    for lag in sorted(lag_states):
        earlier = lag_states[lag]
        recurrence[lag] = [
            float(value)
            for value in (state == earlier).to(torch.float64).mean(dim=(-2, -1)).tolist()
        ]
        coarse_recurrence[lag] = _correlation(current_coarse, _coarse_occupancy(earlier, 64))

    structure = radial_structure_factor(state != DEFAULT_ALPHABET.empty)
    results: list[dict[str, Any]] = []
    for index in range(batch):
        occupancy_cpu = (state[index] != DEFAULT_ALPHABET.empty).detach().cpu()
        occupied_fraction = float(occupied_fractions[index].item())
        growth_rate = None
        if previous_tick is not None and previous_tick < tick:
            previous_fraction = previous_occupied_fractions[index]
            if previous_fraction is not None:
                growth_rate = (occupied_fraction - previous_fraction) / (tick - previous_tick)
        results.append(
            {
                "tick": tick,
                "total_charge": int(charges[index].item()),
                "state_populations": [int(value) for value in populations[index].tolist()],
                "occupied_fraction": occupied_fraction,
                "occupancy_growth_per_phase": growth_rate,
                "activity_one_phase": activity[index],
                "reaction_density_one_phase": reaction[index],
                "entropy_normalized": float(entropy[index].item()),
                "boundary_density": float(boundary[index].item()),
                "coarse_density": coarse_density_statistics(
                    occupancy_cpu,
                    blocks=coarse_blocks,
                    shuffle_base_seed=shuffle_base_seeds[index],
                    tick=tick,
                ),
                "empty_domains": toroidal_empty_domain_stats(~occupancy_cpu.numpy()),
                "structure_factor": structure[index],
                "exact_state_recurrence": {
                    str(lag): recurrence[lag][index] for lag in sorted(recurrence)
                },
                "coarse_occupancy_correlation_block64": {
                    str(lag): coarse_recurrence[lag][index]
                    for lag in sorted(coarse_recurrence)
                },
            }
        )
    return results
