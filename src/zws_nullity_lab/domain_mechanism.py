from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .alphabet import DEFAULT_ALPHABET
from .confirmation_diagnostics import radial_structure_factor
from .margolus import _from_blocks, _to_blocks, encode_blocks
from .rules import ReversibleRule


STATE_MASS = torch.tensor([0, 1, 1, 1], dtype=torch.int64)


def state_composition(state: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    """Return exact four-state counts and derived Q/M composition for one lattice region."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    if mask is None:
        selected = state.reshape(-1)
    else:
        if mask.shape != state.shape:
            raise ValueError("mask and state shapes must match")
        selected = state[mask.to(device=state.device, dtype=torch.bool)]
    counts = torch.bincount(selected.long(), minlength=4).to(dtype=torch.int64)
    total = int(selected.numel())
    fractions = counts.to(torch.float64) / total if total else torch.zeros(4, dtype=torch.float64)
    names = DEFAULT_ALPHABET.state_names
    count_map = {name: int(counts[index].item()) for index, name in enumerate(names)}
    fraction_map = {
        name: float(fractions[index].item()) if total else None
        for index, name in enumerate(names)
    }
    positive = int(counts[DEFAULT_ALPHABET.positive].item())
    negative = int(counts[DEFAULT_ALPHABET.negative].item())
    bound = int(counts[DEFAULT_ALPHABET.bound].item())
    occupied = positive + negative + bound
    nonzero_fractions = fractions[fractions > 0]
    entropy = float(-(nonzero_fractions * torch.log(nonzero_fractions)).sum().item())
    normalized_entropy = entropy / math.log(4.0)
    return {
        "site_count": total,
        "state_counts": count_map,
        "state_fractions": fraction_map,
        "positive_count": positive,
        "negative_count": negative,
        "bound_neutral_count": bound,
        "charged_count": positive + negative,
        "occupied_count": occupied,
        "total_signed_charge": positive - negative,
        "normalized_state_entropy": normalized_entropy,
        "positive_fraction": positive / total if total else None,
        "negative_fraction": negative / total if total else None,
        "bound_neutral_fraction": bound / total if total else None,
        "charged_fraction": (positive + negative) / total if total else None,
        "occupied_fraction": occupied / total if total else None,
    }


def exact_occupancy_mass(state: torch.Tensor) -> torch.Tensor:
    """Return exact nonempty-site count for one lattice or a leading batch."""
    mass = STATE_MASS.to(state.device)[state.long()]
    if state.ndim <= 2:
        return mass.sum()
    return mass.sum(dim=tuple(range(1, state.ndim)))


def toroidal_box_mean(field: torch.Tensor, width: int) -> torch.Tensor:
    """Centered stride-one toroidal box average with the same output shape as ``field``."""
    if field.ndim not in {2, 3}:
        raise ValueError("field must have shape [H,W] or [B,H,W]")
    if width < 1:
        raise ValueError("width must be positive")
    height, world_width = field.shape[-2:]
    if width > min(height, world_width):
        raise ValueError("width cannot exceed either world dimension")
    lower = (width - 1) // 2

    def moving_sum(source: torch.Tensor, dimension: int) -> torch.Tensor:
        shifted = torch.roll(source, shifts=lower, dims=dimension)
        moved = shifted.movedim(dimension, -1)
        extended = torch.cat([moved, moved[..., : width - 1]], dim=-1) if width > 1 else moved
        prefix = torch.cat(
            [
                torch.zeros_like(extended[..., :1]),
                torch.cumsum(extended, dim=-1),
            ],
            dim=-1,
        )
        summed = prefix[..., width:] - prefix[..., :-width]
        return summed.movedim(-1, dimension)

    source = field.to(dtype=torch.float32)
    summed = moving_sum(moving_sum(source, -1), -2)
    return summed / float(width * width)


def _toroidal_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return mask.to(dtype=torch.bool)
    source = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    padded = F.pad(source, (radius, radius, radius, radius), mode="circular")
    return F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1)[0, 0].bool()


def otsu_threshold(values: torch.Tensor, bins: int = 256) -> float | None:
    """Return a deterministic Otsu threshold, or ``None`` for a constant field."""
    flattened = values.detach().to(dtype=torch.float64).reshape(-1).cpu()
    minimum = float(flattened.min().item())
    maximum = float(flattened.max().item())
    if maximum <= minimum:
        return None
    histogram = torch.histc(flattened.to(torch.float32), bins=bins, min=minimum, max=maximum)
    centers = torch.linspace(minimum, maximum, bins, dtype=torch.float64)
    weights_first = torch.cumsum(histogram.to(torch.float64), dim=0)
    weights_second = histogram.sum() - weights_first
    sums_first = torch.cumsum(histogram.to(torch.float64) * centers, dim=0)
    sums_total = sums_first[-1]
    valid = (weights_first > 0) & (weights_second > 0)
    variance = torch.full((bins,), -1.0, dtype=torch.float64)
    means_first = sums_first[valid] / weights_first[valid]
    means_second = (sums_total - sums_first[valid]) / weights_second[valid]
    variance[valid] = weights_first[valid] * weights_second[valid] * (
        means_first - means_second
    ).square()
    return float(centers[int(torch.argmax(variance).item())].item())


def density_environment_masks(
    state: torch.Tensor,
    *,
    width: int,
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    phase_threshold: str = "quantile_midpoint",
    interface_band_radius: int = 1,
    local_density: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Create transparent dense/sparse/interfacial masks from local occupancy."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    occupancy = (state != DEFAULT_ALPHABET.empty).to(torch.float32)
    density = toroidal_box_mean(occupancy, width) if local_density is None else local_density
    if density.shape != state.shape:
        raise ValueError("local_density and state shapes must match")
    quantiles = torch.quantile(
        density.reshape(-1),
        torch.tensor([lower_quantile, upper_quantile], device=density.device),
    )
    lower = float(quantiles[0].item())
    upper = float(quantiles[1].item())
    if phase_threshold == "quantile_midpoint":
        threshold = 0.5 * (lower + upper)
        threshold_fallback = False
    elif phase_threshold == "global_occupancy":
        threshold = float(occupancy.mean().item())
        threshold_fallback = False
    elif phase_threshold == "otsu":
        observed = otsu_threshold(density)
        threshold_fallback = observed is None
        threshold = 0.5 * (lower + upper) if observed is None else observed
    else:
        raise ValueError("phase_threshold must be quantile_midpoint, global_occupancy, or otsu")

    dense = density >= upper
    sparse = density <= lower
    high_phase = density >= threshold
    boundary = torch.zeros_like(high_phase)
    for shifts, dimension in ((1, -2), (-1, -2), (1, -1), (-1, -1)):
        boundary |= high_phase != torch.roll(high_phase, shifts=shifts, dims=dimension)
    interface = _toroidal_dilate(boundary, interface_band_radius)
    dense_interior = dense & ~interface
    sparse_interior = sparse & ~interface
    site_count = state.numel()
    return {
        "local_density": density,
        "dense": dense,
        "sparse": sparse,
        "high_phase": high_phase,
        "interface": interface,
        "dense_interior": dense_interior,
        "sparse_interior": sparse_interior,
        "metadata": {
            "box_width": width,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "lower_threshold": lower,
            "upper_threshold": upper,
            "phase_threshold_method": phase_threshold,
            "phase_threshold": threshold,
            "phase_threshold_fallback": threshold_fallback,
            "interface_band_radius": interface_band_radius,
            "dense_realized_fraction": float(dense.sum().item()) / site_count,
            "sparse_realized_fraction": float(sparse.sum().item()) / site_count,
            "interface_realized_fraction": float(interface.sum().item()) / site_count,
        },
    }


def _sliding_block_codes(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    top_left = state.long()
    top_right = torch.roll(top_left, shifts=-1, dims=-1)
    bottom_left = torch.roll(top_left, shifts=-1, dims=-2)
    bottom_right = torch.roll(top_right, shifts=-1, dims=-2)
    return top_left + 4 * top_right + 16 * bottom_left + 64 * bottom_right


def sliding_block_code_histogram(
    state: torch.Tensor, mask: torch.Tensor | None = None
) -> list[int]:
    """Count all toroidal sliding 2x2 state motifs, optionally by top-left region."""
    codes = _sliding_block_codes(state)
    selected = codes.reshape(-1) if mask is None else codes[mask.bool()]
    return [int(value) for value in torch.bincount(selected, minlength=256).cpu().tolist()]


def _neighbourhood_codes(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    codes = torch.zeros_like(state, dtype=torch.int64)
    power = 1
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            neighbour = torch.roll(
                state.long(), shifts=(-row_shift, -column_shift), dims=(-2, -1)
            )
            codes += power * neighbour
            power *= 4
    return codes


def _top_neighbourhoods_from_codes(
    codes: torch.Tensor, mask: torch.Tensor | None, *, limit: int
) -> dict[str, Any]:
    selected = codes.reshape(-1) if mask is None else codes[mask.bool()]
    counts = torch.bincount(selected, minlength=4**9)
    nonzero = torch.nonzero(counts, as_tuple=False).flatten()
    order = nonzero[torch.argsort(counts[nonzero], descending=True, stable=True)][:limit]
    records = []
    for code_tensor in order:
        code = int(code_tensor.item())
        remainder = code
        pattern = []
        for _ in range(9):
            pattern.append(remainder % 4)
            remainder //= 4
        records.append(
            {
                "code": code,
                "count": int(counts[code].item()),
                "pattern_row_major": [pattern[0:3], pattern[3:6], pattern[6:9]],
            }
        )
    return {"observed_count": int(selected.numel()), "top": records}


def top_neighbourhood_frequencies(
    state: torch.Tensor, mask: torch.Tensor | None = None, *, limit: int = 64
) -> dict[str, Any]:
    """Return the most frequent toroidal 3x3 neighbourhood codes and exact patterns."""
    if limit < 1:
        raise ValueError("limit must be positive")
    return _top_neighbourhoods_from_codes(_neighbourhood_codes(state), mask, limit=limit)


def central_state_conditioned_neighbourhood_frequencies(
    state: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    limit_per_state: int = 64,
) -> dict[str, Any]:
    """Return top toroidal 3x3 patterns separately for each exact central state."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    if limit_per_state < 1:
        raise ValueError("limit_per_state must be positive")
    selected_mask = torch.ones_like(state, dtype=torch.bool) if mask is None else mask.bool()
    if selected_mask.shape != state.shape:
        raise ValueError("mask and state shapes must match")
    codes = _neighbourhood_codes(state)
    output: dict[str, Any] = {}
    for central_state, name in enumerate(DEFAULT_ALPHABET.state_names):
        central_mask = selected_mask & (state == central_state)
        summary = _top_neighbourhoods_from_codes(
            codes,
            central_mask,
            limit=limit_per_state,
        )
        summary["central_state_code"] = central_state
        summary["central_state_name"] = name
        output[name] = summary
    return output


def nonoverlap_local_fields(state: torch.Tensor, block: int = 16) -> dict[str, torch.Tensor]:
    """Return exact Q/M and normalized entropy on nonoverlapping local blocks."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    height, width = state.shape
    if block < 1 or height % block or width % block:
        raise ValueError("block must divide both world dimensions")
    counts = []
    for state_index in range(4):
        indicator = state == state_index
        local = indicator.reshape(height // block, block, width // block, block).sum(
            dim=(1, 3), dtype=torch.int64
        )
        counts.append(local)
    stacked = torch.stack(counts, dim=-1)
    total = float(block * block)
    probabilities = stacked.to(torch.float64) / total
    entropy_terms = torch.where(
        probabilities > 0,
        -probabilities * torch.log(probabilities.clamp_min(torch.finfo(torch.float64).tiny)),
        torch.zeros_like(probabilities),
    )
    return {
        "state_counts": stacked,
        "signed_charge": stacked[..., 1] - stacked[..., 2],
        "occupancy_mass": stacked[..., 1:].sum(dim=-1),
        "normalized_state_entropy": entropy_terms.sum(dim=-1) / math.log(4.0),
    }


def conditional_local_block_distributions(
    state: torch.Tensor, masks: dict[str, torch.Tensor], *, block: int = 16
) -> dict[str, Any]:
    """Summarize exact local Q/M/entropy fields at block centers by environment."""
    fields = nonoverlap_local_fields(state, block)
    height, width = state.shape
    row_start = block // 2
    column_start = block // 2
    output: dict[str, Any] = {}
    quantile_levels = torch.tensor(
        [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], dtype=torch.float64
    )
    for name, mask in masks.items():
        if mask.shape != state.shape:
            raise ValueError("all masks must match the state shape")
        selected_blocks = mask[
            row_start:height:block,
            column_start:width:block,
        ].bool()
        charge = fields["signed_charge"][selected_blocks].to(torch.int64).cpu()
        mass = fields["occupancy_mass"][selected_blocks].to(torch.int64).cpu()
        entropy = fields["normalized_state_entropy"][selected_blocks].to(torch.float64).cpu()
        if not charge.numel():
            output[name] = {"block_count": 0}
            continue
        charge_values, charge_counts = torch.unique(charge, sorted=True, return_counts=True)
        output[name] = {
            "block_count": int(charge.numel()),
            "block_width": block,
            "signed_charge": {
                "minimum": int(charge.min().item()),
                "maximum": int(charge.max().item()),
                "mean": float(charge.to(torch.float64).mean().item()),
                "mean_absolute": float(charge.abs().to(torch.float64).mean().item()),
                "standard_deviation": float(charge.to(torch.float64).std(unbiased=False).item()),
                "histogram": {
                    str(int(value)): int(count)
                    for value, count in zip(charge_values.tolist(), charge_counts.tolist(), strict=True)
                },
            },
            "occupancy_mass_quantiles": {
                str(float(level)): float(value)
                for level, value in zip(
                    quantile_levels.tolist(),
                    torch.quantile(mass.to(torch.float64), quantile_levels).tolist(),
                    strict=True,
                )
            },
            "normalized_state_entropy_quantiles": {
                str(float(level)): float(value)
                for level, value in zip(
                    quantile_levels.tolist(),
                    torch.quantile(entropy, quantile_levels).tolist(),
                    strict=True,
                )
            },
        }
    return output


def reaction_cell_map(state: torch.Tensor, rule: ReversibleRule, phase: int) -> torch.Tensor:
    """Mark all four cells of every state-multiset-changing Margolus block."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    batch = state.unsqueeze(0)
    blocks, offset = _to_blocks(batch, phase)
    codes = encode_blocks(blocks)
    reacted = rule.reaction_mask().to(state.device)[codes]
    expanded = reacted.unsqueeze(-1).expand(*reacted.shape, 4).to(dtype=torch.uint8)
    return _from_blocks(expanded, offset, state.shape[0], state.shape[1])[0].bool()


def state_transition_matrix(
    before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor | None = None
) -> list[list[int]]:
    """Return exact Eulerian site-state transitions within a fixed spatial mask."""
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("before and after must share shape [height,width]")
    codes = before.long() * 4 + after.long()
    selected = codes.reshape(-1) if mask is None else codes[mask.bool()]
    counts = torch.bincount(selected, minlength=16).reshape(4, 4)
    return [[int(value) for value in row] for row in counts.cpu().tolist()]


def interface_geometry(high_phase: torch.Tensor, interface: torch.Tensor) -> dict[str, Any]:
    """Return toroidal edge length and scale-free interface roughness descriptors."""
    if high_phase.shape != interface.shape or high_phase.ndim != 2:
        raise ValueError("phase and interface masks must share shape [height,width]")
    horizontal_edges = int((high_phase != torch.roll(high_phase, -1, -1)).sum().item())
    vertical_edges = int((high_phase != torch.roll(high_phase, -1, -2)).sum().item())
    edge_count = horizontal_edges + vertical_edges
    area = int(high_phase.sum().item())
    complement = high_phase.numel() - area
    reference_area = min(area, complement)
    roughness = None
    if reference_area > 0:
        roughness = edge_count * edge_count / (4.0 * torch.pi * reference_area)
        roughness = float(roughness)
    return {
        "horizontal_crossing_edges": horizontal_edges,
        "vertical_crossing_edges": vertical_edges,
        "total_crossing_edges": edge_count,
        "crossing_edge_density": edge_count / (2.0 * high_phase.numel()),
        "interface_site_count": int(interface.sum().item()),
        "interface_site_fraction": float(interface.float().mean().item()),
        "isoperimetric_roughness_proxy": roughness,
    }


def conditional_environment_diagnostics(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    rule: ReversibleRule,
    phase: int,
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure composition, reaction, motifs, and Eulerian exchange by environment."""
    if before.shape != after.shape:
        raise ValueError("before and after shapes must match")
    reaction = reaction_cell_map(before, rule, phase)
    motif_codes = _sliding_block_codes(before)
    neighbourhood_codes = _neighbourhood_codes(before)
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        selected = mask.bool()
        selected_count = int(selected.sum().item())
        output[name] = {
            "composition": state_composition(before, selected),
            "reaction_cell_fraction": (
                float(reaction[selected].float().mean().item()) if selected_count else None
            ),
            "motif_2x2_histogram": [
                int(value)
                for value in torch.bincount(motif_codes[selected], minlength=256).cpu().tolist()
            ],
            "top_neighbourhoods_3x3": _top_neighbourhoods_from_codes(
                neighbourhood_codes, selected, limit=64
            ),
            "central_state_conditioned_neighbourhoods_3x3": {
                state_name: {
                    **_top_neighbourhoods_from_codes(
                        neighbourhood_codes,
                        selected & (before == state_code),
                        limit=64,
                    ),
                    "central_state_code": state_code,
                    "central_state_name": state_name,
                }
                for state_code, state_name in enumerate(DEFAULT_ALPHABET.state_names)
            },
            "eulerian_state_transition_matrix": state_transition_matrix(before, after, selected),
        }
    return output


def state_specific_structure_factors(state: torch.Tensor) -> dict[str, dict[str, Any]]:
    """Compute the same radial structure-factor diagnostic for each exact state field."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    output: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(DEFAULT_ALPHABET.state_names):
        indicator = (state == index).to(torch.float32).unsqueeze(0)
        output[name] = radial_structure_factor(indicator)[0]
    return output
