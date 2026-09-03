from __future__ import annotations

from typing import Any

import torch

from .domain_mechanism import (
    conditional_environment_diagnostics,
    conditional_local_block_distributions,
    density_environment_masks,
    interface_geometry,
    state_composition,
    state_specific_structure_factors,
    toroidal_box_mean,
)
from .rules import ReversibleRule


def _compact_environment(
    state: torch.Tensor,
    *,
    width: int,
    lower: float,
    upper: float,
    method: str,
    density: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    environment = density_environment_masks(
        state,
        width=width,
        lower_quantile=lower,
        upper_quantile=upper,
        phase_threshold=method,
        interface_band_radius=max(1, width // 16),
        local_density=density,
    )
    masks = {
        "dense_interior": environment["dense_interior"],
        "sparse_interior": environment["sparse_interior"],
        "interface": environment["interface"],
    }
    geometry = interface_geometry(environment["high_phase"], environment["interface"])
    transition = (density > environment["metadata"]["lower_threshold"]) & (
        density < environment["metadata"]["upper_threshold"]
    )
    edge_count = int(geometry["total_crossing_edges"])
    geometry["transition_band_site_count"] = int(transition.sum().item())
    geometry["area_over_edge_thickness_cells"] = (
        float(transition.sum().item()) / edge_count if edge_count else None
    )
    compact = {
        "metadata": environment["metadata"],
        "geometry": geometry,
        "composition": {name: state_composition(state, mask) for name, mask in masks.items()},
    }
    return compact, masks


def analyze_spatial_state(
    state: torch.Tensor,
    *,
    after: torch.Tensor | None,
    rule: ReversibleRule,
    tick: int,
    roles: list[str],
) -> dict[str, Any]:
    """Apply the preregistered robust spatial diagnostics to one exact state."""
    if state.ndim != 2:
        raise ValueError("state must have shape [height,width]")
    if after is not None and after.shape != state.shape:
        raise ValueError("outgoing state must match the exact-state shape")
    robustness = []
    primary_masks: dict[str, torch.Tensor] | None = None
    interface_threshold_robustness = []
    occupancy = (state != 0).to(torch.float32)
    for width in (16, 32, 64):
        if width > min(state.shape):
            raise ValueError("spatial diagnostics require worlds at least 64x64")
        density = toroidal_box_mean(occupancy, width)
        for lower, upper in ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7)):
            compact, masks = _compact_environment(
                state,
                width=width,
                lower=lower,
                upper=upper,
                method="quantile_midpoint",
                density=density,
            )
            compact["threshold_scheme"] = f"q{int(lower * 100):02d}_q{int(upper * 100):02d}"
            robustness.append(compact)
            if width == 32 and lower == 0.2:
                primary_masks = masks
        if width == 32:
            for method in ("quantile_midpoint", "otsu", "global_occupancy"):
                compact, _ = _compact_environment(
                    state,
                    width=width,
                    lower=0.2,
                    upper=0.8,
                    method=method,
                    density=density,
                )
                compact["threshold_scheme"] = "q20_q80"
                interface_threshold_robustness.append(compact)
        del density
    if primary_masks is None:
        raise AssertionError("primary environment masks were not created")

    primary_dynamics: dict[str, Any]
    if after is None:
        primary_dynamics = {
            "estimable": False,
            "reason": "no outgoing phase belongs to the retained trajectory at the final endpoint",
        }
    else:
        primary_dynamics = {
            "estimable": True,
            "outgoing_phase": tick,
            "environments": conditional_environment_diagnostics(
                state,
                after,
                rule=rule,
                phase=tick,
                masks=primary_masks,
            ),
        }
    local_distributions = conditional_local_block_distributions(
        state, primary_masks, block=16
    )
    compute_structure = any(
        role in {"initial", "finite_domain_scale_peak", "final"} for role in roles
    )
    return {
        "schema": "zwslab-domain-spatial-milestone-v1",
        "diagnostic_revision": 2,
        "tick": tick,
        "roles": roles,
        "global_composition": state_composition(state),
        "density_environment_robustness": robustness,
        "interface_threshold_robustness": interface_threshold_robustness,
        "primary_definition": {
            "box_width": 32,
            "quantiles": [0.2, 0.8],
            "phase_threshold": "quantile_midpoint",
        },
        "primary_environment_dynamics": primary_dynamics,
        "primary_local_block_distributions": local_distributions,
        "state_specific_structure_factors": (
            state_specific_structure_factors(state) if compute_structure else None
        ),
        "curvature_status": (
            "No signed-curvature estimate is promoted: the discrete toroidal masks change "
            "substantially with box width. Edge length, thickness, and roughness are reported."
        ),
    }
