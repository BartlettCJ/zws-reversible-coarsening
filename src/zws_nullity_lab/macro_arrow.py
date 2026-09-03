from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np
import torch

from .confirmation_diagnostics import toroidal_empty_domain_stats
from .domain_mechanism import state_composition


def tensor_digest(state: torch.Tensor) -> str:
    array = state.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _single_state(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 3 and state.shape[0] == 1:
        return state[0]
    if state.ndim != 2:
        raise ValueError("state must have shape [H,W] or [1,H,W]")
    return state


def _binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    values = probabilities.to(torch.float64)
    tiny = torch.finfo(torch.float64).tiny
    return -(
        values * torch.log(values.clamp_min(tiny))
        + (1.0 - values) * torch.log((1.0 - values).clamp_min(tiny))
    ) / math.log(2.0)


def checkpoint_macro_observables(
    state: torch.Tensor, *, tick: int, coarse_block: int = 64
) -> dict[str, Any]:
    """Compute the declared coarse observables from one exact discovery checkpoint."""
    state = _single_state(state)
    height, width = state.shape
    if coarse_block < 1 or height % coarse_block or width % coarse_block:
        raise ValueError("coarse_block must divide both world dimensions")
    occupancy = state != 0
    exact_crossings = (state != torch.roll(state, -1, dims=-2)).sum(dtype=torch.int64)
    exact_crossings += (state != torch.roll(state, -1, dims=-1)).sum(dtype=torch.int64)
    occupancy_crossings = (
        occupancy != torch.roll(occupancy, -1, dims=-2)
    ).sum(dtype=torch.int64)
    occupancy_crossings += (
        occupancy != torch.roll(occupancy, -1, dims=-1)
    ).sum(dtype=torch.int64)
    coarse_density = occupancy.to(torch.float32).reshape(
        height // coarse_block,
        coarse_block,
        width // coarse_block,
        coarse_block,
    ).mean(dim=(1, 3))
    composition = state_composition(state)
    empty_domains = toroidal_empty_domain_stats(~occupancy.detach().cpu().numpy())
    return {
        "schema": "zwslab-discovery-checkpoint-macro-observables-v1",
        "tick": int(tick),
        "world": [height, width],
        "exact_state_sha256_uint8": tensor_digest(state),
        "global_composition": composition,
        "occupied_fraction": composition["occupied_fraction"],
        "positive_fraction": composition["positive_fraction"],
        "negative_fraction": composition["negative_fraction"],
        "bound_neutral_fraction": composition["bound_neutral_fraction"],
        "charged_fraction": composition["charged_fraction"],
        "total_signed_charge": composition["total_signed_charge"],
        "normalized_state_entropy": composition["normalized_state_entropy"],
        "coarse_occupancy_entropy_block": coarse_block,
        "coarse_occupancy_entropy_normalized": float(
            _binary_entropy(coarse_density).mean().item()
        ),
        "exact_state_boundary_edge_density": float(exact_crossings.item())
        / (2.0 * state.numel()),
        "occupancy_interface_edge_density": float(occupancy_crossings.item())
        / (2.0 * state.numel()),
        "finite_domain_scale": empty_domains["finite_area_weighted_linear_scale"],
        "empty_domains": empty_domains,
    }


def select_capture_ticks(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(records, key=lambda record: int(record["tick"]))
    if len(ordered) < 3 or int(ordered[0]["tick"]) != 0:
        raise ValueError("capture selection requires a tick-zero series with at least three points")
    finite = [
        record for record in ordered if record.get("finite_domain_scale") is not None
    ]
    if not finite:
        raise ValueError("capture series contains no finite-domain scale")
    peak = max(finite, key=lambda record: (float(record["finite_domain_scale"]), -int(record["tick"])))
    return {
        "initial": int(ordered[0]["tick"]),
        "peak_organisation": int(peak["tick"]),
        "final": int(ordered[-1]["tick"]),
    }


def synchronized_forward_reverse(
    records: Iterable[dict[str, Any]],
    reversibility_audit: dict[str, Any],
) -> dict[str, Any]:
    """Mirror exact checkpoint observables onto forward and reverse traversal axes."""
    ordered = sorted(records, key=lambda record: int(record["tick"]))
    if not ordered:
        raise ValueError("macro-arrow series is empty")
    total_phases = int(reversibility_audit["phases_reversed"])
    if int(ordered[0]["tick"]) != 0 or int(ordered[-1]["tick"]) != total_phases:
        raise ValueError("macro-arrow series does not span the audited trajectory")
    if not reversibility_audit.get("exact_at_every_checkpoint_boundary"):
        raise ValueError("reversibility audit did not pass every checkpoint boundary")
    audit_by_tick = {
        int(boundary["tick"]): boundary for boundary in reversibility_audit["boundaries"]
    }
    if set(audit_by_tick) != {int(record["tick"]) for record in ordered[:-1]}:
        raise ValueError("observable ticks and exact reverse-audit boundaries do not match")
    for record in ordered[:-1]:
        boundary = audit_by_tick[int(record["tick"])]
        if not boundary.get("exact_state_equal") or boundary.get("mismatch_cells") != 0:
            raise ValueError(f"reverse boundary failed at tick {record['tick']}")
        if boundary["reference_state_sha256_uint8"] != record["exact_state_sha256_uint8"]:
            raise ValueError(f"reverse/reference state digest mismatch at tick {record['tick']}")

    def projected(record: dict[str, Any], traversal_phase: int) -> dict[str, Any]:
        return {
            "traversal_phase": traversal_phase,
            "microscopic_phase": int(record["tick"]),
            "exact_state_sha256_uint8": record["exact_state_sha256_uint8"],
            "occupied_fraction": record["occupied_fraction"],
            "finite_domain_scale": record["finite_domain_scale"],
            "occupancy_interface_edge_density": record["occupancy_interface_edge_density"],
            "exact_state_boundary_edge_density": record["exact_state_boundary_edge_density"],
            "normalized_state_entropy": record["normalized_state_entropy"],
            "coarse_occupancy_entropy_normalized": record[
                "coarse_occupancy_entropy_normalized"
            ],
            "positive_fraction": record["positive_fraction"],
            "negative_fraction": record["negative_fraction"],
            "bound_neutral_fraction": record["bound_neutral_fraction"],
            "charged_fraction": record["charged_fraction"],
            "total_signed_charge": record["total_signed_charge"],
        }

    forward = [projected(record, int(record["tick"])) for record in ordered]
    reverse = [
        projected(record, total_phases - int(record["tick"]))
        for record in reversed(ordered)
    ]
    return {
        "schema": "zwslab-reversible-macro-arrow-v1",
        "phases": total_phases,
        "exact_reverse_verified_at_all_retained_boundaries": True,
        "checkpoint_count": len(ordered),
        "capture_ticks": select_capture_ticks(ordered),
        "forward": forward,
        "reverse": reverse,
        "interpretation_boundary": (
            "The reverse series is justified by byte-exact checkpoint recovery. It demonstrates "
            "microscopic reversibility with a history-dependent coarse projection; it is not, by "
            "itself, a thermodynamic arrow."
        ),
    }


def coarse_state_rgb(state: torch.Tensor, block: int = 8) -> np.ndarray:
    """Render block-averaged exact-state composition with a fixed four-state palette."""
    state = _single_state(state)
    height, width = state.shape
    if block < 1 or height % block or width % block:
        raise ValueError("render block must divide both world dimensions")
    palette = torch.tensor(
        [
            [0.02, 0.025, 0.04],
            [0.0, 0.75, 1.0],
            [1.0, 0.15, 0.55],
            [1.0, 0.82, 0.10],
        ],
        dtype=torch.float32,
        device=state.device,
    )
    fractions = []
    for code in range(4):
        fractions.append(
            (state == code)
            .to(torch.float32)
            .reshape(height // block, block, width // block, block)
            .mean(dim=(1, 3))
        )
    mixture = torch.stack(fractions, dim=-1) @ palette
    return mixture.clamp(0.0, 1.0).detach().cpu().numpy()
