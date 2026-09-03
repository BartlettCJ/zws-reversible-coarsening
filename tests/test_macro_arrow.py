from __future__ import annotations

import pytest
import torch

from zws_nullity_lab.macro_arrow import (
    checkpoint_macro_observables,
    coarse_state_rgb,
    select_capture_ticks,
    synchronized_forward_reverse,
)


def _record(tick: int, scale: float, digest: str) -> dict[str, object]:
    return {
        "tick": tick,
        "exact_state_sha256_uint8": digest,
        "occupied_fraction": 0.2,
        "finite_domain_scale": scale,
        "occupancy_interface_edge_density": 0.1,
        "exact_state_boundary_edge_density": 0.2,
        "normalized_state_entropy": 0.3,
        "coarse_occupancy_entropy_normalized": 0.4,
        "positive_fraction": 0.08,
        "negative_fraction": 0.08,
        "bound_neutral_fraction": 0.04,
        "charged_fraction": 0.16,
        "total_signed_charge": 0,
    }


def test_checkpoint_macro_observables_report_exact_q_and_m() -> None:
    state = torch.zeros((64, 64), dtype=torch.uint8)
    state[16:48, 16:48] = 3
    state[20:28, 20:28] = 1
    state[36:44, 36:44] = 2
    result = checkpoint_macro_observables(state, tick=2048)
    assert result["world"] == [64, 64]
    assert result["total_signed_charge"] == 0
    assert result["occupied_fraction"] == pytest.approx(1024 / 4096)
    assert result["occupancy_interface_edge_density"] > 0
    assert result["normalized_state_entropy"] > 0


def test_capture_tick_uses_earliest_exact_peak() -> None:
    records = [_record(0, 1.0, "a"), _record(8, 9.0, "b"), _record(16, 9.0, "c")]
    assert select_capture_ticks(records) == {
        "initial": 0,
        "peak_organisation": 8,
        "final": 16,
    }


def test_synchronized_reverse_is_exact_mirror_of_forward() -> None:
    records = [_record(0, 1.0, "a"), _record(8, 9.0, "b"), _record(16, 3.0, "c")]
    audit = {
        "phases_reversed": 16,
        "exact_at_every_checkpoint_boundary": True,
        "boundaries": [
            {
                "tick": 8,
                "exact_state_equal": True,
                "mismatch_cells": 0,
                "reference_state_sha256_uint8": "b",
            },
            {
                "tick": 0,
                "exact_state_equal": True,
                "mismatch_cells": 0,
                "reference_state_sha256_uint8": "a",
            },
        ],
    }
    result = synchronized_forward_reverse(records, audit)
    assert [row["microscopic_phase"] for row in result["forward"]] == [0, 8, 16]
    assert [row["microscopic_phase"] for row in result["reverse"]] == [16, 8, 0]
    assert [row["traversal_phase"] for row in result["reverse"]] == [0, 8, 16]
    assert result["capture_ticks"]["peak_organisation"] == 8


def test_coarse_state_rgb_preserves_fixed_palette_endpoints() -> None:
    state = torch.zeros((8, 16), dtype=torch.uint8)
    state[:, 8:] = 1
    image = coarse_state_rgb(state, block=8)
    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == pytest.approx([0.02, 0.025, 0.04])
    assert image[0, 1].tolist() == pytest.approx([0.0, 0.75, 1.0])
