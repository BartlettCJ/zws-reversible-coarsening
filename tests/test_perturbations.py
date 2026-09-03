from __future__ import annotations

import pytest
import torch

from zws_nullity_lab.perturbations import (
    apply_perturbation,
    balanced_creation,
    balanced_removal,
    count_preserving_state_swaps,
)


def _balanced_state() -> torch.Tensor:
    state = torch.zeros((64, 64), dtype=torch.uint8)
    state[8:24, 8:24] = 1
    state[24:40, 8:24] = 2
    state[40:56, 8:24] = 3
    return state


def test_balanced_creation_preserves_q_and_increases_exact_mass() -> None:
    state = _balanced_state()
    result, audit = balanced_creation(
        state,
        origin=(0, 0),
        patch_size=64,
        changed_sites=32,
        seed=123,
    )
    assert audit.charge_preserved is True
    assert audit.mass_preserved is False
    assert audit.mass_after - audit.mass_before == 32
    assert audit.observed_changed_sites == 32
    assert torch.equal(result, balanced_creation(
        state,
        origin=(0, 0),
        patch_size=64,
        changed_sites=32,
        seed=123,
    )[0])


def test_balanced_removal_preserves_q_and_decreases_exact_mass() -> None:
    state = _balanced_state()
    _, audit = balanced_removal(
        state,
        origin=(0, 0),
        patch_size=64,
        changed_sites=32,
        seed=456,
    )
    assert audit.charge_preserved is True
    assert audit.mass_preserved is False
    assert audit.mass_before - audit.mass_after == 32
    assert audit.observed_changed_sites == 32


def test_qm_swaps_preserve_every_state_count_and_exact_amplitude() -> None:
    state = _balanced_state()
    _, audit = count_preserving_state_swaps(
        state,
        origin=(60, 60),
        patch_size=64,
        changed_sites=128,
        seed=789,
    )
    assert audit.charge_preserved is True
    assert audit.mass_preserved is True
    assert audit.state_counts_before == audit.state_counts_after
    assert audit.observed_changed_sites == 128


def test_dispatch_and_insufficient_amplitude_fail_loudly() -> None:
    state = _balanced_state()
    _, audit = apply_perturbation(
        state,
        kind="qm_state_swaps",
        origin=(0, 0),
        patch_size=64,
        changed_sites=16,
        seed=1,
    )
    assert audit.kind == "qm_state_swaps"
    with pytest.raises(ValueError, match="empty patch sites"):
        balanced_creation(
            torch.ones((16, 16), dtype=torch.uint8),
            origin=(0, 0),
            patch_size=16,
            changed_sites=16,
            seed=2,
        )
    with pytest.raises(ValueError, match="even"):
        count_preserving_state_swaps(
            state,
            origin=(0, 0),
            patch_size=64,
            changed_sites=15,
            seed=3,
        )
