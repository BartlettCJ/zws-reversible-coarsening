from __future__ import annotations

import torch

from zws_nullity_lab.alphabet import DEFAULT_ALPHABET
from zws_nullity_lab.initial import SeedSpec, balanced_random, paired_impulse, perturb_charge_preserving


def test_block_charge_sectors_cover_all_codes() -> None:
    sectors = DEFAULT_ALPHABET.charge_sectors
    covered = torch.cat(list(sectors.values())).sort().values
    assert torch.equal(covered, torch.arange(256))
    assert set(sectors) == set(range(-4, 5))


def test_two_dimensional_total_charge_is_a_scalar() -> None:
    state = torch.tensor([[1, 2], [1, 0]], dtype=torch.uint8)
    total = DEFAULT_ALPHABET.total_charge(state)
    assert total.ndim == 0
    assert int(total) == 1


def test_balanced_random_is_exactly_zero() -> None:
    state = balanced_random(7, 20, 24, spec=SeedSpec(0.19, 0.11), seed=19)
    assert state.dtype == torch.uint8
    assert torch.equal(DEFAULT_ALPHABET.total_charge(state), torch.zeros(7, dtype=torch.int64))


def test_localized_seed_is_zero_and_outside_empty() -> None:
    spec = SeedSpec(0.2, 0.1, localized=True, localized_fraction=0.5)
    state = balanced_random(1, 24, 24, spec=spec, seed=11)
    assert int(DEFAULT_ALPHABET.total_charge(state)[0]) == 0
    assert int(state[0, 0, 0]) == DEFAULT_ALPHABET.empty


def test_perturbations_preserve_charge_and_change_state() -> None:
    state = balanced_random(3, 16, 16, seed=9)
    for mode in ("pair-conversion", "swap"):
        changed = perturb_charge_preserving(state, seed=7, mode=mode)
        assert torch.equal(DEFAULT_ALPHABET.total_charge(state), DEFAULT_ALPHABET.total_charge(changed))
        assert bool((state != changed).any())


def test_paired_impulse_is_minimal_zero_state() -> None:
    state = paired_impulse(8, 8)
    assert int(DEFAULT_ALPHABET.total_charge(state.unsqueeze(0))[0]) == 0
    assert int((state != 0).sum()) == 2
