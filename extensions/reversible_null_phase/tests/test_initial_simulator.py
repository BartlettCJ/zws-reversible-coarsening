from __future__ import annotations

import torch

from zws_reversible_null_phase.alphabet import DEFAULT_ALPHABET
from zws_reversible_null_phase.initial import inventory_for_density, make_state, matched_arrangements, state_sha256
from zws_reversible_null_phase.rule import ReversibleRule
from zws_reversible_null_phase.simulator import audit_run, reverse_steps, step_rule


def test_exact_inventory_and_matched_arrangements() -> None:
    inventory = inventory_for_density(64, 64, 0.275)
    states = matched_arrangements(64, 64, 0.275, seed=123)
    hashes = set()
    for state in states.values():
        assert tuple(DEFAULT_ALPHABET.state_counts(state).tolist()) == inventory.as_tuple()
        assert int(DEFAULT_ALPHABET.total_charge(state)) == 0
        hashes.add(state_sha256(state))
    assert len(hashes) == len(states)


def test_forward_reverse_exact(winner: ReversibleRule) -> None:
    initial = make_state(32, 32, inventory_for_density(32, 32, 0.28), seed=7)
    state = initial.clone()
    for phase in range(64):
        state = step_rule(state, winner, phase)
    recovered = reverse_steps(state, winner, 64)
    assert torch.equal(initial, recovered)
    audit = audit_run(initial, winner, steps=64)
    assert audit.charge_preserved
    assert audit.reverse_equal
    assert not audit.occupancy_preserved or audit.initial_m == audit.final_m
