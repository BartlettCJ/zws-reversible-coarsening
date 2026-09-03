from __future__ import annotations

import torch

from zws_nullity_lab.alphabet import DEFAULT_ALPHABET
from zws_nullity_lab.initial import balanced_random
from zws_nullity_lab.margolus import (
    audit_trace,
    decode_codes,
    encode_blocks,
    reverse_trajectory,
    run_trajectory,
    step_rule,
)
from zws_nullity_lab.rules import ReversibleRule


def test_encode_decode_all_codes() -> None:
    codes = torch.arange(256)
    assert torch.equal(encode_blocks(decode_codes(codes)), codes)


def test_identity_step_is_identity_both_phases() -> None:
    state = balanced_random(2, 12, 16, seed=5)
    rule = ReversibleRule.identity()
    assert torch.equal(step_rule(state, rule, 0), state)
    assert torch.equal(step_rule(state, rule, 1), state)


def test_charge_preserved_for_long_random_run() -> None:
    state = balanced_random(3, 24, 24, seed=5)
    rule = ReversibleRule.random(torch.Generator().manual_seed(7))
    trajectory = run_trajectory(state, rule, 150)
    charge = DEFAULT_ALPHABET.total_charge(trajectory.flatten(0, 1))
    assert bool((charge == 0).all())


def test_forward_inverse_reconstruction() -> None:
    initial = balanced_random(2, 18, 20, seed=23)
    rule = ReversibleRule.reactive_rotation(24)
    trajectory = run_trajectory(initial, rule, 79, start_phase=1)
    recovered = reverse_trajectory(trajectory[-1], rule, 79, start_phase=1)
    assert torch.equal(recovered, initial)


def test_trace_audit() -> None:
    initial = balanced_random(1, 16, 16, seed=41)
    audit = audit_trace(initial, ReversibleRule.reactive_rotation(12), 50)
    assert audit.charge_preserved
    assert audit.reversible
    assert audit.state_equal_after_reverse
