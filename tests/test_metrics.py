from __future__ import annotations

import torch

from zws_nullity_lab.initial import balanced_random, perturb_charge_preserving
from zws_nullity_lab.metrics import (
    block_entropy,
    evaluate_population,
    spatial_mutual_information,
    state_entropy,
)
from zws_nullity_lab.rules import ReversibleRule, stack_rules


def test_entropy_bounds() -> None:
    state = balanced_random(4, 16, 16, seed=4)
    for metric in (state_entropy(state), block_entropy(state), spatial_mutual_information(state)):
        assert bool(torch.isfinite(metric).all())
        assert bool((metric >= -1e-6).all())
        assert bool((metric <= 1.0 + 1e-6).all())


def test_static_identity_is_dynamics_gated() -> None:
    initial = balanced_random(2, 16, 16, seed=3)
    perturbed = perturb_charge_preserving(initial, seed=4)
    rules = [ReversibleRule.identity(), ReversibleRule.reactive_rotation(12)]
    metrics = evaluate_population(
        initial,
        perturbed,
        stack_rules(rules),
        steps=24,
        burn_in=4,
        sample_every=2,
    )
    assert float(metrics.activity[0]) == 0.0
    assert float(metrics.reaction[0]) == 0.0
    assert float(metrics.quality[0]) < float(metrics.quality[1])


def test_invariant_breaking_control_is_rejected() -> None:
    initial = balanced_random(1, 16, 16, seed=5)
    perturbed = perturb_charge_preserving(initial, seed=6)
    rule = ReversibleRule.random_unconstrained(torch.Generator().manual_seed(123))
    metrics = evaluate_population(
        initial,
        perturbed,
        rule.table.unsqueeze(0),
        steps=12,
        burn_in=2,
        sample_every=2,
    )
    assert not bool(metrics.charge_preserved[0])
    assert float(metrics.quality[0]) < -1e8
