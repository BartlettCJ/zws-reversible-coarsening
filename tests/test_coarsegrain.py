from __future__ import annotations

import torch

from zws_nullity_lab.coarsegrain import audit_coarse_graining, coarse_charge
from zws_nullity_lab.initial import balanced_random
from zws_nullity_lab.margolus import run_trajectory
from zws_nullity_lab.rules import ReversibleRule


def test_coarse_charge_matches_micro_total() -> None:
    state = balanced_random(3, 16, 16, seed=3)
    q2 = coarse_charge(state, 2)
    q4 = coarse_charge(state, 4)
    assert torch.equal(q2.sum(dim=(-2, -1)), q4.sum(dim=(-2, -1)))
    composed = q2.reshape(3, 4, 2, 4, 2).sum(dim=(2, 4))
    assert torch.equal(composed, q4)


def test_coarse_audit_reports_exact_composition() -> None:
    state = balanced_random(1, 16, 16, seed=7)
    trajectory = run_trajectory(state, ReversibleRule.reactive_rotation(12), 30)
    audit = audit_coarse_graining(trajectory, (2, 4))
    assert audit.passed_additive_composition
    assert audit.levels[1].direct_composition_exact is True
    assert 0.0 <= audit.levels[0].majority_prediction_accuracy <= 1.0
