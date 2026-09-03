from __future__ import annotations

import torch

from zws_nullity_lab.rules import ReversibleRule


def test_standard_rules_validate() -> None:
    assert ReversibleRule.identity().is_valid()
    for k in range(4):
        assert ReversibleRule.spatial_rotation(k).is_valid()
    reactive = ReversibleRule.reactive_rotation(12)
    assert reactive.is_valid()
    assert reactive.reaction_fraction() > 0.0


def test_random_rule_is_sector_permutation() -> None:
    generator = torch.Generator().manual_seed(123)
    rule = ReversibleRule.random(generator)
    assert rule.is_permutation()
    assert rule.preserves_charge()


def test_unconstrained_control_breaks_charge_for_fixed_seed() -> None:
    generator = torch.Generator().manual_seed(123)
    rule = ReversibleRule.random_unconstrained(generator)
    assert rule.is_permutation()
    assert not rule.preserves_charge()


def test_mutation_preserves_validity() -> None:
    generator = torch.Generator().manual_seed(17)
    parent = ReversibleRule.reactive_rotation(8)
    child = parent.mutated(100, generator)
    assert child.is_valid()
    assert child.hamming_distance(parent) > 0


def test_inverse_is_exact_permutation_inverse() -> None:
    rule = ReversibleRule.random(torch.Generator().manual_seed(2))
    index = torch.arange(256)
    assert torch.equal(rule.inverse_table[rule.table], index)
    assert torch.equal(rule.table[rule.inverse_table], index)


def test_state_conjugation_is_valid_for_declared_permutations() -> None:
    rule = ReversibleRule.reactive_rotation(16)
    assert rule.conjugated([3, 1, 2, 0]).is_valid()
    assert rule.conjugated([0, 2, 1, 3]).is_valid()


def test_simple_baseline_distance() -> None:
    assert ReversibleRule.identity().simple_baseline_distance() == 0.0
    assert ReversibleRule.spatial_rotation(1).simple_baseline_distance() == 0.0
    assert ReversibleRule.reactive_rotation(8).simple_baseline_distance() > 0.0


def test_physical_symmetry_is_distinct_from_conjugacy() -> None:
    symmetric = ReversibleRule.identity()
    reactive = ReversibleRule.reactive_rotation(12)
    assert symmetric.sign_asymmetry() == 0.0
    assert 0.0 <= reactive.sign_asymmetry() <= 1.0
    # Conjugating is valid regardless of whether the original rule commutes.
    assert reactive.conjugated([0, 2, 1, 3]).is_valid()
