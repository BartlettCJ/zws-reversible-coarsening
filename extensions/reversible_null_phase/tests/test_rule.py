from __future__ import annotations

from zws_reversible_null_phase.rule import ReversibleRule


def test_frozen_rule_identity_and_pairs(winner: ReversibleRule) -> None:
    assert winner.sha256() == "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
    audit = winner.audit()
    assert audit.permutation
    assert audit.charge_preserving
    assert not audit.occupancy_preserving
    assert audit.differing_from_rotation == 24
    assert len(winner.reactive_pairs_relative_to_rotation()) == 12


def test_single_pair_ablation_remains_exact(winner: ReversibleRule) -> None:
    ablated = winner.ablate_pairs([0])
    audit = ablated.audit()
    assert audit.permutation
    assert audit.charge_preserving
    assert audit.differing_from_rotation == 22


def test_qm_sibling_is_reversible_and_double_conserving(winner: ReversibleRule) -> None:
    sibling = winner.closest_qm_preserving_sibling()
    audit = sibling.audit()
    assert audit.permutation
    assert audit.charge_preserving
    assert audit.occupancy_preserving
    assert audit.differing_from_rotation == 24
    assert sibling.sha256() == "ad8c2ad901201e2647c8734563dcca93258e15877743e7c82a7351bd08a72d01"
    assert sibling.metadata["entries_identical_to_source"] == 211
    assert sibling.metadata["entries_identical_to_source"] >= 200
