from __future__ import annotations

import torch

from zws_nullity_lab.alphabet import DEFAULT_ALPHABET
from zws_nullity_lab.double_conservation import (
    assert_double_conserving,
    audit_double_conservation,
    block_mass,
    closest_double_conserving_sibling,
    joint_conservation_sectors,
    joint_rule_space_log10,
    matched_double_conserving_rotation,
)
from zws_nullity_lab.domain_mechanism import exact_occupancy_mass
from zws_nullity_lab.margolus import reverse_trajectory, run_trajectory
from zws_nullity_lab.rules import ReversibleRule


def _winner() -> ReversibleRule:
    return ReversibleRule.load(
        "experiments/phase_separation_confirmation_20260825/rules/winner.json"
    )


def test_joint_sector_partition_and_rule_space() -> None:
    sectors = joint_conservation_sectors()
    assert len(sectors) == 25
    assert sum(int(indices.numel()) for indices in sectors.values()) == 256
    assert int(sectors[(0, 4)].numel()) == 19
    assert int(sectors[(-2, 3)].numel()) == 12
    assert torch.equal(block_mass(), (DEFAULT_ALPHABET.block_digits_cpu != 0).sum(dim=1))
    assert abs(joint_rule_space_log10() - 199.81924459859437) < 1e-12


def test_winner_changes_mass_on_all_24_reactive_inputs() -> None:
    winner = _winner()
    audit = audit_double_conservation(winner)
    assert audit.permutation is True
    assert audit.charge_preserved is True
    assert audit.occupancy_mass_preserved is False
    assert len(audit.occupancy_mass_violation_inputs) == 24
    assert set(audit.occupancy_mass_violation_inputs) == set(
        torch.nonzero(winner.reaction_mask(), as_tuple=False).flatten().tolist()
    )


def test_closest_sibling_is_optimal_in_matched_twelve_swap_family() -> None:
    winner = _winner()
    sibling = closest_double_conserving_sibling(winner)
    rotation = ReversibleRule.spatial_rotation()
    assert_double_conserving(sibling)
    assert int(sibling.reaction_mask().sum().item()) == 24
    assert sibling.hamming_distance(rotation) == 24
    assert sibling.metadata["reactive_support_overlap_with_winner"] == 3
    assert sibling.metadata["reactive_support_outside_winner"] == 21
    assert sibling.hamming_distance(winner) == 45
    assert sibling.metadata["optimal_hamming_distance_from_winner_in_12_swap_family"] == 45
    assert_double_conserving(sibling.inverse())


def test_matched_double_conserving_controls_are_exact_and_reactive() -> None:
    first = matched_double_conserving_rotation(seed=1234, index=0)
    second = matched_double_conserving_rotation(seed=5678, index=1)
    rotation = ReversibleRule.spatial_rotation()
    assert_double_conserving(first)
    assert_double_conserving(second)
    assert int(first.reaction_mask().sum().item()) == 24
    assert first.hamming_distance(rotation) == 24
    assert not torch.equal(first.table, second.table)


def test_q_and_m_are_exact_through_forward_and_inverse_trace() -> None:
    rule = closest_double_conserving_sibling(_winner())
    generator = torch.Generator().manual_seed(42)
    initial = torch.randint(0, 4, (2, 16, 16), generator=generator, dtype=torch.uint8)
    trajectory = run_trajectory(initial, rule, 17)
    initial_q = DEFAULT_ALPHABET.total_charge(initial)
    initial_m = exact_occupancy_mass(initial)
    for state in trajectory:
        assert torch.equal(DEFAULT_ALPHABET.total_charge(state), initial_q)
        assert torch.equal(exact_occupancy_mass(state), initial_m)
    recovered = reverse_trajectory(trajectory[-1], rule, 17)
    assert torch.equal(recovered, initial)
