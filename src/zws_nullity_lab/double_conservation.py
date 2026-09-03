from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET
from .rules import ReversibleRule


STATE_MASS = torch.tensor([0, 1, 1, 1], dtype=torch.int64)


def block_mass(alphabet: Alphabet = DEFAULT_ALPHABET) -> torch.Tensor:
    return STATE_MASS[alphabet.block_digits_cpu].sum(dim=-1)


def joint_conservation_sectors(
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> dict[tuple[int, int], torch.Tensor]:
    charges = alphabet.block_charge_cpu
    masses = block_mass(alphabet)
    keys = sorted({(int(charges[index]), int(masses[index])) for index in range(256)})
    return {
        key: torch.nonzero((charges == key[0]) & (masses == key[1]), as_tuple=False).flatten()
        for key in keys
    }


def joint_rule_space_log10(alphabet: Alphabet = DEFAULT_ALPHABET) -> float:
    return sum(
        math.lgamma(int(indices.numel()) + 1) / math.log(10.0)
        for indices in joint_conservation_sectors(alphabet).values()
    )


@dataclass(frozen=True)
class DoubleConservationAudit:
    permutation: bool
    charge_preserved: bool
    occupancy_mass_preserved: bool
    valid: bool
    charge_violation_inputs: tuple[int, ...]
    occupancy_mass_violation_inputs: tuple[int, ...]
    joint_sector_sizes: dict[str, int]
    joint_rule_space_log10: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_double_conservation(rule: ReversibleRule) -> DoubleConservationAudit:
    charges = rule.alphabet.block_charge_cpu
    masses = block_mass(rule.alphabet)
    charge_bad = torch.nonzero(charges != charges[rule.table], as_tuple=False).flatten()
    mass_bad = torch.nonzero(masses != masses[rule.table], as_tuple=False).flatten()
    permutation = rule.is_permutation()
    charge_preserved = charge_bad.numel() == 0
    mass_preserved = mass_bad.numel() == 0
    sectors = joint_conservation_sectors(rule.alphabet)
    return DoubleConservationAudit(
        permutation=permutation,
        charge_preserved=charge_preserved,
        occupancy_mass_preserved=mass_preserved,
        valid=permutation and charge_preserved and mass_preserved,
        charge_violation_inputs=tuple(int(value) for value in charge_bad.tolist()),
        occupancy_mass_violation_inputs=tuple(int(value) for value in mass_bad.tolist()),
        joint_sector_sizes={f"Q{charge:+d}_M{mass}": int(indices.numel()) for (charge, mass), indices in sectors.items()},
        joint_rule_space_log10=joint_rule_space_log10(rule.alphabet),
    )


def assert_double_conserving(rule: ReversibleRule) -> None:
    audit = audit_double_conservation(rule)
    if not audit.valid:
        raise ValueError(
            "rule is not a joint Q/M sector permutation: "
            f"permutation={audit.permutation}, Q={audit.charge_preserved}, "
            f"M={audit.occupancy_mass_preserved}, "
            f"first Q violation={audit.charge_violation_inputs[:1]}, "
            f"first M violation={audit.occupancy_mass_violation_inputs[:1]}"
        )


def _optimal_twelve_swaps(
    winner_support: set[int],
    *,
    rotation: ReversibleRule,
) -> tuple[list[tuple[int, int]], int]:
    """Maximize winner-support overlap among 12 disjoint reactive joint-sector swaps."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    histograms = rotation.alphabet.block_histogram_cpu
    candidates: list[tuple[int, int]] = []
    for indices in joint_conservation_sectors(rotation.alphabet).values():
        values = indices.tolist()
        for offset, first in enumerate(values):
            for second in values[offset + 1 :]:
                if bool((histograms[first] != histograms[second]).any()):
                    candidates.append((first, second))
    candidates.sort()
    edge_count = len(candidates)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for edge_index, (first, second) in enumerate(candidates):
        rows.extend((first, second, 256))
        columns.extend((edge_index, edge_index, edge_index))
        data.extend((1.0, 1.0, 1.0))
    constraints = coo_matrix((data, (rows, columns)), shape=(257, edge_count)).tocsr()
    lower = np.concatenate([np.zeros(256), np.asarray([12.0])])
    upper = np.concatenate([np.ones(256), np.asarray([12.0])])
    overlap = np.asarray(
        [int(first in winner_support) + int(second in winner_support) for first, second in candidates],
        dtype=np.float64,
    )
    rank = np.arange(edge_count, dtype=np.float64)
    dominance = float(13 * edge_count + 1)
    objective = -dominance * overlap + rank
    result = milp(
        c=objective,
        integrality=np.ones(edge_count),
        bounds=Bounds(np.zeros(edge_count), np.ones(edge_count)),
        constraints=LinearConstraint(constraints, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"joint-sector swap optimization failed: {result.message}")
    selected = [candidates[index] for index, value in enumerate(result.x) if value > 0.5]
    if len(selected) != 12:
        raise RuntimeError(f"optimizer selected {len(selected)} swaps instead of 12")
    overlap_count = len({value for pair in selected for value in pair} & winner_support)
    return selected, overlap_count


def closest_double_conserving_sibling(
    winner: ReversibleRule,
    *,
    name: str = "winner_support_qm_sibling",
) -> ReversibleRule:
    """Find the closest 12-swap reactive-rotation sibling under exact Q/M sectors.

    Every winner table entry that differs from 90-degree rotation changes M and is therefore
    forbidden in the joint-conserving family. Keeping all 24 input locations is infeasible because
    some joint-sector subsets contain only one state histogram and cannot react with one another.
    A binary integer program therefore chooses exactly 12 disjoint reactive image swaps while
    maximizing overlap with the winner's 24 input locations. This is optimal inside the matched
    12-swap reactive-rotation architecture used by the winner and controls.
    """
    rotation = ReversibleRule.spatial_rotation(quarter_turns=1)
    winner_support_list = torch.nonzero(
        winner.table != rotation.table, as_tuple=False
    ).flatten().tolist()
    if len(winner_support_list) != 24:
        raise ValueError(f"expected winner support of 24 inputs, found {len(winner_support_list)}")
    winner_support = set(winner_support_list)
    pairs, overlap_count = _optimal_twelve_swaps(winner_support, rotation=rotation)
    table = rotation.table.clone()
    for first, second in pairs:
        table[first], table[second] = table[second].clone(), table[first].clone()

    rule = ReversibleRule(
        table,
        name=name,
        metadata={
            "construction": "optimal_joint_QM_12_swap_sibling",
            "mass_definition": "count(states != empty)",
            "winner_name": winner.name,
            "winner_reactive_support": winner_support_list,
            "swap_input_pairs": [list(pair) for pair in pairs],
            "reactive_support_overlap_with_winner": overlap_count,
            "reactive_support_outside_winner": 24 - overlap_count,
            "optimization": (
                "binary integer program; exactly 12 disjoint reactive joint-sector swaps; "
                "maximize winner-support overlap; minimize sum of sorted candidate ranks as tie-break"
            ),
            "optimal_hamming_distance_from_winner_in_12_swap_family": 48 - overlap_count,
        },
    )
    assert_double_conserving(rule)
    observed_support = torch.nonzero(rule.table != rotation.table, as_tuple=False).flatten().tolist()
    if int(rule.reaction_mask().sum().item()) != 24:
        raise AssertionError("closest sibling must have exactly 24 reactive inputs")
    if len(observed_support) != 24:
        raise AssertionError("closest sibling must differ from rotation at exactly 24 inputs")
    expected_distance = 48 - overlap_count
    if rule.hamming_distance(winner) != expected_distance:
        raise AssertionError("closest sibling did not attain the optimized 12-swap distance")
    return rule


def matched_double_conserving_rotation(
    *,
    seed: int,
    index: int,
    swaps: int = 12,
) -> ReversibleRule:
    """Create an independent reactive rotation with exact local Q and M conservation."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rotation = ReversibleRule.spatial_rotation(quarter_turns=1)
    histograms = rotation.alphabet.block_histogram_cpu
    candidates: list[tuple[int, int]] = []
    for indices in joint_conservation_sectors(rotation.alphabet).values():
        values = indices.tolist()
        for offset, first in enumerate(values):
            for second in values[offset + 1 :]:
                if bool((histograms[first] != histograms[second]).any()):
                    candidates.append((first, second))
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for candidate_index in torch.randperm(len(candidates), generator=generator).tolist():
        first, second = candidates[candidate_index]
        if first in used or second in used:
            continue
        pairs.append((first, second))
        used.update((first, second))
        if len(pairs) == swaps:
            break
    if len(pairs) != swaps:
        raise RuntimeError(f"constructed only {len(pairs)} of {swaps} requested joint-sector swaps")

    table = rotation.table.clone()
    for first, second in pairs:
        table[first], table[second] = table[second].clone(), table[first].clone()
    rule = ReversibleRule(
        table,
        name=f"matched_double_conserving_rotation_{index:02d}",
        metadata={
            "construction": "independent_matched_double_conserving_rotation",
            "mass_definition": "count(states != empty)",
            "control_index": index,
            "generation_seed": seed,
            "quarter_turns_clockwise": 1,
            "swap_input_pairs": [list(pair) for pair in pairs],
            "swaps": swaps,
        },
    )
    assert_double_conserving(rule)
    if int(rule.reaction_mask().sum().item()) != 2 * swaps:
        raise AssertionError("matched Q/M control has the wrong reactive-input count")
    if rule.hamming_distance(rotation) != 2 * swaps:
        raise AssertionError("matched Q/M control has the wrong rotation distance")
    return rule
