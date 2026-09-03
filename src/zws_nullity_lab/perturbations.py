from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from typing import Literal

import numpy as np
import torch


PerturbationKind = Literal[
    "q_balanced_creation",
    "q_balanced_removal",
    "qm_state_swaps",
]


@dataclass(frozen=True)
class PerturbationAudit:
    kind: str
    selection_seed: int
    patch_origin: tuple[int, int]
    patch_size: int
    requested_changed_sites: int
    observed_changed_sites: int
    charge_before: int
    charge_after: int
    mass_before: int
    mass_after: int
    state_counts_before: tuple[int, int, int, int]
    state_counts_after: tuple[int, int, int, int]
    charge_preserved: bool
    mass_preserved: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _counts(state: torch.Tensor) -> tuple[int, int, int, int]:
    values = torch.bincount(state.long().reshape(-1), minlength=4)
    return tuple(int(value) for value in values.tolist())  # type: ignore[return-value]


def _audit(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    kind: str,
    selection_seed: int,
    patch_origin: tuple[int, int],
    patch_size: int,
    requested_changed_sites: int,
) -> PerturbationAudit:
    before_counts = _counts(before)
    after_counts = _counts(after)
    q_before = before_counts[1] - before_counts[2]
    q_after = after_counts[1] - after_counts[2]
    m_before = sum(before_counts[1:])
    m_after = sum(after_counts[1:])
    return PerturbationAudit(
        kind=kind,
        selection_seed=selection_seed,
        patch_origin=patch_origin,
        patch_size=patch_size,
        requested_changed_sites=requested_changed_sites,
        observed_changed_sites=int((before != after).sum().item()),
        charge_before=q_before,
        charge_after=q_after,
        mass_before=m_before,
        mass_after=m_after,
        state_counts_before=before_counts,
        state_counts_after=after_counts,
        charge_preserved=q_before == q_after,
        mass_preserved=m_before == m_after,
    )


def _patch_coordinates(
    state: torch.Tensor,
    *,
    origin: tuple[int, int],
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.ndim != 2:
        raise ValueError("perturbations require one state with shape [height,width]")
    height, width = state.shape
    if patch_size < 1 or patch_size > min(height, width):
        raise ValueError("patch_size must fit within both world dimensions")
    row_values = (torch.arange(patch_size, device=state.device) + int(origin[0])) % height
    column_values = (torch.arange(patch_size, device=state.device) + int(origin[1])) % width
    rows = row_values[:, None].expand(patch_size, patch_size).reshape(-1)
    columns = column_values[None, :].expand(patch_size, patch_size).reshape(-1)
    patch = state[rows, columns].detach().cpu()
    return rows, columns, patch


def _validate_amplitude(changed_sites: int) -> None:
    if changed_sites < 2 or changed_sites % 2:
        raise ValueError("changed_sites must be a positive even number")


def balanced_creation(
    state: torch.Tensor,
    *,
    origin: tuple[int, int],
    patch_size: int,
    changed_sites: int,
    seed: int,
) -> tuple[torch.Tensor, PerturbationAudit]:
    """Create equal numbers of + and - sites, preserving Q but increasing M exactly."""
    _validate_amplitude(changed_sites)
    rows, columns, patch = _patch_coordinates(state, origin=origin, patch_size=patch_size)
    eligible = torch.nonzero(patch == 0, as_tuple=False).flatten().numpy()
    if eligible.size < changed_sites:
        raise ValueError(
            f"balanced creation needs {changed_sites} empty patch sites, found {eligible.size}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.permutation(eligible)[:changed_sites]
    half = changed_sites // 2
    result = state.clone()
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=state.device)
    result[rows[selected_tensor[:half]], columns[selected_tensor[:half]]] = 1
    result[rows[selected_tensor[half:]], columns[selected_tensor[half:]]] = 2
    audit = _audit(
        state,
        result,
        kind="q_balanced_creation",
        selection_seed=seed,
        patch_origin=origin,
        patch_size=patch_size,
        requested_changed_sites=changed_sites,
    )
    if not audit.charge_preserved or audit.mass_after - audit.mass_before != changed_sites:
        raise AssertionError("balanced creation failed its exact Q/M audit")
    return result, audit


def balanced_removal(
    state: torch.Tensor,
    *,
    origin: tuple[int, int],
    patch_size: int,
    changed_sites: int,
    seed: int,
) -> tuple[torch.Tensor, PerturbationAudit]:
    """Remove equal numbers of + and - sites, preserving Q but decreasing M exactly."""
    _validate_amplitude(changed_sites)
    rows, columns, patch = _patch_coordinates(state, origin=origin, patch_size=patch_size)
    half = changed_sites // 2
    positive = torch.nonzero(patch == 1, as_tuple=False).flatten().numpy()
    negative = torch.nonzero(patch == 2, as_tuple=False).flatten().numpy()
    if positive.size < half or negative.size < half:
        raise ValueError(
            "balanced removal lacks paired charges in patch: "
            f"need {half} each, found +{positive.size}/-{negative.size}"
        )
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [rng.permutation(positive)[:half], rng.permutation(negative)[:half]]
    )
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=state.device)
    result = state.clone()
    result[rows[selected_tensor], columns[selected_tensor]] = 0
    audit = _audit(
        state,
        result,
        kind="q_balanced_removal",
        selection_seed=seed,
        patch_origin=origin,
        patch_size=patch_size,
        requested_changed_sites=changed_sites,
    )
    if not audit.charge_preserved or audit.mass_before - audit.mass_after != changed_sites:
        raise AssertionError("balanced removal failed its exact Q/M audit")
    return result, audit


def count_preserving_state_swaps(
    state: torch.Tensor,
    *,
    origin: tuple[int, int],
    patch_size: int,
    changed_sites: int,
    seed: int,
) -> tuple[torch.Tensor, PerturbationAudit]:
    """Swap unlike patch sites pairwise, preserving every exact state count, Q, and M."""
    _validate_amplitude(changed_sites)
    rows, columns, patch = _patch_coordinates(state, origin=origin, patch_size=patch_size)
    pair_count = changed_sites // 2
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for state_code in range(4):
        indices = torch.nonzero(patch == state_code, as_tuple=False).flatten().numpy()
        groups[state_code] = [int(value) for value in rng.permutation(indices).tolist()]
    maximum_pairs = min(patch.numel() // 2, patch.numel() - max(len(group) for group in groups.values()))
    if pair_count > maximum_pairs:
        raise ValueError(
            f"Q+M swaps need {pair_count} unlike pairs, but patch permits {maximum_pairs}"
        )

    priorities = {state_code: float(rng.random()) for state_code in range(4)}
    heap: list[tuple[int, float, int, int]] = []
    for state_code, indices in groups.items():
        if indices:
            heapq.heappush(heap, (-len(indices), priorities[state_code], state_code, 0))
    pairs: list[tuple[int, int]] = []
    for _ in range(pair_count):
        if len(heap) < 2:
            raise AssertionError("unlike-pair construction exhausted unexpectedly")
        first_count, first_priority, first_code, first_offset = heapq.heappop(heap)
        second_count, second_priority, second_code, second_offset = heapq.heappop(heap)
        pairs.append((groups[first_code][first_offset], groups[second_code][second_offset]))
        first_count += 1
        second_count += 1
        if first_count < 0:
            heapq.heappush(
                heap,
                (first_count, first_priority, first_code, first_offset + 1),
            )
        if second_count < 0:
            heapq.heappush(
                heap,
                (second_count, second_priority, second_code, second_offset + 1),
            )

    result = state.clone()
    for first, second in pairs:
        first_value = result[rows[first], columns[first]].clone()
        result[rows[first], columns[first]] = result[rows[second], columns[second]]
        result[rows[second], columns[second]] = first_value
    audit = _audit(
        state,
        result,
        kind="qm_state_swaps",
        selection_seed=seed,
        patch_origin=origin,
        patch_size=patch_size,
        requested_changed_sites=changed_sites,
    )
    if (
        not audit.charge_preserved
        or not audit.mass_preserved
        or audit.state_counts_before != audit.state_counts_after
        or audit.observed_changed_sites != changed_sites
    ):
        raise AssertionError("Q+M swap perturbation failed its exact audit")
    return result, audit


def apply_perturbation(
    state: torch.Tensor,
    *,
    kind: PerturbationKind,
    origin: tuple[int, int],
    patch_size: int,
    changed_sites: int,
    seed: int,
) -> tuple[torch.Tensor, PerturbationAudit]:
    functions = {
        "q_balanced_creation": balanced_creation,
        "q_balanced_removal": balanced_removal,
        "qm_state_swaps": count_preserving_state_swaps,
    }
    return functions[kind](
        state,
        origin=origin,
        patch_size=patch_size,
        changed_sites=changed_sites,
        seed=seed,
    )
