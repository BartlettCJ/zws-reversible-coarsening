from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET
from .rule import ReversibleRule


_POWERS = torch.tensor([1, 4, 16, 64], dtype=torch.int64)


def encode_blocks(cells: torch.Tensor) -> torch.Tensor:
    if cells.shape[-1] != 4:
        raise ValueError(f"expected final dimension 4, got {tuple(cells.shape)}")
    return (cells.long() * _POWERS.to(cells.device)).sum(dim=-1)


def decode_codes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.long()
    return torch.stack(
        [codes % 4, (codes // 4) % 4, (codes // 16) % 4, (codes // 64) % 4],
        dim=-1,
    )


def to_blocks(state: torch.Tensor, phase: int) -> tuple[torch.Tensor, int]:
    if state.ndim != 3:
        raise ValueError(f"state must be [B,H,W], got {tuple(state.shape)}")
    batch, height, width = state.shape
    if height % 2 or width % 2:
        raise ValueError("Margolus lattices require even height and width")
    offset = phase & 1
    aligned = torch.roll(state, shifts=(-offset, -offset), dims=(-2, -1))
    blocks = (
        aligned.reshape(batch, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, height // 2, width // 2, 4)
    )
    return blocks, offset


def from_blocks(blocks: torch.Tensor, offset: int, height: int, width: int) -> torch.Tensor:
    batch = blocks.shape[0]
    aligned = (
        blocks.reshape(batch, height // 2, width // 2, 2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, height, width)
    )
    return torch.roll(aligned, shifts=(offset, offset), dims=(-2, -1))


def step_batched(
    state: torch.Tensor,
    rule_tables: torch.Tensor,
    phase: int,
    *,
    return_codes: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one alternating Margolus phase.

    `rule_tables` may be [256], [1,256], or [B,256].  When requested, input and
    output block codes are returned for reaction-flux accounting.
    """
    if state.ndim != 3:
        raise ValueError("state must be [B,H,W]")
    batch, height, width = state.shape
    table = torch.as_tensor(rule_tables, dtype=torch.int64, device=state.device)
    if table.ndim == 1:
        table = table.unsqueeze(0)
    if table.shape[0] == 1 and batch > 1:
        table = table.expand(batch, -1)
    if table.shape != (batch, 256):
        raise ValueError(f"rule_tables must be [B,256], got {tuple(table.shape)}")

    blocks, offset = to_blocks(state, phase)
    in_codes = encode_blocks(blocks)
    out_codes = table.gather(1, in_codes.reshape(batch, -1)).reshape_as(in_codes)
    out_blocks = decode_codes(out_codes)
    next_state = from_blocks(out_blocks, offset, height, width).to(state.dtype)
    if return_codes:
        return next_state, in_codes, out_codes
    return next_state


def step_rule(state: torch.Tensor, rule: ReversibleRule, phase: int) -> torch.Tensor:
    batch_state = state.unsqueeze(0) if state.ndim == 2 else state
    result = step_batched(batch_state, rule.table.to(batch_state.device), phase)
    assert isinstance(result, torch.Tensor)
    return result[0] if state.ndim == 2 else result


def reverse_steps(state: torch.Tensor, rule: ReversibleRule, steps: int, *, start_phase: int = 0) -> torch.Tensor:
    current = state.unsqueeze(0) if state.ndim == 2 else state.clone()
    inv = rule.inverse_table.to(current.device)
    for t in reversed(range(steps)):
        result = step_batched(current, inv, start_phase + t)
        assert isinstance(result, torch.Tensor)
        current = result
    return current[0] if state.ndim == 2 else current


@dataclass(frozen=True)
class Sample:
    phase: int
    state: torch.Tensor
    input_histogram: torch.Tensor | None = None
    output_histogram: torch.Tensor | None = None


def sample_trajectory(
    initial: torch.Tensor,
    rule: ReversibleRule,
    *,
    steps: int,
    sample_every: int,
    start_phase: int = 0,
    include_initial: bool = True,
    reaction_histograms: bool = False,
) -> Iterator[Sample]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    state = initial.unsqueeze(0) if initial.ndim == 2 else initial.clone()
    table = rule.table.to(state.device)
    if include_initial:
        yield Sample(start_phase, state.clone())
    accum_in = torch.zeros((state.shape[0], 256), dtype=torch.int64, device=state.device)
    accum_out = torch.zeros_like(accum_in)
    for t in range(steps):
        result = step_batched(state, table, start_phase + t, return_codes=reaction_histograms)
        if reaction_histograms:
            assert isinstance(result, tuple)
            state, in_codes, out_codes = result
            for b in range(state.shape[0]):
                accum_in[b] += torch.bincount(in_codes[b].reshape(-1), minlength=256)
                accum_out[b] += torch.bincount(out_codes[b].reshape(-1), minlength=256)
        else:
            assert isinstance(result, torch.Tensor)
            state = result
        phase = start_phase + t + 1
        if phase % sample_every == 0 or t == steps - 1:
            yield Sample(
                phase=phase,
                state=state.clone(),
                input_histogram=accum_in.clone() if reaction_histograms else None,
                output_histogram=accum_out.clone() if reaction_histograms else None,
            )
            accum_in.zero_()
            accum_out.zero_()


@dataclass(frozen=True)
class ExactAudit:
    initial_q: tuple[int, ...]
    final_q: tuple[int, ...]
    initial_m: tuple[int, ...]
    final_m: tuple[int, ...]
    charge_preserved: bool
    occupancy_preserved: bool
    reverse_equal: bool


def audit_run(
    initial: torch.Tensor,
    rule: ReversibleRule,
    *,
    steps: int,
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> ExactAudit:
    state = initial.unsqueeze(0) if initial.ndim == 2 else initial.clone()
    q0 = alphabet.total_charge(state)
    m0 = alphabet.total_occupancy(state)
    current = state
    table = rule.table.to(current.device)
    for t in range(steps):
        result = step_batched(current, table, t)
        assert isinstance(result, torch.Tensor)
        current = result
    q1 = alphabet.total_charge(current)
    m1 = alphabet.total_occupancy(current)
    recovered = reverse_steps(current, rule, steps)
    if recovered.ndim == 2:
        recovered = recovered.unsqueeze(0)
    return ExactAudit(
        initial_q=tuple(int(x) for x in q0.detach().cpu().reshape(-1).tolist()),
        final_q=tuple(int(x) for x in q1.detach().cpu().reshape(-1).tolist()),
        initial_m=tuple(int(x) for x in m0.detach().cpu().reshape(-1).tolist()),
        final_m=tuple(int(x) for x in m1.detach().cpu().reshape(-1).tolist()),
        charge_preserved=bool(torch.equal(q0, q1)),
        occupancy_preserved=bool(torch.equal(m0, m1)),
        reverse_equal=bool(torch.equal(state.detach().cpu(), recovered.detach().cpu())),
    )
