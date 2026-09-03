from __future__ import annotations

from dataclasses import dataclass

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET
from .rules import ReversibleRule


_POWERS = torch.tensor([1, 4, 16, 64], dtype=torch.int64)


def encode_blocks(cells: torch.Tensor) -> torch.Tensor:
    """Encode row-major four-cell blocks into 0..255 integer codes."""
    if cells.shape[-1] != 4:
        raise ValueError(f"expected final dimension 4, got {cells.shape}")
    return (cells.long() * _POWERS.to(cells.device)).sum(dim=-1)


def decode_codes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.long()
    return torch.stack(
        [codes % 4, (codes // 4) % 4, (codes // 16) % 4, (codes // 64) % 4],
        dim=-1,
    )


def _to_blocks(state: torch.Tensor, phase: int) -> tuple[torch.Tensor, int]:
    if state.ndim != 3:
        raise ValueError(f"state must have shape [batch,height,width], got {state.shape}")
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


def _from_blocks(blocks: torch.Tensor, offset: int, height: int, width: int) -> torch.Tensor:
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
    return_reaction: bool = False,
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply one phase to a batch, one rule table per state.

    Args:
        state: integer tensor [B,H,W]
        rule_tables: integer tensor [B,256], or [1,256] broadcastable to B
        phase: alternating partition phase, normally 0 then 1
        return_reaction: also return per-state fraction of blocks whose state
            multiset changed (pair conversion, binding, etc.)
    """
    batch, height, width = state.shape
    if rule_tables.ndim == 1:
        rule_tables = rule_tables.unsqueeze(0)
    if rule_tables.shape[0] == 1 and batch != 1:
        rule_tables = rule_tables.expand(batch, -1)
    if rule_tables.shape != (batch, 256):
        raise ValueError(f"rule_tables must be [B,256], got {rule_tables.shape}")

    blocks, offset = _to_blocks(state, phase)
    codes = encode_blocks(blocks)
    out_codes = rule_tables.long().gather(1, codes.reshape(batch, -1)).reshape_as(codes)
    out_blocks = decode_codes(out_codes)
    next_state = _from_blocks(out_blocks, offset, height, width).to(dtype=state.dtype)

    if not return_reaction:
        return next_state

    hist = alphabet.block_histogram(state.device)
    in_hist = hist[codes.long()]
    out_hist = hist[out_codes.long()]
    reaction = (in_hist != out_hist).any(dim=-1).float().mean(dim=(-2, -1))
    return next_state, reaction


def step_rule(state: torch.Tensor, rule: ReversibleRule, phase: int) -> torch.Tensor:
    table = rule.table.to(device=state.device).unsqueeze(0)
    return step_batched(state, table, phase)  # type: ignore[return-value]


def run_trajectory(
    initial: torch.Tensor,
    rule: ReversibleRule,
    steps: int,
    *,
    start_phase: int = 0,
    include_initial: bool = True,
) -> torch.Tensor:
    """Run a single rule; returns [T,B,H,W]."""
    if initial.ndim == 2:
        state = initial.unsqueeze(0)
    elif initial.ndim == 3:
        state = initial
    else:
        raise ValueError("initial must be [H,W] or [B,H,W]")
    frames: list[torch.Tensor] = [state.clone()] if include_initial else []
    table = rule.table.to(state.device).unsqueeze(0).expand(state.shape[0], -1)
    for t in range(steps):
        state = step_batched(state, table, start_phase + t)  # type: ignore[assignment]
        frames.append(state.clone())
    return torch.stack(frames, dim=0)


def reverse_trajectory(final: torch.Tensor, rule: ReversibleRule, steps: int, start_phase: int = 0) -> torch.Tensor:
    """Reverse exactly the same number of phases used by run_trajectory."""
    state = final.unsqueeze(0) if final.ndim == 2 else final
    inv = rule.inverse_table.to(state.device).unsqueeze(0).expand(state.shape[0], -1)
    for t in reversed(range(steps)):
        state = step_batched(state, inv, start_phase + t)  # type: ignore[assignment]
    return state


@dataclass(frozen=True)
class TraceAudit:
    initial_charge: list[int]
    final_charge: list[int]
    charge_preserved: bool
    reversible: bool
    state_equal_after_reverse: bool


def audit_trace(initial: torch.Tensor, rule: ReversibleRule, steps: int) -> TraceAudit:
    alphabet = rule.alphabet
    trajectory = run_trajectory(initial, rule, steps)
    first = trajectory[0]
    last = trajectory[-1]
    recovered = reverse_trajectory(last, rule, steps)
    q0 = alphabet.total_charge(first).cpu().tolist()
    q1 = alphabet.total_charge(last).cpu().tolist()
    same = bool(torch.equal(first.cpu(), recovered.cpu()))
    return TraceAudit(
        initial_charge=[int(x) for x in q0],
        final_charge=[int(x) for x in q1],
        charge_preserved=q0 == q1,
        reversible=rule.is_permutation(),
        state_equal_after_reverse=same,
    )
