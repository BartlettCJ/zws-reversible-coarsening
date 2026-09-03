from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET


def _check_block(state: torch.Tensor, block: int) -> None:
    if state.shape[-2] % block or state.shape[-1] % block:
        raise ValueError(f"spatial dimensions must be divisible by block={block}")


def coarse_charge(
    state: torch.Tensor, block: int = 2, alphabet: Alphabet = DEFAULT_ALPHABET
) -> torch.Tensor:
    """Exact additive coarse-graining of signed charge."""
    if state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3:
        raise ValueError("state must be [H,W] or [B,H,W]")
    _check_block(state, block)
    charge = alphabet.state_charge(state)
    batch, height, width = charge.shape
    return charge.reshape(batch, height // block, block, width // block, block).sum(
        dim=(2, 4)
    )


def coarse_histogram_symbol(state: torch.Tensor, block: int = 2) -> torch.Tensor:
    """Position-insensitive macro symbol retaining the four state counts."""
    if state.ndim == 2:
        state = state.unsqueeze(0)
    _check_block(state, block)
    batch, height, width = state.shape
    cells = (
        state.reshape(batch, height // block, block, width // block, block)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, height // block, width // block, block * block)
    )
    counts = torch.nn.functional.one_hot(cells.long(), num_classes=4).sum(dim=-2)
    base = block * block + 1
    powers = torch.tensor([1, base, base**2, base**3], device=state.device)
    return (counts * powers).sum(dim=-1).long()


def conditional_entropy_and_accuracy(current: torch.Tensor, following: torch.Tensor) -> tuple[float, float]:
    """Empirical H(Y|X)/log|Y| and majority-transition prediction accuracy."""
    x = current.detach().cpu().flatten().long()
    y = following.detach().cpu().flatten().long()
    if x.numel() != y.numel() or x.numel() == 0:
        raise ValueError("current/following must be nonempty and equal size")
    pairs = torch.stack([x, y], dim=1)
    unique_pairs, pair_counts = torch.unique(pairs, dim=0, return_counts=True)
    unique_x, x_counts = torch.unique(x, return_counts=True)
    x_lookup = {int(v): int(c) for v, c in zip(unique_x, x_counts, strict=True)}
    conditional = 0.0
    best_correct = 0
    by_x: dict[int, list[int]] = {}
    for pair, count in zip(unique_pairs, pair_counts, strict=True):
        xv, _ = int(pair[0]), int(pair[1])
        by_x.setdefault(xv, []).append(int(count))
        pxy = int(count) / x.numel()
        py_given_x = int(count) / x_lookup[xv]
        conditional -= pxy * math.log(max(py_given_x, 1e-30))
    for counts in by_x.values():
        best_correct += max(counts)
    distinct_y = max(2, int(torch.unique(y).numel()))
    return conditional / math.log(distinct_y), best_correct / x.numel()


@dataclass(frozen=True)
class CoarseLevelResult:
    block: int
    total_charge_exact: bool
    direct_composition_exact: bool | None
    conditional_entropy: float
    majority_prediction_accuracy: float
    macro_symbols_observed: int


@dataclass(frozen=True)
class CoarseGrainAudit:
    passed_additive_composition: bool
    levels: list[CoarseLevelResult]
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed_additive_composition": self.passed_additive_composition,
            "levels": [asdict(level) for level in self.levels],
            "interpretation": self.interpretation,
        }


def audit_coarse_graining(
    trajectory: torch.Tensor,
    blocks: tuple[int, ...] = (2, 4),
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> CoarseGrainAudit:
    """Test exact additive composition and empirical macro-dynamic closure.

    Exact charge composition is a real formal result for this map. Low
    conditional entropy is only evidence that the selected macro vocabulary
    approximates a Markovian level on this trace.
    """
    if trajectory.ndim == 4:
        if trajectory.shape[1] != 1:
            raise ValueError("coarse audit accepts one trajectory batch item")
        frames = trajectory[:, 0]
    elif trajectory.ndim == 3:
        frames = trajectory
    else:
        raise ValueError("trajectory must be [T,H,W] or [T,1,H,W]")
    micro_total = alphabet.total_charge(frames).cpu()
    levels: list[CoarseLevelResult] = []
    all_exact = True
    for block in blocks:
        macro_q = coarse_charge(frames, block=block, alphabet=alphabet)
        total_exact = bool(torch.equal(macro_q.sum(dim=(-2, -1)).cpu(), micro_total))
        composition: bool | None = None
        if block % 2 == 0 and block >= 4:
            q2 = coarse_charge(frames, block=2, alphabet=alphabet)
            factor = block // 2
            composed = q2.reshape(
                q2.shape[0], q2.shape[1] // factor, factor, q2.shape[2] // factor, factor
            ).sum(dim=(2, 4))
            composition = bool(torch.equal(composed.cpu(), macro_q.cpu()))
        symbols = coarse_histogram_symbol(frames, block=block)
        cond_h, accuracy = conditional_entropy_and_accuracy(symbols[:-1], symbols[1:])
        levels.append(
            CoarseLevelResult(
                block=block,
                total_charge_exact=total_exact,
                direct_composition_exact=composition,
                conditional_entropy=cond_h,
                majority_prediction_accuracy=accuracy,
                macro_symbols_observed=int(torch.unique(symbols).numel()),
            )
        )
        all_exact &= total_exact and (composition is not False)
    return CoarseGrainAudit(
        passed_additive_composition=all_exact,
        levels=levels,
        interpretation=(
            "Charge coarse-graining composes exactly. Macro-symbol closure is empirical and "
            "vocabulary-relative; conditional entropy near zero would support, not prove, a "
            "stable autonomous higher-level description."
        ),
    )
