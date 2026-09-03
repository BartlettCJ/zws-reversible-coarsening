from __future__ import annotations

from dataclasses import dataclass

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET


@dataclass(frozen=True)
class SeedSpec:
    """Parameters for exactly charge-balanced finite initial states."""

    positive_fraction: float = 0.12
    bound_fraction: float = 0.08
    localized: bool = False
    localized_fraction: float = 0.45

    def validate(self) -> None:
        if not 0.0 <= self.positive_fraction <= 0.5:
            raise ValueError("positive_fraction must be in [0, 0.5]")
        if not 0.0 <= self.bound_fraction <= 1.0:
            raise ValueError("bound_fraction must be in [0, 1]")
        if 2.0 * self.positive_fraction + self.bound_fraction > 1.0:
            raise ValueError("2*positive_fraction + bound_fraction must not exceed 1")
        if not 0.0 < self.localized_fraction <= 1.0:
            raise ValueError("localized_fraction must be in (0, 1]")


def _balanced_flat(
    cells: int,
    spec: SeedSpec,
    generator: torch.Generator | None,
    alphabet: Alphabet,
) -> torch.Tensor:
    spec.validate()
    n_pairs = int(cells * spec.positive_fraction)
    n_bound = int(cells * spec.bound_fraction)
    while 2 * n_pairs + n_bound > cells:
        n_bound = max(0, n_bound - 1)
    flat = torch.full((cells,), alphabet.empty, dtype=torch.uint8)
    flat[:n_pairs] = alphabet.positive
    flat[n_pairs : 2 * n_pairs] = alphabet.negative
    flat[2 * n_pairs : 2 * n_pairs + n_bound] = alphabet.bound
    return flat[torch.randperm(cells, generator=generator)]


def balanced_random(
    batch: int,
    height: int,
    width: int,
    *,
    spec: SeedSpec = SeedSpec(),
    seed: int = 0,
    device: torch.device | str = "cpu",
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> torch.Tensor:
    """Create [B,H,W] states with exact integer total charge zero.

    Positive and negative particles are inserted in equal numbers; no floating
    correction or post-update rebalancing is ever used.
    """
    if batch < 1 or height < 2 or width < 2:
        raise ValueError("batch >= 1 and spatial dimensions >= 2 are required")
    if height % 2 or width % 2:
        raise ValueError("Margolus lattices require even height and width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    states = torch.empty((batch, height, width), dtype=torch.uint8)
    if not spec.localized:
        for b in range(batch):
            states[b] = _balanced_flat(height * width, spec, generator, alphabet).reshape(
                height, width
            )
    else:
        local_h = max(2, int(round(height * spec.localized_fraction)))
        local_w = max(2, int(round(width * spec.localized_fraction)))
        local_h -= local_h % 2
        local_w -= local_w % 2
        local_h = min(height, max(2, local_h))
        local_w = min(width, max(2, local_w))
        top = (height - local_h) // 2
        left = (width - local_w) // 2
        states.fill_(alphabet.empty)
        for b in range(batch):
            local = _balanced_flat(local_h * local_w, spec, generator, alphabet)
            states[b, top : top + local_h, left : left + local_w] = local.reshape(
                local_h, local_w
            )
    assert bool((alphabet.total_charge(states) == 0).all())
    return states.to(device=device)


def paired_impulse(
    height: int,
    width: int,
    *,
    separation: int = 2,
    device: torch.device | str = "cpu",
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> torch.Tensor:
    """Minimal nonempty zero-charge state: one + and one - particle."""
    if height % 2 or width % 2:
        raise ValueError("Margolus lattices require even dimensions")
    state = torch.zeros((height, width), dtype=torch.uint8)
    y, x = height // 2, width // 2
    state[y, (x - separation // 2) % width] = alphabet.positive
    state[y, (x + (separation + 1) // 2) % width] = alphabet.negative
    return state.to(device=device)


def perturb_charge_preserving(
    state: torch.Tensor,
    *,
    seed: int = 1,
    mode: str = "pair-conversion",
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> torch.Tensor:
    """Return a small exact-charge-preserving perturbation.

    ``pair-conversion`` converts one (+,-) pair to (bound,empty), or the
    reverse if no charged pair is available. ``swap`` exchanges two cells.
    Both preserve total charge exactly.
    """
    original_ndim = state.ndim
    if original_ndim == 2:
        work = state.unsqueeze(0).clone().cpu()
    elif original_ndim == 3:
        work = state.clone().cpu()
    else:
        raise ValueError("state must be [H,W] or [B,H,W]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for b in range(work.shape[0]):
        flat = work[b].flatten()
        if mode == "swap":
            pair = torch.randperm(flat.numel(), generator=generator)[:2]
            a, c = int(pair[0]), int(pair[1])
            flat[a], flat[c] = flat[c].clone(), flat[a].clone()
        elif mode == "pair-conversion":
            pos = torch.nonzero(flat == alphabet.positive, as_tuple=False).flatten()
            neg = torch.nonzero(flat == alphabet.negative, as_tuple=False).flatten()
            if pos.numel() and neg.numel():
                i = int(pos[torch.randint(pos.numel(), (1,), generator=generator)])
                j = int(neg[torch.randint(neg.numel(), (1,), generator=generator)])
                flat[i], flat[j] = alphabet.bound, alphabet.empty
            else:
                zeros = torch.nonzero(
                    (flat == alphabet.empty) | (flat == alphabet.bound), as_tuple=False
                ).flatten()
                if zeros.numel() < 2:
                    raise ValueError("state lacks a charge-preserving two-cell perturbation")
                pair = zeros[torch.randperm(zeros.numel(), generator=generator)[:2]]
                flat[int(pair[0])], flat[int(pair[1])] = alphabet.positive, alphabet.negative
    q0 = alphabet.total_charge(state.cpu() if original_ndim == 3 else state.cpu().unsqueeze(0))
    q1 = alphabet.total_charge(work)
    if not torch.equal(q0, q1):
        raise AssertionError("perturbation changed exact total charge")
    result = work.to(state.device)
    return result[0] if original_ndim == 2 else result
