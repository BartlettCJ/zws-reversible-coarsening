from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch


@dataclass(frozen=True)
class Alphabet:
    """Four-state alphabet with an exact signed charge.

    States:
      0: empty neutral cell, q=0
      1: positive quantum, q=+1
      2: negative quantum, q=-1
      3: bound/neutral excitation, q=0

    Having two distinct zero-charge states prevents the invariant from
    exhausting the state description. Local rules may convert (+,-) pairs
    into neutral excitations and back while preserving charge exactly.
    """

    empty: int = 0
    positive: int = 1
    negative: int = 2
    bound: int = 3
    radix: int = 4

    @cached_property
    def charges_cpu(self) -> torch.Tensor:
        return torch.tensor([0, 1, -1, 0], dtype=torch.int64)

    def charges(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.charges_cpu.to(device=device)

    @cached_property
    def state_names(self) -> tuple[str, ...]:
        return ("empty", "positive", "negative", "bound")

    @cached_property
    def block_digits_cpu(self) -> torch.Tensor:
        codes = torch.arange(self.radix**4, dtype=torch.int64)
        return torch.stack(
            [
                codes % self.radix,
                (codes // self.radix) % self.radix,
                (codes // (self.radix**2)) % self.radix,
                (codes // (self.radix**3)) % self.radix,
            ],
            dim=-1,
        )

    def block_digits(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.block_digits_cpu.to(device=device)

    @cached_property
    def block_charge_cpu(self) -> torch.Tensor:
        return self.charges_cpu[self.block_digits_cpu].sum(dim=-1)

    def block_charge(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.block_charge_cpu.to(device=device)

    @cached_property
    def block_histogram_cpu(self) -> torch.Tensor:
        digits = self.block_digits_cpu
        return torch.nn.functional.one_hot(digits, num_classes=self.radix).sum(dim=1)

    def block_histogram(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.block_histogram_cpu.to(device=device)

    @cached_property
    def charge_sectors(self) -> dict[int, torch.Tensor]:
        charges = self.block_charge_cpu
        return {
            int(q): torch.nonzero(charges == q, as_tuple=False).flatten()
            for q in range(-4, 5)
            if bool((charges == q).any())
        }

    def state_charge(self, state: torch.Tensor) -> torch.Tensor:
        return self.charges(state.device)[state.long()]

    def total_charge(self, state: torch.Tensor) -> torch.Tensor:
        """Return exact total charge for one lattice or a leading batch.

        A two-dimensional tensor is one ``[H,W]`` lattice and therefore
        returns a scalar. Tensors with three or more dimensions retain the
        leading axis and sum every remaining axis, matching the package's
        ``[B,H,W]`` convention.
        """
        q = self.state_charge(state)
        if state.ndim <= 2:
            return q.sum()
        return q.sum(dim=tuple(range(1, state.ndim)))


DEFAULT_ALPHABET = Alphabet()
