from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Alphabet:
    """Four-state alphabet used by the frozen v0.1 winner.

    State 0 is empty, 1 is a +1 excitation, 2 is a -1 excitation, and 3 is a
    bound/neutral excitation.  Signed charge Q and unsigned occupancy M are
    deliberately independent:

        Q = N(+)-N(-)
        M = N(+)+N(-)+N(bound)
    """

    names: tuple[str, ...] = ("empty", "positive", "negative", "bound")
    charges: tuple[int, ...] = (0, 1, -1, 0)
    occupancy: tuple[int, ...] = (0, 1, 1, 1)

    @property
    def radix(self) -> int:
        return len(self.names)

    def charge_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.charges, dtype=torch.int64, device=device)

    def occupancy_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.occupancy, dtype=torch.int64, device=device)

    def total_charge(self, state: torch.Tensor) -> torch.Tensor:
        lookup = self.charge_tensor(state.device)
        if state.ndim == 2:
            return lookup[state.long()].sum()
        if state.ndim == 3:
            return lookup[state.long()].sum(dim=(-2, -1))
        raise ValueError(f"state must be [H,W] or [B,H,W], got {tuple(state.shape)}")

    def total_occupancy(self, state: torch.Tensor) -> torch.Tensor:
        lookup = self.occupancy_tensor(state.device)
        if state.ndim == 2:
            return lookup[state.long()].sum()
        if state.ndim == 3:
            return lookup[state.long()].sum(dim=(-2, -1))
        raise ValueError(f"state must be [H,W] or [B,H,W], got {tuple(state.shape)}")

    def state_counts(self, state: torch.Tensor) -> torch.Tensor:
        """Return counts with shape [4] or [B,4]."""
        if state.ndim == 2:
            return torch.bincount(state.long().reshape(-1), minlength=self.radix)
        if state.ndim != 3:
            raise ValueError(f"state must be [H,W] or [B,H,W], got {tuple(state.shape)}")
        out = torch.zeros((state.shape[0], self.radix), dtype=torch.int64, device=state.device)
        flat = state.long().reshape(state.shape[0], -1)
        out.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.int64))
        return out

    def block_digits_numpy(self) -> np.ndarray:
        codes = np.arange(self.radix**4, dtype=np.int64)
        return np.stack(
            [codes % 4, (codes // 4) % 4, (codes // 16) % 4, (codes // 64) % 4],
            axis=1,
        )


DEFAULT_ALPHABET = Alphabet()
