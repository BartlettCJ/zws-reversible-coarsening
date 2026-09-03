from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch

from .alphabet import DEFAULT_ALPHABET
from .margolus import step_rule
from .rules import ReversibleRule


@dataclass(frozen=True)
class CycleResult:
    found: bool
    transient: int | None
    period: int | None
    ticks_examined: int
    charge: int
    start_digest: str
    repeat_digest: str | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def state_digest(state: torch.Tensor) -> str:
    data = state.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _state_key(state: torch.Tensor) -> bytes:
    return state.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()


def full_tick(state: torch.Tensor, rule: ReversibleRule) -> torch.Tensor:
    """One autonomous two-partition update: phase 0 then phase 1."""
    out = step_rule(state.unsqueeze(0) if state.ndim == 2 else state, rule, 0)
    out = step_rule(out, rule, 1)
    return out[0] if state.ndim == 2 else out


def find_cycle(
    initial: torch.Tensor,
    rule: ReversibleRule,
    *,
    max_ticks: int = 200_000,
) -> CycleResult:
    """Find an exact finite-torus orbit repeat by hash plus byte equality.

    A valid full-tick rule is a bijection on a finite state space, so every
    state lies on a cycle and the true transient is zero. The cap may prevent
    witnessing a long period; failure under the cap is not non-periodicity.
    """
    if initial.ndim != 2:
        raise ValueError("cycle search expects one [H,W] state")
    if not rule.is_valid():
        raise ValueError("cycle search requires a valid charge-preserving permutation")
    state = initial.clone()
    first_digest = state_digest(state)
    seen: dict[bytes, int] = {_state_key(state): 0}
    for tick in range(1, max_ticks + 1):
        state = full_tick(state, rule)
        key = _state_key(state)
        previous = seen.get(key)
        if previous is not None:
            period = tick - previous
            note = (
                "Exact repeated state. Because the full-tick map is bijective, "
                "a nonzero transient would signal an implementation error."
            )
            return CycleResult(
                found=True,
                transient=previous,
                period=period,
                ticks_examined=tick,
                charge=int(DEFAULT_ALPHABET.total_charge(initial.unsqueeze(0))[0]),
                start_digest=first_digest,
                repeat_digest=state_digest(state),
                note=note,
            )
        seen[key] = tick
    return CycleResult(
        found=False,
        transient=None,
        period=None,
        ticks_examined=max_ticks,
        charge=int(DEFAULT_ALPHABET.total_charge(initial.unsqueeze(0))[0]),
        start_digest=first_digest,
        repeat_digest=None,
        note="No repeat within the declared cap; finite bijectivity still entails a cycle.",
    )
