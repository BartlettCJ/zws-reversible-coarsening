from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .alphabet import DEFAULT_ALPHABET
from .rule import ReversibleRule


@dataclass(frozen=True)
class ReactionEntry:
    input_code: int
    output_code: int
    input_digits: tuple[int, int, int, int]
    output_digits: tuple[int, int, int, int]
    delta_q: int
    delta_m: int
    delta_bound: int
    delta_charged: int
    scaffold_input_partner: int | None
    reactive_pair_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reaction_catalog(rule: ReversibleRule) -> list[ReactionEntry]:
    digits = DEFAULT_ALPHABET.block_digits_numpy()
    table = rule.table.numpy()
    deltas = rule.transition_deltas()
    pair_lookup: dict[int, tuple[int, int]] = {}
    try:
        for index, (a, b) in enumerate(rule.reactive_pairs_relative_to_rotation()):
            pair_lookup[a] = (index, b)
            pair_lookup[b] = (index, a)
    except ValueError:
        pair_lookup = {}
    entries: list[ReactionEntry] = []
    for code in range(256):
        out = int(table[code])
        info = pair_lookup.get(code)
        entries.append(
            ReactionEntry(
                input_code=code,
                output_code=out,
                input_digits=tuple(int(x) for x in digits[code]),
                output_digits=tuple(int(x) for x in digits[out]),
                delta_q=int(deltas["delta_q"][code]),
                delta_m=int(deltas["delta_m"][code]),
                delta_bound=int(deltas["delta_bound"][code]),
                delta_charged=int(deltas["delta_charged"][code]),
                scaffold_input_partner=info[1] if info else None,
                reactive_pair_index=info[0] if info else None,
            )
        )
    return entries


def summarize_flux(
    rule: ReversibleRule,
    input_histogram: np.ndarray,
) -> dict[str, Any]:
    hist = np.asarray(input_histogram, dtype=np.int64).reshape(256)
    catalog = reaction_catalog(rule)
    total_blocks = int(hist.sum())
    delta_m_flux = int(sum(hist[e.input_code] * e.delta_m for e in catalog))
    delta_bound_flux = int(sum(hist[e.input_code] * e.delta_bound for e in catalog))
    delta_charged_flux = int(sum(hist[e.input_code] * e.delta_charged for e in catalog))
    reactive_visits = int(
        sum(hist[e.input_code] for e in catalog if e.reactive_pair_index is not None)
    )
    pair_flux: dict[int, int] = {}
    for entry in catalog:
        if entry.reactive_pair_index is not None:
            pair_flux[entry.reactive_pair_index] = pair_flux.get(entry.reactive_pair_index, 0) + int(
                hist[entry.input_code]
            )
    return {
        "total_blocks": total_blocks,
        "reactive_visits": reactive_visits,
        "reactive_fraction": reactive_visits / total_blocks if total_blocks else 0.0,
        "delta_m_flux": delta_m_flux,
        "delta_bound_flux": delta_bound_flux,
        "delta_charged_flux": delta_charged_flux,
        "pair_flux": {str(k): int(v) for k, v in sorted(pair_flux.items())},
    }
