from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET


_POWERS_NP = np.asarray([1, 4, 16, 64], dtype=np.int64)


def encode_digits_numpy(digits: np.ndarray) -> np.ndarray:
    arr = np.asarray(digits, dtype=np.int64)
    if arr.shape[-1] != 4:
        raise ValueError("final dimension must contain four cells")
    return np.sum(arr * _POWERS_NP, axis=-1, dtype=np.int64)


def rotate_table(quarter_turns: int = 1) -> np.ndarray:
    digits = DEFAULT_ALPHABET.block_digits_numpy().reshape(256, 2, 2)
    rotated = np.rot90(digits, k=-(quarter_turns % 4), axes=(1, 2)).reshape(256, 4)
    return encode_digits_numpy(rotated)


@dataclass(frozen=True)
class RuleAudit:
    permutation: bool
    charge_preserving: bool
    occupancy_preserving: bool
    differing_from_rotation: int
    sha256: str


@dataclass
class ReversibleRule:
    table: torch.Tensor
    name: str = "unnamed"
    metadata: dict[str, Any] | None = None
    alphabet: Alphabet = DEFAULT_ALPHABET

    def __post_init__(self) -> None:
        table = torch.as_tensor(self.table, dtype=torch.int64, device="cpu").flatten()
        if table.numel() != 256:
            raise ValueError(f"rule table must have 256 entries, got {table.numel()}")
        if int(table.min()) < 0 or int(table.max()) > 255:
            raise ValueError("rule outputs must be in 0..255")
        self.table = table
        self.metadata = dict(self.metadata or {})

    @classmethod
    def load(cls, path: str | Path) -> "ReversibleRule":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            torch.tensor(payload["table"], dtype=torch.int64),
            name=str(payload.get("name", Path(path).stem)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "zws-nullity-rule-v1",
            "name": self.name,
            "alphabet": {
                "states": list(self.alphabet.names),
                "charges": list(self.alphabet.charges),
            },
            "metadata": dict(self.metadata or {}),
            "table": [int(x) for x in self.table.tolist()],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @property
    def inverse_table(self) -> torch.Tensor:
        inv = torch.empty_like(self.table)
        inv[self.table] = torch.arange(256, dtype=torch.int64)
        return inv

    def sha256(self) -> str:
        """Canonical scientific rule identity: SHA-256 of 256 uint8 outputs.

        This matches the frozen v0.1.4 rule identifier d1fe99... and is
        independent of JSON formatting or metadata.
        """
        return hashlib.sha256(self.table.numpy().astype(np.uint8).tobytes()).hexdigest()

    def definition_sha256(self) -> str:
        canonical = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def audit(self) -> RuleAudit:
        table = self.table.numpy()
        digits = self.alphabet.block_digits_numpy()
        q_lookup = np.asarray(self.alphabet.charges, dtype=np.int64)
        m_lookup = np.asarray(self.alphabet.occupancy, dtype=np.int64)
        q = q_lookup[digits].sum(axis=1)
        m = m_lookup[digits].sum(axis=1)
        permutation = np.array_equal(np.sort(table), np.arange(256, dtype=np.int64))
        charge_ok = bool(np.array_equal(q, q[table]))
        occupancy_ok = bool(np.array_equal(m, m[table]))
        rot = rotate_table(1)
        return RuleAudit(
            permutation=permutation,
            charge_preserving=charge_ok,
            occupancy_preserving=occupancy_ok,
            differing_from_rotation=int(np.count_nonzero(table != rot)),
            sha256=self.sha256(),
        )

    def assert_valid(self, *, require_occupancy: bool = False) -> None:
        audit = self.audit()
        if not audit.permutation:
            raise ValueError("rule table is not a permutation")
        if not audit.charge_preserving:
            raise ValueError("rule does not preserve exact local charge")
        if require_occupancy and not audit.occupancy_preserving:
            raise ValueError("rule does not preserve exact local occupancy")

    def reactive_pairs_relative_to_rotation(self) -> list[tuple[int, int]]:
        """Recover transposed input pairs relative to the 90-degree rotation scaffold.

        The frozen winner was made by swapping the rotation images of input pairs.
        This method verifies that structure rather than assuming metadata is honest.
        """
        rot = rotate_table(1)
        table = self.table.numpy()
        differing = set(int(x) for x in np.flatnonzero(table != rot))
        pairs: list[tuple[int, int]] = []
        while differing:
            a = min(differing)
            # Find b such that winner[a]=rot[b] and winner[b]=rot[a].
            candidates = [
                b
                for b in differing
                if b != a and table[a] == rot[b] and table[b] == rot[a]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"rule difference at input {a} is not a unique scaffold transposition"
                )
            b = candidates[0]
            pairs.append((a, b))
            differing.remove(a)
            differing.remove(b)
        return pairs

    def ablate_pairs(self, pair_indices: Iterable[int], *, name: str | None = None) -> "ReversibleRule":
        pairs = self.reactive_pairs_relative_to_rotation()
        selected = set(int(i) for i in pair_indices)
        if any(i < 0 or i >= len(pairs) for i in selected):
            raise IndexError("reactive-pair index out of range")
        table = self.table.numpy().copy()
        rot = rotate_table(1)
        for idx in selected:
            a, b = pairs[idx]
            table[a] = rot[a]
            table[b] = rot[b]
        rule = ReversibleRule(
            torch.tensor(table, dtype=torch.int64),
            name=name or f"{self.name}_ablate_{'-'.join(map(str, sorted(selected)))}",
            metadata={**dict(self.metadata or {}), "ablated_pairs": sorted(selected)},
        )
        rule.assert_valid()
        return rule

    def closest_qm_preserving_sibling(self, *, name: str | None = None) -> "ReversibleRule":
        """Return the frozen reactive Q+M sibling from the mechanism study.

        The winner's 24 non-rotational mappings all change occupancy, so a
        greedy projection onto joint (Q, M) sectors collapses to pure rotation
        and silently duplicates that control.  The completed v0.1.4 mechanism
        study instead preregistered an exact twelve-swap optimum: it maximizes
        overlap with the winner's reactive support while remaining a reactive
        permutation inside joint (Q, M) sectors.  Reuse those frozen pairs here
        and audit the resulting table exhaustively.
        """
        expected_winner = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
        if self.sha256() != expected_winner:
            raise ValueError("the frozen Q+M sibling is defined only for the v0.1.4 winner")

        pairs = (
            (6, 195),
            (9, 51),
            (15, 33),
            (18, 204),
            (22, 55),
            (24, 60),
            (25, 31),
            (26, 47),
            (27, 63),
            (106, 175),
            (154, 187),
            (166, 190),
        )
        winner = self.table.numpy()
        out = rotate_table(1).copy()
        for first, second in pairs:
            out[first], out[second] = out[second], out[first]
        sibling = ReversibleRule(
            torch.tensor(out, dtype=torch.int64),
            name=name or f"{self.name}_qm_sibling",
            metadata={
                **dict(self.metadata or {}),
                "construction": "frozen_optimal_joint_QM_12_swap_sibling",
                "source_rule_sha256": self.sha256(),
                "frozen_prior_rule_sha256": (
                    "ad8c2ad901201e2647c8734563dcca93258e15877743e7c82a7351bd08a72d01"
                ),
                "swap_input_pairs": [list(pair) for pair in pairs],
                "entries_identical_to_source": int(np.count_nonzero(out == winner)),
            },
        )
        sibling.assert_valid(require_occupancy=True)
        audit = sibling.audit()
        if audit.differing_from_rotation != 24:
            raise AssertionError("frozen Q+M sibling must differ from rotation at 24 inputs")
        if sibling.sha256() != sibling.metadata["frozen_prior_rule_sha256"]:
            raise AssertionError("frozen Q+M sibling table changed")
        return sibling

    def transition_deltas(self) -> dict[str, np.ndarray]:
        digits = self.alphabet.block_digits_numpy()
        out_digits = digits[self.table.numpy()]
        q_lookup = np.asarray(self.alphabet.charges, dtype=np.int64)
        m_lookup = np.asarray(self.alphabet.occupancy, dtype=np.int64)
        return {
            "delta_q": q_lookup[out_digits].sum(axis=1) - q_lookup[digits].sum(axis=1),
            "delta_m": m_lookup[out_digits].sum(axis=1) - m_lookup[digits].sum(axis=1),
            "delta_bound": (out_digits == 3).sum(axis=1) - (digits == 3).sum(axis=1),
            "delta_charged": np.isin(out_digits, [1, 2]).sum(axis=1)
            - np.isin(digits, [1, 2]).sum(axis=1),
        }
