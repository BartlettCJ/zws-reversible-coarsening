from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET
from .atomic import atomic_write_json


@dataclass
class ReversibleRule:
    """A 2x2 Margolus block permutation.

    The lookup table maps every four-cell input block code (0..255) to an
    output block code. A valid signed rule is a permutation *within each total
    charge sector*. This gives three properties by construction:

    1. every local update preserves exact integer charge;
    2. every partition update is bijective;
    3. the inverse rule exists and is exact.
    """

    table: torch.Tensor
    name: str = "unnamed"
    metadata: dict[str, Any] | None = None
    alphabet: Alphabet = DEFAULT_ALPHABET

    def __post_init__(self) -> None:
        table = torch.as_tensor(self.table, dtype=torch.int64, device="cpu").flatten()
        if table.numel() != self.alphabet.radix**4:
            raise ValueError(f"rule table must contain 256 entries, got {table.numel()}")
        self.table = table
        self.metadata = dict(self.metadata or {})

    @classmethod
    def identity(cls, name: str = "identity") -> "ReversibleRule":
        return cls(torch.arange(256, dtype=torch.int64), name=name)

    @classmethod
    def spatial_rotation(cls, quarter_turns: int = 1, name: str | None = None) -> "ReversibleRule":
        """Rotate each 2x2 block, a nonreactive charge-preserving baseline."""
        k = quarter_turns % 4
        digits = DEFAULT_ALPHABET.block_digits_cpu.reshape(256, 2, 2)
        out = torch.rot90(digits, k=-k, dims=(1, 2)).reshape(256, 4)
        powers = torch.tensor([1, 4, 16, 64], dtype=torch.int64)
        table = (out * powers).sum(dim=-1)
        return cls(table, name=name or f"rotate_{90 * k}")

    @classmethod
    def reactive_rotation(
        cls,
        swaps: int = 16,
        quarter_turns: int = 1,
        name: str | None = None,
        alphabet: Alphabet = DEFAULT_ALPHABET,
    ) -> "ReversibleRule":
        """Low-description-length reactive seed built from block rotation.

        Images of selected equal-charge inputs with different state multisets
        are exchanged. The result remains a charge-sector permutation, but it
        contains explicit pair-conversion/binding reactions rather than the
        hot random soup produced by a fully random 256-entry permutation.
        """
        base = cls.spatial_rotation(quarter_turns).table.clone()
        hist = alphabet.block_histogram_cpu
        charge = alphabet.block_charge_cpu
        used: set[int] = set()
        completed = 0
        # Deterministic lexicographic construction makes this a reproducible
        # baseline rather than a lucky search result.
        for q in sorted(alphabet.charge_sectors):
            indices = alphabet.charge_sectors[q].tolist()
            for i, a in enumerate(indices):
                if a in used:
                    continue
                partner = None
                for b in indices[i + 1 :]:
                    if b in used:
                        continue
                    if bool((hist[a] != hist[b]).any()) and int(charge[a]) == int(charge[b]):
                        partner = b
                        break
                if partner is None:
                    continue
                base[a], base[partner] = base[partner].clone(), base[a].clone()
                used.add(a)
                used.add(partner)
                completed += 1
                if completed >= swaps:
                    rule = cls(
                        base,
                        name=name or f"reactive_rotate_{90 * (quarter_turns % 4)}_s{completed}",
                        metadata={"construction": "reactive_rotation", "swaps": completed},
                        alphabet=alphabet,
                    )
                    rule.assert_valid()
                    return rule
        rule = cls(
            base,
            name=name or f"reactive_rotate_{90 * (quarter_turns % 4)}_s{completed}",
            metadata={"construction": "reactive_rotation", "swaps": completed},
            alphabet=alphabet,
        )
        rule.assert_valid()
        return rule

    def state_symmetry_error(
        self, state_permutation: list[int] | torch.Tensor
    ) -> float:
        """Fraction of inputs for which F(Cx) != C(Fx).

        This tests whether the *same rule* has a physical state symmetry. It is
        distinct from conjugating the rule, which is a harmless-redescription
        audit and must always be exact.
        """
        p = torch.as_tensor(state_permutation, dtype=torch.int64)
        if p.numel() != 4 or not torch.equal(torch.sort(p).values, torch.arange(4)):
            raise ValueError("state_permutation must be a permutation of [0,1,2,3]")
        digits = self.alphabet.block_digits_cpu
        powers = torch.tensor([1, 4, 16, 64], dtype=torch.int64)
        c = (p[digits] * powers).sum(dim=-1)
        actual = self.table[c]
        expected = c[self.table]
        return float((actual != expected).float().mean())

    def sign_asymmetry(self) -> float:
        return self.state_symmetry_error([0, 2, 1, 3])

    def neutral_label_asymmetry(self) -> float:
        return self.state_symmetry_error([3, 1, 2, 0])

    def hamming_distance(self, other: "ReversibleRule") -> int:
        return int((self.table != other.table).sum())

    def simple_baseline_distance(self) -> float:
        """Minimum normalized table distance from identity or a block rotation."""
        baselines = [self.identity()] + [self.spatial_rotation(k) for k in (1, 2, 3)]
        return min(self.hamming_distance(base) for base in baselines) / 256.0

    @classmethod
    def random(
        cls,
        generator: torch.Generator | None = None,
        name: str = "random_charge_preserving",
        alphabet: Alphabet = DEFAULT_ALPHABET,
    ) -> "ReversibleRule":
        table = torch.empty(alphabet.radix**4, dtype=torch.int64)
        for indices in alphabet.charge_sectors.values():
            perm = indices[torch.randperm(indices.numel(), generator=generator)]
            table[indices] = perm
        rule = cls(table, name=name, alphabet=alphabet)
        rule.assert_valid()
        return rule

    @classmethod
    def random_unconstrained(
        cls, generator: torch.Generator | None = None, name: str = "random_unconstrained"
    ) -> "ReversibleRule":
        return cls(torch.randperm(256, generator=generator), name=name)

    @property
    def inverse_table(self) -> torch.Tensor:
        inv = torch.empty_like(self.table)
        inv[self.table] = torch.arange(self.table.numel(), dtype=torch.int64)
        return inv

    def inverse(self) -> "ReversibleRule":
        return ReversibleRule(
            self.inverse_table,
            name=f"{self.name}_inverse",
            metadata={**self.metadata, "inverse_of": self.name},
            alphabet=self.alphabet,
        )

    def is_permutation(self) -> bool:
        return bool(torch.equal(torch.sort(self.table).values, torch.arange(256)))

    def preserves_charge(self) -> bool:
        charges = self.alphabet.block_charge_cpu
        return bool(torch.equal(charges, charges[self.table]))

    def is_valid(self) -> bool:
        return self.is_permutation() and self.preserves_charge()

    def assert_valid(self) -> None:
        if not self.is_permutation():
            raise ValueError("rule table is not a permutation")
        if not self.preserves_charge():
            charges = self.alphabet.block_charge_cpu
            bad = torch.nonzero(charges != charges[self.table], as_tuple=False).flatten()
            first = int(bad[0]) if bad.numel() else -1
            raise ValueError(
                f"rule violates charge preservation at input {first}: "
                f"{int(charges[first])} -> {int(charges[self.table[first]])}"
            )

    def reaction_mask(self) -> torch.Tensor:
        """Inputs whose output changes the state multiset, not merely positions."""
        hist = self.alphabet.block_histogram_cpu
        return (hist != hist[self.table]).any(dim=-1)

    def reaction_fraction(self) -> float:
        return float(self.reaction_mask().float().mean())

    def mutated(
        self,
        swaps: int,
        generator: torch.Generator | None = None,
        name: str | None = None,
    ) -> "ReversibleRule":
        """Swap images of inputs in equal-charge sectors.

        Each swap preserves both bijectivity and charge-sector membership.
        """
        table = self.table.clone()
        sector_keys = sorted(self.alphabet.charge_sectors)
        sector_weights = torch.tensor(
            [self.alphabet.charge_sectors[q].numel() for q in sector_keys], dtype=torch.float32
        )
        for _ in range(swaps):
            sector_idx = int(torch.multinomial(sector_weights, 1, generator=generator))
            indices = self.alphabet.charge_sectors[sector_keys[sector_idx]]
            if indices.numel() < 2:
                continue
            pair = indices[torch.randperm(indices.numel(), generator=generator)[:2]]
            a, b = int(pair[0]), int(pair[1])
            table[a], table[b] = table[b].clone(), table[a].clone()
        child = ReversibleRule(
            table,
            name=name or f"{self.name}_m{swaps}",
            metadata={**self.metadata, "parent": self.name, "mutation_swaps": swaps},
            alphabet=self.alphabet,
        )
        child.assert_valid()
        return child

    def conjugated(self, state_permutation: list[int] | torch.Tensor, name: str | None = None) -> "ReversibleRule":
        """Apply a state relabelling to both inputs and outputs.

        The permutation must preserve charge up to either identity or global sign
        reversal. Neutral swap [3,1,2,0] and sign flip [0,2,1,3] are useful
        harmless-redescription tests.
        """
        p = torch.as_tensor(state_permutation, dtype=torch.int64)
        if p.numel() != 4 or not torch.equal(torch.sort(p).values, torch.arange(4)):
            raise ValueError("state_permutation must be a permutation of [0,1,2,3]")
        inv_p = torch.empty_like(p)
        inv_p[p] = torch.arange(4)
        digits = self.alphabet.block_digits_cpu
        powers = torch.tensor([1, 4, 16, 64], dtype=torch.int64)

        # Encoded relabel map C: old block -> relabelled block.
        c = (p[digits] * powers).sum(dim=-1)
        c_inv = (inv_p[digits] * powers).sum(dim=-1)
        # Conjugate: C o F o C^{-1}.
        table = c[self.table[c_inv]]
        return ReversibleRule(
            table,
            name=name or f"{self.name}_conjugated",
            metadata={**self.metadata, "conjugated_by": p.tolist()},
            alphabet=self.alphabet,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "zws-nullity-rule-v1",
            "name": self.name,
            "alphabet": {
                "states": list(self.alphabet.state_names),
                "charges": self.alphabet.charges_cpu.tolist(),
            },
            "table": self.table.tolist(),
            "metadata": self.metadata,
            "valid": self.is_valid(),
            "reaction_fraction": self.reaction_fraction(),
        }

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ReversibleRule":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rule = cls(data["table"], name=data.get("name", "loaded"), metadata=data.get("metadata"))
        if data.get("schema") not in (None, "zws-nullity-rule-v1"):
            raise ValueError(f"unsupported rule schema: {data.get('schema')}")
        return rule


def stack_rules(rules: list[ReversibleRule], device: torch.device | str = "cpu") -> torch.Tensor:
    if not rules:
        raise ValueError("at least one rule is required")
    return torch.stack([r.table for r in rules], dim=0).to(device=device)
