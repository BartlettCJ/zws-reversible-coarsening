from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .margolus import run_trajectory
from .measures import trajectory_candidate_values
from .metrics import block_entropy, spatial_mutual_information, state_entropy
from .rules import ReversibleRule


NEUTRAL_SWAP = torch.tensor([3, 1, 2, 0], dtype=torch.int64)
SIGN_FLIP = torch.tensor([0, 2, 1, 3], dtype=torch.int64)


def relabel_state(state: torch.Tensor, permutation: torch.Tensor | list[int]) -> torch.Tensor:
    p = torch.as_tensor(permutation, device=state.device, dtype=torch.int64)
    return p[state.long()].to(dtype=state.dtype)


@dataclass(frozen=True)
class RedescriptionCase:
    name: str
    exact_trace_equivariance: bool
    max_state_entropy_error: float
    max_block_entropy_error: float
    max_spatial_mi_error: float
    active_density_error: float
    conditional_entropy_error: float
    compression_ratio_error: float
    note: str


@dataclass(frozen=True)
class RedescriptionAudit:
    passed: bool
    cases: list[RedescriptionCase]

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "cases": [asdict(case) for case in self.cases]}


def audit_state_relabellings(
    initial: torch.Tensor,
    rule: ReversibleRule,
    *,
    steps: int = 64,
) -> RedescriptionAudit:
    """Test conjugacy, not merely similar-looking output.

    For a state relabelling C and conjugated rule C F C^-1, every frame must be
    byte-exactly C applied to the original frame. Neutral-label exchange and
    global sign reversal are the two declared harmless redescriptions.
    """
    base = run_trajectory(initial, rule, steps)
    base_candidates = trajectory_candidate_values(base)
    cases: list[RedescriptionCase] = []
    for name, permutation, note in (
        (
            "neutral-label-swap",
            NEUTRAL_SWAP,
            "Exchanges the two zero-charge states; exact Q is unchanged.",
        ),
        (
            "global-sign-flip",
            SIGN_FLIP,
            "Exchanges + and -; Q changes sign, so a Q=0 trace remains Q=0.",
        ),
    ):
        transformed_initial = relabel_state(initial, permutation)
        transformed_rule = rule.conjugated(permutation, name=f"{rule.name}_{name}")
        transformed = run_trajectory(transformed_initial, transformed_rule, steps)
        expected = relabel_state(base, permutation)
        exact = bool(torch.equal(transformed.cpu(), expected.cpu()))
        # Compare the original trace with its bijectively redescribed trace.
        # This is stronger than comparing transformed with expected (which is
        # already covered by exact conjugacy) and directly tests measure
        # sensitivity to a harmless code change.
        se = float(
            (
                state_entropy(transformed.flatten(0, 1))
                - state_entropy(base.flatten(0, 1))
            )
            .abs()
            .max()
        )
        be = float(
            (
                block_entropy(transformed.flatten(0, 1))
                - block_entropy(base.flatten(0, 1))
            )
            .abs()
            .max()
        )
        smi = float(
            (
                spatial_mutual_information(transformed.flatten(0, 1))
                - spatial_mutual_information(base.flatten(0, 1))
            )
            .abs()
            .max()
        )
        candidates = trajectory_candidate_values(transformed)
        cases.append(
            RedescriptionCase(
                name=name,
                exact_trace_equivariance=exact,
                max_state_entropy_error=se,
                max_block_entropy_error=be,
                max_spatial_mi_error=smi,
                active_density_error=abs(
                    candidates["active_density"] - base_candidates["active_density"]
                ),
                conditional_entropy_error=abs(
                    candidates["conditional_entropy"]
                    - base_candidates["conditional_entropy"]
                ),
                compression_ratio_error=abs(
                    candidates["compression_ratio"] - base_candidates["compression_ratio"]
                ),
                note=note,
            )
        )

    # Even torus translations preserve both Margolus partitions exactly.
    translated_initial = torch.roll(initial, shifts=(2, -2), dims=(-2, -1))
    translated = run_trajectory(translated_initial, rule, steps)
    expected_translation = torch.roll(base, shifts=(2, -2), dims=(-2, -1))
    exact_translation = bool(torch.equal(translated.cpu(), expected_translation.cpu()))
    translated_candidates = trajectory_candidate_values(translated)
    cases.append(
        RedescriptionCase(
            name="even-torus-translation",
            exact_trace_equivariance=exact_translation,
            max_state_entropy_error=float(
                (
                    state_entropy(translated.flatten(0, 1))
                    - state_entropy(base.flatten(0, 1))
                )
                .abs()
                .max()
            ),
            max_block_entropy_error=float(
                (
                    block_entropy(translated.flatten(0, 1))
                    - block_entropy(base.flatten(0, 1))
                )
                .abs()
                .max()
            ),
            max_spatial_mi_error=float(
                (
                    spatial_mutual_information(translated.flatten(0, 1))
                    - spatial_mutual_information(base.flatten(0, 1))
                )
                .abs()
                .max()
            ),
            active_density_error=abs(
                translated_candidates["active_density"] - base_candidates["active_density"]
            ),
            conditional_entropy_error=abs(
                translated_candidates["conditional_entropy"]
                - base_candidates["conditional_entropy"]
            ),
            compression_ratio_error=abs(
                translated_candidates["compression_ratio"]
                - base_candidates["compression_ratio"]
            ),
            note="A shift by whole 2x2 block periods is a partition-preserving coordinate change.",
        )
    )
    return RedescriptionAudit(
        passed=all(case.exact_trace_equivariance for case in cases), cases=cases
    )
