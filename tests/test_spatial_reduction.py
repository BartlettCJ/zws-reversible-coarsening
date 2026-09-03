from __future__ import annotations

import torch

from zws_nullity_lab.margolus import step_rule
from zws_nullity_lab.rules import ReversibleRule
from zws_nullity_lab.spatial_analysis import analyze_spatial_state
from zws_nullity_lab.spatial_reduction import (
    composition_contrast_rows,
    contrast_robustness_summary,
    environment_composition_rows,
    local_block_distribution_rows,
    primary_dynamics_rows,
)


def _milestone(*, final: bool = False) -> dict[str, object]:
    state = torch.zeros((128, 128), dtype=torch.uint8)
    state[:, :64] = 3
    state[16:48, 16:48] = 1
    state[80:112, 16:48] = 2
    rule = ReversibleRule.spatial_rotation()
    after = None if final else step_rule(state.unsqueeze(0), rule, 0)[0]
    record = analyze_spatial_state(
        state,
        after=after,
        rule=rule,
        tick=16 if final else 0,
        roles=["final"] if final else ["initial"],
    )
    record.update(
        {
            "stage_id": "smoke",
            "seed_index": 0,
            "initial_seed": 123,
            "exact_state_sha256_uint8": "a" * 64,
        }
    )
    return record


def test_spatial_reducer_retains_every_robust_environment_definition() -> None:
    record = _milestone()
    environments = environment_composition_rows([record])
    contrasts = composition_contrast_rows([record])
    assert len(environments) == 36
    assert len(contrasts) == 9
    assert {row["box_width"] for row in contrasts} == {16, 32, 64}
    assert {row["threshold_scheme"] for row in contrasts} == {
        "q10_q90",
        "q20_q80",
        "q30_q70",
    }
    assert all("dense_minus_sparse_occupied_fraction" in row for row in contrasts)


def test_spatial_reducer_exports_all_motifs_and_conditioned_neighbourhoods() -> None:
    dynamics, motifs, neighbourhoods = primary_dynamics_rows([_milestone()])
    assert len(dynamics) == 3
    assert len(motifs) == 3 * 256
    assert {row["motif_code"] for row in motifs} == set(range(256))
    assert neighbourhoods
    assert {row["central_state"] for row in neighbourhoods} <= {
        "empty",
        "positive",
        "negative",
        "bound",
    }
    assert all("conditional_fraction" in row for row in neighbourhoods)


def test_spatial_reducer_keeps_eulerian_language_and_final_nonestimate() -> None:
    dynamics, _, _ = primary_dynamics_rows([_milestone(), _milestone(final=True)])
    estimated = [row for row in dynamics if row["estimable"]]
    unestimated = [row for row in dynamics if not row["estimable"]]
    assert estimated
    assert len(unestimated) == 1
    assert "positive_eulerian_net_count_change" in estimated[0]
    assert "no outgoing phase" in unestimated[0]["reason"]


def test_spatial_contrast_summary_reports_threshold_sensitivity_without_fitness() -> None:
    contrasts = composition_contrast_rows([_milestone()])
    summary = contrast_robustness_summary(contrasts)
    assert summary
    assert {row["role_group"] for row in summary} == {"initial"}
    assert all("sign_robustness" in row for row in summary)
    assert all("fitness" not in row for row in summary)


def test_local_block_reduction_retains_charge_mass_and_entropy() -> None:
    rows = local_block_distribution_rows([_milestone()])
    assert len(rows) == 3
    nonempty = [row for row in rows if row["block_count"]]
    assert nonempty
    assert all("mean_absolute_signed_charge" in row for row in nonempty)
    assert all("median_occupancy_mass" in row for row in nonempty)
    assert all("median_normalized_state_entropy" in row for row in nonempty)
