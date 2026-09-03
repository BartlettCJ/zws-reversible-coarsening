from __future__ import annotations

import torch

from zws_nullity_lab.margolus import step_rule
from zws_nullity_lab.rules import ReversibleRule
from zws_nullity_lab.spatial_analysis import analyze_spatial_state


def _structured_state() -> torch.Tensor:
    state = torch.zeros((64, 64), dtype=torch.uint8)
    state[8:56, 8:56] = 3
    state[16:32, 16:32] = 1
    state[32:48, 32:48] = 2
    return state


def test_spatial_milestone_retains_all_frozen_robustness_definitions() -> None:
    rule = ReversibleRule.spatial_rotation()
    state = _structured_state()
    after = step_rule(state.unsqueeze(0), rule, 0)[0]
    result = analyze_spatial_state(
        state,
        after=after,
        rule=rule,
        tick=0,
        roles=["initial"],
    )
    assert result["diagnostic_revision"] == 2
    assert len(result["density_environment_robustness"]) == 9
    assert len(result["interface_threshold_robustness"]) == 3
    assert result["primary_definition"] == {
        "box_width": 32,
        "quantiles": [0.2, 0.8],
        "phase_threshold": "quantile_midpoint",
    }
    assert result["primary_environment_dynamics"]["estimable"] is True
    assert set(result["primary_local_block_distributions"]) == {
        "dense_interior",
        "sparse_interior",
        "interface",
    }
    assert set(result["state_specific_structure_factors"]) == {
        "empty",
        "positive",
        "negative",
        "bound",
    }


def test_final_spatial_milestone_does_not_invent_outgoing_flux() -> None:
    result = analyze_spatial_state(
        _structured_state(),
        after=None,
        rule=ReversibleRule.identity(),
        tick=16,
        roles=["final"],
    )
    dynamics = result["primary_environment_dynamics"]
    assert dynamics["estimable"] is False
    assert "no outgoing phase" in dynamics["reason"]
