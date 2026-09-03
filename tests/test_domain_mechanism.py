from __future__ import annotations

import torch

from zws_nullity_lab.domain_mechanism import (
    central_state_conditioned_neighbourhood_frequencies,
    conditional_environment_diagnostics,
    density_environment_masks,
    exact_occupancy_mass,
    interface_geometry,
    conditional_local_block_distributions,
    nonoverlap_local_fields,
    otsu_threshold,
    reaction_cell_map,
    sliding_block_code_histogram,
    state_composition,
    state_transition_matrix,
    top_neighbourhood_frequencies,
    toroidal_box_mean,
)
from zws_nullity_lab.margolus import step_rule
from zws_nullity_lab.rules import ReversibleRule


def test_state_composition_and_exact_mass() -> None:
    state = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 3]], dtype=torch.uint8)
    result = state_composition(state)
    assert result["state_counts"] == {"empty": 1, "positive": 2, "negative": 2, "bound": 3}
    assert result["total_signed_charge"] == 0
    assert result["occupied_count"] == 7
    assert int(exact_occupancy_mass(state).item()) == 7
    batch = torch.stack([state, torch.zeros_like(state)])
    assert exact_occupancy_mass(batch).tolist() == [7, 0]


def test_toroidal_box_mean_preserves_constant_and_total() -> None:
    constant = torch.ones((8, 8), dtype=torch.float32)
    assert torch.equal(toroidal_box_mean(constant, 4), constant)
    impulse = torch.zeros((8, 8), dtype=torch.float32)
    impulse[0, 0] = 1.0
    averaged = toroidal_box_mean(impulse, 4)
    assert averaged.shape == impulse.shape
    assert torch.isclose(averaged.sum(), torch.tensor(1.0))
    assert int((averaged > 0).sum().item()) == 16


def test_density_regions_and_interface_are_toroidal_and_transparent() -> None:
    state = torch.zeros((16, 16), dtype=torch.uint8)
    state[:, :8] = 3
    result = density_environment_masks(
        state,
        width=3,
        lower_quantile=0.2,
        upper_quantile=0.8,
        interface_band_radius=1,
    )
    assert result["dense"].any()
    assert result["sparse"].any()
    assert result["interface"].any()
    assert not bool((result["dense_interior"] & result["interface"]).any())
    assert not bool((result["sparse_interior"] & result["interface"]).any())
    assert result["metadata"]["lower_threshold"] <= result["metadata"]["upper_threshold"]


def test_otsu_constant_and_bimodal_fields() -> None:
    assert otsu_threshold(torch.ones((4, 4))) is None
    values = torch.cat([torch.zeros(64), torch.ones(64)])
    threshold = otsu_threshold(values)
    assert threshold is not None
    assert 0.0 <= threshold < 1.0


def test_sliding_motif_and_neighbourhood_counts() -> None:
    empty = torch.zeros((4, 4), dtype=torch.uint8)
    motif = sliding_block_code_histogram(empty)
    assert motif[0] == 16
    assert sum(motif) == 16

    bound = torch.full((4, 4), 3, dtype=torch.uint8)
    neighbourhoods = top_neighbourhood_frequencies(bound, limit=2)
    assert neighbourhoods["observed_count"] == 16
    assert neighbourhoods["top"][0]["code"] == 4**9 - 1
    assert neighbourhoods["top"][0]["count"] == 16

    mixed = torch.tensor(
        [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]],
        dtype=torch.uint8,
    )
    conditioned = central_state_conditioned_neighbourhood_frequencies(
        mixed, limit_per_state=2
    )
    assert set(conditioned) == {"empty", "positive", "negative", "bound"}
    assert [conditioned[name]["observed_count"] for name in conditioned] == [4, 4, 4, 4]
    for code, name in enumerate(conditioned):
        assert conditioned[name]["central_state_code"] == code
        for pattern in conditioned[name]["top"]:
            assert pattern["pattern_row_major"][1][1] == code


def test_exact_nonoverlap_local_fields_and_conditioning() -> None:
    state = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 0, 0, 0],
            [1, 1, 2, 2],
            [3, 3, 0, 0],
        ],
        dtype=torch.uint8,
    )
    fields = nonoverlap_local_fields(state, block=2)
    assert fields["signed_charge"].tolist() == [[0, 0], [2, -2]]
    assert fields["occupancy_mass"].tolist() == [[3, 0], [4, 2]]
    masks = {
        "all": torch.ones_like(state, dtype=torch.bool),
        "bottom": torch.arange(4).unsqueeze(1).expand(4, 4) >= 2,
    }
    result = conditional_local_block_distributions(state, masks, block=2)
    assert result["all"]["block_count"] == 4
    assert result["all"]["signed_charge"]["histogram"] == {"-2": 1, "0": 2, "2": 1}
    assert result["bottom"]["block_count"] == 2
    assert result["bottom"]["signed_charge"]["mean_absolute"] == 2.0


def test_reaction_map_and_conditional_diagnostics() -> None:
    rule = ReversibleRule.reactive_rotation(swaps=1)
    reaction_input = int(torch.nonzero(rule.reaction_mask(), as_tuple=False)[0].item())
    state = rule.alphabet.block_digits_cpu[reaction_input].reshape(2, 2).to(torch.uint8)
    after = step_rule(state.unsqueeze(0), rule, 0)[0]
    reaction = reaction_cell_map(state, rule, 0)
    assert bool(reaction.all())
    result = conditional_environment_diagnostics(
        state,
        after,
        rule=rule,
        phase=0,
        masks={"all": torch.ones_like(state, dtype=torch.bool)},
    )
    assert result["all"]["reaction_cell_fraction"] == 1.0
    assert sum(sum(row) for row in result["all"]["eulerian_state_transition_matrix"]) == 4
    conditioned = result["all"]["central_state_conditioned_neighbourhoods_3x3"]
    assert sum(value["observed_count"] for value in conditioned.values()) == 4

    rotation = ReversibleRule.spatial_rotation()
    assert not bool(reaction_cell_map(state, rotation, 0).any())


def test_state_transition_matrix_and_interface_geometry() -> None:
    before = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8)
    after = torch.tensor([[1, 1], [0, 3]], dtype=torch.uint8)
    transitions = state_transition_matrix(before, after)
    assert transitions[0][1] == 1
    assert transitions[1][1] == 1
    assert transitions[2][0] == 1
    assert transitions[3][3] == 1

    phase = torch.zeros((4, 4), dtype=torch.bool)
    phase[:, :2] = True
    interface = phase != torch.roll(phase, shifts=1, dims=-1)
    geometry = interface_geometry(phase, interface)
    assert geometry["horizontal_crossing_edges"] == 8
    assert geometry["vertical_crossing_edges"] == 0
    assert geometry["total_crossing_edges"] == 8
