from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zws_nullity_lab.control import WorkloadPaused
from zws_nullity_lab.explorer import load_trajectory_bundle
from zws_nullity_lab.initial import SeedSpec, balanced_random
from zws_nullity_lab.margolus import run_trajectory
from zws_nullity_lab.observe import _reconstruct_clip, _torus_crop, observe_rule
from zws_nullity_lab.rules import ReversibleRule


def _options() -> dict:
    return {
        "height": 24,
        "width": 24,
        "steps": 30,
        "device": "cpu",
        "seed": 7711,
        "sample_every": 2,
        "checkpoint_every": 10,
        "event_count": 2,
        "event_min_separation": 4,
        "event_before": 3,
        "event_after": 2,
        "crop_size": 8,
        "tile_size": 6,
    }


def test_sparse_observation_reconstructs_exact_local_clips(tmp_path: Path) -> None:
    rule = ReversibleRule.reactive_rotation(12, name="sparse-test")
    root = tmp_path / "observation"
    result = observe_rule(rule, root, **_options())

    assert result["status"] == "complete"
    assert result["world"] == [24, 24]
    assert result["steps"] == 30
    assert len(result["events"]) == 2
    preview, metadata, _ = load_trajectory_bundle(root)
    assert preview.shape[1:] == (8, 8)
    assert len(metadata["ticks"]) == preview.shape[0]
    assert metadata["context"]["sparse_observation"] is True

    direct = run_trajectory(
        balanced_random(1, 24, 24, spec=SeedSpec(), seed=7711), rule, 30
    )
    for record in result["events"]:
        clip = np.load(record["states_npy"], allow_pickle=False)
        for index, tick in enumerate(record["clip_ticks"]):
            expected = _torus_crop(
                direct[tick],
                int(record["hotspot_x"]),
                int(record["hotspot_y"]),
                8,
            )
            assert np.array_equal(clip[index], expected.numpy())

    reverse_clip, reverse_ticks, reconstruction = _reconstruct_clip(
        root,
        rule,
        event_tick=12,
        center_x=5,
        center_y=7,
        before=3,
        after=2,
        crop_size=8,
        final_tick=30,
        device=torch.device("cpu"),
    )
    assert reconstruction["mode"] == "backward"
    for index, tick in enumerate(reverse_ticks):
        assert torch.equal(reverse_clip[index], _torus_crop(direct[tick], 5, 7, 8))


def test_sparse_observation_resume_matches_uninterrupted(tmp_path: Path) -> None:
    rule = ReversibleRule.reactive_rotation(12, name="resume-sparse-test")
    options = _options()
    interrupted = tmp_path / "interrupted"
    with pytest.raises(WorkloadPaused, match="exact tick 13"):
        observe_rule(rule, interrupted, stop_after_tick=13, **options)
    resumed = observe_rule(rule, interrupted, resume=True, **options)
    direct_root = tmp_path / "direct"
    direct = observe_rule(rule, direct_root, **options)

    resumed_metrics = json.loads((interrupted / "metrics.json").read_text(encoding="utf-8"))
    direct_metrics = json.loads((direct_root / "metrics.json").read_text(encoding="utf-8"))
    assert resumed_metrics == direct_metrics
    assert [event["tick"] for event in resumed["events"]] == [
        event["tick"] for event in direct["events"]
    ]

    resumed_final = torch.load(
        interrupted / "checkpoints" / "states" / "state_t000000000030.pt",
        map_location="cpu",
        weights_only=False,
    )["state"]
    direct_final = torch.load(
        direct_root / "checkpoints" / "states" / "state_t000000000030.pt",
        map_location="cpu",
        weights_only=False,
    )["state"]
    assert torch.equal(resumed_final, direct_final)
