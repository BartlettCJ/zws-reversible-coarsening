from __future__ import annotations

import hashlib

import torch

from zws_nullity_lab.confirmation_campaign import (
    DEFAULT_RESOURCE_CONFIG,
    _initial_batch,
    rule_table_digest,
    run_condition_batch,
    tensor_digest,
)
from zws_nullity_lab.control import RunControl, SignalLatch
from zws_nullity_lab.resources import ResourceGovernor
from zws_nullity_lab.rules import ReversibleRule


def test_initial_batch_is_paired_and_independent_of_condition() -> None:
    config = {
        "positive_fraction": 0.12,
        "bound_fraction": 0.08,
        "localized": True,
        "localized_fraction": 0.45,
    }
    seeds = [101, 202, 303]
    first = _initial_batch(seeds, height=64, width=64, initial_config=config)
    second = _initial_batch(seeds, height=64, width=64, initial_config=config)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert [int((item == 1).sum()) for item in first] == [94, 94, 94]
    assert [int((item == 2).sum()) for item in first] == [94, 94, 94]


def test_tensor_digest_uses_raw_uint8_bytes() -> None:
    state = torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8)
    assert tensor_digest(state) == hashlib.sha256(bytes([0, 1, 2, 3])).hexdigest()


def test_confirmation_governor_uses_wddm_memory_pressure_not_process_names() -> None:
    assert DEFAULT_RESOURCE_CONFIG["yield_for_external_gpu_processes"] is False
    assert DEFAULT_RESOURCE_CONFIG["external_gpu_memory_mib"] == 4096
    assert DEFAULT_RESOURCE_CONFIG["gpu_driver_allowance_mib"] == 3072


def test_condition_runner_checkpoints_and_writes_paired_trajectories(tmp_path) -> None:
    rule = ReversibleRule.identity()
    rule_path = tmp_path / "identity.json"
    rule.save(rule_path)
    root = tmp_path / "output"
    rows = [
        {
            "run_id": "smoke__identity__seed00",
            "condition_id": "identity",
            "seed_index": 0,
            "initial_seed": 1001,
            "shuffle_base_seed": 2001,
        },
        {
            "run_id": "smoke__identity__seed01",
            "condition_id": "identity",
            "seed_index": 1,
            "initial_seed": 1002,
            "shuffle_base_seed": 2002,
        },
    ]
    condition = {
        "condition_id": "identity",
        "rule_file": str(rule_path),
        "rule_table_sha256_uint8": rule_table_digest(rule),
    }
    initial = {
        "positive_fraction": 0.12,
        "bound_fraction": 0.08,
        "localized": True,
        "localized_fraction": 0.45,
    }
    diagnostics = {
        "sample_interval_phases": 8,
        "recurrence_lags_phases": [1, 2, 4, 8],
        "coarse_blocks": [8, 16, 32, 64],
    }
    governor = ResourceGovernor({"enabled": False}, run_directory=root)
    control = RunControl(root)
    with SignalLatch() as signal_latch:
        completion = run_condition_batch(
            root=root,
            stage_id="smoke",
            condition=condition,
            trajectory_rows=rows,
            initial_config=initial,
            diagnostics_config=diagnostics,
            height=64,
            width=64,
            phases=8,
            manifest_digest="a" * 64,
            device=torch.device("cpu"),
            resume=True,
            governor=governor,
            control=control,
            signal_latch=signal_latch,
        )
    assert completion["status"] == "complete"
    assert (root / "smoke" / "identity" / "checkpoint.pt.sha256").is_file()
    for row in rows:
        trajectory = root / "trajectories" / f"{row['run_id']}.json"
        assert trajectory.is_file()
        text = trajectory.read_text(encoding="utf-8")
        assert '"charge_violation_count": 0' in text
        assert '"8": 1.0' in text
