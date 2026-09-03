from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from zws_nullity_lab.checkpoint import available_checkpoints, load_checkpoint
from zws_nullity_lab.config import load_config, validate_config
from zws_nullity_lab.control import WorkloadPaused
from zws_nullity_lab.rules import ReversibleRule
from zws_nullity_lab.search import run_search


def _chunked_smoke() -> dict:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.yaml")
    config["resilience"]["candidate_batch"] = 2
    config["resilience"]["checkpoint_keep"] = 8
    validate_config(config)
    return config


def test_mid_generation_resume_is_exact(tmp_path: Path) -> None:
    config = _chunked_smoke()
    uninterrupted = run_search(deepcopy(config), tmp_path / "uninterrupted")
    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(WorkloadPaused):
        run_search(deepcopy(config), interrupted_root, stop_after_chunks=1)
    resumed = run_search(deepcopy(config), interrupted_root, resume=True)

    direct_rule = ReversibleRule.load(uninterrupted.best_rule_path)
    resumed_rule = ReversibleRule.load(resumed.best_rule_path)
    assert direct_rule.table.tolist() == resumed_rule.table.tolist()
    assert [item.__dict__ for item in uninterrupted.generations] == [
        item.__dict__ for item in resumed.generations
    ]
    assert uninterrupted.unique_rules_evaluated == resumed.unique_rules_evaluated
    assert resumed.resumed


def test_corrupt_latest_checkpoint_falls_back(tmp_path: Path) -> None:
    config = _chunked_smoke()
    root = tmp_path / "run"
    with pytest.raises(WorkloadPaused):
        run_search(config, root, stop_after_chunks=1)
    checkpoints = available_checkpoints(root)
    assert len(checkpoints) >= 2
    checkpoints[0].write_bytes(b"deliberately-corrupt")
    payload, selected = load_checkpoint(root)
    assert selected != checkpoints[0]
    assert payload["schema"] == "zws-nullity-search-checkpoint-v2"


def test_resume_rejects_changed_config_without_overwriting_resolved_file(tmp_path: Path) -> None:
    config = _chunked_smoke()
    root = tmp_path / "run"
    with pytest.raises(WorkloadPaused):
        run_search(deepcopy(config), root, stop_after_chunks=1)
    resolved_before = (root / "config.resolved.yaml").read_bytes()
    changed = deepcopy(config)
    changed["search"]["generations"] = int(changed["search"]["generations"]) + 1
    with pytest.raises(ValueError, match="configuration hash"):
        run_search(changed, root, resume=True)
    assert (root / "config.resolved.yaml").read_bytes() == resolved_before
