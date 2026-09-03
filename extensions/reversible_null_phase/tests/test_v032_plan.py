from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "reversible_null_phase_v032_history_erasure_20260829"
RULE_HASH = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"


def _load(name: str) -> dict:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_v032_frozen_plan_has_exact_bounded_scope() -> None:
    plan = _load("FROZEN_PLAN.json")
    manifest = _load("INTERVENTION_MANIFEST.json")
    assert plan["winner_raw_uint8_sha256"] == RULE_HASH
    assert manifest["winner_raw_uint8_sha256"] == RULE_HASH
    assert len(plan["source_tasks"]) == 16
    assert manifest["entry_count"] == 72
    assert plan["counts"]["maximum_scientific_trajectories"] == 88
    assert {task["seed"] for task in plan["source_tasks"]} == set(range(6201, 6209))
    assert {task["density"] for task in plan["source_tasks"]} == {0.20, 0.24}
    all_ids = [task["task_id"] for task in plan["source_tasks"]] + [
        task["task_id"] for task in manifest["entries"]
    ]
    assert len(all_ids) == len(set(all_ids)) == 88


def test_v032_payload_hashes_and_sidecars_verify() -> None:
    for name in ("FROZEN_PLAN.json", "INTERVENTION_MANIFEST.json"):
        payload = _load(name)
        claimed = payload.pop("payload_sha256")
        assert _digest(payload) == claimed
    for sidecar in EXPERIMENT.glob("*.sha256"):
        expected, filename = sidecar.read_text(encoding="ascii").strip().split("  ", 1)
        assert hashlib.sha256((sidecar.parent / filename).read_bytes()).hexdigest() == expected


def test_v032_interventions_are_frozen_and_phase_compatible() -> None:
    manifest = _load("INTERVENTION_MANIFEST.json")
    counts: dict[tuple[str, str], int] = {}
    for entry in manifest["entries"]:
        key = (entry["branch_point"], entry["intervention"])
        counts[key] = counts.get(key, 0) + 1
        if "temporary_even_offset_yx" in entry:
            assert all(value % 2 == 0 for value in entry["temporary_even_offset_yx"])
        if "translation_offset_yx" in entry:
            assert all(value % 2 == 0 for value in entry["translation_offset_yx"])
        if entry["intervention"].startswith("tile_"):
            tile_size = entry["tile_size"]
            tile_count = (1024 // tile_size) ** 2
            assert sorted(entry["tile_permutation"]) == list(range(tile_count))
            values = np.asarray(entry["tile_permutation"], dtype="<i8")
            assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == entry["tile_permutation_sha256"]
    assert counts == {
        ("A", "global_shuffle"): 16,
        ("B", "global_shuffle"): 20,
        ("B", "tile_32"): 16,
        ("B", "tile_128"): 16,
        ("B", "even_translation_null"): 4,
    }


def test_v032_global_permutation_digests_reproduce() -> None:
    manifest = _load("INTERVENTION_MANIFEST.json")
    for entry in manifest["entries"]:
        if entry["intervention"] != "global_shuffle":
            continue
        rng = np.random.Generator(np.random.PCG64(entry["intervention_seed"]))
        values = rng.permutation(1024 * 1024).astype("<i8", copy=False)
        assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == entry["permutation_sha256"]
