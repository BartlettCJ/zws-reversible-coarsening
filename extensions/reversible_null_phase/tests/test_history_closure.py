from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import zws_reversible_null_phase.history_closure as closure
from zws_reversible_null_phase.atomic import atomic_write_text, sha256_file
from zws_reversible_null_phase.history_closure import (
    _event_counts,
    inspect_companion_archive,
    resource_summary,
)
from zws_reversible_null_phase.history_publication import RULE_SHA256


def test_resource_summary_preserves_governor_evidence(tmp_path: Path) -> None:
    path = tmp_path / "resource_log.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "utc",
                "cpu_percent",
                "available_ram_gib",
                "gpu_temperature_c",
                "external_gpu_memory_mib",
                "reasons",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "utc": "2026-08-29T00:00:00+00:00",
                    "cpu_percent": "20",
                    "available_ram_gib": "30",
                    "gpu_temperature_c": "50",
                    "external_gpu_memory_mib": "",
                    "reasons": "",
                },
                {
                    "utc": "2026-08-29T01:00:00+00:00",
                    "cpu_percent": "60",
                    "available_ram_gib": "20",
                    "gpu_temperature_c": "70",
                    "external_gpu_memory_mib": "2048",
                    "reasons": "foreground_gpu",
                },
            ]
        )
    result = resource_summary(path)
    assert result["samples"] == 2
    assert result["wall_hours"] == 1.0
    assert result["max_cpu_percent"] == 60.0
    assert result["min_available_ram_gib"] == 20.0
    assert result["median_gpu_temperature_c"] == 60.0
    assert result["resource_yield_samples"] == 1


def test_event_counts_keep_censoring_separate_from_events() -> None:
    rows = [
        {"branch_point": "B", "intervention": "global_shuffle", "hit_observed": "True"},
        {"branch_point": "B", "intervention": "global_shuffle", "hit_observed": "False"},
        {"branch_point": "B", "intervention": "tile_32", "hit_observed": "False"},
    ]
    result = _event_counts(rows)
    global_row = next(row for row in result if row["intervention"] == "global_shuffle")
    assert global_row == {
        "branch_point": "B",
        "intervention": "global_shuffle",
        "n": 2,
        "events": 1,
        "right_censored": 1,
    }


def test_companion_archive_verifies_every_manifest_entry(tmp_path: Path) -> None:
    root = "ZWS_Reversible_Null_Phase_Companion_v0.2"
    content = {
        "README.md": b"readme",
        "WITHDRAWN_CLAIM_NOTICE.md": b"withdrawn classifier claim",
        "DATA_VOLUME_NOTE.md": b"data",
        "environment_freeze.txt": b"environment",
        "CITATION.cff": b"citation",
        "LICENSE": b"license",
        ".zenodo.json": b"{}",
        "paper/main.tex": b"paper",
        "evidence/v032_history_erasure/INTEGRITY_AUDIT.json": b"{}",
        "software/src/model.py": b"pass\n",
        "software/eshkol/parity.esh": b"parity",
        "software/rules/winner_d1fe99.json": b"rule",
    }
    entries = [
        {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        for name, raw in sorted(content.items())
    ]
    manifest = {
        "schema": "zws-reversible-null-phase-companion-archive-v2",
        "rule_hash": RULE_SHA256,
        "entry_count": len(entries),
        "entries": entries,
    }
    archive_path = tmp_path / "companion.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, raw in content.items():
            archive.writestr(f"{root}/{name}", raw)
        archive.writestr(f"{root}/ARCHIVE_MANIFEST.json", json.dumps(manifest))
    atomic_write_text(
        archive_path.with_suffix(".zip.sha256"),
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
    )
    result = inspect_companion_archive(archive_path)
    assert result["passed"] is True
    assert result["entry_count"] == len(entries)


def test_completion_record_includes_claim_inventory_and_generator_digest(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    experiment = repo / "experiments/v032"
    paper = repo / "paper"
    release = repo / "release"
    experiment.mkdir(parents=True)
    paper.mkdir(parents=True)
    run.mkdir(parents=True)
    atomic_write_text(
        repo / "FINAL_RESEARCH_STOP_DECISION.md",
        "A. THE MECHANISM IS SUFFICIENTLY RESOLVED FOR THE COMPANION PAPER.\n",
    )

    branch = {
        "classification": "INVENTORY-EQUIVALENT",
        "pooled": {
            "paired_rmst_delay": {
                "point_estimate": 0.0,
                "ci90": [-100.0, 100.0],
                "ci95": [-200.0, 200.0],
            }
        },
    }
    decision = {
        "branch_A": branch,
        "branch_B": branch,
        "combined_mechanism": "A. INVENTORY-DOMINATED THROUGHOUT",
        "tile_interpretation": "inventory_or_very_local_statistics_dominate",
        "density_direction_agreement": True,
        "density_direction_agreement_by_branch": {"A": True, "B": True},
    }
    figures = {
        "figures": [
            {
                "figure_id": f"FIGURE_{index:02d}",
                "outputs": [{"format": "png", "sha256": f"figure-{index}"}],
            }
            for index in range(1, 9)
        ]
    }
    videos = {
        "videos": [
            {"name": name, "sha256": f"video-{index}"}
            for index, name in enumerate(
                (
                    "hero_forward_reverse_exact.mp4",
                    "history_erasure_density020.mp4",
                    "history_erasure_density024.mp4",
                    "history_erasure_short_preview.gif",
                )
            )
        ]
    }
    payloads = {
        "FROZEN_PLAN.json": {
            "starting_commit": "start",
            "payload_sha256": "plan",
            "source_seed_freshness": {"seeds": [6201]},
            "counts": {"maximum_scientific_trajectories": 88},
        },
        "REALIZED_BRANCH_PLAN.json": {
            "payload_sha256": "realized",
            "ineligible_count": 0,
        },
        "INTEGRITY_AUDIT.json": {"realized_scientific_trajectory_count": 17},
        "SCIENTIFIC_DECISION_V032.json": decision,
        "FIGURE_MANIFEST.json": figures,
        "VIDEO_MANIFEST.json": videos,
        "FINAL_VERIFICATION_V032.json": {
            "checks": {"tests": {"passed": True, "summary": "passed"}},
            "failures_and_repairs": [],
            "pause_resume_events": [],
        },
        "INTERVENTION_MANIFEST.json": {
            "entries": [
                {
                    "task_id": "branch1",
                    "permutation_sha256": "permutation-digest",
                }
            ]
        },
        "environment.json": {"python": "test"},
    }

    def fake_load(path: str | Path):
        return payloads[Path(path).name]

    def fake_csv(path: Path):
        if path.name == "SOURCE_TRAJECTORIES.csv":
            return [
                {
                    "source_task_id": "source1",
                    "density": "0.20",
                    "seed": "6201",
                    "branch_A_eligible": "True",
                    "branch_A_phase": "4096",
                    "branch_B_eligible": "True",
                    "branch_B_phase": "8192",
                }
            ]
        if path.name == "INTERVENTION_AUDIT.csv":
            return [
                {
                    "task_id": "branch1",
                    "source_task_id": "source1",
                    "branch_point": "A",
                    "branch_phase": "4096",
                    "intervention": "global_shuffle",
                    "realization": "1",
                    "source_M": "400000",
                    "source_Q": "0",
                    "source_histogram": "[648576,150000,150000,100000]",
                    "altered_state_sha256": "altered-state",
                }
            ]
        if path.name == "RESIDUAL_FIRST_PASSAGE.csv":
            return [
                {
                    "branch_point": "A",
                    "intervention": "global_shuffle",
                    "hit_observed": "True",
                }
            ]
        if path.name == "PRIOR_ART_MATRIX.csv":
            return [{"source": "one"}, {"source": "two"}]
        raise AssertionError(path)

    monkeypatch.setattr(closure, "audit_final_history_package", lambda **_: {"archive": {"path": str(release / "archive.zip"), "sha256": "archive-hash"}})
    monkeypatch.setattr(closure, "load_verified_json", fake_load)
    monkeypatch.setattr(closure, "_read_csv", fake_csv)
    monkeypatch.setattr(closure, "resource_summary", lambda _: {
        "samples": 2,
        "wall_hours": 1.0,
        "max_cpu_percent": 50.0,
        "min_available_ram_gib": 20.0,
        "median_gpu_temperature_c": 55.0,
        "max_gpu_temperature_c": 60.0,
        "resource_yield_samples": 0,
    })
    monkeypatch.setattr(
        closure,
        "_git",
        lambda _repo, *args: "branch-v032" if args[:2] == ("branch", "--show-current") else "artifact-head",
    )
    monkeypatch.setattr(closure, "sha256_file", lambda _: "synthetic-hash")

    result = closure.build_completion_record(
        repo_root=repo,
        run_path=run,
        experiment_path=experiment,
        paper_path=paper,
        release_path=release,
        verification_path=experiment / "FINAL_VERIFICATION_V032.json",
    )
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "Precise scientific claim" in text
    assert "[648576,150000,150000,100000]" in text
    assert "permutation-digest" in text
    assert "altered-state" in text
