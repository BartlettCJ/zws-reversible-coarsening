from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic import atomic_write_json, sha256_file
from .history_figures import (
    _branch_entry,
    _read_json,
    _realized,
    _representative_for_point,
    _representatives,
    _state_panel_from_record,
    _source_result,
    figure_01_hero,
    figure_02_history_branches,
    figure_03_fixed_occupancy,
    figure_04_event_proximal,
    figure_05_multiscale_memory,
    figure_06_micro_macro,
    figure_07_mechanism_ledger,
    figure_08_research_story,
)
from .history_media import reconstruct_untouched_media_states
from .history_publication import (
    build_history_dashboard,
    build_history_explorer,
    explorer_frame,
    read_verified_csv,
)
from .history_reporting import build_history_report
from .history_video import _history_exact_records, build_required_videos


def _require_completed_run(run: Path) -> dict[str, Any]:
    status_path = run / "status.json"
    if not status_path.is_file():
        raise RuntimeError("history campaign status is missing")
    status = __import__("json").loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "branches_complete":
        raise RuntimeError(
            f"publication is sealed until the governed run reaches branches_complete; current={status.get('state')}"
        )
    return status


def _metric_for_frame(
    *,
    run: Path,
    source_id: str,
    intervention: str,
    elapsed: int,
    phase: int,
    realized: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if intervention == "untouched":
        source = _source_result(run, source_id)
        metric = next((row for row in source["series"] if int(row["phase"]) == phase), {})
        hit = source["time_to_event"]["T_hit"]
        branch_phase = int(source["branch_points"]["B"]["phase"])
        event_elapsed = None if hit is None else int(hit) - branch_phase
    else:
        entry = _branch_entry(realized, source_id, "B", intervention)
        branch = _read_json(run / "branches" / entry["task_id"] / "branch_result.json")
        metric = next((row for row in branch["series"] if int(row["elapsed"]) == elapsed), {})
        event_elapsed = branch["residual_time_to_event"]["R_hit"]
    events = []
    if event_elapsed is not None:
        events.append({"label": "T_hit", "elapsed": int(event_elapsed)})
    return {
        "low_k_power": metric.get("low_k_power"),
        "low_k_fraction": metric.get("low_k_fraction"),
        "finite_domain_scale": metric.get("finite_empty_scale"),
        "interface_density": metric.get("interface_density"),
        "strong_domain": metric.get("strong_domain"),
    }, events


def _build_explorer_payload(
    run: Path,
    experiment: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, int]]]:
    realized = _realized(run)
    frames_by_density: dict[str, dict[str, list[dict[str, Any]]]] = {}
    fixed_crops: dict[str, dict[str, int]] = {}
    for representative in _representatives(experiment):
        source_id = str(representative["source_task_id"])
        density = float(representative["density"])
        seed = int(representative["seed"])
        density_key = f"{density:.2f}"
        by_branch, crop, _ = _history_exact_records(run, experiment, representative)
        common = sorted(set.intersection(*(set(records) for records in by_branch.values())))
        fixed_crops[density_key] = {
            "x": crop[0],
            "y": crop[1],
            "width": crop[2],
            "height": crop[3],
        }
        frames_by_density[density_key] = {}
        for intervention, records in by_branch.items():
            frames: list[dict[str, Any]] = []
            for elapsed in common:
                state, record = records[elapsed]
                panel = _state_panel_from_record(
                    record,
                    source_task=source_id,
                    seed=seed,
                    density=density,
                    branch_point="B",
                    intervention=intervention,
                    requested_elapsed=elapsed,
                    extra={"panel": "interactive_explorer"},
                )
                metrics, events = _metric_for_frame(
                    run=run,
                    source_id=source_id,
                    intervention=intervention,
                    elapsed=elapsed,
                    phase=int(record["phase"]),
                    realized=realized,
                )
                frames.append(explorer_frame(state, panel, metrics=metrics, events=events))
            frames_by_density[density_key][intervention.replace("_", " ")] = frames
    return frames_by_density, fixed_crops


def build_history_publication_package(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    run = Path(run_path).resolve()
    experiment = Path(experiment_path).resolve()
    paper = Path(paper_path).resolve()
    _require_completed_run(run)

    report_result = build_history_report(run, experiment)
    integrity = _read_json(experiment / "INTEGRITY_AUDIT.json")
    if not integrity["passed"]:
        raise RuntimeError("publication stopped because the frozen integrity audit failed")

    representatives = _representatives(experiment)
    reconstruct_untouched_media_states(
        config_path,
        run,
        [str(row["source_task_id"]) for row in representatives],
        branch_point="B",
    )
    realized = _realized(run)
    representative_a = _representative_for_point(run, experiment, realized, "A")
    reconstruct_untouched_media_states(
        config_path,
        run,
        [str(representative_a["source_task_id"])],
        branch_point="A",
    )

    figure_dir = paper / "figures"
    figures = [
        figure_01_hero(repo, figure_dir),
        figure_02_history_branches(run, experiment, figure_dir),
        figure_03_fixed_occupancy(run, experiment, figure_dir),
        figure_04_event_proximal(run, experiment, figure_dir),
        figure_05_multiscale_memory(experiment, figure_dir),
        figure_06_micro_macro(run, experiment, figure_dir),
        figure_07_mechanism_ledger(figure_dir),
        figure_08_research_story(repo, experiment, figure_dir),
    ]
    if len(figures) != 8 or len({entry["figure_id"] for entry in figures}) != 8:
        raise RuntimeError("required eight-figure set is incomplete or duplicated")
    figure_manifest = {
        "schema": "zws-reversible-null-phase-v032-figure-manifest-v1",
        "rule_hash": report_result.get("winner_hash", None)
        or "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c",
        "rendering_spec_path": str((experiment / "MEDIA_RENDERING_SPEC.json").resolve()),
        "rendering_spec_sha256": sha256_file(experiment / "MEDIA_RENDERING_SPEC.json"),
        "figures": figures,
    }
    atomic_write_json(experiment / "FIGURE_MANIFEST.json", figure_manifest)

    media_dir = paper / "media"
    videos = build_required_videos(repo, run, experiment, media_dir)
    if {entry["name"] for entry in videos} != {
        "hero_forward_reverse_exact.mp4",
        "history_erasure_density020.mp4",
        "history_erasure_density024.mp4",
        "history_erasure_short_preview.gif",
    }:
        raise RuntimeError("required video set is incomplete")
    video_manifest = {
        "schema": "zws-reversible-null-phase-v032-video-manifest-v1",
        "rule_hash": figure_manifest["rule_hash"],
        "videos": videos,
    }
    atomic_write_json(experiment / "VIDEO_MANIFEST.json", video_manifest)

    decision = _read_json(experiment / "SCIENTIFIC_DECISION_V032.json")
    frames_by_density, fixed_crops = _build_explorer_payload(run, experiment)
    explorer = build_history_explorer(
        experiment / "history_erasure_explorer.html",
        frames_by_density=frames_by_density,
        fixed_crops=fixed_crops,
        title=f"ZWSLab v0.3.2 history erasure · {decision['combined_mechanism']}",
    )
    dashboard = build_history_dashboard(
        experiment / "dashboard.html",
        decision=decision,
        results_rows=read_verified_csv(experiment / "HISTORY_ERASURE_RESULTS.csv"),
        survival_rows=read_verified_csv(experiment / "SURVIVAL_SUMMARY.csv"),
        integrity=integrity,
    )
    return {
        "report": report_result,
        "figure_manifest": str((experiment / "FIGURE_MANIFEST.json").resolve()),
        "video_manifest": str((experiment / "VIDEO_MANIFEST.json").resolve()),
        "figures": len(figures),
        "videos": len(videos),
        "dashboard": dashboard,
        "explorer": explorer,
    }
