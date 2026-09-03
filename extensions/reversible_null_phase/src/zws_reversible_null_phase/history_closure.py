from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
from typing import Any, Iterable
import zipfile

from PIL import Image

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_manuscript import result_language
from .history_media import load_verified_json, verify_sidecar
from .history_publication import RULE_SHA256, validate_panel_record
from .rule import ReversibleRule


EXPECTED_TAGS = {
    "discovery-d1fe99b5735d": {
        "object": "1712ae575d40db2decc2ca8f0e267a681bee8b66",
        "peeled": "74b4243c9518ec6f95b906de1a374bc277a2a5d5",
    },
    "v0.1.4-windows-discovery-freeze": {
        "object": "4c17ff8b2408f8ef0d43569029399b4e42013544",
        "peeled": "74b4243c9518ec6f95b906de1a374bc277a2a5d5",
    },
    "overnight-phase-separation-confirmation-20260825": {
        "object": "eb73e793acff16133e8d5fa58dfa65fda6469364",
        "peeled": "970d1f118836d04fb2d5ab64448c418ca05094b2",
    },
}

REQUIRED_EXPERIMENT_ARTIFACTS = (
    "PREREGISTRATION.md",
    "FROZEN_PLAN.json",
    "INTERVENTION_MANIFEST.json",
    "ANALYSIS_PLAN.md",
    "FIGURE_PLAN.md",
    "FINAL_SCOPE.md",
    "MEDIA_RENDERING_SPEC.json",
    "HISTORY_ERASURE_REPORT.md",
    "HISTORY_ERASURE_RESULTS.csv",
    "SOURCE_TRAJECTORIES.csv",
    "INTERVENTION_AUDIT.csv",
    "RESIDUAL_FIRST_PASSAGE.csv",
    "SURVIVAL_SUMMARY.csv",
    "MULTISCALE_MEMORY.csv",
    "ENTROPY_HIERARCHY.csv",
    "ANISOTROPY_RESULTS.csv",
    "REPRESENTATIVE_STATE_INDEX.csv",
    "INTEGRITY_AUDIT.json",
    "SCIENTIFIC_DECISION_V032.json",
    "FIGURE_MANIFEST.json",
    "VIDEO_MANIFEST.json",
    "dashboard.html",
    "history_erasure_explorer.html",
    "ERRATUM_Q_AND_M.md",
)

REQUIRED_PRIOR_ART = (
    "PRIOR_ART_REVIEW.md",
    "PRIOR_ART_MATRIX.csv",
    "SEARCH_LOG.md",
    "references.bib",
    "DOI_AUDIT.csv",
)

REQUIRED_PAPER_ARTIFACTS = (
    "main.tex",
    "supplement.tex",
    "references.bib",
    "manuscript.md",
    "abstract_options.md",
    "title_options.md",
    "lay_summary.md",
    "highlights.md",
    "graphical_abstract.png",
    "ONE_PAGE_LAY_SUMMARY.pdf",
    "ONE_PAGE_LAY_SUMMARY.png",
    "LAY_ARTIFACT_METADATA.json",
    "ZWS_COMPANION_INSERT.md",
    "ZWS_REVISION_NOTES.md",
    "README.md",
    "tables/history_primary.tex",
    "build/manuscript.pdf",
    "build/supplement.pdf",
)

REQUIRED_FIGURES = {
    "FIGURE_01_ZERO_BALANCE_REVERSIBLE_HERO",
    "FIGURE_02_HISTORY_ERASURE_BRANCHES",
    "FIGURE_03_FIXED_OCCUPANCY_HISTORY_ERASURE",
    "FIGURE_04_EVENT_PROXIMAL_HISTORY_ERASURE",
    "FIGURE_05_MULTISCALE_MEMORY",
    "FIGURE_06_MICRO_DISORDER_MACRO_ORDER",
    "FIGURE_07_MECHANISM_LEDGER",
    "FIGURE_08_COMPLETE_RESEARCH_STORY",
}

REQUIRED_VIDEOS = {
    "hero_forward_reverse_exact.mp4",
    "history_erasure_density020.mp4",
    "history_erasure_density024.mp4",
    "history_erasure_short_preview.gif",
}

REQUIRED_VERIFICATION_GATES = {
    "extension_tests",
    "root_regression_tests",
    "ruff",
    "compileall",
    "smoke_intervention",
    "planned_task_ids",
    "sidecars",
    "nonfinite_values",
    "duplicate_missing_results",
    "interrupted_resume_equivalence",
    "campaign_resume_replay",
    "resource_governor",
    "eshkol_parity",
    "pdf_render_qa",
    "video_render_qa",
    "browser_dashboard_explorer",
    "github_private_push",
}


def _write_checked_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    verify_sidecar(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "passed", "pass"}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _tag_checks(repo: Path) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_TAGS.items():
        observed_object = _git(repo, "rev-parse", name)
        observed_peeled = _git(repo, "rev-parse", f"{name}^{{}}")
        checks[name] = {
            "expected_object": expected["object"],
            "observed_object": observed_object,
            "expected_peeled": expected["peeled"],
            "observed_peeled": observed_peeled,
            "passed": observed_object == expected["object"] and observed_peeled == expected["peeled"],
        }
    return checks


def resource_summary(resource_log: str | Path) -> dict[str, Any]:
    path = Path(resource_log)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("resource log contains no samples")

    def numbers(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name, "").strip()]

    timestamps = [datetime.fromisoformat(row["utc"]) for row in rows]
    temperatures = numbers("gpu_temperature_c")
    reason_rows = [row for row in rows if row.get("reasons", "").strip()]
    return {
        "schema": "zws-reversible-null-phase-v032-resource-summary-v1",
        "samples": len(rows),
        "first_utc": min(timestamps).astimezone(timezone.utc).isoformat(),
        "last_utc": max(timestamps).astimezone(timezone.utc).isoformat(),
        "wall_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600.0,
        "max_cpu_percent": max(numbers("cpu_percent")),
        "min_available_ram_gib": min(numbers("available_ram_gib")),
        "median_gpu_temperature_c": statistics.median(temperatures) if temperatures else None,
        "max_gpu_temperature_c": max(temperatures) if temperatures else None,
        "resource_yield_samples": len(reason_rows),
        "resource_yield_reasons": sorted({row["reasons"] for row in reason_rows}),
    }


def _manifest_file(path_text: str, *, roots: Iterable[Path]) -> Path:
    path = Path(path_text).resolve()
    if not any(path == root or root in path.parents for root in roots):
        raise RuntimeError(f"manifest path lies outside governed roots: {path}")
    return path


def inspect_companion_archive(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    verify_sidecar(target)
    blocked_parts = {".git", ".venv", ".venv-v030", "__pycache__", ".pytest_cache", ".ruff_cache"}
    with zipfile.ZipFile(target, "r") as archive:
        names = archive.namelist()
        if not names:
            raise RuntimeError("companion archive is empty")
        normalized = [PurePosixPath(name) for name in names]
        if any(".." in item.parts or item.is_absolute() for item in normalized):
            raise RuntimeError("companion archive contains an unsafe path")
        roots = {item.parts[0] for item in normalized if item.parts}
        if len(roots) != 1:
            raise RuntimeError("companion archive must contain exactly one top-level directory")
        root = next(iter(roots))
        lowered = [tuple(part.lower() for part in item.parts) for item in normalized]
        if any(any(part in blocked_parts for part in parts) for parts in lowered):
            raise RuntimeError("companion archive contains a forbidden cache or repository directory")
        if any(parts[-1] in {".env", ".env.local", "credentials.json"} for parts in lowered):
            raise RuntimeError("companion archive contains a credential-like filename")
        manifest_name = f"{root}/ARCHIVE_MANIFEST.json"
        manifest = json.loads(archive.read(manifest_name))
        entries = manifest["entries"]
        if manifest["entry_count"] != len(entries):
            raise RuntimeError("archive manifest entry count is inconsistent")
        for entry in entries:
            member = f"{root}/{entry['path']}"
            raw = archive.read(member)
            if len(raw) != int(entry["size"]):
                raise RuntimeError(f"archive size mismatch: {member}")
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"archive hash mismatch: {member}")
        entry_paths = {entry["path"] for entry in entries}
        required_exact = {
            "README.md",
            "WITHDRAWN_CLAIM_NOTICE.md",
            "DATA_VOLUME_NOTE.md",
            "environment_freeze.txt",
            "CITATION.cff",
            "LICENSE",
            ".zenodo.json",
        }
        if not required_exact.issubset(entry_paths):
            raise RuntimeError(f"archive lacks required roots: {sorted(required_exact - entry_paths)}")
        required_prefixes = {
            "paper/",
            "evidence/v032_history_erasure/",
            "software/src/",
            "software/eshkol/",
            "software/rules/",
        }
        missing_prefixes = [prefix for prefix in required_prefixes if not any(p.startswith(prefix) for p in entry_paths)]
        if missing_prefixes:
            raise RuntimeError(f"archive lacks required content classes: {missing_prefixes}")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size": target.stat().st_size,
        "entry_count": manifest["entry_count"],
        "manifest_rule_hash": manifest["rule_hash"],
        "passed": manifest["rule_hash"] == RULE_SHA256,
    }


def _artifact_checks(paths: Iterable[Path]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    failures: list[str] = []
    for path in paths:
        try:
            verify_sidecar(path)
            hashes[str(path.resolve())] = sha256_file(path)
        except Exception as exc:  # noqa: BLE001 - collect a complete final audit
            failures.append(f"{path}: {exc}")
    return hashes, failures


def _verification_gate_passed(value: Any) -> bool:
    if isinstance(value, dict):
        return _truth(value.get("passed", False))
    return _truth(value)


def audit_final_history_package(
    *,
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
    release_path: str | Path,
    verification_path: str | Path,
) -> dict[str, Any]:
    repo, run = Path(repo_root).resolve(), Path(run_path).resolve()
    experiment, paper, release = (
        Path(experiment_path).resolve(),
        Path(paper_path).resolve(),
        Path(release_path).resolve(),
    )
    status = load_verified_json(run / "status.json")
    plan = load_verified_json(experiment / "FROZEN_PLAN.json")
    realized = load_verified_json(run / "REALIZED_BRANCH_PLAN.json")
    integrity = load_verified_json(experiment / "INTEGRITY_AUDIT.json")
    decision = load_verified_json(experiment / "SCIENTIFIC_DECISION_V032.json")
    figures = load_verified_json(experiment / "FIGURE_MANIFEST.json")
    videos = load_verified_json(experiment / "VIDEO_MANIFEST.json")
    verification = load_verified_json(verification_path)

    required_paths = [experiment / name for name in REQUIRED_EXPERIMENT_ARTIFACTS]
    required_paths.extend(paper / "prior_art" / name for name in REQUIRED_PRIOR_ART)
    required_paths.extend(paper / name for name in REQUIRED_PAPER_ARTIFACTS)
    required_paths.append(repo / "FINAL_RESEARCH_STOP_DECISION.md")
    artifact_hashes, artifact_failures = _artifact_checks(required_paths)

    figure_failures: list[str] = []
    figure_ids = {entry.get("figure_id") for entry in figures.get("figures", [])}
    for figure in figures.get("figures", []):
        for panel in figure.get("panels", []):
            try:
                validate_panel_record(panel)
            except Exception as exc:  # noqa: BLE001
                figure_failures.append(f"{figure.get('figure_id')} panel: {exc}")
        formats = {output.get("format") for output in figure.get("outputs", [])}
        if formats != {"svg", "pdf", "png"}:
            figure_failures.append(f"{figure.get('figure_id')} formats={sorted(formats)}")
        for output in figure.get("outputs", []):
            try:
                path = _manifest_file(output["path"], roots=(repo, run))
                verify_sidecar(path)
                if sha256_file(path) != output["sha256"]:
                    raise RuntimeError("manifest hash mismatch")
                if output["format"] == "png" and figure.get("major_morphology"):
                    with Image.open(path) as image:
                        if image.width < 3840 or image.height < 2160:
                            raise RuntimeError(f"major morphology dimensions are {image.width}x{image.height}")
                artifact_hashes[str(path)] = output["sha256"]
            except Exception as exc:  # noqa: BLE001
                figure_failures.append(f"{figure.get('figure_id')} output: {exc}")

    video_failures: list[str] = []
    video_names = {entry.get("name") for entry in videos.get("videos", [])}
    for entry in videos.get("videos", []):
        try:
            path = _manifest_file(entry["path"], roots=(repo, run))
            verify_sidecar(path)
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError("manifest hash mismatch")
            if entry.get("intermediate_automaton_states_synthesized") is not False:
                raise RuntimeError("video does not explicitly forbid synthesized automaton states")
            artifact_hashes[str(path)] = entry["sha256"]
        except Exception as exc:  # noqa: BLE001
            video_failures.append(f"{entry.get('name')}: {exc}")

    rule = ReversibleRule.load(repo / "extensions/reversible_null_phase/rules/winner_d1fe99.json")
    rule_audit = rule.audit()
    tag_results = _tag_checks(repo)
    changed_paths = set(_git(repo, "diff", "--name-only", f"{plan['starting_commit']}..HEAD").splitlines())
    forbidden_changes = sorted(
        changed_paths
        & {
            "extensions/reversible_null_phase/rules/winner_d1fe99.json",
            "extensions/reversible_null_phase/configs/rtx3060_powered.yaml",
            "extensions/reversible_null_phase/scripts/run_powered_managed.ps1",
        }
    )
    forbidden_changes.extend(sorted(path for path in changed_paths if "causal_nullity_ecology" in path.lower()))

    archive = inspect_companion_archive(release / "ZWS_Reversible_Null_Phase_Companion_v0.2.zip")
    pdf_outputs = [
        output
        for figure in figures.get("figures", [])
        for output in figure.get("outputs", [])
        if output.get("format") == "pdf"
    ]
    pdf_outputs.extend(
        {"path": str(path)}
        for path in (
            paper / "build/manuscript.pdf",
            paper / "build/supplement.pdf",
            paper / "ONE_PAGE_LAY_SUMMARY.pdf",
        )
    )

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for root in (experiment, paper)
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".tex", ".json", ".html"}
        and "corrected_v031" not in {part.lower() for part in path.parts}
        and path.name != "ERRATUM_Q_AND_M.md"
    )
    bad_qm_phrase = bool(re.search(r"Q\s*=\s*M\s*=\s*0", public_text, flags=re.IGNORECASE))
    stop_text = (repo / "FINAL_RESEARCH_STOP_DECISION.md").read_text(encoding="utf-8")
    stop_exact = (
        "A. THE MECHANISM IS SUFFICIENTLY RESOLVED FOR THE COMPANION PAPER." in stop_text
        or "B. THE RESULT REMAINS INCONCLUSIVE, BUT THE PRESENT FINITE CAMPAIGN IS COMPLETE AND THE UNCERTAINTY WILL BE REPORTED IN THE PAPER."
        in " ".join(stop_text.split())
    )
    verification_checks = verification.get("checks", {})
    missing_verification = sorted(REQUIRED_VERIFICATION_GATES - set(verification_checks))
    failed_verification = sorted(
        name for name in REQUIRED_VERIFICATION_GATES if not _verification_gate_passed(verification_checks.get(name))
    )

    checks = {
        "run_reached_branches_complete": status.get("state") == "branches_complete",
        "frozen_rule_hash_exact": rule.sha256() == RULE_SHA256 == plan["winner_raw_uint8_sha256"],
        "rule_is_permutation_and_Q_preserving": rule_audit.permutation and rule_audit.charge_preserving,
        "winner_does_not_falsely_claim_M_conservation": not rule_audit.occupancy_preserving,
        "final_integrity_audit_passed": bool(integrity.get("passed")),
        "trajectory_count_within_frozen_maximum": int(integrity["realized_scientific_trajectory_count"])
        <= int(plan["counts"]["maximum_scientific_trajectories"]),
        "source_positive_control_gate_passed": bool(realized["source_gate"]["passed"]),
        "translation_nulls_exact": bool(integrity["checks"]["translation_nulls_exact"]),
        "all_required_artifacts_sidecar_verified": not artifact_failures,
        "exact_eight_figure_set_verified": figure_ids == REQUIRED_FIGURES and not figure_failures,
        "exact_four_video_set_verified": video_names == REQUIRED_VIDEOS and not video_failures,
        "exact_eleven_pdf_outputs": len(pdf_outputs) == 11,
        "verification_matrix_complete_and_passed": not missing_verification and not failed_verification,
        "archive_verified": bool(archive["passed"]),
        "immutable_tags_unchanged": all(value["passed"] for value in tag_results.values()),
        "forbidden_scope_unchanged": not forbidden_changes,
        "new_public_copy_avoids_false_Q_M_zero_phrase": not bad_qm_phrase,
        "final_stop_decision_uses_frozen_wording": stop_exact,
        "decision_payload_is_finite": _finite(decision),
    }
    audit = {
        "schema": "zws-reversible-null-phase-v032-final-requirement-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "failures": {
            "artifact": artifact_failures,
            "figure": figure_failures,
            "video": video_failures,
            "missing_verification": missing_verification,
            "failed_verification": failed_verification,
            "forbidden_changes": forbidden_changes,
        },
        "rule_audit": {
            "sha256": rule_audit.sha256,
            "permutation": rule_audit.permutation,
            "charge_preserving": rule_audit.charge_preserving,
            "occupancy_preserving": rule_audit.occupancy_preserving,
        },
        "tags": tag_results,
        "archive": archive,
        "artifact_hashes": artifact_hashes,
        "figure_count": len(figures.get("figures", [])),
        "video_count": len(videos.get("videos", [])),
        "pdf_count": len(pdf_outputs),
        "realized_scientific_trajectory_count": integrity["realized_scientific_trajectory_count"],
        "combined_mechanism": decision["combined_mechanism"],
    }
    atomic_write_json(experiment / "FINAL_REQUIREMENT_AUDIT_V032.json", audit)
    rows = [
        "# v0.3.2 final requirement audit",
        "",
        f"Overall: **{'PASS' if audit['passed'] else 'FAIL'}**.",
        "",
        "| Gate | Result |",
        "|---|---|",
        *[f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items()],
        "",
    ]
    _write_checked_text(experiment / "FINAL_REQUIREMENT_AUDIT_V032.md", "\n".join(rows))
    if not audit["passed"]:
        raise RuntimeError("v0.3.2 final requirement audit failed; inspect FINAL_REQUIREMENT_AUDIT_V032.json")
    return audit


def _event_counts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["branch_point"], row["intervention"]), []).append(row)
    return [
        {
            "branch_point": point,
            "intervention": intervention,
            "n": len(group),
            "events": sum(_truth(row["hit_observed"]) for row in group),
            "right_censored": sum(not _truth(row["hit_observed"]) for row in group),
        }
        for (point, intervention), group in sorted(groups.items())
    ]


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def build_completion_record(
    *,
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
    release_path: str | Path,
    verification_path: str | Path,
) -> dict[str, Any]:
    repo, run = Path(repo_root).resolve(), Path(run_path).resolve()
    experiment, paper = Path(experiment_path).resolve(), Path(paper_path).resolve()
    audit = audit_final_history_package(
        repo_root=repo,
        run_path=run,
        experiment_path=experiment,
        paper_path=paper,
        release_path=release_path,
        verification_path=verification_path,
    )
    plan = load_verified_json(experiment / "FROZEN_PLAN.json")
    realized = load_verified_json(run / "REALIZED_BRANCH_PLAN.json")
    integrity = load_verified_json(experiment / "INTEGRITY_AUDIT.json")
    decision = load_verified_json(experiment / "SCIENTIFIC_DECISION_V032.json")
    figures = load_verified_json(experiment / "FIGURE_MANIFEST.json")
    videos = load_verified_json(experiment / "VIDEO_MANIFEST.json")
    verification = load_verified_json(verification_path)
    intervention_manifest = load_verified_json(experiment / "INTERVENTION_MANIFEST.json")
    sources = _read_csv(experiment / "SOURCE_TRAJECTORIES.csv")
    interventions = _read_csv(experiment / "INTERVENTION_AUDIT.csv")
    residuals = _read_csv(experiment / "RESIDUAL_FIRST_PASSAGE.csv")
    resources = resource_summary(run / "resource_log.csv")
    atomic_write_json(experiment / "RESOURCE_SUMMARY_V032.json", resources)
    language = result_language(decision)
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    archive = audit["archive"]

    inventory_by_source_point = {
        (row["source_task_id"], row["branch_point"]): row["source_histogram"]
        for row in interventions
    }
    source_rows = [
        (
            row["source_task_id"],
            row["density"],
            row["seed"],
            row["branch_A_eligible"],
            row["branch_A_phase"] or "—",
            inventory_by_source_point.get((row["source_task_id"], "A"), "—"),
            row["branch_B_eligible"],
            row["branch_B_phase"] or "—",
            inventory_by_source_point.get((row["source_task_id"], "B"), "—"),
        )
        for row in sources
    ]
    manifest_by_task = {entry["task_id"]: entry for entry in intervention_manifest["entries"]}

    def generator_record(task_id: str) -> str:
        entry = manifest_by_task[task_id]
        if "permutation_sha256" in entry:
            return str(entry["permutation_sha256"])
        if "tile_permutation_sha256" in entry:
            return str(entry["tile_permutation_sha256"])
        return "even offset yx=" + json.dumps(entry.get("translation_offset_yx"), separators=(",", ":"))

    intervention_rows = [
        (
            row["task_id"],
            row["source_task_id"],
            row["branch_point"],
            row["branch_phase"],
            row["intervention"],
            row["realization"],
            row["source_M"],
            row["source_Q"],
            generator_record(row["task_id"]),
            row["altered_state_sha256"],
        )
        for row in interventions
    ]
    event_rows = [
        (row["branch_point"], row["intervention"], row["events"], row["right_censored"], row["n"])
        for row in _event_counts(residuals)
    ]
    artifact_rows: list[tuple[str, str]] = []
    for path in (
        experiment / "HISTORY_ERASURE_REPORT.md",
        experiment / "INTEGRITY_AUDIT.json",
        experiment / "SCIENTIFIC_DECISION_V032.json",
        experiment / "FIGURE_MANIFEST.json",
        experiment / "VIDEO_MANIFEST.json",
        paper / "build/manuscript.pdf",
        paper / "build/supplement.pdf",
        paper / "ONE_PAGE_LAY_SUMMARY.pdf",
        paper / "graphical_abstract.png",
    ):
        artifact_rows.append((str(path.relative_to(repo)), sha256_file(path)))
    for entry in figures["figures"]:
        png = next(output for output in entry["outputs"] if output["format"] == "png")
        artifact_rows.append((f"paper/figures/{entry['figure_id']}.png", png["sha256"]))
    for entry in videos["videos"]:
        artifact_rows.append((f"paper/media/{entry['name']}", entry["sha256"]))
    artifact_rows.append((Path(archive["path"]).name, archive["sha256"]))

    environment = load_verified_json(run / "environment.json")
    failures_repairs = verification.get("failures_and_repairs", [])
    pause_resume = verification.get("pause_resume_events", [])
    lines = [
        "# ZWS Nullity Lab v0.3.2 completion record",
        "",
        f"Completed on branch `{branch}`. The compact artifact commit immediately before this record is `{head}`; this record is committed as a metadata successor, so its own commit cannot contain its self-referential hash.",
        "",
        "## Outcome",
        "",
        f"The final preregistered history-erasure campaign is complete. Branch A was **{decision['branch_A']['classification']}**; Branch B was **{decision['branch_B']['classification']}**. The frozen combined classification is **{decision['combined_mechanism']}**. Tile-scale interpretation: **{decision['tile_interpretation'].replace('_', ' ')}**.",
        "",
        f"**Precise scientific claim:** {language['claim']}",
        "",
        f"**Lay headline:** {language['headline']}",
        "",
        "This is the final planned mechanism campaign for the frozen rule. No rule search, powered profile, broad density sweep, 4096² campaign, unplanned Stage 2, or Causal Nullity Ecology run was launched.",
        "",
        "## Frozen provenance",
        "",
        f"- Starting commit: `{plan['starting_commit']}`",
        f"- Compact artifact commit before this record: `{head}`",
        f"- Branch: `{branch}`",
        f"- Frozen raw uint8 rule SHA-256: `{RULE_SHA256}`",
        f"- Frozen-plan payload SHA-256: `{plan['payload_sha256']}`",
        f"- Realized-plan payload SHA-256: `{realized['payload_sha256']}`",
        f"- Source seeds: `{', '.join(str(seed) for seed in plan['source_seed_freshness']['seeds'])}` at each density 0.20 and 0.24",
        f"- Realized scientific trajectories: **{integrity['realized_scientific_trajectory_count']}** / maximum **{plan['counts']['maximum_scientific_trajectories']}**",
        f"- Ineligible planned branch tasks: **{realized['ineligible_count']}**",
        "",
        "All immutable discovery/freeze/confirmation tags retained their recorded tag objects and peeled commits.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(environment, indent=2, sort_keys=True),
        "```",
        "",
        "## Exact source eligibility, branch phases, and inventories",
        "",
        *_markdown_table(
            [
                "Source",
                "density",
                "seed",
                "A eligible",
                "A phase",
                "A histogram [0,+,-,bound]",
                "B eligible",
                "B phase",
                "B histogram [0,+,-,bound]",
            ],
            source_rows,
        ),
        "",
        "## Exact intervention identities and altered-state hashes",
        "",
        *_markdown_table(
            [
                "Task",
                "Source",
                "point",
                "phase",
                "intervention",
                "realization",
                "M",
                "Q",
                "generator digest / offset",
                "altered state SHA-256",
            ],
            intervention_rows,
        ),
        "",
        "Every row preserved the exact four-state histogram, Q, M, single-site entropy, and Margolus phase. Generator seeds and permutation/tile/translation digests remain in `INTERVENTION_MANIFEST.json` and the checksummed per-task intervention audits.",
        "",
        "## Event and censor counts",
        "",
        *_markdown_table(["Point", "Intervention", "events", "right-censored", "N"], event_rows),
        "",
        "Missing first passages remain right-censored; no horizon was substituted as an event time.",
        "",
        "## Primary mechanism result",
        "",
        f"- Branch A: **{decision['branch_A']['classification']}** — RMST delay `{decision['branch_A']['pooled']['paired_rmst_delay']['point_estimate']}` phases; 90% interval `{decision['branch_A']['pooled']['paired_rmst_delay']['ci90']}`; 95% interval `{decision['branch_A']['pooled']['paired_rmst_delay']['ci95']}`.",
        f"- Branch B: **{decision['branch_B']['classification']}** — RMST delay `{decision['branch_B']['pooled']['paired_rmst_delay']['point_estimate']}` phases; 90% interval `{decision['branch_B']['pooled']['paired_rmst_delay']['ci90']}`; 95% interval `{decision['branch_B']['pooled']['paired_rmst_delay']['ci95']}`.",
        f"- Tile-scale result: **{decision['tile_interpretation'].replace('_', ' ')}**.",
        f"- Density-specific direction agreement: Branch A **{decision['density_direction_agreement_by_branch']['A']}**; Branch B **{decision['density_direction_agreement_by_branch']['B']}**; both-branch summary **{decision['density_direction_agreement']}**.",
        "- Multiscale entropy, mutual information, low-k, interface, and anisotropy results are preserved in `MULTISCALE_MEMORY.csv`, `ENTROPY_HIERARCHY.csv`, and `ANISOTROPY_RESULTS.csv` with SHA-256 sidecars.",
        "",
        "## Q/M correction",
        "",
        "The winner preserves `Q = N_positive - N_negative = 0` exactly. It does not conserve `M = N_positive + N_negative + N_bound`; M changes during dynamics. Original v0.3.1 checksummed prose remains historical, with a versioned erratum and corrected derivatives. No simulation data changed.",
        "",
        "## Graphics, video, dashboard, explorer, and literature",
        "",
        f"- Figures: {len(figures['figures'])}/8, each SVG/vector-PDF/300-dpi PNG with machine-readable panel provenance.",
        f"- Videos/preview: {len(videos['videos'])}/4, exact saved states only; no synthetic automaton frames.",
        "- Local self-contained dashboard and explorer: verified, including synchronization, exact metadata, exports, and browser console checks.",
        f"- Prior-art matrix: {len(_read_csv(paper / 'prior_art/PRIOR_ART_MATRIX.csv'))} close-source records; DOI audit and search log complete.",
        "- Manuscript, supplement, one-page lay summary, graphical abstract, ZWS insert, and revision notes are complete.",
        "",
        "## Verification matrix",
        "",
        *_markdown_table(
            ["Gate", "Result", "Summary"],
            (
                (
                    name,
                    "PASS" if _verification_gate_passed(value) else "FAIL",
                    value.get("summary", "") if isinstance(value, dict) else "",
                )
                for name, value in verification["checks"].items()
            ),
        ),
        "",
        "## Resource use",
        "",
        f"- Samples: {resources['samples']}",
        f"- Managed wall span: {resources['wall_hours']:.3f} hours",
        f"- Maximum host CPU: {resources['max_cpu_percent']:.1f}%",
        f"- Minimum available RAM: {resources['min_available_ram_gib']:.2f} GiB",
        f"- Median / maximum GPU temperature: {resources['median_gpu_temperature_c']} / {resources['max_gpu_temperature_c']} °C",
        f"- Resource-yield samples: {resources['resource_yield_samples']}",
        "",
        "## Pause, resume, failures, and repairs",
        "",
        *(f"- {item}" for item in pause_resume),
        *(f"- {item}" for item in failures_repairs),
        "",
        "## Compact artifacts and hashes",
        "",
        *_markdown_table(["Artifact", "SHA-256"], artifact_rows),
        "",
        "## Contribution to Zero Without Stasis",
        "",
        "The work moves signed nullity from static compatibility to a reproducible dynamical example: exact global signed zero and exact microscopic reversibility coexist with reversible local reactions, delayed density- and history-dependent macroscopic organisation, and persistent microscopic disorder. The proposed Section 14.2 insert labels this a formal computational result.",
        "",
        "It does not establish a physical cosmology, a natural invariant of our universe, life, heredity, consciousness, primitive occurrence, ETNE, an equilibrium thermodynamic phase transition, an infinite-time density boundary, or universality across cellular automata.",
        "",
        "## Final programme decision",
        "",
        (repo / "FINAL_RESEARCH_STOP_DECISION.md").read_text(encoding="utf-8").strip(),
        "",
    ]
    completion = repo / "CODEX_COMPLETION_V032.md"
    _write_checked_text(completion, "\n".join(lines))
    return {
        "path": str(completion),
        "sha256": sha256_file(completion),
        "audit": str(experiment / "FINAL_REQUIREMENT_AUDIT_V032.json"),
        "passed": True,
    }
