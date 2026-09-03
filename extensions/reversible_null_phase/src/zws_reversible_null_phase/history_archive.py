from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import zipfile

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_media import load_verified_json, verify_sidecar
from .history_publication import RULE_SHA256


ARCHIVE_VERSION = "0.2.0"
ARCHIVE_NAME = "ZWS_Reversible_Null_Phase_Companion_v0.2"
CANONICAL_TITLE = (
    "Delayed Distributed Coarsening in an Exactly Reversible "
    "Signed-Charge-Conserving Block Cellular Automaton"
)


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    ".venv-v030",
    "checkpoints",
    "raw",
    "cache",
    "caches",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log"}
SECRET_NAME_TERMS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "credentials",
    "password",
    "secret_key",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
)


def _allowed_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return (
        path.is_file()
        and not any(part.lower() in EXCLUDED_PARTS for part in path.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
        and not lower_name.endswith(".zip")
        and lower_name not in {".env", ".env.local", "credentials.json"}
        and not any(term in lower_name for term in SECRET_NAME_TERMS)
    )


def _tree_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if _allowed_file(path)) if root.is_dir() else []


def _copy(source: Path, stage: Path, relative: Path) -> dict[str, Any]:
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(target),
        "size": target.stat().st_size,
    }


def _add_tree(
    entries: list[tuple[Path, Path]],
    source_root: Path,
    destination_root: Path,
) -> None:
    for source in _tree_files(source_root):
        entries.append((source, destination_root / source.relative_to(source_root)))


def _explicit_existing(repo: Path, paths: Iterable[str]) -> list[Path]:
    return [repo / value for value in paths if (repo / value).is_file()]


def collect_archive_candidates(
    *,
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
) -> list[tuple[Path, Path]]:
    repo, run = Path(repo_root).resolve(), Path(run_path).resolve()
    experiment, paper = Path(experiment_path).resolve(), Path(paper_path).resolve()
    entries: list[tuple[Path, Path]] = []
    _add_tree(entries, paper, Path("paper"))
    entries = [
        pair
        for pair in entries
        if pair[0].name != "Zero_Without_Stasis_v1.209_glossed.html"
        and not any(part.startswith("pdf_qa") for part in pair[1].parts)
        and "media" not in pair[1].parts
        and pair[0].suffix.lower() not in {".log", ".blg"}
    ]
    _add_tree(entries, experiment, Path("evidence/v032_history_erasure"))
    entries = [
        pair
        for pair in entries
        if pair[0].name not in {"history_erasure_explorer.html", "dashboard.html"}
    ]
    extension = repo / "extensions/reversible_null_phase"
    for directory in ("src", "tests", "docs", "eshkol"):
        _add_tree(entries, extension / directory, Path("software") / directory)
    for source in _explicit_existing(
        extension,
        (
            "pyproject.toml",
            "README.md",
            "AGENTS.md",
            "rules/winner_d1fe99.json",
            "configs/v032_history_erasure.yaml",
            "scripts/run_v032_history_managed.ps1",
            "scripts/audit_results.py",
        ),
    ):
        entries.append((source, Path("software") / source.relative_to(extension)))
    for source in _explicit_existing(
        repo,
        (
            "LICENSE",
            "CITATION.cff",
            "README.md",
            "CODEX_COMPLETION_V030.md",
            "CODEX_COMPLETION_V031.md",
            "FINAL_RESEARCH_STOP_DECISION.md",
            "FINAL_RESEARCH_STOP_DECISION_AMENDED.md",
            "WITHDRAWN_CLAIM_GREP_AUDIT.json",
            "WITHDRAWN_CLAIM_GREP_AUDIT.md",
            "CODEX_COMPLETION_COMPANION_V02_REPAIR.md",
            "OWNER_README_NEXT_STEPS.md",
            "SCIENTIFIC_SUMMARY_V030.md",
            "experiments/domain_mechanism_20260825/DOMAIN_MECHANISM_REPORT.md",
            "experiments/domain_mechanism_20260825/PREREGISTRATION.md",
            "experiments/domain_mechanism_20260825/artifacts/double_conservation_screen/CONDITION_SUMMARIES.csv",
            "experiments/domain_mechanism_20260825/artifacts/double_conservation_screen/DOUBLE_CONSERVATION_CLASSIFICATION.json",
            "experiments/domain_mechanism_20260825/artifacts/prevalence_screen/PREVALENCE_CLASSIFICATION.json",
            "experiments/domain_mechanism_20260825/artifacts/prevalence_screen/PREVALENCE_RULE_SUMMARIES.csv",
            "experiments/homogeneous_ic_20260826/HOMOGENEOUS_IC_REPORT.md",
            "experiments/homogeneous_density_threshold_20260826/DENSITY_THRESHOLD_REPORT.md",
            "experiments/homogeneous_density_threshold_20260826/DENSITY_SUMMARY.csv",
            "experiments/reversible_null_phase_v030_20260827/balanced/report/REPORT.md",
            "experiments/reversible_null_phase_v030_20260827/balanced/report/SUMMARY.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/NUCLEATION_REPORT.md",
            "experiments/reversible_null_phase_v031_nucleation_20260827/FIRST_PASSAGE_RESULTS.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/SURVIVAL_SUMMARY.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/AREA_SCALING.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/EVENT_WINDOW_SUMMARY.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/SEEDED_BASIN_RESULTS.csv",
            "experiments/reversible_null_phase_v031_nucleation_20260827/INTEGRITY_AUDIT.json",
            "experiments/reversible_null_phase_v032_history_erasure_20260829/corrected_v031/NUCLEATION_REPORT_QM_CORRECTED_V032.md",
        ),
    ):
        entries.append((source, Path("background") / source.relative_to(repo)))

    figure_manifest = load_verified_json(experiment / "FIGURE_MANIFEST.json")
    state_paths: dict[str, Path] = {}
    for figure in figure_manifest["figures"]:
        for panel in figure["panels"]:
            raw = panel.get("source_path")
            if not raw:
                continue
            path = Path(raw).resolve()
            if path.suffix.lower() not in {".npz", ".pt"}:
                continue
            try:
                path.relative_to(run)
            except ValueError:
                try:
                    path.relative_to(repo)
                except ValueError as exc:
                    raise RuntimeError(f"representative state lies outside governed roots: {path}") from exc
            verify_sidecar(path)
            state_paths[sha256_file(path)] = path
    # Exact-state hashes and source paths remain in the governed manifests, but
    # multi-megabyte representative tensors are excluded from the compact v0.2
    # archive along with trajectories and checkpoints.

    unique: dict[str, tuple[Path, Path]] = {}
    for source, relative in entries:
        representative = bool(relative.parts and relative.parts[0] == "representative_states")
        if not _allowed_file(source) and not (representative and source.is_file()):
            continue
        key = relative.as_posix().lower()
        if key in unique and sha256_file(unique[key][0]) != sha256_file(source):
            raise RuntimeError(f"archive destination collision: {relative}")
        unique[key] = (source, relative)
    return sorted(unique.values(), key=lambda pair: pair[1].as_posix())


def _command_output(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return (
        f"$ {' '.join(command)}\n"
        f"exit={completed.returncode}\n"
        f"{completed.stdout.rstrip()}\n"
        f"{completed.stderr.rstrip()}\n"
    )


def _environment_freeze(repo: Path) -> str:
    sections = [
        f"python_executable={sys.executable}\npython_version={sys.version}\n",
        _command_output([sys.executable, "-m", "pip", "freeze", "--all"], cwd=repo),
        _command_output(["git", "rev-parse", "HEAD"], cwd=repo),
        _command_output(["git", "branch", "--show-current"], cwd=repo),
    ]
    if shutil.which("nvidia-smi"):
        sections.append(
            _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                cwd=repo,
            )
        )
    return "\n".join(sections)


def _archive_readme() -> str:
    return f"""# ZWS Reversible Null Phase companion archive v0.2

This is a compact, review-ready package for the frozen reversible signed-charge-conserving block cellular automaton and the corrected companion paper.

## Correction notice

The original v0.3.2 history-erasure causal headline is withdrawn. Its intervention executed correctly, but the endpoint misclassified shuffled connected-empty null geometry. Historical files are preserved for audit and governed by their additive erratum and corrected derivative. They are not evidence of causal spatial memory, history inhibition, or accelerated recovery.

The raw uint8 winner rule SHA-256 is `{RULE_SHA256}`.

Contents include manuscript and supplement source/PDF, publication figures and exact-state metadata, frozen manifests, compact CSV/JSON evidence, Eshkol parity source/result, source and tests required for the reported analyses, environment freeze, and integrity audit.

The package is preparation only. It has not been published and does not make the private repository public.

## Reproduction

Install the extension from `software/` and preserve the frozen rule hash. Full raw trajectories are deterministic but intentionally excluded. See `DATA_VOLUME_NOTE.md` for regeneration boundaries. Do not use the historical v0.3.2 endpoint as a validated intervention classifier.
"""


def _withdrawal_notice() -> str:
    return """# Withdrawn v0.3.2 causal headline

The v0.3.2 state-matched shuffles were executed and audited correctly, but the
strong-domain endpoint failed under the shuffled null distribution. It could be
triggered by random connected-empty geometry while long-wave power and local
mutual information remained near shuffled levels. Therefore no causal claim about
hidden spatial memory, history inhibition, acceleration, or recovery time is
supported by that first-passage comparison.

Use `paper/V032_CLASSIFIER_REVIEW_20260831.md`, the experiment erratum, and the
corrected report for current interpretation. Historical source files remain
unaltered for auditability.
"""


def _data_volume_note(run: Path) -> str:
    return f"""# Data-volume note

Full trajectories and rotating checkpoints are excluded from this compact archive. They remain in the governed local raw-result directory during project closure and are not required to verify the compact tables, panel hashes, or representative states.

Logical governed run name: `{run.name}`. The machine-specific absolute path is intentionally omitted from this public-facing package.

The frozen configuration, rule, source/intervention manifests, source seeds, exact initial hashes, observation/checkpoint schedules, software, and environment freeze permit deterministic regeneration. Exact representative tensors, videos, and full interactive explorers are excluded for compactness; their source hashes and rendering provenance remain in the figure and video manifests.
"""


def assert_no_staged_credentials(stage: Path) -> None:
    """Fail closed if an archive candidate contains a recognizable live credential."""
    for path in sorted(candidate for candidate in stage.rglob("*") if candidate.is_file()):
        lower_name = path.name.lower()
        if lower_name in {".env", ".env.local", "credentials.json"} or any(
            term in lower_name for term in SECRET_NAME_TERMS
        ):
            raise RuntimeError(f"credential-like filename blocked from archive: {path.relative_to(stage)}")
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        raw = path.read_bytes()
        if any(pattern.search(raw) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"credential-like content blocked from archive: {path.relative_to(stage)}")


def _zenodo(language_title: str) -> dict[str, Any]:
    return {
        "title": language_title,
        "upload_type": "dataset",
        "description": (
            "Compact code, evidence, figures, and companion-paper "
            "materials for the ZWSLab reversible charge-balanced cellular automaton."
        ),
        "creators": [
            {
                "name": "Bartlett, Chris",
                "orcid": "0009-0001-3323-0131",
            }
        ],
        "license": "MIT",
        "publication_date": date.today().isoformat(),
        "keywords": [
            "reversible cellular automata",
            "Margolus block automata",
            "conservation law",
            "coarsening",
            "first passage",
            "classifier correction",
            "structured nullity",
        ],
        "notes": "Prepared for review; DOI intentionally absent until explicit publication.",
    }


def _archive_citation(language_title: str) -> str:
    return f"""cff-version: 1.2.0
title: {json.dumps(language_title)}
message: Please cite this companion dataset and paper package.
type: dataset
authors:
  - family-names: Bartlett
    given-names: Chris
    orcid: https://orcid.org/0009-0001-3323-0131
version: 0.2.0
date-released: {date.today().isoformat()}
license: MIT
keywords:
  - reversible cellular automata
  - Margolus block automata
  - conservation law
  - coarsening
  - first passage
  - classifier correction
  - structured nullity
"""


def build_companion_archive(
    *,
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
    release_path: str | Path,
) -> dict[str, Any]:
    repo, run = Path(repo_root).resolve(), Path(run_path).resolve()
    experiment, paper, release = (
        Path(experiment_path).resolve(),
        Path(paper_path).resolve(),
        Path(release_path).resolve(),
    )
    integrity = load_verified_json(experiment / "INTEGRITY_AUDIT.json")
    if not integrity["passed"]:
        raise RuntimeError("archive stopped because the final integrity audit failed")
    for required in (
        paper / "build/manuscript.pdf",
        paper / "build/supplement.pdf",
        paper / "ONE_PAGE_LAY_SUMMARY.pdf",
        experiment / "FIGURE_MANIFEST.json",
        experiment / "VIDEO_MANIFEST.json",
    ):
        verify_sidecar(required)
    candidates = collect_archive_candidates(
        repo_root=repo,
        run_path=run,
        experiment_path=experiment,
        paper_path=paper,
    )
    release.mkdir(parents=True, exist_ok=True)
    zip_path = release / f"{ARCHIVE_NAME}.zip"
    with tempfile.TemporaryDirectory(prefix="zws_companion_archive_") as temporary:
        stage = Path(temporary) / ARCHIVE_NAME
        stage.mkdir(parents=True)
        manifest_entries = [_copy(source, stage, relative) for source, relative in candidates]

        def _write_text_stage(relative: str, content: str) -> None:
            atomic_write_text(stage / relative, content)

        _write_text_stage("README.md", _archive_readme())
        _write_text_stage("WITHDRAWN_CLAIM_NOTICE.md", _withdrawal_notice())
        _write_text_stage("DATA_VOLUME_NOTE.md", _data_volume_note(run))
        _write_text_stage("environment_freeze.txt", _environment_freeze(repo))
        license_file = repo / "LICENSE"
        language_title = CANONICAL_TITLE
        atomic_write_text(stage / "CITATION.cff", _archive_citation(language_title))
        shutil.copy2(license_file, stage / "LICENSE")
        atomic_write_json(stage / ".zenodo.json", _zenodo(language_title))
        for relative in (
            Path("README.md"),
            Path("WITHDRAWN_CLAIM_NOTICE.md"),
            Path("DATA_VOLUME_NOTE.md"),
            Path("environment_freeze.txt"),
            Path("CITATION.cff"),
            Path("LICENSE"),
            Path(".zenodo.json"),
            Path(".zenodo.json.sha256"),
        ):
            target = stage / relative
            manifest_entries.append(
                {"path": relative.as_posix(), "sha256": sha256_file(target), "size": target.stat().st_size}
            )
        manifest_entries = sorted(
            {entry["path"]: entry for entry in manifest_entries}.values(),
            key=lambda entry: entry["path"],
        )
        manifest = {
            "schema": "zws-reversible-null-phase-companion-archive-v2",
            "version": ARCHIVE_VERSION,
            "rule_hash": RULE_SHA256,
            "entry_count": len(manifest_entries),
            "total_uncompressed_bytes": sum(int(entry["size"]) for entry in manifest_entries),
            "entries": manifest_entries,
            "exclusions": [
                "full raw trajectories",
                "rotating transient checkpoints not used as representative states",
                "representative exact-state tensors",
                "videos, GIF previews, and interactive explorer HTML",
                "rendered PDF-QA page images",
                "virtual environments and caches",
                "credentials and private repository metadata",
                "unrelated project files",
            ],
        }
        atomic_write_json(stage / "ARCHIVE_MANIFEST.json", manifest)
        assert_no_staged_credentials(stage)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(stage.name) / path.relative_to(stage))
    atomic_write_text(
        zip_path.with_suffix(zip_path.suffix + ".sha256"),
        f"{sha256_file(zip_path)}  {zip_path.name}\n",
    )
    manifest_path = release / "ARCHIVE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "path": str(zip_path),
        "sha256": sha256_file(zip_path),
        "size": zip_path.stat().st_size,
        "manifest": str(manifest_path),
        "candidate_count": len(candidates),
        "integrity_passed": True,
    }
