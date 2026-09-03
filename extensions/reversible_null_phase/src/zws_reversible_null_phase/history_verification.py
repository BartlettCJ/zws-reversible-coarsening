from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .atomic import atomic_write_json, sha256_file
from .history_media import load_verified_json, verify_sidecar
from .history_publication import RULE_SHA256


def record_manual_qa(
    *,
    output_path: str | Path,
    kind: str,
    summary: str,
    details: Iterable[str],
    artifacts: Iterable[str | Path],
    passed: bool,
) -> dict[str, Any]:
    if kind not in {"pdf_visual", "video_visual", "browser"}:
        raise ValueError("manual QA kind must be pdf_visual, video_visual, or browser")
    artifact_records: list[dict[str, Any]] = []
    for raw in artifacts:
        path = Path(raw).resolve()
        if not path.is_file():
            raise RuntimeError(f"manual QA artifact is missing: {path}")
        artifact_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    payload = {
        "schema": f"zws-reversible-null-phase-v032-{kind}-qa-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "visual inspection",
        "passed": bool(passed),
        "summary": summary,
        "details": list(details),
        "artifacts": artifact_records,
    }
    atomic_write_json(output_path, payload)
    return payload


def _run(command: list[str], *, cwd: Path, timeout: int = 1800) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return {
        "command": command,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": stdout[-6000:],
        "stderr_tail": stderr[-6000:],
    }


def _verification_python_paths(repo: Path) -> tuple[Path, Path]:
    root_python = repo / ".venv/Scripts/python.exe"
    extension_python = repo / ".venv-v030/Scripts/python.exe"
    missing = [path for path in (root_python, extension_python) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "verification Python environment is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return root_python, extension_python


def verify_sidecar_tree(roots: Iterable[str | Path]) -> dict[str, Any]:
    checked = 0
    failures: list[str] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        for sidecar in sorted(root.rglob("*.sha256")):
            resolved = sidecar.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            target = Path(str(sidecar)[: -len(".sha256")])
            try:
                verify_sidecar(target)
                checked += 1
            except Exception as exc:  # noqa: BLE001 - preserve all final audit failures
                failures.append(f"{target}: {exc}")
    return {"passed": not failures, "checked": checked, "failures": failures}


def resource_governor_gate(status: dict[str, Any], stderr_path: str | Path) -> dict[str, Any]:
    """Accept only terminal completion plus documented managed-pause diagnostics."""
    path = Path(stderr_path).resolve()
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    prefix = "managed pause: resource pressure: "
    managed_pauses = [line for line in lines if line.startswith(prefix) and line != prefix]
    unexpected = [line for line in lines if line not in managed_pauses]
    terminal = status.get("state") == "branches_complete"
    return {
        "passed": terminal and not unexpected,
        "summary": (
            f"terminal state={status.get('state')}; managed resource pauses="
            f"{len(managed_pauses)}; unexpected stderr lines={len(unexpected)}"
        ),
        "managed_pause_count": len(managed_pauses),
        "managed_pause_lines": managed_pauses,
        "unexpected_stderr_lines": unexpected,
    }


def _github_gate(repo: Path, branch: str) -> dict[str, Any]:
    view = _run(
        [
            "gh",
            "repo",
            "view",
            "<public-repository>",
            "--json",
            "visibility,nameWithOwner,url",
        ],
        cwd=repo,
    )
    details = json.loads(view["stdout_tail"]) if view["passed"] else {}
    local = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    remote = _run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo,
    )
    local_head = local["stdout_tail"].splitlines()[-1] if local["passed"] else ""
    remote_head = remote["stdout_tail"].split()[0] if remote["passed"] and remote["stdout_tail"] else ""
    passed = (
        view["passed"]
        and details.get("visibility") == "PRIVATE"
        and local["passed"]
        and remote["passed"]
        and local_head == remote_head
    )
    return {
        "passed": passed,
        "summary": f"{details.get('nameWithOwner', 'repository')} visibility={details.get('visibility')}; local/remote head match={local_head == remote_head}",
        "repository": details,
        "local_head": local_head,
        "remote_head": remote_head,
        "commands": [view, local, remote],
    }


def build_final_verification(
    *,
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
    smoke_path: str | Path,
    pdf_technical_qa_path: str | Path,
    pdf_visual_qa_path: str | Path,
    video_technical_qa_path: str | Path,
    video_visual_qa_path: str | Path,
    browser_qa_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo, run = Path(repo_root).resolve(), Path(run_path).resolve()
    experiment, paper = Path(experiment_path).resolve(), Path(paper_path).resolve()
    root_python, extension_python = _verification_python_paths(repo)

    commands = {
        "extension_tests": _run(
            [
                str(extension_python),
                "-m",
                "pytest",
                "-q",
                "extensions/reversible_null_phase/tests",
            ],
            cwd=repo,
        ),
        "root_regression_tests": _run(
            [str(root_python), "-m", "pytest", "-q", "tests"], cwd=repo
        ),
        "ruff": _run(
            [
                str(extension_python),
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts",
                "extensions/reversible_null_phase/src",
                "extensions/reversible_null_phase/tests",
            ],
            cwd=repo,
        ),
        "compileall": _run(
            [
                str(extension_python),
                "-m",
                "compileall",
                "-q",
                "src",
                "scripts",
                "extensions/reversible_null_phase/src",
            ],
            cwd=repo,
        ),
    }

    smoke = load_verified_json(Path(smoke_path) / "SMOKE_REPORT.json")
    integrity = load_verified_json(experiment / "INTEGRITY_AUDIT.json")
    resume_audit_path = experiment / "RESUME_REPLAY_AUDIT.json"
    resume_audit = load_verified_json(resume_audit_path)
    status = load_verified_json(run / "status.json")
    pdf_technical = load_verified_json(pdf_technical_qa_path)
    pdf_visual = load_verified_json(pdf_visual_qa_path)
    video_technical = load_verified_json(video_technical_qa_path)
    video_visual = load_verified_json(video_visual_qa_path)
    browser = load_verified_json(browser_qa_path)
    parity_path = repo / "extensions/reversible_null_phase/eshkol/winner_parity.parity.json"
    parity = load_verified_json(parity_path)
    sidecars = verify_sidecar_tree((run, experiment, paper, Path(smoke_path)))
    stderr_path = run / "supervisor.stderr.log"
    resource_governor = resource_governor_gate(status, stderr_path)
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    github = _github_gate(repo, branch)

    integrity_checks = integrity["checks"]
    checks: dict[str, dict[str, Any]] = {
        "extension_tests": {
            **commands["extension_tests"],
            "summary": commands["extension_tests"]["stdout_tail"].splitlines()[-1],
        },
        "root_regression_tests": {
            **commands["root_regression_tests"],
            "summary": commands["root_regression_tests"]["stdout_tail"].splitlines()[-1],
        },
        "ruff": {**commands["ruff"], "summary": "repository and extension lint"},
        "compileall": {**commands["compileall"], "summary": "root and extension Python byte-compilation"},
        "smoke_intervention": {
            "passed": bool(smoke["translation_null_passed"] and smoke["resume_exact"]),
            "summary": f"512^2 CUDA smoke; translation mismatches={smoke['translation_mismatch_count']}; exact resume={smoke['resume_exact']}",
            "report": str((Path(smoke_path) / "SMOKE_REPORT.json").resolve()),
        },
        "planned_task_ids": {
            "passed": bool(
                integrity_checks["planned_ids_unique"]
                and integrity_checks["no_missing_result_ids"]
                and integrity_checks["result_ids_unique"]
            ),
            "summary": f"{integrity['realized_scientific_trajectory_count']} realized scientific trajectories; unique/planned identities exact",
        },
        "sidecars": {
            **sidecars,
            "summary": f"{sidecars['checked']} SHA-256 sidecars verified",
        },
        "nonfinite_values": {
            "passed": bool(integrity_checks["all_payload_values_finite"]),
            "summary": "all source and branch payload values finite",
        },
        "duplicate_missing_results": {
            "passed": bool(
                integrity_checks["no_missing_result_ids"]
                and integrity_checks["result_ids_unique"]
                and integrity_checks["branch_result_count_exact"]
                and integrity_checks["source_count_exact"]
            ),
            "summary": "no missing, duplicate, unexpected, or silently omitted result identities",
        },
        "interrupted_resume_equivalence": {
            "passed": bool(smoke["checkpoint_verified"] and smoke["worker_unloaded_before_resume"] and smoke["resume_exact"]),
            "summary": f"checkpoint verified={smoke['checkpoint_verified']}; worker unloaded={smoke['worker_unloaded_before_resume']}; resumed hash={smoke['resumed_final_sha256']}",
        },
        "campaign_resume_replay": {
            "passed": bool(
                resume_audit["passed"]
                and not any(int(value) for value in resume_audit["mismatch_counts"])
                and not any(
                    int(value) for value in resume_audit["reference_mismatch_counts"]
                )
                and integrity_checks["realized_plan_lineage_valid"]
                and integrity_checks["branch_tasks_match_realized_scientific_core"]
            ),
            "summary": (
                f"governed 1024^2 checkpoint replayed through phase "
                f"{resume_audit['checkpoint_elapsed']}; state mismatches="
                f"{sum(int(value) for value in resume_audit['mismatch_counts'])}; "
                f"accepted plan lineage={integrity_checks['realized_plan_lineage_valid']}"
            ),
            "report": str(resume_audit_path.resolve()),
        },
        "resource_governor": resource_governor,
        "eshkol_parity": {
            "passed": bool(
                parity["passed"]
                and parity["mismatch_count"] == 0
                and parity["cell_count"] == 256
                and parity["actual_final_sha256"] == parity["expected_final_sha256"]
            ),
            "summary": "pinned Eshkol/Python complete 16x16 whole-lattice parity: 256 cells, zero mismatches",
            "report": str(parity_path.resolve()),
        },
        "pdf_render_qa": {
            "passed": bool(
                pdf_technical["pdf_count"] == 11
                and pdf_technical["blank_page_count"] == 0
                and pdf_technical["all_sidecars_and_page_counts_passed"]
                and pdf_visual["passed"]
            ),
            "summary": f"11 PDFs, {pdf_technical['rendered_page_count']} rendered pages, zero blank pages; visual review={pdf_visual['passed']}",
        },
        "video_render_qa": {
            "passed": bool(
                video_technical["video_count"] == 4
                and video_technical["all_sidecars_and_decoding_passed"]
                and video_visual["passed"]
            ),
            "summary": f"4 videos/previews decoded and sampled; visual review={video_visual['passed']}",
        },
        "browser_dashboard_explorer": {
            "passed": bool(browser["passed"]),
            "summary": browser["summary"],
        },
        "github_private_push": github,
    }
    payload = {
        "schema": "zws-reversible-null-phase-v032-final-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "winner_raw_uint8_sha256": RULE_SHA256,
        "passed": all(value["passed"] for value in checks.values()),
        "checks": checks,
        "pause_resume_events": [
            "The final deterministic CUDA smoke saved a phase-16 checkpoint, unloaded the worker, reloaded the verified checkpoint, and matched uninterrupted phase 32 byte-for-byte.",
            "The governed campaign checkpoint for task IDs fdea77124c6aebfd38a8 and a15332445c6a146186d2 was independently replayed through elapsed phase 24,576 with zero whole-lattice mismatches before the worker resumed and advanced the same batch.",
            "The governed campaign retained atomic checkpoints and exact-resume semantics; no scientific seed was replaced or restarted as a new replicate.",
        ],
        "failures_and_repairs": [
            "Preregistration Amendment 001 froze missing estimator definitions before any v0.3.2 trajectory was evolved or outcome opened; it changed no seed, task, intervention, horizon, or decision rule.",
            "The v0.3.1 public wording error conflating Q and M was corrected additively through ERRATUM_Q_AND_M.md and checksummed derivatives; original historical artifacts and all simulation data remain unchanged.",
            "A managed retry regenerated volatile realized-plan metadata and correctly stopped on a checkpoint-lineage mismatch. RESUME_REPAIR_001.md and RESUME_REPLAY_AUDIT.json preserve the failure, root cause, exact two-lattice replay, zero mismatch proof, idempotent preparation fix, and successful resume.",
            f"No v0.3.2 scientific trajectory failure or stderr error was silently omitted; the repaired terminal supervisor invocation completed with {resource_governor['managed_pause_count']} documented resource-governor pause diagnostics and zero unexpected stderr lines.",
        ],
        "evidence": {
            "smoke": str((Path(smoke_path) / "SMOKE_REPORT.json").resolve()),
            "integrity": str((experiment / "INTEGRITY_AUDIT.json").resolve()),
            "campaign_resume_replay": str(resume_audit_path.resolve()),
            "pdf_technical_qa": str(Path(pdf_technical_qa_path).resolve()),
            "pdf_visual_qa": str(Path(pdf_visual_qa_path).resolve()),
            "video_technical_qa": str(Path(video_technical_qa_path).resolve()),
            "video_visual_qa": str(Path(video_visual_qa_path).resolve()),
            "browser_qa": str(Path(browser_qa_path).resolve()),
            "eshkol_parity": str(parity_path.resolve()),
        },
    }
    atomic_write_json(output_path, payload)
    if not payload["passed"]:
        failed = [name for name, value in checks.items() if not value["passed"]]
        raise RuntimeError(f"final verification failed gates: {', '.join(failed)}")
    return payload
