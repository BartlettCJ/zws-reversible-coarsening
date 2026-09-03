from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .atomic import atomic_write_json, atomic_write_text
from .config import StudyConfig
from .doctor import environment_report
from .eshkol import export_parity_case, run_eshkol_parity
from .history_runner import (
    audit_pending_branch_resume,
    prepare_history_branches,
    run_history_branches,
    run_history_full,
    run_history_smoke,
    run_history_sources,
)
from .history_archive import build_companion_archive
from .history_closure import audit_final_history_package, build_completion_record
from .history_package import build_history_publication_package
from .history_pdf_qa import render_pdf_qa
from .history_manuscript import build_manuscript_sources
from .history_lay import build_lay_artifacts
from .history_reporting import build_history_report
from .history_verification import build_final_verification, record_manual_qa
from .history_video_qa import render_video_qa
from .nucleation_reporting import build_nucleation_report
from .reporting import build_report
from .rule import ReversibleRule
from .runner import ManagedPause, run_study, write_plan
from .stage2 import (
    evaluate_primary_gate,
    freeze_stage2_addendum,
    run_stage2_continuation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zws-null-phase",
        description="Powered study of the frozen reversible null-phase onset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--rule")
    doctor.add_argument("--output")

    validate = sub.add_parser("validate")
    validate.add_argument("rule")
    validate.add_argument("--require-occupancy", action="store_true")

    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--managed", action="store_true")
    run.add_argument("--stage", action="append")

    full = sub.add_parser("full")
    full.add_argument("--config", required=True)
    full.add_argument("--output", required=True)
    full.add_argument("--managed", action="store_true")
    full.add_argument("--stage", action="append")

    report = sub.add_parser("report")
    report.add_argument("--output", required=True)

    nucleation_report = sub.add_parser("report-nucleation")
    nucleation_report.add_argument("--run", required=True)
    nucleation_report.add_argument("--experiment", required=True)

    primary_gate = sub.add_parser("primary-gate")
    primary_gate.add_argument("--run", required=True)
    primary_gate.add_argument("--experiment", required=True)

    freeze_stage2 = sub.add_parser("freeze-stage2")
    freeze_stage2.add_argument("--run", required=True)
    freeze_stage2.add_argument("--experiment", required=True)

    run_stage2 = sub.add_parser("run-stage2")
    run_stage2.add_argument("--run", required=True)
    run_stage2.add_argument("--experiment", required=True)
    run_stage2.add_argument("--managed", action="store_true")

    for name in ("history-sources", "history-prepare", "history-branches", "history-full"):
        history = sub.add_parser(name)
        history.add_argument("--config", required=True)
        history.add_argument("--experiment", required=True)
        history.add_argument("--output", required=True)
        if name != "history-prepare":
            history.add_argument("--managed", action="store_true")

    history_resume_audit = sub.add_parser("history-audit-resume")
    history_resume_audit.add_argument("--config", required=True)
    history_resume_audit.add_argument("--experiment", required=True)
    history_resume_audit.add_argument("--output", required=True)

    history_smoke = sub.add_parser("history-smoke")
    history_smoke.add_argument("--config", required=True)
    history_smoke.add_argument("--output", required=True)

    history_report = sub.add_parser("report-history")
    history_report.add_argument("--run", required=True)
    history_report.add_argument("--experiment", required=True)

    history_publish = sub.add_parser("publish-history")
    history_publish.add_argument("--repo", required=True)
    history_publish.add_argument("--config", required=True)
    history_publish.add_argument("--run", required=True)
    history_publish.add_argument("--experiment", required=True)
    history_publish.add_argument("--paper", required=True)

    history_manuscript = sub.add_parser("manuscript-history")
    history_manuscript.add_argument("--repo", required=True)
    history_manuscript.add_argument("--experiment", required=True)
    history_manuscript.add_argument("--paper", required=True)

    history_lay = sub.add_parser("lay-history")
    history_lay.add_argument("--experiment", required=True)
    history_lay.add_argument("--paper", required=True)

    history_archive = sub.add_parser("archive-history")
    history_archive.add_argument("--repo", required=True)
    history_archive.add_argument("--run", required=True)
    history_archive.add_argument("--experiment", required=True)
    history_archive.add_argument("--paper", required=True)
    history_archive.add_argument("--release", required=True)

    for name in ("audit-history", "complete-history"):
        history_closure = sub.add_parser(name)
        history_closure.add_argument("--repo", required=True)
        history_closure.add_argument("--run", required=True)
        history_closure.add_argument("--experiment", required=True)
        history_closure.add_argument("--paper", required=True)
        history_closure.add_argument("--release", required=True)
        history_closure.add_argument("--verification", required=True)

    history_pdf_qa = sub.add_parser("qa-history-pdfs")
    history_pdf_qa.add_argument("--paper", required=True)
    history_pdf_qa.add_argument("--output", required=True)
    history_pdf_qa.add_argument("--pdftoppm")
    history_pdf_qa.add_argument("--pdfinfo")

    history_manual_qa = sub.add_parser("record-history-qa")
    history_manual_qa.add_argument("--output", required=True)
    history_manual_qa.add_argument(
        "--kind", choices=("pdf_visual", "video_visual", "browser"), required=True
    )
    history_manual_qa.add_argument("--summary", required=True)
    history_manual_qa.add_argument("--detail", action="append", default=[])
    history_manual_qa.add_argument("--artifact", action="append", default=[])
    history_manual_qa.add_argument("--passed", action="store_true")

    history_verify = sub.add_parser("verify-history-final")
    history_verify.add_argument("--repo", required=True)
    history_verify.add_argument("--run", required=True)
    history_verify.add_argument("--experiment", required=True)
    history_verify.add_argument("--paper", required=True)
    history_verify.add_argument("--smoke", required=True)
    history_verify.add_argument("--pdf-technical-qa", required=True)
    history_verify.add_argument("--pdf-visual-qa", required=True)
    history_verify.add_argument("--video-technical-qa", required=True)
    history_verify.add_argument("--video-visual-qa", required=True)
    history_verify.add_argument("--browser-qa", required=True)
    history_verify.add_argument("--output", required=True)

    history_video_qa = sub.add_parser("qa-history-videos")
    history_video_qa.add_argument("--paper", required=True)
    history_video_qa.add_argument("--output", required=True)
    history_video_qa.add_argument("--ffmpeg")
    history_video_qa.add_argument("--ffprobe")

    status = sub.add_parser("status")
    status.add_argument("--output", required=True)

    control = sub.add_parser("control")
    control.add_argument("--output", required=True)
    control.add_argument("action", choices=["pause", "resume", "hibernate", "stop"])

    export = sub.add_parser("export-eshkol")
    export.add_argument("--rule", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--size", type=int, default=16)
    export.add_argument("--density", type=float, default=0.28)
    export.add_argument("--seed", type=int, default=303)
    export.add_argument("--steps", type=int, default=256)

    parity = sub.add_parser("run-eshkol")
    parity.add_argument("source")
    parity.add_argument("--eshkol-run", default="eshkol-run")

    return parser


def _control(output: Path, action: str) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    names = ["PAUSE.request", "HIBERNATE.request", "STOP.request", "RESOURCE_PAUSE.request"]
    if action == "resume":
        for name in names:
            (output / name).unlink(missing_ok=True)
        atomic_write_text(output / "RESUME.request", "resume\n")
    else:
        filename = {
            "pause": "PAUSE.request",
            "hibernate": "HIBERNATE.request",
            "stop": "STOP.request",
        }[action]
        atomic_write_text(output / filename, action + "\n")
    return {"action": action, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            payload = environment_report(args.rule)
            if args.output:
                atomic_write_json(args.output, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "validate":
            rule = ReversibleRule.load(args.rule)
            rule.assert_valid(require_occupancy=args.require_occupancy)
            payload = {
                "name": rule.name,
                "audit": rule.audit().__dict__,
                "reactive_pairs": rule.reactive_pairs_relative_to_rotation(),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "plan":
            config = StudyConfig.load(args.config)
            tasks = write_plan(config, Path(args.output))
            print(json.dumps({"tasks": len(tasks), "output": str(Path(args.output).resolve())}, indent=2))
            return 0
        if args.command in {"run", "full"}:
            config = StudyConfig.load(args.config)
            result = run_study(
                config,
                args.output,
                managed=args.managed,
                only_stages=set(args.stage) if args.stage else None,
            )
            if args.command == "full":
                result["report"] = build_report(args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "report":
            print(json.dumps(build_report(args.output), indent=2, sort_keys=True))
            return 0
        if args.command == "report-nucleation":
            payload = build_nucleation_report(args.run, args.experiment)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "primary-gate":
            print(
                json.dumps(
                    evaluate_primary_gate(args.run, args.experiment),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "freeze-stage2":
            print(
                json.dumps(
                    freeze_stage2_addendum(args.run, args.experiment),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run-stage2":
            print(
                json.dumps(
                    run_stage2_continuation(
                        args.run,
                        args.experiment,
                        managed=args.managed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-sources":
            print(
                json.dumps(
                    run_history_sources(
                        args.config,
                        args.experiment,
                        args.output,
                        managed=args.managed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-prepare":
            print(
                json.dumps(
                    prepare_history_branches(args.config, args.experiment, args.output),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-branches":
            print(
                json.dumps(
                    run_history_branches(
                        args.config,
                        args.experiment,
                        args.output,
                        managed=args.managed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-full":
            print(
                json.dumps(
                    run_history_full(
                        args.config,
                        args.experiment,
                        args.output,
                        managed=args.managed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-audit-resume":
            print(
                json.dumps(
                    audit_pending_branch_resume(
                        args.config,
                        args.experiment,
                        args.output,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "history-smoke":
            print(
                json.dumps(
                    run_history_smoke(args.config, args.output),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "report-history":
            print(
                json.dumps(
                    build_history_report(args.run, args.experiment),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "publish-history":
            print(
                json.dumps(
                    build_history_publication_package(
                        repo_root=args.repo,
                        config_path=args.config,
                        run_path=args.run,
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "manuscript-history":
            print(
                json.dumps(
                    build_manuscript_sources(
                        repo_root=args.repo,
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "lay-history":
            print(
                json.dumps(
                    build_lay_artifacts(
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "archive-history":
            print(
                json.dumps(
                    build_companion_archive(
                        repo_root=args.repo,
                        run_path=args.run,
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                        release_path=args.release,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {"audit-history", "complete-history"}:
            function = (
                audit_final_history_package
                if args.command == "audit-history"
                else build_completion_record
            )
            print(
                json.dumps(
                    function(
                        repo_root=args.repo,
                        run_path=args.run,
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                        release_path=args.release,
                        verification_path=args.verification,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "qa-history-pdfs":
            print(
                json.dumps(
                    render_pdf_qa(
                        paper_root=args.paper,
                        output_path=args.output,
                        pdftoppm=args.pdftoppm,
                        pdfinfo=args.pdfinfo,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "record-history-qa":
            print(
                json.dumps(
                    record_manual_qa(
                        output_path=args.output,
                        kind=args.kind,
                        summary=args.summary,
                        details=args.detail,
                        artifacts=args.artifact,
                        passed=args.passed,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "verify-history-final":
            print(
                json.dumps(
                    build_final_verification(
                        repo_root=args.repo,
                        run_path=args.run,
                        experiment_path=args.experiment,
                        paper_path=args.paper,
                        smoke_path=args.smoke,
                        pdf_technical_qa_path=args.pdf_technical_qa,
                        pdf_visual_qa_path=args.pdf_visual_qa,
                        video_technical_qa_path=args.video_technical_qa,
                        video_visual_qa_path=args.video_visual_qa,
                        browser_qa_path=args.browser_qa,
                        output_path=args.output,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "qa-history-videos":
            print(
                json.dumps(
                    render_video_qa(
                        paper_root=args.paper,
                        output_path=args.output,
                        ffmpeg=args.ffmpeg,
                        ffprobe=args.ffprobe,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "status":
            path = Path(args.output) / "status.json"
            if not path.exists():
                print(json.dumps({"state": "not_started", "output": str(Path(args.output).resolve())}, indent=2))
            else:
                print(path.read_text(encoding="utf-8"))
            return 0
        if args.command == "control":
            print(json.dumps(_control(Path(args.output), args.action), indent=2))
            return 0
        if args.command == "export-eshkol":
            rule = ReversibleRule.load(args.rule)
            payload = export_parity_case(
                rule,
                args.output,
                size=args.size,
                density=args.density,
                seed=args.seed,
                steps=args.steps,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "run-eshkol":
            print(json.dumps(run_eshkol_parity(args.source, eshkol_run=args.eshkol_run), indent=2, sort_keys=True))
            return 0
    except ManagedPause as exc:
        print(f"managed pause: {exc}", file=sys.stderr)
        return 75
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
