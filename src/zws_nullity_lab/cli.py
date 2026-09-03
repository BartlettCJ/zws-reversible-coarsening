from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .analysis import analyze_rule, local_rule_audit
from .atomic import atomic_write_json
from .checkpoint import checkpoint_summary
from .config import load_config
from .control import RunControl, WorkloadHibernate, WorkloadPaused, WorkloadStop
from .cycle import find_cycle
from .doctor import doctor_report, write_doctor_report
from .eshkol_export import export_eshkol_oracle
from .explorer import capture_bundle, serve_explorer
from .initial import paired_impulse
from .observe import observe_rule
from .resources import ResourceGovernor
from .rules import ReversibleRule
from .search import run_search


EXIT_PAUSED = 75
EXIT_HIBERNATE = 76
EXIT_STOPPED = 77


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zwslab",
        description="ZWSLab - exact signed-nullity cellular-automata research laboratory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="inspect Python/CUDA/Eshkol environment")
    doctor.add_argument("--output", type=Path)

    validate = sub.add_parser("validate", help="exhaustively validate one rule")
    validate.add_argument("rule", type=Path)

    search = sub.add_parser("search", help="run or resume evolutionary rule search")
    search.add_argument("--config", type=Path, default=Path("configs/workstation.yaml"))
    search.add_argument("--output", type=Path)
    search.add_argument("--resume", action="store_true")

    analyze = sub.add_parser("analyze", help="run or resume the adversarial analysis suite")
    analyze.add_argument("rule", type=Path)
    analyze.add_argument("--config", type=Path, default=Path("configs/workstation.yaml"))
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--resume", action="store_true")

    full = sub.add_parser("full", help="search, then analyze the frozen best rule")
    full.add_argument("--config", type=Path, default=Path("configs/workstation.yaml"))
    full.add_argument("--output", type=Path)
    full.add_argument("--resume", action="store_true")

    managed = sub.add_parser(
        "managed",
        help="supervise a full run and restart resumably after resource hibernation",
    )
    managed.add_argument("--config", type=Path, default=Path("configs/workstation.yaml"))
    managed.add_argument("--output", type=Path)

    resume = sub.add_parser("resume", help="resume a run from its resolved config and checkpoint")
    resume.add_argument("run_directory", type=Path)

    control = sub.add_parser("control", help="pause, resume, hibernate, or stop a live run")
    control.add_argument("run_directory", type=Path)
    control.add_argument("action", choices=["pause", "resume", "hibernate", "stop", "clear-stop"])

    status = sub.add_parser("status", help="show heartbeat and checkpoint state")
    status.add_argument("run_directory", type=Path)

    resources = sub.add_parser("resources", help="sample the cooperative resource governor")
    resources.add_argument("--config", type=Path, default=Path("configs/workstation.yaml"))
    resources.add_argument("--run-directory", type=Path)

    explore = sub.add_parser("explore", help="open the smooth offline trajectory explorer")
    explore.add_argument("path", type=Path)
    explore.add_argument("--host", default="127.0.0.1")
    explore.add_argument("--port", type=int, default=0)
    explore.add_argument("--no-browser", action="store_true")

    capture = sub.add_parser("capture", help="export an exact frame, crop, state, and GIF clip")
    capture.add_argument("path", type=Path)
    capture.add_argument("--frame", type=int, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--before", type=int, default=24)
    capture.add_argument("--after", type=int, default=48)
    capture.add_argument("--crop", type=str, help="x,y,width,height")
    capture.add_argument("--scale", type=int, default=6)
    capture.add_argument("--fps", type=int, default=20)

    smoke = sub.add_parser("smoke", help="small CPU/GPU end-to-end acceptance run")
    smoke.add_argument("--output", type=Path, default=Path("results/smoke"))
    smoke.add_argument("--resume", action="store_true")

    observe = sub.add_parser(
        "observe",
        help="stream a huge reversible world and retain sparse local event clips",
    )
    observe.add_argument("rule", type=Path)
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--height", type=int, default=2048)
    observe.add_argument("--width", type=int, default=2048)
    observe.add_argument("--steps", type=int, default=32768)
    observe.add_argument("--device", default="auto")
    observe.add_argument("--seed", type=int, default=260824)
    observe.add_argument("--sample-every", type=int, default=8)
    observe.add_argument("--checkpoint-every", type=int, default=2048)
    observe.add_argument("--event-count", type=int, default=24)
    observe.add_argument("--event-min-separation", type=int, default=512)
    observe.add_argument("--event-before", type=int, default=128)
    observe.add_argument("--event-after", type=int, default=128)
    observe.add_argument("--crop-size", type=int, default=256)
    observe.add_argument("--tile-size", type=int, default=128)
    observe.add_argument("--positive-fraction", type=float, default=0.12)
    observe.add_argument("--bound-fraction", type=float, default=0.08)
    observe.add_argument("--localized", action="store_true")
    observe.add_argument("--resume", action="store_true")

    cycle = sub.add_parser("cycle", help="search for an exact finite orbit repeat")
    cycle.add_argument("rule", type=Path)
    cycle.add_argument("--height", type=int, default=4)
    cycle.add_argument("--width", type=int, default=4)
    cycle.add_argument("--max-ticks", type=int, default=200000)

    oracle = sub.add_parser("export-eshkol", help="emit an exact Eshkol rule oracle")
    oracle.add_argument("rule", type=Path)
    oracle.add_argument("--output", type=Path, required=True)
    return parser


def _default_run_directory(config: dict[str, object]) -> Path:
    run = config["run"]  # type: ignore[index]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(str(run["output_root"])) / f"{run['name']}_{stamp}"  # type: ignore[index]


def _run_full(config: dict[str, object], output: Path | None, *, resume: bool) -> dict[str, object]:
    search_result = run_search(config, output, resume=resume)  # type: ignore[arg-type]
    analysis_dir = Path(search_result.run_directory) / "analysis"
    analysis = analyze_rule(
        ReversibleRule.load(search_result.best_rule_path),
        config,  # type: ignore[arg-type]
        analysis_dir,
        resume=resume,
    )
    return {
        "search": search_result.to_dict(),
        "analysis_paths": analysis["paths"],
    }


def _active_supervisor_record(root: Path) -> dict[str, object] | None:
    path = root / "supervisor.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        pid = int(value["pid"])
        expected_create_time = float(value["process_create_time"])
        if str(value.get("status")) in {"complete", "stopped", "failed"}:
            return None
        process = psutil.Process(pid)
        if not process.is_running():
            return None
        if abs(float(process.create_time()) - expected_create_time) > 1.0:
            return None
        return value
    except (OSError, ValueError, KeyError, json.JSONDecodeError, psutil.Error):
        return None


def _managed(config_path: Path, output: Path | None) -> int:
    config = load_config(config_path)
    root = (output or _default_run_directory(config)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = _active_supervisor_record(root)
    if existing is not None and int(existing["pid"]) != os.getpid():
        raise RuntimeError(
            f"a managed supervisor is already active for {root} (PID {existing['pid']})"
        )
    control = RunControl(root)
    governor = ResourceGovernor(config["resources"], run_directory=root)
    process_create_time = float(psutil.Process(os.getpid()).create_time())

    def publish(status: str, message: str | None = None) -> None:
        atomic_write_json(
            root / "supervisor.json",
            {
                "schema": "zws-nullity-supervisor-v1",
                "heartbeat_utc": datetime.now().astimezone().isoformat(),
                "pid": os.getpid(),
                "process_create_time": process_create_time,
                "status": status,
                "message": message,
                "run_directory": str(root),
                "config": str(config_path.resolve()),
            },
        )

    first = not (root / "config.resolved.yaml").exists()
    print(f"Managed ZWS run directory: {root}")
    publish("starting")
    try:
        while True:
            flags = control.read()
            if flags.stop:
                print("STOP file present; managed supervisor is exiting.")
                publish("stopped", "STOP file present")
                return EXIT_STOPPED
            if flags.pause or flags.hibernate:
                mode = "paused" if flags.pause else "hibernated"
                publish(mode, "manual control latch present; waiting for resume")
                print("Run is manually paused/hibernated. Use 'zwslab control <run> resume'.")
                time.sleep(governor.poll_seconds)
                continue
            command = [
                sys.executable,
                "-m",
                "zws_nullity_lab.cli",
                "full",
                "--config",
                str(config_path.resolve()),
                "--output",
                str(root),
            ]
            if not first:
                command.append("--resume")
            first = False
            publish("worker_running", "resumable worker subprocess active")
            print("Starting resumable worker...")
            try:
                completed = subprocess.run(command, check=False)
            except KeyboardInterrupt:
                control.request("pause")
                publish("paused", "console interrupt requested safe pause")
                print("Pause requested. The worker will checkpoint at its next safe chunk boundary.")
                return EXIT_PAUSED
            if completed.returncode == 0:
                publish("complete", "managed run completed normally")
                print("Managed run completed.")
                return 0
            if completed.returncode == EXIT_STOPPED:
                publish("stopped", "worker stopped safely")
                return EXIT_STOPPED
            if completed.returncode not in {EXIT_PAUSED, EXIT_HIBERNATE}:
                publish("failed", f"worker exit code {completed.returncode}")
                print(f"Worker failed with exit code {completed.returncode}.", file=sys.stderr)
                return completed.returncode
            if completed.returncode == EXIT_PAUSED:
                # A worker may pause because of Ctrl+C/Ctrl+Break without a PAUSE
                # control file having been written. Materialise one here so the
                # supervisor cannot immediately restart and defeat the pause.
                if not control.read().pause:
                    control.request("pause")
                publish("paused", "worker checkpointed; explicit resume required")
                print("Worker paused safely; waiting for an explicit resume request.")
                while control.read().pause and not control.read().stop:
                    publish("paused", "worker checkpointed; explicit resume required")
                    time.sleep(governor.poll_seconds)
                continue
            publish("resource_wait", "worker hibernated after external resource pressure")
            print("Worker hibernated after external resource pressure; waiting for quiet resources.")
            stable = 0
            required = max(1, int(config["resources"].get("idle_confirm_samples", 3)))
            while True:
                flags = control.read()
                if flags.stop:
                    publish("stopped", "STOP observed during resource wait")
                    return EXIT_STOPPED
                if flags.pause or flags.hibernate:
                    publish("paused" if flags.pause else "hibernated", "manual control latch present")
                    time.sleep(governor.poll_seconds)
                    continue
                decision = governor.observe()
                publish("resource_wait", "; ".join(decision.reasons) or "quiet sample")
                print(
                    json.dumps(
                        {
                            "runnable": decision.runnable,
                            "reasons": list(decision.reasons),
                            "cpu": decision.snapshot.cpu_percent,
                            "gpu_util": decision.snapshot.gpu_utilization_percent,
                            "external_gpu_mib": decision.snapshot.external_gpu_memory_mib,
                        }
                    )
                )
                stable = stable + 1 if decision.runnable else 0
                if stable >= required:
                    break
                time.sleep(governor.poll_seconds)
    except KeyboardInterrupt:
        control.request("pause")
        publish("paused", "supervisor interrupted; explicit resume required")
        return EXIT_PAUSED
    except Exception as exc:
        publish("failed", f"{type(exc).__name__}: {exc}")
        raise


def _parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x,y,width,height")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor_report()
            if args.output:
                write_doctor_report(args.output)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "validate":
            rule = ReversibleRule.load(args.rule)
            audit = local_rule_audit(rule)
            print(json.dumps(audit, indent=2, sort_keys=True))
            return 0 if audit["valid"] else 2
        if args.command == "search":
            config = load_config(args.config)
            result = run_search(config, args.output, resume=args.resume)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "analyze":
            config = load_config(args.config)
            rule = ReversibleRule.load(args.rule)
            result = analyze_rule(rule, config, args.output, resume=args.resume)
            print(json.dumps(result["paths"], indent=2, sort_keys=True))
            return 0
        if args.command == "full":
            config = load_config(args.config)
            print(json.dumps(_run_full(config, args.output, resume=args.resume), indent=2, sort_keys=True))
            return 0
        if args.command == "managed":
            return _managed(args.config, args.output)
        if args.command == "resume":
            root = args.run_directory.resolve()
            config_path = root / "config.resolved.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(f"resolved configuration missing: {config_path}")
            config = load_config(config_path)
            print(json.dumps(_run_full(config, root, resume=True), indent=2, sort_keys=True))
            return 0
        if args.command == "control":
            path = RunControl(args.run_directory).request(args.action)
            print(json.dumps({"action": args.action, "path": str(path)}, indent=2))
            return 0
        if args.command == "status":
            root = args.run_directory.resolve()
            status_path = root / "status.json"
            live = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
            print(
                json.dumps(
                    {
                        "run_directory": str(root),
                        "live": live,
                        "supervisor": (
                            json.loads((root / "supervisor.json").read_text(encoding="utf-8"))
                            if (root / "supervisor.json").is_file()
                            else None
                        ),
                        "supervisor_active": _active_supervisor_record(root) is not None,
                        "checkpoint": checkpoint_summary(root),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "resources":
            config = load_config(args.config)
            decision = ResourceGovernor(
                config["resources"], run_directory=args.run_directory
            ).observe()
            print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
            return 0 if decision.runnable else 4
        if args.command == "explore":
            serve_explorer(
                args.path,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
        if args.command == "capture":
            result = capture_bundle(
                args.path,
                frame=args.frame,
                output=args.output,
                before=args.before,
                after=args.after,
                crop=_parse_crop(args.crop),
                scale=args.scale,
                fps=args.fps,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "smoke":
            config_path = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
            if not config_path.exists():
                config_path = Path("configs/smoke.yaml")
            config = load_config(config_path)
            print(json.dumps(_run_full(config, args.output, resume=args.resume), indent=2))
            return 0
        if args.command == "observe":
            result = observe_rule(
                ReversibleRule.load(args.rule),
                args.output,
                height=args.height,
                width=args.width,
                steps=args.steps,
                device=args.device,
                seed=args.seed,
                sample_every=args.sample_every,
                checkpoint_every=args.checkpoint_every,
                event_count=args.event_count,
                event_min_separation=args.event_min_separation,
                event_before=args.event_before,
                event_after=args.event_after,
                crop_size=args.crop_size,
                tile_size=args.tile_size,
                positive_fraction=args.positive_fraction,
                bound_fraction=args.bound_fraction,
                localized=args.localized,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "world": result["world"],
                        "steps": result["steps"],
                        "samples": result["samples"],
                        "checkpoints": result["checkpoints"],
                        "events": len(result["events"]),
                        "performance": result["performance"],
                        "explorer": result["explorer"]["html"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "cycle":
            rule = ReversibleRule.load(args.rule)
            seed = paired_impulse(args.height, args.width, separation=1)
            result = find_cycle(seed, rule, max_ticks=args.max_ticks)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.found else 3
        if args.command == "export-eshkol":
            path = export_eshkol_oracle(ReversibleRule.load(args.rule), args.output)
            print(path)
            return 0
    except WorkloadPaused as exc:
        print(f"PAUSED SAFELY: {exc}", file=sys.stderr)
        return EXIT_PAUSED
    except WorkloadHibernate as exc:
        print(f"HIBERNATED SAFELY: {exc}", file=sys.stderr)
        return EXIT_HIBERNATE
    except WorkloadStop as exc:
        print(f"STOPPED SAFELY: {exc}", file=sys.stderr)
        return EXIT_STOPPED
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
