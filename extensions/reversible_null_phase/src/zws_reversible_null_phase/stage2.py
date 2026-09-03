from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .config import StudyConfig
from .initial import state_sha256
from .metrics import StrongDomainCriterion
from .resources import wait_for_quiet
from .rule import ReversibleRule
from .runner import (
    _checkpoint_path,
    _chunks,
    _load_verified_torch,
    _run_batch,
    _save_batch_checkpoint,
    _status,
    _task_result_path,
    _task_result_valid,
    _variant_rules,
    resolve_device,
)
from .tasks import Task, generate_tasks


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected != sha256_file(path):
        raise RuntimeError(f"checksum mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_stage2_bracket(
    event_counts: dict[float, int],
    cohort_sizes: dict[float, int],
    *,
    density_ceiling: float = 0.24,
    minimum_events: int = 2,
    lower_fraction_max_exclusive: float = 0.25,
) -> tuple[float, float] | None:
    densities = sorted(event_counts)
    for index, density in enumerate(densities):
        if index == 0 or density > density_ceiling:
            continue
        lower = densities[index - 1]
        lower_fraction = event_counts[lower] / cohort_sizes[lower]
        if (
            event_counts[density] >= minimum_events
            and lower_fraction < lower_fraction_max_exclusive
        ):
            return lower, density
    return None


def evaluate_primary_gate(run_dir: str | Path, experiment_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    experiment = Path(experiment_dir).resolve()
    frozen = _verified_json(experiment / "FROZEN_PLAN.json")
    primary = [
        task for task in frozen["tasks"] if task["stage"] == "primary_homogeneous_1024"
    ]
    if len(primary) != 84:
        raise RuntimeError(f"frozen primary matrix is not 84 tasks: {len(primary)}")
    results: list[dict[str, Any]] = []
    for task in primary:
        path = run / "tasks" / task["stage"] / f"{task['id']}.json"
        payload = _verified_json(path)
        if payload["config_sha256"] != frozen["config_sha256"]:
            raise RuntimeError(f"config mismatch: {task['id']}")
        if payload["rule"]["sha256"] != frozen["winner_raw_uint8_sha256"]:
            raise RuntimeError(f"winner mismatch: {task['id']}")
        if payload["initial"]["state_sha256"] != task["initial_state_sha256"]:
            raise RuntimeError(f"initial-state mismatch: {task['id']}")
        if any(int(row["charge"]) != 0 for row in payload["topological_series"]):
            raise RuntimeError(f"exact-Q violation: {task['id']}")
        results.append(payload)

    densities = sorted({float(item["task"]["density"]) for item in results})
    event_counts = {
        density: sum(
            item["time_to_event"]["T_hit"] is not None
            for item in results
            if float(item["task"]["density"]) == density
        )
        for density in densities
    }
    cohort_sizes = {
        density: sum(float(item["task"]["density"]) == density for item in results)
        for density in densities
    }
    settings = frozen["config"]["nucleation"]
    positive = settings["positive_control"]
    positive_events = sum(
        item["time_to_event"]["T_hit"] is not None
        and int(item["time_to_event"]["T_hit"]) <= int(positive["phase"])
        for item in results
        if float(item["task"]["density"]) == float(positive["density"])
    )
    positive_pass = positive_events >= int(positive["minimum_events"])
    gate = settings["stage2"]
    bracket = choose_stage2_bracket(
        event_counts,
        cohort_sizes,
        density_ceiling=float(gate["event_density_ceiling"]),
        minimum_events=int(gate["minimum_events"]),
        lower_fraction_max_exclusive=float(
            gate["lower_density_event_fraction_exclusive_max"]
        ),
    )
    selected_ids: list[str] = []
    if bracket is not None and positive_pass:
        selected_ids = [
            item["task"]["id"]
            for item in results
            if float(item["task"]["density"]) in bracket
            and item["time_to_event"]["T_hit"] is None
        ]
    status = (
        "positive_control_failed"
        if not positive_pass
        else "stage2_eligible"
        if bracket is not None
        else "stage2_not_eligible"
    )
    decision = {
        "schema": "zws-reversible-null-phase-v031-primary-gate-v1",
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "primary_results_verified": len(results),
        "config_sha256": frozen["config_sha256"],
        "winner_sha256": frozen["winner_raw_uint8_sha256"],
        "event_counts": {f"{density:.2f}": event_counts[density] for density in densities},
        "cohort_sizes": {f"{density:.2f}": cohort_sizes[density] for density in densities},
        "event_fractions": {
            f"{density:.2f}": event_counts[density] / cohort_sizes[density]
            for density in densities
        },
        "positive_control": {
            "density": positive["density"],
            "phase": positive["phase"],
            "events": positive_events,
            "minimum_events": positive["minimum_events"],
            "pass": positive_pass,
        },
        "selected_bracket": list(bracket) if bracket else None,
        "eligible_censored_source_task_ids": selected_ids,
        "target_phase": int(gate["maximum_phase"]),
        "optional_new_seeds_added": 0,
    }
    atomic_write_json(experiment / "PRIMARY_GATE.json", decision)
    return decision


def _source_batches(config: StudyConfig) -> dict[str, list[Task]]:
    primary = [
        task
        for task in generate_tasks(config)
        if task.stage == str(config.nucleation["primary_stage"])
    ]
    batch_size = config.stages[str(config.nucleation["primary_stage"])].batch_size
    mapping: dict[str, list[Task]] = {}
    for batch in _chunks(primary, batch_size):
        for task in batch:
            mapping[task.id] = batch
    return mapping


def freeze_stage2_addendum(
    run_dir: str | Path, experiment_dir: str | Path
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    experiment = Path(experiment_dir).resolve()
    decision = _verified_json(experiment / "PRIMARY_GATE.json")
    if decision["status"] != "stage2_eligible":
        raise RuntimeError(f"Stage 2 is not eligible: {decision['status']}")
    frozen = _verified_json(experiment / "FROZEN_PLAN.json")
    config = StudyConfig.load(frozen["config_path"])
    source_tasks = {task.id: task for task in generate_tasks(config)}
    batches = _source_batches(config)
    target_phase = int(decision["target_phase"])
    entries: list[dict[str, Any]] = []
    checkpoint_cache: dict[Path, dict[str, Any]] = {}
    for source_id in decision["eligible_censored_source_task_ids"]:
        source_task = source_tasks[source_id]
        result_path = run / "tasks" / source_task.stage / f"{source_id}.json"
        result = _verified_json(result_path)
        if result["time_to_event"]["T_hit"] is not None:
            raise RuntimeError(f"source is not censored: {source_id}")
        source_batch = batches[source_id]
        checkpoint_path = _checkpoint_path(run, source_batch)
        saved = checkpoint_cache.setdefault(checkpoint_path, _load_verified_torch(checkpoint_path))
        if int(saved["phase"]) != source_task.steps:
            raise RuntimeError(f"source checkpoint phase mismatch: {source_id}")
        source_index = saved["task_ids"].index(source_id)
        if state_sha256(saved["states"][source_index]) != result["final"]["state_sha256"]:
            raise RuntimeError(f"source checkpoint state mismatch: {source_id}")
        continuation = Task(
            stage="stage2_extension_1024",
            size=source_task.size,
            density=source_task.density,
            seed=source_task.seed,
            steps=target_phase,
            sample_every=source_task.sample_every,
            topology_every=source_task.topology_every,
            checkpoint_every=source_task.checkpoint_every,
            arrangement=source_task.arrangement,
            rule_variant=source_task.rule_variant,
            early_fit_start=source_task.early_fit_start,
            early_fit_end=source_task.early_fit_end,
            shells=source_task.shells,
        )
        entries.append(
            {
                "source_task_id": source_id,
                "source_task": source_task.to_dict(),
                "source_result_path": str(result_path),
                "source_result_sha256": sha256_file(result_path),
                "source_final_state_sha256": result["final"]["state_sha256"],
                "source_checkpoint_path": str(checkpoint_path),
                "source_checkpoint_sha256": sha256_file(checkpoint_path),
                "source_checkpoint_task_ids": saved["task_ids"],
                "source_checkpoint_index": source_index,
                "continuation_task": continuation.to_dict(),
            }
        )
    plan = {
        "schema": "zws-reversible-null-phase-v031-stage2-plan-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": frozen["config_path"],
        "config_sha256": frozen["config_sha256"],
        "winner_sha256": frozen["winner_raw_uint8_sha256"],
        "source_phase": max(int(entry["source_task"]["steps"]) for entry in entries),
        "target_phase": target_phase,
        "selected_bracket": decision["selected_bracket"],
        "optional_new_seeds_added": 0,
        "task_count": len(entries),
        "entries": entries,
    }
    atomic_write_json(experiment / "STAGE2_PLAN.json", plan)
    lines = [
        "# v0.3.1 Stage 2 Addendum",
        "",
        "Frozen after the complete primary campaign and before any extended outcome was generated.",
        "",
        f"- Gate: {decision['status']}",
        f"- Bracket densities: {decision['selected_bracket']}",
        f"- Exact continuation: phase {plan['source_phase']:,} to {target_phase:,}",
        f"- Censored source trajectories: {len(entries)}",
        "- Optional new seeds: 0",
        "- Restart from phase zero: forbidden",
        "",
        "The exact source task/result/checkpoint hashes and continuation task IDs are frozen in",
        "STAGE2_PLAN.json. Only the listed censored trajectories may be extended.",
        "",
        "## Frozen source IDs",
        "",
        *[
            f"- {entry['source_task_id']} -> {entry['continuation_task']['id']}"
            for entry in entries
        ],
        "",
    ]
    atomic_write_text(experiment / "STAGE2_ADDENDUM.md", "\n".join(lines))
    atomic_write_text(
        experiment / "STAGE2_ADDENDUM.md.sha256",
        f"{sha256_file(experiment / 'STAGE2_ADDENDUM.md')}  STAGE2_ADDENDUM.md\n",
    )
    return plan


def _require_committed(experiment: Path) -> None:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=experiment,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    for name in ("STAGE2_ADDENDUM.md", "STAGE2_PLAN.json"):
        path = experiment / name
        relative = str(path.relative_to(Path(root)))
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            text=True,
            capture_output=True,
        )
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=root
        )
        if tracked.returncode or clean.returncode:
            raise RuntimeError(f"Stage 2 prerequisite is not committed at HEAD: {relative}")


def run_stage2_continuation(
    run_dir: str | Path,
    experiment_dir: str | Path,
    *,
    managed: bool = True,
    require_committed: bool = True,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    experiment = Path(experiment_dir).resolve()
    if require_committed:
        _require_committed(experiment)
    plan = _verified_json(experiment / "STAGE2_PLAN.json")
    config = StudyConfig.load(plan["config_path"])
    # Stage-2 plans frozen by package <= 0.3.1 carry scheme-1 hashes; accept either.
    if plan["config_sha256"] not in {config.sha256(), config.legacy_sha256()}:
        raise RuntimeError("Stage 2 configuration hash changed")
    winner = ReversibleRule.load(config.rule_path)
    if winner.sha256() != plan["winner_sha256"]:
        raise RuntimeError("Stage 2 winner hash changed")
    output = run / "stage2"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "PLAN.json",
        {
            "schema": plan["schema"],
            "config_sha256": plan["config_sha256"],
            "task_count": plan["task_count"],
            "tasks": [entry["continuation_task"] for entry in plan["entries"]],
        },
    )
    entries = {
        entry["continuation_task"]["id"]: entry for entry in plan["entries"]
    }
    continuation_tasks = [
        Task(**{key: value for key, value in entry["continuation_task"].items() if key != "id"})
        for entry in plan["entries"]
    ]
    pending = [
        task
        for task in continuation_tasks
        if not _task_result_valid(_task_result_path(output, task))
    ]
    device = resolve_device(config.device)
    criterion = StrongDomainCriterion(**config.criterion)
    variants = _variant_rules(winner)
    completed = len(continuation_tasks) - len(pending)
    _status(
        output,
        {
            "state": "starting",
            "task_count": len(continuation_tasks),
            "completed": completed,
            "pending": len(pending),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    checkpoint_cache: dict[Path, dict[str, Any]] = {}
    for batch in _chunks(pending, 2):
        source_payloads: list[tuple[dict[str, Any], int]] = []
        for task in batch:
            entry = entries[task.id]
            checkpoint_path = Path(entry["source_checkpoint_path"])
            if sha256_file(checkpoint_path) != entry["source_checkpoint_sha256"]:
                raise RuntimeError(f"source checkpoint changed: {checkpoint_path}")
            saved = checkpoint_cache.setdefault(
                checkpoint_path, _load_verified_torch(checkpoint_path)
            )
            index = int(entry["source_checkpoint_index"])
            if saved["task_ids"][index] != entry["source_task_id"]:
                raise RuntimeError(f"source checkpoint task ordering changed: {task.id}")
            if state_sha256(saved["states"][index]) != entry["source_final_state_sha256"]:
                raise RuntimeError(f"source continuation state changed: {task.id}")
            source_payloads.append((saved, index))

        target_checkpoint = _checkpoint_path(output, batch)
        if not target_checkpoint.exists():
            _save_batch_checkpoint(
                target_checkpoint,
                config_sha256=config.sha256(),
                tasks=batch,
                phase=int(plan["source_phase"]),
                states=torch.stack(
                    [saved["states"][index] for saved, index in source_payloads]
                ),
                initial_states=torch.stack(
                    [saved["initial_states"][index] for saved, index in source_payloads]
                ),
                topological_series=[
                    saved["topological_series"][index]
                    for saved, index in source_payloads
                ],
                temporal_series=[
                    saved["temporal_series"][index] for saved, index in source_payloads
                ],
                cumulative_input_hist=torch.stack(
                    [
                        saved["cumulative_input_hist"][index]
                        for saved, index in source_payloads
                    ]
                ),
                interval_reaction_counts=torch.stack(
                    [
                        saved["interval_reaction_counts"][index]
                        for saved, index in source_payloads
                    ]
                ),
                interval_delta_m=torch.stack(
                    [saved["interval_delta_m"][index] for saved, index in source_payloads]
                ),
                previous_sample_states=torch.stack(
                    [
                        saved["previous_sample_states"][index]
                        for saved, index in source_payloads
                    ]
                ),
                peak_scales=[
                    saved["peak_scales"][index] for saved, index in source_payloads
                ],
                peak_states=[
                    saved["peak_states"][index] for saved, index in source_payloads
                ],
                nucleation_tracking=[
                    saved["nucleation_tracking"][index]
                    for saved, index in source_payloads
                ],
            )
        wait_for_quiet(
            config.resource,
            log_path=output / "resource_log.csv",
            stop_file=output / "STOP.request",
        )
        _run_batch(
            config,
            output,
            batch,
            variants,
            device,
            criterion,
            managed=managed,
        )
        completed += len(batch)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _status(
            output,
            {
                "state": "running",
                "task_count": len(continuation_tasks),
                "completed": completed,
                "pending": len(continuation_tasks) - completed,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    _status(
        output,
        {
            "state": "complete",
            "task_count": len(continuation_tasks),
            "completed": completed,
            "pending": len(continuation_tasks) - completed,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "task_count": len(continuation_tasks),
        "completed": completed,
        "output": str(output),
        "source_phase": plan["source_phase"],
        "target_phase": plan["target_phase"],
    }
