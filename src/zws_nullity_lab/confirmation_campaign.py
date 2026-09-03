from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .atomic import atomic_torch_save, atomic_write_json, atomic_write_text, sha256_file
from .confirmation_diagnostics import sample_diagnostics
from .control import RunControl, SignalLatch
from .initial import SeedSpec, balanced_random
from .margolus import step_batched
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule


DEFAULT_RESOURCE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "sleep",
    "gpu_index": 0,
    "poll_seconds": 5,
    "sample_interval_seconds": 0.75,
    "busy_confirm_samples": 1,
    "idle_confirm_samples": 3,
    "cpu_busy_percent": 92,
    "cpu_resume_percent": 70,
    "min_available_ram_gib": 10,
    # WDDM reports ordinary desktop/CEF processes as compute clients, usually
    # with unavailable per-process memory. Do not yield merely because those
    # names exist; instead yield when total non-worker VRAM rises materially
    # above the measured Windows desktop/driver baseline.
    "yield_for_external_gpu_processes": False,
    "external_gpu_memory_mib": 4096,
    "gpu_driver_allowance_mib": 3072,
    # Total GPU utilization includes this worker, so it cannot distinguish a
    # competing application. External process/memory and temperature checks can.
    "yield_for_gpu_utilization": False,
    "gpu_busy_temperature_c": 83,
    "gpu_resume_temperature_c": 76,
    "yield_when_user_active": False,
    "release_cuda_cache": True,
    "below_normal_priority": True,
    "leave_logical_cores": 4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rule_table_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(bytes(int(value) for value in rule.table.tolist())).hexdigest()


def tensor_digest(state: torch.Tensor) -> str:
    array = state.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(normalized)


def load_and_verify_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path).resolve()
    digest = sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"manifest sidecar is missing: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if declared != digest:
        raise RuntimeError(f"manifest hash mismatch: expected {declared}, observed {digest}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != "zwslab-phase-separation-confirmation-manifest-v1":
        raise ValueError("unsupported confirmation manifest schema")
    return value, digest


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint sidecar is missing: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(path)
    if declared != observed:
        raise RuntimeError(f"checkpoint hash mismatch: expected {declared}, observed {observed}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "zwslab-confirmation-condition-checkpoint-v1":
        raise ValueError("unsupported confirmation checkpoint schema")
    return value


def _initial_batch(
    seeds: list[int],
    *,
    height: int,
    width: int,
    initial_config: dict[str, Any],
) -> torch.Tensor:
    spec = SeedSpec(
        positive_fraction=float(initial_config["positive_fraction"]),
        bound_fraction=float(initial_config["bound_fraction"]),
        localized=bool(initial_config["localized"]),
        localized_fraction=float(initial_config["localized_fraction"]),
    )
    states = [
        balanced_random(1, height, width, spec=spec, seed=seed, device="cpu")
        for seed in seeds
    ]
    return torch.cat(states, dim=0)


def _publish_status(root: Path, **values: Any) -> None:
    atomic_write_json(
        root / "status.json",
        {
            "schema": "zwslab-confirmation-campaign-status-v1",
            "updated_utc": utc_now(),
            "pid": os.getpid(),
            **values,
        },
    )


def _wait_for_resources(
    governor: ResourceGovernor,
    root: Path,
    *,
    stage_id: str,
    condition_id: str,
    control: RunControl,
    signal_latch: SignalLatch,
) -> str:
    while True:
        flags = control.read()
        if flags.stop:
            return "stopped"
        if flags.pause or flags.hibernate or signal_latch.requested.is_set():
            return "paused"
        decision = governor.observe()
        if decision.runnable:
            return "runnable"
        _publish_status(
            root,
            status="resource_wait",
            stage=stage_id,
            condition=condition_id,
            reasons=list(decision.reasons),
            resource_snapshot=decision.snapshot.to_dict(),
        )
        if torch.cuda.is_available() and bool(governor.config.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()
        time.sleep(governor.poll_seconds)


def run_condition_batch(
    *,
    root: Path,
    stage_id: str,
    condition: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    initial_config: dict[str, Any],
    diagnostics_config: dict[str, Any],
    height: int,
    width: int,
    phases: int,
    manifest_digest: str,
    device: torch.device,
    resume: bool,
    governor: ResourceGovernor,
    control: RunControl,
    signal_latch: SignalLatch,
) -> dict[str, Any]:
    condition_id = str(condition["condition_id"])
    condition_dir = root / stage_id / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = condition_dir / "checkpoint.pt"
    completion_path = condition_dir / "completion.json"
    if resume and completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete" and completion.get("manifest_sha256") == manifest_digest:
            return completion

    rule_path = root.parent.parent / str(condition["rule_file"])
    if not rule_path.is_file():
        # Normal runs place results under the repository, but callers may use
        # an arbitrary output path. The manifest source path is stored below.
        rule_path = Path(str(condition["rule_file"])).resolve()
    rule = ReversibleRule.load(rule_path)
    observed_rule_digest = rule_table_digest(rule)
    if observed_rule_digest != condition["rule_table_sha256_uint8"]:
        raise RuntimeError(f"rule digest mismatch for {condition_id}")
    rule.assert_valid()
    rule_table = rule.table.to(device=device).unsqueeze(0)
    seeds = [int(row["initial_seed"]) for row in trajectory_rows]
    shuffle_base_seeds = [int(row["shuffle_base_seed"]) for row in trajectory_rows]
    initial_hashes: list[str]
    samples: list[list[dict[str, Any]]]
    previous_occupied: list[float | None]
    previous_tick: int | None
    charge_violations: list[int]

    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint["manifest_sha256"] != manifest_digest:
            raise RuntimeError("checkpoint belongs to a different manifest")
        if checkpoint["condition_id"] != condition_id:
            raise RuntimeError("checkpoint condition does not match requested condition")
        state = checkpoint["state"].to(device=device)
        tick = int(checkpoint["tick"])
        samples = checkpoint["samples"]
        initial_hashes = list(checkpoint["initial_state_sha256_uint8"])
        previous_occupied = list(checkpoint["previous_occupied_fractions"])
        previous_tick = checkpoint["previous_tick"]
        charge_violations = list(checkpoint["charge_violations"])
    else:
        state_cpu = _initial_batch(seeds, height=height, width=width, initial_config=initial_config)
        initial_hashes = [tensor_digest(item) for item in state_cpu]
        state = state_cpu.to(device=device)
        tick = 0
        samples = [[] for _ in seeds]
        previous_occupied = [None for _ in seeds]
        previous_tick = None
        charge_violations = [0 for _ in seeds]

    sample_interval = int(diagnostics_config["sample_interval_phases"])
    lags = [int(value) for value in diagnostics_config["recurrence_lags_phases"]]
    coarse_blocks = [int(value) for value in diagnostics_config["coarse_blocks"]]
    if phases % sample_interval:
        raise ValueError("campaign phases must be divisible by the diagnostic interval")
    started = time.monotonic()
    if tick == 0 and not samples[0]:
        diagnostics = sample_diagnostics(
            state,
            tick=0,
            rule=rule,
            lag_states={},
            coarse_blocks=coarse_blocks,
            shuffle_base_seeds=shuffle_base_seeds,
            previous_occupied_fractions=previous_occupied,
            previous_tick=previous_tick,
        )
        for index, record in enumerate(diagnostics):
            samples[index].append(record)
            previous_occupied[index] = record["occupied_fraction"]
            charge_violations[index] += int(record["total_charge"] != 0)
        previous_tick = 0
        _save_checkpoint(
            checkpoint_path,
            {
                "schema": "zwslab-confirmation-condition-checkpoint-v1",
                "manifest_sha256": manifest_digest,
                "condition_id": condition_id,
                "tick": tick,
                "state": state.detach().cpu(),
                "samples": samples,
                "initial_state_sha256_uint8": initial_hashes,
                "previous_occupied_fractions": previous_occupied,
                "previous_tick": previous_tick,
                "charge_violations": charge_violations,
            },
        )

    with torch.inference_mode():
        while tick < phases:
            target_tick = min(phases, tick + sample_interval)
            lag_states: dict[int, torch.Tensor] = {}
            while tick < target_tick:
                remaining = target_tick - tick
                if remaining in lags:
                    lag_states[remaining] = state.clone()
                state = step_batched(state, rule_table, tick)  # type: ignore[assignment]
                tick += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            diagnostics = sample_diagnostics(
                state,
                tick=tick,
                rule=rule,
                lag_states=lag_states,
                coarse_blocks=coarse_blocks,
                shuffle_base_seeds=shuffle_base_seeds,
                previous_occupied_fractions=previous_occupied,
                previous_tick=previous_tick,
            )
            for index, record in enumerate(diagnostics):
                samples[index].append(record)
                previous_occupied[index] = record["occupied_fraction"]
                charge_violations[index] += int(record["total_charge"] != 0)
            previous_tick = tick
            checkpoint_payload = {
                "schema": "zwslab-confirmation-condition-checkpoint-v1",
                "manifest_sha256": manifest_digest,
                "condition_id": condition_id,
                "tick": tick,
                "state": state.detach().cpu(),
                "samples": samples,
                "initial_state_sha256_uint8": initial_hashes,
                "previous_occupied_fractions": previous_occupied,
                "previous_tick": previous_tick,
                "charge_violations": charge_violations,
            }
            _save_checkpoint(checkpoint_path, checkpoint_payload)
            elapsed = max(time.monotonic() - started, 1e-9)
            _publish_status(
                root,
                status="running",
                stage=stage_id,
                condition=condition_id,
                tick=tick,
                phases=phases,
                batch_size=len(seeds),
                phases_per_second=tick / elapsed,
                checkpoint_sha256=sha256_file(checkpoint_path),
            )
            resource_state = _wait_for_resources(
                governor,
                root,
                stage_id=stage_id,
                condition_id=condition_id,
                control=control,
                signal_latch=signal_latch,
            )
            if resource_state != "runnable":
                return {
                    "schema": "zwslab-confirmation-condition-completion-v1",
                    "status": resource_state,
                    "condition_id": condition_id,
                    "tick": tick,
                    "phases": phases,
                    "manifest_sha256": manifest_digest,
                }

    final_hashes = [tensor_digest(item) for item in state]
    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_paths = []
    for index, row in enumerate(trajectory_rows):
        trajectory_record = {
            "schema": "zwslab-confirmation-trajectory-v1",
            "status": "complete",
            "stage_id": stage_id,
            "run_id": row["run_id"],
            "condition_id": condition_id,
            "seed_index": row["seed_index"],
            "initial_seed": row["initial_seed"],
            "shuffle_base_seed": row["shuffle_base_seed"],
            "world": [height, width],
            "phases": phases,
            "rule_table_sha256_uint8": observed_rule_digest,
            "manifest_sha256": manifest_digest,
            "initial_state_sha256_uint8": initial_hashes[index],
            "final_state_sha256_uint8": final_hashes[index],
            "charge_violation_count": charge_violations[index],
            "samples": samples[index],
        }
        path = trajectory_dir / f"{row['run_id']}.json"
        atomic_write_json(path, trajectory_record)
        trajectory_paths.append(
            {"run_id": row["run_id"], "path": str(path), "sha256": sha256_file(path)}
        )

    completion = {
        "schema": "zwslab-confirmation-condition-completion-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "condition_id": condition_id,
        "tick": tick,
        "phases": phases,
        "world": [height, width],
        "batch_size": len(seeds),
        "manifest_sha256": manifest_digest,
        "rule_table_sha256_uint8": observed_rule_digest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trajectories": trajectory_paths,
        "elapsed_seconds_this_process": time.monotonic() - started,
    }
    atomic_write_json(completion_path, completion)
    return completion


def _failed_trajectory_records(
    root: Path,
    *,
    stage_id: str,
    condition_id: str,
    rows: list[dict[str, Any]],
    manifest_digest: str,
    error: Exception,
) -> None:
    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = trajectory_dir / f"{row['run_id']}.json"
        atomic_write_json(
            path,
            {
                "schema": "zwslab-confirmation-trajectory-v1",
                "status": "failed",
                "stage_id": stage_id,
                "run_id": row["run_id"],
                "condition_id": condition_id,
                "seed_index": row["seed_index"],
                "initial_seed": row["initial_seed"],
                "manifest_sha256": manifest_digest,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def run_stage(
    manifest_path: str | Path,
    output: str | Path,
    *,
    stage_id: str,
    device: str = "auto",
    resume: bool = True,
    include_optional_stage_b_seeds: bool = False,
    resource_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, manifest_digest = load_and_verify_manifest(manifest_path)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    stage_key = "stage_a" if stage_id == "stage7a" else "stage_b"
    if stage_id not in {"stage7a", "stage7b"}:
        raise ValueError("stage_id must be stage7a or stage7b")
    stage = manifest["configuration"][stage_key]
    if stage["stage_id"] != stage_id:
        raise RuntimeError("stage identifier mismatch in manifest")
    rows_key = "stage_a_trajectories" if stage_id == "stage7a" else "stage_b_trajectories"
    all_rows = list(manifest[rows_key])
    if stage_id == "stage7b" and not include_optional_stage_b_seeds:
        all_rows = [row for row in all_rows if not row.get("optional_extension", False)]

    resource_values = dict(DEFAULT_RESOURCE_CONFIG)
    resource_values.update(resource_config or {})
    atomic_write_json(root / "resource_config.resolved.json", resource_values)
    atomic_write_json(root / "manifest_snapshot.json", manifest)
    atomic_write_json(
        root / "campaign_metadata.json",
        {
            "schema": "zwslab-confirmation-campaign-v1",
            "started_or_resumed_utc": utc_now(),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": manifest_digest,
            "stage_id": stage_id,
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else None
            ),
            "resource_config": resource_values,
        },
    )
    if resource_values.get("below_normal_priority", True):
        atomic_write_json(
            root / "process_policy.json",
            set_background_priority(
                leave_logical_cores=int(resource_values.get("leave_logical_cores", 0))
            ),
        )

    conditions_by_id = {
        str(condition["condition_id"]): condition for condition in manifest["conditions"]
    }
    condition_ids = list(stage["condition_ids"])
    governor = ResourceGovernor(resource_values, run_directory=root)
    control = RunControl(root)
    completed: list[str] = []
    failed: list[dict[str, str]] = []
    with SignalLatch() as signal_latch:
        for condition_id in condition_ids:
            rows = [row for row in all_rows if row["condition_id"] == condition_id]
            if not rows:
                continue
            _publish_status(root, status="starting_condition", stage=stage_id, condition=condition_id)
            try:
                result = run_condition_batch(
                    root=root,
                    stage_id=stage_id,
                    condition=conditions_by_id[condition_id],
                    trajectory_rows=rows,
                    initial_config=manifest["configuration"]["initial_condition"],
                    diagnostics_config=manifest["configuration"]["diagnostics"],
                    height=int(stage["height"]),
                    width=int(stage["width"]),
                    phases=int(stage["phases"]),
                    manifest_digest=manifest_digest,
                    device=resolved_device,
                    resume=resume,
                    governor=governor,
                    control=control,
                    signal_latch=signal_latch,
                )
            except Exception as exc:
                failed.append({"condition_id": condition_id, "error": f"{type(exc).__name__}: {exc}"})
                _failed_trajectory_records(
                    root,
                    stage_id=stage_id,
                    condition_id=condition_id,
                    rows=rows,
                    manifest_digest=manifest_digest,
                    error=exc,
                )
                atomic_write_json(
                    root / stage_id / condition_id / "completion.json",
                    {
                        "schema": "zwslab-confirmation-condition-completion-v1",
                        "status": "failed",
                        "condition_id": condition_id,
                        "manifest_sha256": manifest_digest,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            if result["status"] == "complete":
                completed.append(condition_id)
            else:
                summary = {
                    "schema": "zwslab-confirmation-stage-summary-v1",
                    "status": result["status"],
                    "stage_id": stage_id,
                    "manifest_sha256": manifest_digest,
                    "completed_conditions": completed,
                    "failed_conditions": failed,
                    "paused_condition": condition_id,
                }
                atomic_write_json(root / f"{stage_id}_summary.json", summary)
                return summary

    expected = [condition_id for condition_id in condition_ids if any(row["condition_id"] == condition_id for row in all_rows)]
    status = "complete" if len(completed) + len(failed) == len(expected) else "incomplete"
    summary = {
        "schema": "zwslab-confirmation-stage-summary-v1",
        "status": status,
        "stage_id": stage_id,
        "manifest_sha256": manifest_digest,
        "completed_utc": utc_now(),
        "completed_conditions": completed,
        "failed_conditions": failed,
        "planned_condition_count": len(expected),
        "planned_trajectory_count": len(all_rows),
    }
    atomic_write_json(root / f"{stage_id}_summary.json", summary)
    _publish_status(root, status=status, stage=stage_id, summary=summary)
    return summary
