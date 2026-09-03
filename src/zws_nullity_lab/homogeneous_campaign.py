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
from .confirmation_campaign import DEFAULT_RESOURCE_CONFIG, resolve_device
from .confirmation_diagnostics import sample_diagnostics
from .control import RunControl, SignalLatch
from .density_threshold import exact_inventory_homogeneous
from .homogeneous_ic import (
    MATCHED_ARMS,
    assert_matched_inventories,
    construct_initial_state,
    low_wavenumber_diagnostics,
    state_inventory,
    tensor_digest,
)
from .margolus import step_batched
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule


SCHEMA = "zwslab-homogeneous-ic-manifest-v1"
CHECKPOINT_SCHEMA = "zwslab-homogeneous-arm-checkpoint-v1"
TRAJECTORY_SCHEMA = "zwslab-homogeneous-ic-trajectory-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root(manifest_path: Path) -> Path:
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {manifest_path}")


def _rule_table_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(bytes(int(value) for value in rule.table.tolist())).hexdigest()


def _verified_json(path: Path, *, schema: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if declared != digest:
        raise RuntimeError(f"digest mismatch for {path}: {declared} != {digest}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"unexpected schema in {path}: {value.get('schema')!r}")
    return value, digest


def load_and_verify_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str, Path]:
    manifest_path = Path(path).resolve()
    manifest, digest = _verified_json(manifest_path, schema=SCHEMA)
    repo_root = _repo_root(manifest_path)
    preregistration = repo_root / str(manifest["preregistration_path"])
    if sha256_file(preregistration) != manifest["preregistration_sha256"]:
        raise RuntimeError("preregistration changed after the input freeze")
    rule_path = repo_root / str(manifest["winner_rule_file"])
    rule = ReversibleRule.load(rule_path)
    rule.assert_valid()
    observed_rule = _rule_table_digest(rule)
    if observed_rule != manifest["winner_rule_table_sha256_uint8"]:
        raise RuntimeError("winner rule differs from the frozen table digest")

    audit_path = manifest_path.with_name("INITIAL_STATE_AUDIT.json")
    audit, _ = _verified_json(audit_path, schema="zwslab-homogeneous-ic-input-audit-v1")
    if audit.get("manifest_sha256") != digest:
        raise RuntimeError("input audit belongs to a different manifest")
    if not audit.get("all_matched_inventories_exact") or not audit.get("all_total_charge_zero"):
        raise RuntimeError("input audit did not establish exact inventories and Q=0")

    run_ids: list[str] = []
    for stage_id, stage in manifest["stages"].items():
        if stage_id != stage["stage_id"]:
            raise ValueError("stage mapping key differs from stage_id")
        height, width = (int(value) for value in stage["world"])
        interval = int(stage["sample_interval_phases"])
        if height < 64 or width < 64 or height % 64 or width % 64:
            raise ValueError("world dimensions must be divisible by 64")
        if int(stage["phases"]) % interval:
            raise ValueError("duration must be a multiple of the sample interval")
        if int(stage["checkpoint_interval_phases"]) != interval:
            raise ValueError("every sample boundary must be a checkpoint boundary")
        rows = list(stage["planned_runs"])
        if len(rows) != int(stage["planned_run_count"]):
            raise ValueError("planned run count differs from run rows")
        run_ids.extend(str(row["run_id"]) for row in rows)
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for row in rows:
            if row["stage_id"] != stage_id:
                raise ValueError("run row belongs to another stage")
            by_seed.setdefault(int(row["seed_index"]), {})[str(row["arm_id"])] = row
            if int(row["total_charge"]) != 0:
                raise RuntimeError("frozen input row has nonzero charge")
        for seed_index, records in by_seed.items():
            if not set(MATCHED_ARMS).issubset(records):
                raise ValueError(f"paired records missing at seed {seed_index}")
            counts = [records[arm]["state_counts"] for arm in MATCHED_ARMS]
            if any(value != counts[0] for value in counts[1:]):
                raise RuntimeError(f"frozen paired inventories differ at seed {seed_index}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run identifiers are not unique")
    return manifest, digest, repo_root


def _save_torch(path: Path, payload: dict[str, Any]) -> str:
    atomic_torch_save(path, payload)
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def _load_torch(path: Path, *, schema: str) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(path)
    if declared != observed:
        raise RuntimeError(f"checkpoint digest mismatch: {declared} != {observed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported torch artifact schema: {payload.get('schema')!r}")
    return payload


def _publish_status(root: Path, **values: Any) -> None:
    atomic_write_json(
        root / "status.json",
        {
            "schema": "zwslab-homogeneous-campaign-status-v1",
            "updated_utc": utc_now(),
            "pid": os.getpid(),
            **values,
        },
    )


def _resource_gate(
    governor: ResourceGovernor,
    root: Path,
    *,
    arm_id: str,
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
            arm_id=arm_id,
            reasons=list(decision.reasons),
            resource_snapshot=decision.snapshot.to_dict(),
        )
        if torch.cuda.is_available() and bool(governor.config.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()
        time.sleep(governor.poll_seconds)


def _initial_batch(
    stage: dict[str, Any],
    arm_id: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rows = [row for row in stage["planned_runs"] if row["arm_id"] == arm_id]
    rows.sort(key=lambda row: int(row["seed_index"]))
    height, width = (int(value) for value in stage["world"])
    config = stage["initial_condition"]
    states: list[torch.Tensor] = []
    for row in rows:
        if config.get("construction") == "uniform_exact_inventory_nested":
            state = exact_inventory_homogeneous(
                height,
                width,
                state_counts=row["state_counts"],
                arrangement_seed=int(row["arrangement_seed"]),
            )
        else:
            state = construct_initial_state(
                arm_id,
                height=height,
                width=width,
                inventory_seed=int(row["inventory_seed"]),
                arrangement_seed=int(row["arrangement_seed"]),
                localized_fraction=float(config["localized_fraction"]),
                positive_fraction=float(config["positive_fraction"]),
                bound_fraction=float(config["bound_fraction"]),
                modulation_amplitude=float(config["weak_gradient_modulation_amplitude"]),
            )
        observed_digest = tensor_digest(state)
        if observed_digest != row["initial_state_sha256_uint8"]:
            raise RuntimeError(f"regenerated input hash differs for {row['run_id']}")
        inventory = state_inventory(state)
        if inventory["state_counts"] != row["state_counts"]:
            raise RuntimeError(f"regenerated input inventory differs for {row['run_id']}")
        states.append(state)
    return torch.stack(states), rows


def verify_stage_inputs(stage: dict[str, Any]) -> dict[str, Any]:
    batches: dict[str, torch.Tensor] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_id in stage["arm_ids"]:
        batch, rows = _initial_batch(stage, arm_id)
        batches[arm_id] = batch
        rows_by_arm[arm_id] = rows
    paired_audits = []
    for seed_index in range(len(stage["inventory_seeds"])):
        states = {arm: batches[arm][seed_index] for arm in MATCHED_ARMS}
        paired_audits.append({"seed_index": seed_index, **assert_matched_inventories(states)})
    return {
        "schema": "zwslab-homogeneous-stage-input-preflight-v1",
        "stage_id": stage["stage_id"],
        "run_count": sum(len(value) for value in rows_by_arm.values()),
        "paired_audits": paired_audits,
        "all_regenerated_hashes_exact": True,
    }


def _augment_samples(
    state: torch.Tensor,
    *,
    tick: int,
    rule: ReversibleRule,
    lag_states: dict[int, torch.Tensor],
    coarse_blocks: list[int],
    diagnostic_seeds: list[int],
    previous_occupied: list[float | None],
    previous_tick: int | None,
) -> list[dict[str, Any]]:
    records = sample_diagnostics(
        state,
        tick=tick,
        rule=rule,
        lag_states=lag_states,
        coarse_blocks=coarse_blocks,
        shuffle_base_seeds=diagnostic_seeds,
        previous_occupied_fractions=previous_occupied,
        previous_tick=previous_tick,
    )
    low_k = low_wavenumber_diagnostics(state)
    population = state.shape[-2] * state.shape[-1]
    for record, spectral in zip(records, low_k, strict=True):
        domains = record["empty_domains"]
        record["low_wavenumber"] = spectral
        record["occupancy_interface_edge_density"] = domains["total_perimeter"] / (
            2.0 * population
        )
    return records


def run_arm_batch(
    *,
    root: Path,
    repo_root: Path,
    manifest: dict[str, Any],
    manifest_digest: str,
    stage: dict[str, Any],
    arm_id: str,
    device: torch.device,
    governor: ResourceGovernor,
    control: RunControl,
    signal_latch: SignalLatch,
) -> dict[str, Any]:
    arm_dir = root / "arms" / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_dir / "checkpoint.pt"
    completion_path = arm_dir / "completion.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("manifest_sha256") != manifest_digest:
            raise RuntimeError(f"existing completion belongs to another manifest: {arm_id}")
        if completion.get("status") == "complete":
            return completion
        raise RuntimeError(f"non-complete preserved arm result requires review: {arm_id}")

    rule_path = repo_root / str(manifest["winner_rule_file"])
    rule = ReversibleRule.load(rule_path)
    rule_table = rule.table.to(device=device).unsqueeze(0)
    height, width = (int(value) for value in stage["world"])
    phases = int(stage["phases"])
    interval = int(stage["sample_interval_phases"])
    lags = [int(value) for value in stage["recurrence_lags_phases"]]
    coarse_blocks = [int(value) for value in stage["coarse_blocks"]]
    early_tick = int(stage["early_snapshot_phase"])

    if checkpoint_path.is_file():
        checkpoint = _load_torch(checkpoint_path, schema=CHECKPOINT_SCHEMA)
        if checkpoint["manifest_sha256"] != manifest_digest:
            raise RuntimeError("checkpoint belongs to another manifest")
        if checkpoint["stage_id"] != stage["stage_id"] or checkpoint["arm_id"] != arm_id:
            raise RuntimeError("checkpoint stage/arm differs")
        rows = list(checkpoint["run_rows"])
        state = checkpoint["state"].to(device=device)
        tick = int(checkpoint["tick"])
        samples = list(checkpoint["samples"])
        initial_hashes = list(checkpoint["initial_hashes"])
        initial_inventories = list(checkpoint["initial_inventories"])
        previous_occupied = list(checkpoint["previous_occupied"])
        previous_tick = checkpoint["previous_tick"]
        peak_states = checkpoint["peak_states"].clone()
        peak_scales = list(checkpoint["peak_scales"])
        peak_ticks = [int(value) for value in checkpoint["peak_ticks"]]
    else:
        state_cpu, rows = _initial_batch(stage, arm_id)
        initial_hashes = [tensor_digest(item) for item in state_cpu]
        initial_inventories = [state_inventory(item) for item in state_cpu]
        state = state_cpu.to(device=device)
        tick = 0
        samples = [[] for _ in rows]
        previous_occupied = [None for _ in rows]
        previous_tick = None
        peak_states = state_cpu.clone()
        peak_scales: list[float | None] = [None for _ in rows]
        peak_ticks = [0 for _ in rows]
        snapshot_dir = arm_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _save_torch(
            snapshot_dir / "initial.pt",
            {
                "schema": "zwslab-homogeneous-state-snapshot-v1",
                "manifest_sha256": manifest_digest,
                "stage_id": stage["stage_id"],
                "arm_id": arm_id,
                "role": "initial",
                "tick": 0,
                "state": state_cpu,
            },
        )

    diagnostic_seeds = [int(row["diagnostic_seed"]) for row in rows]

    def save_checkpoint() -> str:
        return _save_torch(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "manifest_sha256": manifest_digest,
                "stage_id": stage["stage_id"],
                "arm_id": arm_id,
                "run_rows": rows,
                "tick": tick,
                "state": state.detach().cpu(),
                "samples": samples,
                "initial_hashes": initial_hashes,
                "initial_inventories": initial_inventories,
                "previous_occupied": previous_occupied,
                "previous_tick": previous_tick,
                "peak_states": peak_states,
                "peak_scales": peak_scales,
                "peak_ticks": peak_ticks,
            },
        )

    def record(records: list[dict[str, Any]]) -> None:
        nonlocal previous_tick
        for index, value in enumerate(records):
            if int(value["total_charge"]) != 0:
                raise RuntimeError(f"exact Q violation in {arm_id} seed {index} at tick {tick}")
            samples[index].append(value)
            previous_occupied[index] = float(value["occupied_fraction"])
            scale = value["empty_domains"]["finite_area_weighted_linear_scale"]
            if scale is not None and (peak_scales[index] is None or float(scale) > peak_scales[index]):
                peak_scales[index] = float(scale)
                peak_ticks[index] = tick
                peak_states[index] = state[index].detach().cpu()
        previous_tick = tick

    if tick == 0 and not samples[0]:
        record(
            _augment_samples(
                state,
                tick=0,
                rule=rule,
                lag_states={},
                coarse_blocks=coarse_blocks,
                diagnostic_seeds=diagnostic_seeds,
                previous_occupied=previous_occupied,
                previous_tick=previous_tick,
            )
        )
        save_checkpoint()

    started = time.monotonic()
    with torch.inference_mode():
        while tick < phases:
            target = min(phases, tick + interval)
            lag_states: dict[int, torch.Tensor] = {}
            while tick < target:
                remaining = target - tick
                if remaining in lags:
                    lag_states[remaining] = state.clone()
                state = step_batched(state, rule_table, tick)  # type: ignore[assignment]
                tick += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record(
                _augment_samples(
                    state,
                    tick=tick,
                    rule=rule,
                    lag_states=lag_states,
                    coarse_blocks=coarse_blocks,
                    diagnostic_seeds=diagnostic_seeds,
                    previous_occupied=previous_occupied,
                    previous_tick=previous_tick,
                )
            )
            if tick == early_tick:
                _save_torch(
                    arm_dir / "snapshots" / f"early_t{tick:06d}.pt",
                    {
                        "schema": "zwslab-homogeneous-state-snapshot-v1",
                        "manifest_sha256": manifest_digest,
                        "stage_id": stage["stage_id"],
                        "arm_id": arm_id,
                        "role": "early",
                        "tick": tick,
                        "state": state.detach().cpu(),
                    },
                )
            checkpoint_digest = save_checkpoint()
            elapsed = max(time.monotonic() - started, 1e-9)
            _publish_status(
                root,
                status="running",
                stage_id=stage["stage_id"],
                arm_id=arm_id,
                tick=tick,
                phases=phases,
                batch_size=len(rows),
                phases_per_second=tick / elapsed,
                checkpoint_sha256=checkpoint_digest,
            )
            gate = _resource_gate(
                governor,
                root,
                arm_id=arm_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                return {
                    "schema": "zwslab-homogeneous-arm-completion-v1",
                    "status": gate,
                    "stage_id": stage["stage_id"],
                    "arm_id": arm_id,
                    "tick": tick,
                    "manifest_sha256": manifest_digest,
                }

    snapshot_dir = arm_dir / "snapshots"
    peak_snapshot_sha = _save_torch(
        snapshot_dir / "peak.pt",
        {
            "schema": "zwslab-homogeneous-state-snapshot-v1",
            "manifest_sha256": manifest_digest,
            "stage_id": stage["stage_id"],
            "arm_id": arm_id,
            "role": "per_seed_peak",
            "tick_by_seed": peak_ticks,
            "state": peak_states,
        },
    )
    final_snapshot_sha = _save_torch(
        snapshot_dir / "final.pt",
        {
            "schema": "zwslab-homogeneous-state-snapshot-v1",
            "manifest_sha256": manifest_digest,
            "stage_id": stage["stage_id"],
            "arm_id": arm_id,
            "role": "final",
            "tick": tick,
            "state": state.detach().cpu(),
        },
    )

    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_entries = []
    for index, row in enumerate(rows):
        trajectory = {
            "schema": TRAJECTORY_SCHEMA,
            "status": "complete",
            "run_id": row["run_id"],
            "stage_id": stage["stage_id"],
            "arm_id": arm_id,
            "seed_index": row["seed_index"],
            "inventory_seed": row["inventory_seed"],
            "arrangement_seed": row["arrangement_seed"],
            "diagnostic_seed": row["diagnostic_seed"],
            "world": [height, width],
            "phases": phases,
            "manifest_sha256": manifest_digest,
            "winner_rule_table_sha256_uint8": manifest["winner_rule_table_sha256_uint8"],
            "initial_state_sha256_uint8": initial_hashes[index],
            "final_state_sha256_uint8": tensor_digest(state[index]),
            "peak_state_sha256_uint8": tensor_digest(peak_states[index]),
            "initial_inventory": initial_inventories[index],
            "peak_finite_domain_scale": peak_scales[index],
            "peak_tick": peak_ticks[index],
            "samples": samples[index],
        }
        path = trajectory_dir / f"{row['run_id']}.json"
        if path.is_file():
            raise RuntimeError(f"refusing to replace trajectory: {path}")
        atomic_write_json(path, trajectory)
        trajectory_entries.append(
            {"run_id": row["run_id"], "path": str(path), "sha256": sha256_file(path)}
        )

    completion = {
        "schema": "zwslab-homogeneous-arm-completion-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "stage_id": stage["stage_id"],
        "arm_id": arm_id,
        "tick": tick,
        "phases": phases,
        "world": [height, width],
        "batch_size": len(rows),
        "manifest_sha256": manifest_digest,
        "peak_snapshot_sha256": peak_snapshot_sha,
        "final_snapshot_sha256": final_snapshot_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trajectories": trajectory_entries,
        "elapsed_seconds_this_process": time.monotonic() - started,
    }
    atomic_write_json(completion_path, completion)
    return completion


def run_campaign(
    manifest_path: str | Path,
    output: str | Path,
    *,
    stage_id: str,
    device: str = "auto",
    resource_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, manifest_digest, repo_root = load_and_verify_manifest(manifest_path)
    if stage_id not in manifest["stages"]:
        raise ValueError(f"unknown stage: {stage_id}")
    stage = manifest["stages"][stage_id]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "manifest_snapshot.json"
    if snapshot_path.is_file():
        if json.loads(snapshot_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("output directory contains another manifest snapshot")
    else:
        atomic_write_json(snapshot_path, manifest)

    preflight_path = root / "input_preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("manifest_sha256") != manifest_digest:
            raise RuntimeError("existing input preflight belongs to another manifest")
    else:
        preflight = verify_stage_inputs(stage)
        preflight["manifest_sha256"] = manifest_digest
        atomic_write_json(preflight_path, preflight)

    resolved_device = resolve_device(device)
    resource_values = dict(DEFAULT_RESOURCE_CONFIG)
    resource_values.update(resource_config or {})
    atomic_write_json(root / "resource_config.resolved.json", resource_values)
    atomic_write_json(
        root / "campaign_metadata.json",
        {
            "schema": "zwslab-homogeneous-campaign-v1",
            "started_or_resumed_utc": utc_now(),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": manifest_digest,
            "stage_id": stage_id,
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
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

    governor = ResourceGovernor(resource_values, run_directory=root)
    control = RunControl(root)
    completed: list[str] = []
    with SignalLatch() as signal_latch:
        for arm_id in stage["arm_ids"]:
            _publish_status(root, status="preflight", stage_id=stage_id, arm_id=arm_id)
            gate = _resource_gate(
                governor,
                root,
                arm_id=arm_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                summary = {
                    "schema": "zwslab-homogeneous-campaign-summary-v1",
                    "status": gate,
                    "stage_id": stage_id,
                    "manifest_sha256": manifest_digest,
                    "completed_arms": completed,
                    "paused_arm": arm_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status=gate, summary=summary)
                return summary
            try:
                result = run_arm_batch(
                    root=root,
                    repo_root=repo_root,
                    manifest=manifest,
                    manifest_digest=manifest_digest,
                    stage=stage,
                    arm_id=arm_id,
                    device=resolved_device,
                    governor=governor,
                    control=control,
                    signal_latch=signal_latch,
                )
            except Exception as exc:
                failure = {
                    "schema": "zwslab-homogeneous-arm-completion-v1",
                    "status": "failed",
                    "failed_utc": utc_now(),
                    "stage_id": stage_id,
                    "arm_id": arm_id,
                    "manifest_sha256": manifest_digest,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                arm_dir = root / "arms" / arm_id
                arm_dir.mkdir(parents=True, exist_ok=True)
                failure_path = arm_dir / "completion.json"
                if not failure_path.exists():
                    atomic_write_json(failure_path, failure)
                summary = {
                    "schema": "zwslab-homogeneous-campaign-summary-v1",
                    "status": "failed",
                    "stage_id": stage_id,
                    "manifest_sha256": manifest_digest,
                    "completed_arms": completed,
                    "failed_arm": arm_id,
                    "error": failure["error"],
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status="failed", summary=summary)
                return summary
            if result["status"] != "complete":
                summary = {
                    "schema": "zwslab-homogeneous-campaign-summary-v1",
                    "status": result["status"],
                    "stage_id": stage_id,
                    "manifest_sha256": manifest_digest,
                    "completed_arms": completed,
                    "paused_arm": arm_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status=result["status"], summary=summary)
                return summary
            completed.append(arm_id)

    summary = {
        "schema": "zwslab-homogeneous-campaign-summary-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "stage_id": stage_id,
        "manifest_sha256": manifest_digest,
        "completed_arms": completed,
        "planned_arm_count": len(stage["arm_ids"]),
        "planned_run_count": stage["planned_run_count"],
    }
    atomic_write_json(root / "campaign_summary.json", summary)
    _publish_status(root, status="complete", summary=summary)
    return summary
