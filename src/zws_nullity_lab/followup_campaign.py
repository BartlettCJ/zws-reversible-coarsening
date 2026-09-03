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
from .confirmation_campaign import DEFAULT_RESOURCE_CONFIG, _initial_batch, resolve_device
from .confirmation_diagnostics import sample_diagnostics
from .control import RunControl, SignalLatch
from .margolus import step_batched
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule


SUPPORTED_SCHEMAS = {
    "zwslab-double-conservation-screen-manifest-v1",
    "zwslab-prevalence-screen-manifest-v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rule_table_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(bytes(int(value) for value in rule.table.tolist())).hexdigest()


def tensor_digest(state: torch.Tensor) -> str:
    array = state.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _repo_root(manifest_path: Path) -> Path:
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {manifest_path}")


def _rule_path(repo_root: Path, condition: dict[str, Any]) -> Path:
    declared = Path(str(condition["rule_file"]))
    path = declared if declared.is_absolute() else repo_root / declared
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rule file is missing: {path}")
    return path


def load_and_verify_manifest(path: str | Path) -> tuple[dict[str, Any], str, Path]:
    manifest_path = Path(path).resolve()
    digest = sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"manifest sidecar is missing: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if declared != digest:
        raise RuntimeError(f"manifest hash mismatch: expected {declared}, observed {digest}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported follow-up manifest schema: {manifest.get('schema')!r}")

    conditions = list(manifest["conditions"])
    rows = list(manifest["planned_trajectories"])
    condition_ids = [str(condition["condition_id"]) for condition in conditions]
    run_ids = [str(row["run_id"]) for row in rows]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("condition identifiers are not unique")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run identifiers are not unique")
    if len(rows) != int(manifest["planned_trajectory_count"]):
        raise ValueError("planned trajectory count does not match row count")
    unknown = sorted({str(row["condition_id"]) for row in rows} - set(condition_ids))
    if unknown:
        raise ValueError(f"trajectory rows reference unknown conditions: {unknown}")

    config = manifest["configuration"]
    height, width = (int(value) for value in config["world"])
    phases = int(config["phases"])
    sample_interval = int(config["sample_interval_phases"])
    checkpoint_interval = int(config["checkpoint_interval_phases"])
    if height < 64 or width < 64 or height % 64 or width % 64:
        raise ValueError("follow-up worlds must be at least 64x64 and divisible by 64")
    if phases < 1 or phases % sample_interval:
        raise ValueError("phases must be a positive multiple of the sample interval")
    if checkpoint_interval != sample_interval:
        raise ValueError("this runner requires a checkpoint at every saved-sample boundary")
    lags = [int(value) for value in config["recurrence_lags_phases"]]
    if not lags or min(lags) < 1 or max(lags) > sample_interval or 1 not in lags:
        raise ValueError("recurrence lags must include 1 and fit within a sample interval")
    blocks = [int(value) for value in config["coarse_blocks"]]
    if any(height % block or width % block for block in blocks):
        raise ValueError("every coarse block must divide the world dimensions")

    repo_root = _repo_root(manifest_path)
    observed_rule_hashes: set[str] = set()
    for condition in conditions:
        rule = ReversibleRule.load(_rule_path(repo_root, condition))
        rule.assert_valid()
        observed = rule_table_digest(rule)
        expected = str(condition["rule_table_sha256_uint8"])
        if observed != expected:
            raise RuntimeError(
                f"rule digest mismatch for {condition['condition_id']}: "
                f"expected {expected}, observed {observed}"
            )
        if observed in observed_rule_hashes:
            raise ValueError("follow-up manifest contains duplicate rule tables")
        observed_rule_hashes.add(observed)
    return manifest, digest, repo_root


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
    if value.get("schema") != "zwslab-followup-condition-checkpoint-v1":
        raise ValueError("unsupported follow-up checkpoint schema")
    return value


def _publish_status(root: Path, **values: Any) -> None:
    atomic_write_json(
        root / "status.json",
        {
            "schema": "zwslab-followup-campaign-status-v1",
            "updated_utc": utc_now(),
            "pid": os.getpid(),
            **values,
        },
    )


def _resource_gate(
    governor: ResourceGovernor,
    root: Path,
    *,
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
            condition=condition_id,
            reasons=list(decision.reasons),
            resource_snapshot=decision.snapshot.to_dict(),
        )
        if torch.cuda.is_available() and bool(governor.config.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()
        time.sleep(governor.poll_seconds)


def _required_invariants(condition: dict[str, Any]) -> tuple[str, ...]:
    conservation = str(condition.get("conservation", "Q_only"))
    return ("Q", "M") if conservation == "Q_and_M" else ("Q",)


def _invariant_values(record: dict[str, Any]) -> tuple[int, int]:
    populations = [int(value) for value in record["state_populations"]]
    return int(record["total_charge"]), sum(populations[1:])


def run_condition_batch(
    *,
    root: Path,
    repo_root: Path,
    condition: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_digest: str,
    device: torch.device,
    governor: ResourceGovernor,
    control: RunControl,
    signal_latch: SignalLatch,
) -> dict[str, Any]:
    condition_id = str(condition["condition_id"])
    condition_dir = root / "conditions" / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = condition_dir / "checkpoint.pt"
    completion_path = condition_dir / "completion.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("manifest_sha256") != manifest_digest:
            raise RuntimeError(f"existing completion belongs to another manifest: {condition_id}")
        if completion.get("status") == "complete":
            return completion
        raise RuntimeError(
            f"existing non-complete condition record requires manual preservation/review: "
            f"{condition_id} ({completion.get('status')})"
        )

    rule = ReversibleRule.load(_rule_path(repo_root, condition))
    observed_rule_digest = rule_table_digest(rule)
    if observed_rule_digest != condition["rule_table_sha256_uint8"]:
        raise RuntimeError(f"rule digest mismatch for {condition_id}")
    rule.assert_valid()
    rule_table = rule.table.to(device=device).unsqueeze(0)
    required = _required_invariants(condition)
    height, width = (int(value) for value in config["world"])
    phases = int(config["phases"])
    sample_interval = int(config["sample_interval_phases"])
    seeds = [int(row["initial_seed"]) for row in trajectory_rows]
    shuffle_base_seeds = [int(row["shuffle_base_seed"]) for row in trajectory_rows]

    if checkpoint_path.is_file():
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint["manifest_sha256"] != manifest_digest:
            raise RuntimeError("checkpoint belongs to a different manifest")
        if checkpoint["condition_id"] != condition_id:
            raise RuntimeError("checkpoint condition does not match requested condition")
        if list(checkpoint["initial_seeds"]) != seeds:
            raise RuntimeError("checkpoint seed block does not match manifest")
        state = checkpoint["state"].to(device=device)
        tick = int(checkpoint["tick"])
        samples = list(checkpoint["samples"])
        initial_hashes = list(checkpoint["initial_state_sha256_uint8"])
        initial_q = [int(value) for value in checkpoint["initial_charge"]]
        initial_m = [int(value) for value in checkpoint["initial_mass"]]
        violation_counts = [dict(value) for value in checkpoint["invariant_violation_counts"]]
        previous_occupied = list(checkpoint["previous_occupied_fractions"])
        previous_tick = checkpoint["previous_tick"]
    else:
        state_cpu = _initial_batch(
            seeds,
            height=height,
            width=width,
            initial_config=config["initial_condition"],
        )
        initial_hashes = [tensor_digest(item) for item in state_cpu]
        populations = torch.stack(
            [torch.bincount(item.long().flatten(), minlength=4) for item in state_cpu]
        )
        initial_q = [int(value) for value in (populations[:, 1] - populations[:, 2]).tolist()]
        initial_m = [int(value) for value in populations[:, 1:].sum(dim=1).tolist()]
        state = state_cpu.to(device=device)
        tick = 0
        samples = [[] for _ in seeds]
        violation_counts = [{name: 0 for name in required} for _ in seeds]
        previous_occupied = [None for _ in seeds]
        previous_tick = None

    def save_current_checkpoint() -> None:
        _save_checkpoint(
            checkpoint_path,
            {
                "schema": "zwslab-followup-condition-checkpoint-v1",
                "manifest_sha256": manifest_digest,
                "condition_id": condition_id,
                "rule_table_sha256_uint8": observed_rule_digest,
                "initial_seeds": seeds,
                "tick": tick,
                "state": state.detach().cpu(),
                "samples": samples,
                "initial_state_sha256_uint8": initial_hashes,
                "initial_charge": initial_q,
                "initial_mass": initial_m,
                "invariant_violation_counts": violation_counts,
                "previous_occupied_fractions": previous_occupied,
                "previous_tick": previous_tick,
            },
        )

    def record_sample(records: list[dict[str, Any]]) -> None:
        nonlocal previous_tick
        for index, record in enumerate(records):
            q_value, m_value = _invariant_values(record)
            record["required_invariants"] = list(required)
            record["occupancy_mass"] = m_value
            record["initial_occupancy_mass"] = initial_m[index]
            q_bad = q_value != initial_q[index]
            m_bad = m_value != initial_m[index]
            record["charge_invariant_violation"] = q_bad
            record["mass_invariant_violation"] = m_bad if "M" in required else None
            violation_counts[index]["Q"] += int(q_bad)
            if "M" in required:
                violation_counts[index]["M"] += int(m_bad)
            samples[index].append(record)
            previous_occupied[index] = float(record["occupied_fraction"])
        previous_tick = tick

    coarse_blocks = [int(value) for value in config["coarse_blocks"]]
    lags = [int(value) for value in config["recurrence_lags_phases"]]
    if tick == 0 and not samples[0]:
        record_sample(
            sample_diagnostics(
                state,
                tick=0,
                rule=rule,
                lag_states={},
                coarse_blocks=coarse_blocks,
                shuffle_base_seeds=shuffle_base_seeds,
                previous_occupied_fractions=previous_occupied,
                previous_tick=previous_tick,
            )
        )
        save_current_checkpoint()

    started = time.monotonic()
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
            record_sample(
                sample_diagnostics(
                    state,
                    tick=tick,
                    rule=rule,
                    lag_states=lag_states,
                    coarse_blocks=coarse_blocks,
                    shuffle_base_seeds=shuffle_base_seeds,
                    previous_occupied_fractions=previous_occupied,
                    previous_tick=previous_tick,
                )
            )
            save_current_checkpoint()
            invariant_errors = [
                {name: count for name, count in counts.items() if count}
                for counts in violation_counts
            ]
            if any(invariant_errors):
                raise RuntimeError(
                    f"required invariant violation at tick {tick} for {condition_id}: "
                    f"{invariant_errors}"
                )
            elapsed = max(time.monotonic() - started, 1e-9)
            _publish_status(
                root,
                status="running",
                condition=condition_id,
                tick=tick,
                phases=phases,
                batch_size=len(seeds),
                phases_per_second=tick / elapsed,
                checkpoint_sha256=sha256_file(checkpoint_path),
            )
            gate = _resource_gate(
                governor,
                root,
                condition_id=condition_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                return {
                    "schema": "zwslab-followup-condition-completion-v1",
                    "status": gate,
                    "condition_id": condition_id,
                    "tick": tick,
                    "phases": phases,
                    "manifest_sha256": manifest_digest,
                }

    final_hashes = [tensor_digest(item) for item in state]
    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_paths: list[dict[str, str]] = []
    for index, row in enumerate(trajectory_rows):
        record = {
            "schema": "zwslab-followup-trajectory-v1",
            "status": "complete",
            "run_id": row["run_id"],
            "condition_id": condition_id,
            "conservation": condition.get("conservation", "Q_only"),
            "required_invariants": list(required),
            "seed_index": row["seed_index"],
            "initial_seed": row["initial_seed"],
            "shuffle_base_seed": row["shuffle_base_seed"],
            "world": [height, width],
            "phases": phases,
            "rule_table_sha256_uint8": observed_rule_digest,
            "manifest_sha256": manifest_digest,
            "initial_state_sha256_uint8": initial_hashes[index],
            "final_state_sha256_uint8": final_hashes[index],
            "initial_charge": initial_q[index],
            "initial_occupancy_mass": initial_m[index],
            "invariant_violation_counts": violation_counts[index],
            "samples": samples[index],
        }
        path = trajectory_dir / f"{row['run_id']}.json"
        if path.exists():
            raise RuntimeError(f"refusing to replace existing trajectory: {path}")
        atomic_write_json(path, record)
        trajectory_paths.append(
            {"run_id": str(row["run_id"]), "path": str(path), "sha256": sha256_file(path)}
        )

    completion = {
        "schema": "zwslab-followup-condition-completion-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "condition_id": condition_id,
        "tick": tick,
        "phases": phases,
        "world": [height, width],
        "batch_size": len(seeds),
        "required_invariants": list(required),
        "manifest_sha256": manifest_digest,
        "rule_table_sha256_uint8": observed_rule_digest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trajectories": trajectory_paths,
        "elapsed_seconds_this_process": time.monotonic() - started,
    }
    atomic_write_json(completion_path, completion)
    return completion


def run_campaign(
    manifest_path: str | Path,
    output: str | Path,
    *,
    device: str = "auto",
    resource_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, manifest_digest, repo_root = load_and_verify_manifest(manifest_path)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "manifest_snapshot.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot != manifest:
            raise RuntimeError("output directory contains a different manifest snapshot")
    else:
        atomic_write_json(snapshot_path, manifest)

    resolved_device = resolve_device(device)
    resource_values = dict(DEFAULT_RESOURCE_CONFIG)
    resource_values.update(resource_config or {})
    atomic_write_json(root / "resource_config.resolved.json", resource_values)
    atomic_write_json(
        root / "campaign_metadata.json",
        {
            "schema": "zwslab-followup-campaign-v1",
            "started_or_resumed_utc": utc_now(),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": manifest_digest,
            "manifest_schema": manifest["schema"],
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

    conditions = list(manifest["conditions"])
    rows = list(manifest["planned_trajectories"])
    governor = ResourceGovernor(resource_values, run_directory=root)
    control = RunControl(root)
    completed: list[str] = []
    with SignalLatch() as signal_latch:
        for condition in conditions:
            condition_id = str(condition["condition_id"])
            condition_rows = [row for row in rows if row["condition_id"] == condition_id]
            _publish_status(root, status="preflight", condition=condition_id)
            gate = _resource_gate(
                governor,
                root,
                condition_id=condition_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                summary = {
                    "schema": "zwslab-followup-campaign-summary-v1",
                    "status": gate,
                    "manifest_sha256": manifest_digest,
                    "completed_conditions": completed,
                    "paused_condition": condition_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status=gate, summary=summary)
                return summary
            try:
                result = run_condition_batch(
                    root=root,
                    repo_root=repo_root,
                    condition=condition,
                    trajectory_rows=condition_rows,
                    config=manifest["configuration"],
                    manifest_digest=manifest_digest,
                    device=resolved_device,
                    governor=governor,
                    control=control,
                    signal_latch=signal_latch,
                )
            except Exception as exc:
                failure = {
                    "schema": "zwslab-followup-condition-completion-v1",
                    "status": "failed",
                    "failed_utc": utc_now(),
                    "condition_id": condition_id,
                    "manifest_sha256": manifest_digest,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                condition_dir = root / "conditions" / condition_id
                condition_dir.mkdir(parents=True, exist_ok=True)
                failure_path = condition_dir / "completion.json"
                if not failure_path.exists():
                    atomic_write_json(failure_path, failure)
                summary = {
                    "schema": "zwslab-followup-campaign-summary-v1",
                    "status": "failed",
                    "manifest_sha256": manifest_digest,
                    "completed_conditions": completed,
                    "failed_condition": condition_id,
                    "error": failure["error"],
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status="failed", summary=summary)
                return summary
            if result["status"] != "complete":
                summary = {
                    "schema": "zwslab-followup-campaign-summary-v1",
                    "status": result["status"],
                    "manifest_sha256": manifest_digest,
                    "completed_conditions": completed,
                    "paused_condition": condition_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status=result["status"], summary=summary)
                return summary
            completed.append(condition_id)

    summary = {
        "schema": "zwslab-followup-campaign-summary-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "manifest_sha256": manifest_digest,
        "completed_conditions": completed,
        "planned_condition_count": len(conditions),
        "planned_trajectory_count": len(rows),
    }
    atomic_write_json(root / "campaign_summary.json", summary)
    _publish_status(root, status="complete", summary=summary)
    return summary
