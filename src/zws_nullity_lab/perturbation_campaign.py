from __future__ import annotations

from datetime import datetime, timezone
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
from .followup_campaign import rule_table_digest, tensor_digest
from .margolus import step_batched
from .perturbations import apply_perturbation
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule


SCHEMA = "zwslab-perturbation-response-manifest-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"manifest sidecar is missing: {sidecar}")
    digest = sha256_file(manifest_path)
    declared = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if digest != declared:
        raise RuntimeError(f"manifest hash mismatch: expected {declared}, observed {digest}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported perturbation manifest schema: {manifest.get('schema')!r}")
    repo_root = _repo_root(manifest_path)
    baseline_path = (repo_root / manifest["baseline_manifest_path"]).resolve()
    if sha256_file(baseline_path) != manifest["baseline_manifest_sha256"]:
        raise RuntimeError("frozen baseline manifest digest changed")
    branches = list(manifest["planned_branches"])
    if len(branches) != int(manifest["planned_branch_count"]):
        raise ValueError("planned perturbation branch count does not match rows")
    run_ids = [str(row["run_id"]) for row in branches]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("perturbation run identifiers are not unique")
    condition_ids = {str(condition["condition_id"]) for condition in manifest["conditions"]}
    if {str(row["condition_id"]) for row in branches} - condition_ids:
        raise ValueError("perturbation branches reference unknown conditions")
    if set(int(row["initial_seed"]) for row in branches) != set(manifest["initial_seeds"]):
        raise ValueError("perturbation branch seed block differs from manifest")
    for condition in manifest["conditions"]:
        rule = ReversibleRule.load(_rule_path(repo_root, condition))
        rule.assert_valid()
        if rule_table_digest(rule) != condition["rule_table_sha256_uint8"]:
            raise RuntimeError(f"perturbation rule digest changed: {condition['condition_id']}")
    config = manifest["configuration"]
    height, width = (int(value) for value in config["world"])
    sample_interval = int(config["sample_interval_phases"])
    phases = int(config["phases"])
    if height < 64 or width < 64 or height % 64 or width % 64:
        raise ValueError("perturbation worlds must be at least 64x64 and divisible by 64")
    if phases % sample_interval:
        raise ValueError("perturbation phases must divide into saved-sample intervals")
    if int(config["checkpoint_interval_phases"]) != sample_interval:
        raise ValueError("perturbation runner checkpoints at every saved-sample boundary")
    for tick in config["intervention_ticks"]:
        if int(tick) <= 0 or int(tick) >= phases or int(tick) % sample_interval:
            raise ValueError("intervention ticks must be interior saved-sample boundaries")
    expected_branch_keys = {
        (condition_id, int(tick), str(kind), int(seed))
        for condition_id in condition_ids
        for tick in config["intervention_ticks"]
        for kind in config["intervention"]["kinds"]
        for seed in manifest["initial_seeds"]
    }
    observed_branch_keys = {
        (
            str(row["condition_id"]),
            int(row["intervention_tick"]),
            str(row["intervention_kind"]),
            int(row["initial_seed"]),
        )
        for row in branches
    }
    if observed_branch_keys != expected_branch_keys or len(branches) != len(expected_branch_keys):
        raise ValueError("perturbation manifest is not the complete balanced branch product")
    for row in branches:
        evaluation_ticks = [int(value) for value in row["evaluation_ticks"]]
        if (
            evaluation_ticks != sorted(set(evaluation_ticks))
            or evaluation_ticks[0] != int(row["intervention_tick"])
            or evaluation_ticks[-1] != phases
            or any(tick % sample_interval for tick in evaluation_ticks)
        ):
            raise ValueError(f"invalid perturbation evaluation ticks: {row['run_id']}")
    return manifest, digest, repo_root


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"checkpoint digest mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "zwslab-perturbation-group-checkpoint-v1":
        raise ValueError("unsupported perturbation checkpoint schema")
    return payload


def _publish_status(root: Path, **values: Any) -> None:
    atomic_write_json(
        root / "status.json",
        {
            "schema": "zwslab-perturbation-campaign-status-v1",
            "updated_utc": utc_now(),
            "pid": os.getpid(),
            **values,
        },
    )


def _resource_gate(
    governor: ResourceGovernor,
    root: Path,
    *,
    group_id: str,
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
            group_id=group_id,
            reasons=list(decision.reasons),
            resource_snapshot=decision.snapshot.to_dict(),
        )
        if torch.cuda.is_available() and bool(governor.config.get("release_cuda_cache", True)):
            torch.cuda.empty_cache()
        time.sleep(governor.poll_seconds)


def _group_id(condition_id: str, intervention_tick: int) -> str:
    return f"{condition_id}__t{intervention_tick:05d}"


def _state_q_m(state: torch.Tensor) -> tuple[list[int], list[int]]:
    populations = torch.stack(
        [torch.bincount(item.long().reshape(-1), minlength=4) for item in state]
    )
    charge = [int(value) for value in (populations[:, 1] - populations[:, 2]).tolist()]
    mass = [int(value) for value in populations[:, 1:].sum(dim=1).tolist()]
    return charge, mass


def run_group(
    *,
    root: Path,
    repo_root: Path,
    condition: dict[str, Any],
    branches: list[dict[str, Any]],
    config: dict[str, Any],
    manifest_digest: str,
    device: torch.device,
    governor: ResourceGovernor,
    control: RunControl,
    signal_latch: SignalLatch,
) -> dict[str, Any]:
    condition_id = str(condition["condition_id"])
    intervention_ticks = {int(row["intervention_tick"]) for row in branches}
    if len(intervention_ticks) != 1:
        raise ValueError("one perturbation group must share an intervention tick")
    intervention_tick = intervention_ticks.pop()
    group_id = _group_id(condition_id, intervention_tick)
    group_dir = root / "groups" / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = group_dir / "checkpoint.pt"
    completion_path = group_dir / "completion.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("manifest_sha256") != manifest_digest:
            raise RuntimeError(f"group completion belongs to another manifest: {group_id}")
        if completion.get("status") == "complete":
            return completion
        raise RuntimeError(f"existing failed/non-complete group requires review: {group_id}")

    rule = ReversibleRule.load(_rule_path(repo_root, condition))
    observed_rule_digest = rule_table_digest(rule)
    if observed_rule_digest != condition["rule_table_sha256_uint8"]:
        raise RuntimeError(f"rule digest mismatch: {condition_id}")
    rule_table = rule.table.to(device=device).unsqueeze(0)
    height, width = (int(value) for value in config["world"])
    phases = int(config["phases"])
    sample_interval = int(config["sample_interval_phases"])
    coarse_blocks = [int(value) for value in config["coarse_blocks"]]
    recurrence_lags = [int(value) for value in config["recurrence_lags_phases"]]
    base_seeds = sorted({int(row["initial_seed"]) for row in branches})
    seed_to_index = {seed: index for index, seed in enumerate(base_seeds)}
    ordered_branches = sorted(
        branches,
        key=lambda row: (str(row["intervention_kind"]), int(row["seed_index"])),
    )

    if checkpoint_path.is_file():
        saved = _load_checkpoint(checkpoint_path)
        if saved["manifest_sha256"] != manifest_digest or saved["group_id"] != group_id:
            raise RuntimeError("perturbation checkpoint identity mismatch")
        state = saved["state"].to(device=device)
        tick = int(saved["tick"])
        checkpoint_stage = str(saved["checkpoint_stage"])
        initial_hashes = list(saved["initial_state_sha256_uint8"])
        initial_q = [int(value) for value in saved["initial_charge"]]
        initial_m = [int(value) for value in saved["initial_mass"]]
        audits = list(saved.get("perturbation_audits", []))
        post_hashes = list(saved.get("post_perturb_state_sha256_uint8", []))
        post_q = [int(value) for value in saved.get("post_perturb_charge", [])]
        post_m = [int(value) for value in saved.get("post_perturb_mass", [])]
        samples = list(saved.get("samples", []))
        previous_occupied = list(saved.get("previous_occupied_fractions", []))
        previous_tick = saved.get("previous_tick")
        violation_counts = [dict(value) for value in saved.get("invariant_violation_counts", [])]
    else:
        state_cpu = _initial_batch(
            base_seeds,
            height=height,
            width=width,
            initial_config=config["initial_condition"],
        )
        initial_hashes = [tensor_digest(item) for item in state_cpu]
        initial_q, initial_m = _state_q_m(state_cpu)
        state = state_cpu.to(device=device)
        tick = 0
        checkpoint_stage = "pre_intervention"
        audits = []
        post_hashes = []
        post_q = []
        post_m = []
        samples = []
        previous_occupied = []
        previous_tick = None
        violation_counts = []

    def checkpoint() -> None:
        _save_checkpoint(
            checkpoint_path,
            {
                "schema": "zwslab-perturbation-group-checkpoint-v1",
                "manifest_sha256": manifest_digest,
                "group_id": group_id,
                "condition_id": condition_id,
                "intervention_tick": intervention_tick,
                "checkpoint_stage": checkpoint_stage,
                "tick": tick,
                "state": state.detach().cpu(),
                "initial_state_sha256_uint8": initial_hashes,
                "initial_charge": initial_q,
                "initial_mass": initial_m,
                "perturbation_audits": audits,
                "post_perturb_state_sha256_uint8": post_hashes,
                "post_perturb_charge": post_q,
                "post_perturb_mass": post_m,
                "samples": samples,
                "previous_occupied_fractions": previous_occupied,
                "previous_tick": previous_tick,
                "invariant_violation_counts": violation_counts,
            },
        )

    started = time.monotonic()
    with torch.inference_mode():
        if checkpoint_stage == "pre_intervention":
            while tick < intervention_tick:
                target = min(intervention_tick, tick + sample_interval)
                while tick < target:
                    state = step_batched(state, rule_table, tick)  # type: ignore[assignment]
                    tick += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                checkpoint()
                _publish_status(
                    root,
                    status="advancing_to_intervention",
                    group_id=group_id,
                    tick=tick,
                    intervention_tick=intervention_tick,
                    checkpoint_sha256=sha256_file(checkpoint_path),
                )
                gate = _resource_gate(
                    governor,
                    root,
                    group_id=group_id,
                    control=control,
                    signal_latch=signal_latch,
                )
                if gate != "runnable":
                    return {
                        "schema": "zwslab-perturbation-group-completion-v1",
                        "status": gate,
                        "group_id": group_id,
                        "tick": tick,
                        "manifest_sha256": manifest_digest,
                    }

            intervention = config["intervention"]
            origin = tuple(int(value) for value in intervention["patch_origin_row_column"])
            patch_size = int(intervention["patch_size"])
            changed_sites = int(intervention["changed_sites"])
            branch_states = []
            audits = []
            for row in ordered_branches:
                base_index = seed_to_index[int(row["initial_seed"])]
                perturbed, audit = apply_perturbation(
                    state[base_index],
                    kind=str(row["intervention_kind"]),  # type: ignore[arg-type]
                    origin=origin,  # type: ignore[arg-type]
                    patch_size=patch_size,
                    changed_sites=changed_sites,
                    seed=int(row["selection_seed"]),
                )
                kind_config = intervention["kinds"][str(row["intervention_kind"])]
                observed_mass_delta = audit.mass_after - audit.mass_before
                if (
                    audit.observed_changed_sites != changed_sites
                    or not audit.charge_preserved
                    or observed_mass_delta != int(kind_config["exact_mass_delta"])
                    or (
                        "M" in kind_config["required_exact_invariants"]
                        and not audit.mass_preserved
                    )
                    or (
                        "all_state_counts" in kind_config["required_exact_invariants"]
                        and audit.state_counts_before != audit.state_counts_after
                    )
                ):
                    raise RuntimeError(f"frozen intervention audit failed: {row['run_id']}")
                branch_states.append(perturbed)
                audits.append(audit.to_dict())
            state = torch.stack(branch_states)
            del branch_states
            post_hashes = [tensor_digest(item) for item in state]
            post_q, post_m = _state_q_m(state)
            samples = [[] for _ in ordered_branches]
            previous_occupied = [None for _ in ordered_branches]
            previous_tick = None
            required = ("Q", "M") if condition["conservation"] == "Q_and_M" else ("Q",)
            violation_counts = [{name: 0 for name in required} for _ in ordered_branches]
            diagnostics = sample_diagnostics(
                state,
                tick=tick,
                rule=rule,
                lag_states={},
                coarse_blocks=coarse_blocks,
                shuffle_base_seeds=[int(row["shuffle_base_seed"]) for row in ordered_branches],
                previous_occupied_fractions=previous_occupied,
                previous_tick=previous_tick,
            )
            for index, record in enumerate(diagnostics):
                record["sample_role"] = "immediate_post_intervention"
                record["occupancy_mass"] = sum(int(value) for value in record["state_populations"][1:])
                samples[index].append(record)
                previous_occupied[index] = float(record["occupied_fraction"])
            previous_tick = tick
            checkpoint_stage = "post_intervention"
            checkpoint()

        required = ("Q", "M") if condition["conservation"] == "Q_and_M" else ("Q",)
        if checkpoint_stage != "post_intervention":
            raise RuntimeError(f"unknown perturbation checkpoint stage: {checkpoint_stage}")
        shuffle_seeds = [int(row["shuffle_base_seed"]) for row in ordered_branches]
        while tick < phases:
            target = min(phases, tick + sample_interval)
            lag_states: dict[int, torch.Tensor] = {}
            while tick < target:
                remaining = target - tick
                if remaining in recurrence_lags:
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
                shuffle_base_seeds=shuffle_seeds,
                previous_occupied_fractions=previous_occupied,
                previous_tick=previous_tick,
            )
            for index, record in enumerate(diagnostics):
                populations = [int(value) for value in record["state_populations"]]
                mass = sum(populations[1:])
                q_bad = int(record["total_charge"]) != post_q[index]
                m_bad = mass != post_m[index]
                record["sample_role"] = "post_intervention_response"
                record["occupancy_mass"] = mass
                record["charge_invariant_violation"] = q_bad
                record["mass_invariant_violation"] = m_bad if "M" in required else None
                violation_counts[index]["Q"] += int(q_bad)
                if "M" in required:
                    violation_counts[index]["M"] += int(m_bad)
                samples[index].append(record)
                previous_occupied[index] = float(record["occupied_fraction"])
            previous_tick = tick
            checkpoint()
            if any(any(counts.values()) for counts in violation_counts):
                raise RuntimeError(f"required post-intervention invariant failed in {group_id}")
            _publish_status(
                root,
                status="measuring_response",
                group_id=group_id,
                tick=tick,
                phases=phases,
                checkpoint_sha256=sha256_file(checkpoint_path),
            )
            gate = _resource_gate(
                governor,
                root,
                group_id=group_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                return {
                    "schema": "zwslab-perturbation-group-completion-v1",
                    "status": gate,
                    "group_id": group_id,
                    "tick": tick,
                    "manifest_sha256": manifest_digest,
                }

    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    final_hashes = [tensor_digest(item) for item in state]
    trajectory_paths = []
    for index, row in enumerate(ordered_branches):
        base_index = seed_to_index[int(row["initial_seed"])]
        path = trajectory_dir / f"{row['run_id']}.json"
        if path.exists():
            raise RuntimeError(f"refusing to replace perturbation trajectory: {path}")
        record = {
            "schema": "zwslab-perturbation-response-trajectory-v1",
            "status": "complete",
            "run_id": row["run_id"],
            "condition_id": condition_id,
            "conservation": condition["conservation"],
            "seed_index": row["seed_index"],
            "initial_seed": row["initial_seed"],
            "baseline_run_id": row["baseline_run_id"],
            "intervention_tick": intervention_tick,
            "intervention_kind": row["intervention_kind"],
            "selection_seed": row["selection_seed"],
            "evaluation_ticks": row["evaluation_ticks"],
            "world": [height, width],
            "phases": phases,
            "manifest_sha256": manifest_digest,
            "rule_table_sha256_uint8": observed_rule_digest,
            "initial_state_sha256_uint8": initial_hashes[base_index],
            "post_perturb_state_sha256_uint8": post_hashes[index],
            "final_state_sha256_uint8": final_hashes[index],
            "perturbation_audit": audits[index],
            "required_dynamical_invariants": list(required),
            "invariant_violation_counts": violation_counts[index],
            "samples": samples[index],
        }
        atomic_write_json(path, record)
        trajectory_paths.append(
            {"run_id": row["run_id"], "path": str(path), "sha256": sha256_file(path)}
        )
    completion = {
        "schema": "zwslab-perturbation-group-completion-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "group_id": group_id,
        "condition_id": condition_id,
        "intervention_tick": intervention_tick,
        "tick": tick,
        "phases": phases,
        "manifest_sha256": manifest_digest,
        "rule_table_sha256_uint8": observed_rule_digest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trajectory_count": len(trajectory_paths),
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
        if json.loads(snapshot_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("perturbation output contains a different manifest snapshot")
    else:
        atomic_write_json(snapshot_path, manifest)
    resolved_device = resolve_device(device)
    resources = dict(DEFAULT_RESOURCE_CONFIG)
    resources.update(resource_config or {})
    atomic_write_json(root / "resource_config.resolved.json", resources)
    if resources.get("below_normal_priority", True):
        atomic_write_json(
            root / "process_policy.json",
            set_background_priority(leave_logical_cores=int(resources.get("leave_logical_cores", 0))),
        )
    conditions = {
        str(condition["condition_id"]): condition for condition in manifest["conditions"]
    }
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in manifest["planned_branches"]:
        key = (str(row["condition_id"]), int(row["intervention_tick"]))
        groups.setdefault(key, []).append(row)
    governor = ResourceGovernor(resources, run_directory=root)
    control = RunControl(root)
    completed = []
    with SignalLatch() as signal_latch:
        for (condition_id, intervention_tick), branches in groups.items():
            group_id = _group_id(condition_id, intervention_tick)
            gate = _resource_gate(
                governor,
                root,
                group_id=group_id,
                control=control,
                signal_latch=signal_latch,
            )
            if gate != "runnable":
                summary = {
                    "schema": "zwslab-perturbation-campaign-summary-v1",
                    "status": gate,
                    "manifest_sha256": manifest_digest,
                    "completed_groups": completed,
                    "paused_group": group_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                return summary
            try:
                result = run_group(
                    root=root,
                    repo_root=repo_root,
                    condition=conditions[condition_id],
                    branches=branches,
                    config=manifest["configuration"],
                    manifest_digest=manifest_digest,
                    device=resolved_device,
                    governor=governor,
                    control=control,
                    signal_latch=signal_latch,
                )
            except Exception as exc:
                summary = {
                    "schema": "zwslab-perturbation-campaign-summary-v1",
                    "status": "failed",
                    "manifest_sha256": manifest_digest,
                    "completed_groups": completed,
                    "failed_group": group_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                group_dir = root / "groups" / group_id
                group_dir.mkdir(parents=True, exist_ok=True)
                failure_path = group_dir / "completion.json"
                if not failure_path.exists():
                    atomic_write_json(failure_path, summary)
                atomic_write_json(root / "campaign_summary.json", summary)
                _publish_status(root, status="failed", summary=summary)
                return summary
            if result["status"] != "complete":
                summary = {
                    "schema": "zwslab-perturbation-campaign-summary-v1",
                    "status": result["status"],
                    "manifest_sha256": manifest_digest,
                    "completed_groups": completed,
                    "paused_group": group_id,
                }
                atomic_write_json(root / "campaign_summary.json", summary)
                return summary
            completed.append(group_id)
    summary = {
        "schema": "zwslab-perturbation-campaign-summary-v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "manifest_sha256": manifest_digest,
        "completed_groups": completed,
        "planned_group_count": len(groups),
        "planned_branch_count": len(manifest["planned_branches"]),
    }
    atomic_write_json(root / "campaign_summary.json", summary)
    _publish_status(root, status="complete", summary=summary)
    return summary
