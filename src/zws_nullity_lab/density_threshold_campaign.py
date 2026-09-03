from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .atomic import atomic_write_json, sha256_file
from .confirmation_campaign import DEFAULT_RESOURCE_CONFIG, resolve_device
from .control import RunControl, SignalLatch
from .density_threshold import density_arm_id, exact_density_inventory
from .homogeneous_campaign import (
    _initial_batch,
    _publish_status,
    _resource_gate,
    run_arm_batch,
    utc_now,
)
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule


SCHEMA = "zwslab-density-threshold-manifest-v1"
AUDIT_SCHEMA = "zwslab-density-threshold-input-audit-v1"


def _repo_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {path}")


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
    if digest != declared:
        raise RuntimeError(f"digest mismatch for {path}: {declared} != {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema in {path}: {payload.get('schema')!r}")
    return payload, digest


def load_and_verify_density_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str, Path]:
    manifest_path = Path(path).resolve()
    manifest, digest = _verified_json(manifest_path, schema=SCHEMA)
    repo_root = _repo_root(manifest_path)
    preregistration = repo_root / str(manifest["preregistration_path"])
    if sha256_file(preregistration) != manifest["preregistration_sha256"]:
        raise RuntimeError("preregistration changed after density inputs were frozen")
    rule_path = repo_root / str(manifest["winner_rule_file"])
    rule = ReversibleRule.load(rule_path)
    rule.assert_valid()
    if _rule_table_digest(rule) != manifest["winner_rule_table_sha256_uint8"]:
        raise RuntimeError("winner rule differs from the frozen density-threshold table")
    audit_path = repo_root / str(manifest["input_audit_path"])
    audit, _ = _verified_json(audit_path, schema=AUDIT_SCHEMA)
    if (
        audit.get("manifest_sha256") != digest
        or not audit.get("all_regenerated_hashes_exact")
        or not audit.get("all_nested_supports_exact")
        or not audit.get("all_total_charge_zero")
    ):
        raise RuntimeError("density input audit is incomplete or belongs to another manifest")

    run_ids: list[str] = []
    initial_hashes: list[str] = []
    for stage_id, stage in manifest["stages"].items():
        if stage_id != stage.get("stage_id"):
            raise ValueError("stage mapping key differs from stage_id")
        height, width = (int(value) for value in stage["world"])
        if height < 64 or width < 64 or height % 64 or width % 64:
            raise ValueError("density worlds must be at least 64 and divisible by 64")
        interval = int(stage["sample_interval_phases"])
        if int(stage["phases"]) % interval or int(stage["checkpoint_interval_phases"]) != interval:
            raise ValueError("density duration/checkpoint cadence is inconsistent")
        rows = list(stage["planned_runs"])
        if len(rows) != int(stage["planned_run_count"]):
            raise ValueError("planned density run count differs from run rows")
        arm_ids = list(stage["arm_ids"])
        targets = [float(value) for value in stage["density_targets"]]
        if arm_ids != [density_arm_id(value) for value in targets]:
            raise ValueError("density arm identifiers differ from the frozen target grid")
        rows_by_arm = {arm_id: [] for arm_id in arm_ids}
        for row in rows:
            if row.get("stage_id") != stage_id or row.get("arm_id") not in rows_by_arm:
                raise ValueError("density run row belongs to another stage or arm")
            expected = exact_density_inventory(height, width, float(row["target_occupancy"]))
            if (
                row["state_counts"] != expected["state_counts"]
                or int(row["occupancy_mass"]) != expected["occupancy_mass"]
                or float(row["realized_occupancy"]) != expected["realized_occupancy"]
                or int(row["total_charge"]) != 0
            ):
                raise RuntimeError(f"frozen density inventory differs for {row['run_id']}")
            rows_by_arm[str(row["arm_id"])].append(row)
            run_ids.append(str(row["run_id"]))
            initial_hashes.append(str(row["initial_state_sha256_uint8"]))
        expected_seed_ids = [int(value) for value in stage["inventory_seeds"]]
        reference_seeds: dict[int, tuple[int, int, int]] | None = None
        for arm_id in arm_ids:
            arm_rows = sorted(rows_by_arm[arm_id], key=lambda row: int(row["seed_index"]))
            if [int(row["inventory_seed"]) for row in arm_rows] != expected_seed_ids:
                raise RuntimeError(f"blocked seed identities differ in {arm_id}")
            observed_seeds = {
                int(row["seed_index"]): (
                    int(row["inventory_seed"]),
                    int(row["arrangement_seed"]),
                    int(row["diagnostic_seed"]),
                )
                for row in arm_rows
            }
            if reference_seeds is None:
                reference_seeds = observed_seeds
            elif observed_seeds != reference_seeds:
                raise RuntimeError("seed/arrangement/diagnostic blocking differs across density")
    if len(run_ids) != len(set(run_ids)) or len(initial_hashes) != len(set(initial_hashes)):
        raise RuntimeError("density run identifiers or initial hashes are not unique")
    if int(audit.get("run_count", -1)) != len(run_ids):
        raise RuntimeError("input audit does not cover every frozen density run")
    return manifest, digest, repo_root


def verify_density_stage_inputs(stage: dict[str, Any]) -> dict[str, Any]:
    seed_count = len(stage["inventory_seeds"])
    previous_supports: list[torch.Tensor | None] = [None] * seed_count
    pair_audits = []
    run_count = 0
    for arm_id in stage["arm_ids"]:
        batch, rows = _initial_batch(stage, arm_id)
        run_count += len(rows)
        for index, (state, row) in enumerate(zip(batch, rows, strict=True)):
            occupied = state != 0
            previous = previous_supports[index]
            if previous is not None and bool((previous & ~occupied).any().item()):
                raise RuntimeError(f"nested support preflight failed for {row['run_id']}")
            previous_supports[index] = occupied
            pair_audits.append(
                {
                    "run_id": row["run_id"],
                    "seed_index": int(row["seed_index"]),
                    "arm_id": arm_id,
                    "initial_state_sha256_uint8": row["initial_state_sha256_uint8"],
                    "state_counts": row["state_counts"],
                    "total_charge": 0,
                    "nested_with_previous_density": previous is not None,
                }
            )
    return {
        "schema": "zwslab-density-threshold-stage-input-preflight-v1",
        "stage_id": stage["stage_id"],
        "run_count": run_count,
        "all_regenerated_hashes_exact": True,
        "all_nested_supports_exact": True,
        "all_total_charge_zero": True,
        "run_audits": pair_audits,
    }


def run_density_campaign(
    manifest_path: str | Path,
    output: str | Path,
    *,
    stage_id: str,
    device: str = "auto",
    resource_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, manifest_digest, repo_root = load_and_verify_density_manifest(manifest_path)
    if stage_id not in manifest["stages"]:
        raise ValueError(f"unknown density stage: {stage_id}")
    stage = manifest["stages"][stage_id]
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "manifest_snapshot.json"
    if snapshot_path.is_file():
        if json.loads(snapshot_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("output directory contains another density manifest snapshot")
    else:
        atomic_write_json(snapshot_path, manifest)

    preflight_path = root / "input_preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("manifest_sha256") != manifest_digest:
            raise RuntimeError("existing density preflight belongs to another manifest")
    else:
        preflight = verify_density_stage_inputs(stage)
        preflight["manifest_sha256"] = manifest_digest
        atomic_write_json(preflight_path, preflight)

    resolved_device = resolve_device(device)
    resource_values = dict(DEFAULT_RESOURCE_CONFIG)
    resource_values.update(resource_config or {})
    atomic_write_json(root / "resource_config.resolved.json", resource_values)
    atomic_write_json(
        root / "campaign_metadata.json",
        {
            "schema": "zwslab-density-threshold-campaign-v1",
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
