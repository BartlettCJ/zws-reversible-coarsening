from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from .atomic import atomic_write_bytes, atomic_write_json, atomic_write_text, sha256_file
from .config import ResourceConfig
from .history_erasure import (
    HistoryConfig,
    SourceCalibration,
    apply_global_shuffle,
    apply_tile_permutation,
    as_uint8_state,
    global_permutation,
    intervention_audit,
    load_frozen_json,
    multiscale_summary,
    payload_sha256,
    permutation_digest,
    state_histogram,
    strong_domain_prefix,
    time_to_event,
    translate_state,
)
from .initial import inventory_for_density, make_state, state_sha256
from .metrics import snapshot_metrics
from .resources import append_resource_log, sample_pressure, wait_for_quiet
from .rule import ReversibleRule
from .runner import (
    ManagedPause,
    _atomic_torch_save,
    _check_controls,
    _load_verified_torch,
    _rule_tensors,
    resolve_device,
)
from .simulator import step_batched


SOURCE_BATCH_SIZE = 2
BRANCH_BATCH_SIZE = 2
CHECKPOINT_GENERATIONS = 3


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verified_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        return False
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        return expected == sha256_file(path)
    except (OSError, IndexError):
        return False


def _write_sidecar(path: Path) -> None:
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def _save_npz(path: Path, **arrays: Any) -> dict[str, Any]:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue())
    _write_sidecar(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _load_state_npz(path: Path) -> np.ndarray:
    if not _verified_sidecar(path):
        raise RuntimeError(f"state sidecar verification failed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.uint8)
    return np.ascontiguousarray(state)


def _save_exact_state(path: Path, state: torch.Tensor | np.ndarray, *, phase: int, elapsed: int | None = None) -> dict[str, Any]:
    array = as_uint8_state(state)
    payload = _save_npz(
        path,
        state=array,
        phase=np.asarray([int(phase)], dtype=np.int64),
        elapsed=np.asarray([-1 if elapsed is None else int(elapsed)], dtype=np.int64),
    )
    payload.update({"state_sha256": state_sha256(array), "phase": int(phase), "elapsed": elapsed})
    return payload


def _checkpoint_available(path: Path) -> bool:
    return any(
        (path if generation == 0 else path.with_name(f"{path.stem}.prev{generation}{path.suffix}")).exists()
        for generation in range(CHECKPOINT_GENERATIONS)
    )


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _source_dir(output: Path, task_id: str) -> Path:
    return output / "sources" / task_id


def _source_result_path(output: Path, task_id: str) -> Path:
    return _source_dir(output, task_id) / "source_result.json"


def _branch_dir(output: Path, task_id: str) -> Path:
    return output / "branches" / task_id


def _branch_result_path(output: Path, task_id: str) -> Path:
    return _branch_dir(output, task_id) / "branch_result.json"


def _status(output: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(output / "status.json", {**payload, "updated_utc": _utc()})


def _load_protocol(
    config_path: str | Path,
    experiment_path: str | Path,
) -> tuple[HistoryConfig, dict[str, Any], dict[str, Any], Path]:
    config = HistoryConfig.load(config_path)
    experiment = Path(experiment_path).resolve()
    plan = load_frozen_json(experiment / "FROZEN_PLAN.json")
    manifest = load_frozen_json(experiment / "INTERVENTION_MANIFEST.json")
    if plan["winner_raw_uint8_sha256"] != config.rule_sha256_uint8:
        raise RuntimeError("config and frozen plan winner hashes differ")
    if plan["config_file_sha256"] != config.file_sha256():
        raise RuntimeError("configuration differs from the frozen plan")
    if plan["intervention_manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise RuntimeError("intervention manifest differs from the frozen plan")
    if plan["counts"]["maximum_scientific_trajectories"] != 88 or manifest["entry_count"] != 72:
        raise RuntimeError("frozen task-count audit failed")
    return config, plan, manifest, experiment


def _initialize_output(
    output: Path,
    config: HistoryConfig,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    experiment: Path,
    device: torch.device,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)
    for source in (
        config.source_path,
        config.source_path.with_suffix(config.source_path.suffix + ".sha256"),
        experiment / "FROZEN_PLAN.json",
        experiment / "FROZEN_PLAN.json.sha256",
        experiment / "INTERVENTION_MANIFEST.json",
        experiment / "INTERVENTION_MANIFEST.json.sha256",
        experiment / "PREREGISTRATION.md",
        experiment / "PREREGISTRATION_AMENDMENT_001.md",
    ):
        if source.exists():
            target = frozen / source.name
            if not target.exists() or sha256_file(target) != sha256_file(source):
                atomic_write_bytes(target, source.read_bytes())
                _write_sidecar(target)
    atomic_write_json(
        output / "environment.json",
        {
            "schema": "zws-reversible-null-phase-v032-environment-v1",
            "created_utc": _utc(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "config_sha256": config.file_sha256(),
            "frozen_plan_payload_sha256": plan["payload_sha256"],
            "intervention_manifest_payload_sha256": manifest["payload_sha256"],
            "winner_sha256": config.rule_sha256_uint8,
        },
    )


def _source_checkpoint_path(output: Path, batch: list[dict[str, Any]]) -> Path:
    joined = "_".join(task["task_id"] for task in batch)
    digest = __import__("hashlib").sha256(joined.encode("ascii")).hexdigest()[:20]
    return output / "checkpoints" / f"source_{digest}.pt"


def _metric_row(state: torch.Tensor, *, phase: int) -> dict[str, Any]:
    metric = snapshot_metrics(state, phase=phase, low_k_max=8.0, topology=True).to_dict()
    metric["state_sha256"] = state_sha256(state)
    occupancy = state != 0
    metric["interface_density"] = float(
        (
            (occupancy != torch.roll(occupancy, shifts=-1, dims=0)).float().mean()
            + (occupancy != torch.roll(occupancy, shifts=-1, dims=1)).float().mean()
        )
        / 2.0
    )
    return metric


_TRANSLATION_NULL_METRIC_FIELDS = (
    "charge",
    "occupancy_count",
    "occupancy_fraction",
    "empty_fraction",
    "positive_fraction",
    "negative_fraction",
    "bound_fraction",
    "normalized_state_entropy",
    "empty_component_count",
    "empty_finite_component_count",
    "empty_wrap_x",
    "empty_wrap_y",
    "largest_empty_fraction",
    "largest_finite_empty_fraction",
    "finite_empty_scale",
    "occupied_component_count",
    "occupied_wrap_x",
    "occupied_wrap_y",
    "largest_occupied_fraction",
    "occupied_perimeter_density",
    "interface_density",
)


def _translation_null_check(
    state: torch.Tensor,
    reference_state: torch.Tensor,
    *,
    offset_yx: tuple[int, int],
    elapsed: int,
    absolute_phase: int,
    metric: dict[str, Any],
) -> dict[str, Any]:
    recovered = translate_state(state, (-offset_yx[0], -offset_yx[1]))
    reference = as_uint8_state(reference_state)
    mismatch = int(np.count_nonzero(recovered != reference))
    reference_metric = _metric_row(reference_state, phase=absolute_phase)
    metric_mismatches = [
        field
        for field in _TRANSLATION_NULL_METRIC_FIELDS
        if metric[field] != reference_metric[field]
    ]
    check = {
        "elapsed": int(elapsed),
        "absolute_phase": int(absolute_phase),
        "mismatch_count": mismatch,
        "inverse_translated_sha256": state_sha256(recovered),
        "untouched_reference_sha256": state_sha256(reference),
        "metric_match": not metric_mismatches,
        "metric_mismatch_fields": metric_mismatches,
    }
    if mismatch:
        raise RuntimeError(
            f"translation null state failed at elapsed {elapsed}: {mismatch} mismatches"
        )
    if metric_mismatches:
        raise RuntimeError(
            "translation null metric invariance failed at elapsed "
            f"{elapsed}: {', '.join(metric_mismatches)}"
        )
    return check


def _save_source_checkpoint(
    path: Path,
    *,
    plan_hash: str,
    task_ids: list[str],
    phase: int,
    states: torch.Tensor,
    initial_states: torch.Tensor,
    series: list[list[dict[str, Any]]],
    flags: list[list[bool]],
    multiscale: list[list[dict[str, Any]]],
    branch_points: list[dict[str, Any]],
    rings: list[list[dict[str, Any]]],
    saved_event_phases: list[list[int]],
    previous_states: torch.Tensor,
    interval_reactions: torch.Tensor,
    interval_delta_m: torch.Tensor,
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema": "zws-reversible-null-phase-v032-source-checkpoint-v1",
            "plan_payload_sha256": plan_hash,
            "task_ids": task_ids,
            "phase": int(phase),
            "states": states.detach().cpu(),
            "initial_states": initial_states.detach().cpu(),
            "series": series,
            "flags": flags,
            "multiscale": multiscale,
            "branch_points": branch_points,
            "rings": rings,
            "saved_event_phases": saved_event_phases,
            "previous_states": previous_states.detach().cpu(),
            "interval_reactions": interval_reactions.detach().cpu(),
            "interval_delta_m": interval_delta_m.detach().cpu(),
            "saved_utc": _utc(),
        },
    )


def _source_exact_capture(
    output: Path,
    task_id: str,
    state: torch.Tensor | np.ndarray,
    *,
    phase: int,
    label: str,
) -> dict[str, Any]:
    path = _source_dir(output, task_id) / "states" / f"{label}_phase_{phase:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return _save_exact_state(path, state, phase=phase)


def _run_source_batch(
    config: HistoryConfig,
    plan: dict[str, Any],
    output: Path,
    batch: list[dict[str, Any]],
    rule: ReversibleRule,
    device: torch.device,
    *,
    managed: bool,
) -> None:
    checkpoint = _source_checkpoint_path(output, batch)
    task_ids = [task["task_id"] for task in batch]
    if _checkpoint_available(checkpoint):
        saved = _load_verified_torch(checkpoint)
        if saved.get("plan_payload_sha256") != plan["payload_sha256"] or saved.get("task_ids") != task_ids:
            raise RuntimeError("source checkpoint does not match frozen plan/batch")
        phase = int(saved["phase"])
        states = saved["states"].to(device=device)
        initial_states = saved["initial_states"].to(device=device)
        series = saved["series"]
        flags = saved["flags"]
        multiscale = saved["multiscale"]
        branch_points = saved["branch_points"]
        rings = saved["rings"]
        saved_event_phases = saved["saved_event_phases"]
        previous_states = saved["previous_states"].to(device=device)
        interval_reactions = saved["interval_reactions"].to(device=device)
        interval_delta_m = saved["interval_delta_m"].to(device=device)
    else:
        phase = 0
        initial = []
        for task in batch:
            inventory = inventory_for_density(config.lattice_size, config.lattice_size, float(task["density"]))
            expected = task["initial_inventory"]
            if inventory.empty != expected["empty"] or inventory.occupied != expected["M"] or inventory.charge != 0:
                raise RuntimeError("source inventory differs from frozen plan")
            initial.append(
                make_state(
                    config.lattice_size,
                    config.lattice_size,
                    inventory,
                    seed=int(task["seed"]),
                    arrangement="homogeneous",
                    device=device,
                )
            )
        states = torch.stack(initial)
        initial_states = states.clone()
        series = []
        flags = []
        multiscale = []
        branch_points = []
        rings = []
        saved_event_phases = []
        for index, task in enumerate(batch):
            task_id = task["task_id"]
            task_dir = _source_dir(output, task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            initial_capture = _source_exact_capture(output, task_id, states[index], phase=0, label="initial")
            metric = _metric_row(states[index], phase=0)
            calibration = SourceCalibration.from_initial_metric(metric)
            classification = strong_domain_prefix([metric], calibration)
            metric.update(classification)
            series.append([metric])
            flags.append([bool(classification["strong_domain"])])
            summary, _ = multiscale_summary(states[index])
            multiscale.append([{"phase": 0, **summary}])
            branch_points.append({"A": None, "B": None, "initial": initial_capture})
            rings.append([{"phase": 0, "state": states[index].detach().cpu()}])
            saved_event_phases.append([])
        previous_states = states.clone()
        interval_reactions = torch.zeros(len(batch), dtype=torch.int64, device=device)
        interval_delta_m = torch.zeros(len(batch), dtype=torch.int64, device=device)

    table = torch.stack([rule.table for _ in batch]).to(device=device)
    aux = _rule_tensors(rule, device)
    reaction_mask = torch.stack([aux["reaction"] for _ in batch])
    delta_m_lookup = torch.stack([aux["delta_m"] for _ in batch])
    blocks_per_phase = (config.lattice_size // 2) ** 2
    calibrations = [SourceCalibration.from_initial_metric(rows[0]) for rows in series]
    resource_policy = ResourceConfig.from_mapping(config.resource)

    while phase < config.source_horizon:
        evolved = step_batched(states, table, phase, return_codes=True)
        assert isinstance(evolved, tuple)
        states, input_codes, _ = evolved
        flat_codes = input_codes.reshape(len(batch), -1)
        interval_reactions += reaction_mask.gather(1, flat_codes).sum(dim=1)
        interval_delta_m += delta_m_lookup.gather(1, flat_codes).sum(dim=1)
        phase += 1

        if phase % config.observation_every == 0 or phase == config.source_horizon:
            for index, task in enumerate(batch):
                task_id = task["task_id"]
                metric = _metric_row(states[index], phase=phase)
                metric["activity_from_previous"] = float(
                    (states[index] != previous_states[index]).float().mean()
                )
                denominator = config.observation_every * blocks_per_phase
                metric["reaction_fraction"] = float(interval_reactions[index]) / denominator
                metric["delta_m_flux"] = int(interval_delta_m[index])
                classification = strong_domain_prefix(series[index] + [metric], calibrations[index])
                metric.update(classification)
                series[index].append(metric)
                flags[index].append(bool(classification["strong_domain"]))
                if int(metric["charge"]) != 0:
                    raise RuntimeError(f"exact Q violation in source {task_id} at phase {phase}")

                state_cpu = states[index].detach().cpu()
                rings[index].append({"phase": phase, "state": state_cpu})
                rings[index] = [row for row in rings[index] if phase - int(row["phase"]) <= 32768]

                passage = time_to_event(
                    [int(row["phase"]) for row in series[index]],
                    flags[index],
                    horizon=phase,
                    persistence_followups=config.persistence_followups,
                )
                if branch_points[index]["A"] is None and (
                    float(metric["occupancy_fraction"]) >= config.branch_a_occupancy_fraction
                    and not bool(classification["strong_domain"])
                    and passage["T_persistent"] is None
                    and int(metric["charge"]) == 0
                ):
                    capture = _source_exact_capture(output, task_id, state_cpu, phase=phase, label="branch_A")
                    branch_points[index]["A"] = {
                        **capture,
                        "eligible": True,
                        "histogram": state_histogram(state_cpu).tolist(),
                        "occupancy_fraction": float(metric["occupancy_fraction"]),
                    }

                first_hit_now = passage["T_hit"] == phase and not any(flags[index][:-1])
                if first_hit_now and phase >= config.branch_b_minimum_hit_phase:
                    target = phase - config.branch_b_lead_phases
                    candidate = next((row for row in rings[index] if int(row["phase"]) == target), None)
                    source_metric = next((row for row in series[index] if int(row["phase"]) == target), None)
                    if candidate is None or source_metric is None:
                        raise RuntimeError(f"source {task_id} lacks exact Branch-B ring state at {target}")
                    valid_hash = state_sha256(candidate["state"]) == source_metric["state_sha256"]
                    eligible = bool(
                        not source_metric["strong_domain"]
                        and target % config.observation_every == 0
                        and valid_hash
                    )
                    if eligible:
                        capture = _source_exact_capture(
                            output, task_id, candidate["state"], phase=target, label="branch_B"
                        )
                        branch_points[index]["B"] = {
                            **capture,
                            "eligible": True,
                            "histogram": state_histogram(candidate["state"]).tolist(),
                            "occupancy_fraction": float(source_metric["occupancy_fraction"]),
                            "untouched_T_hit": phase,
                            "untouched_residual_T_hit": config.branch_b_lead_phases,
                        }

                hit_phase = passage["T_hit"]
                if hit_phase is not None:
                    capture_start = int(hit_phase) - 16384
                    capture_end = int(hit_phase) + 16384
                    for row in rings[index]:
                        row_phase = int(row["phase"])
                        if capture_start <= row_phase <= capture_end and row_phase not in saved_event_phases[index]:
                            _source_exact_capture(output, task_id, row["state"], phase=row_phase, label="event")
                            saved_event_phases[index].append(row_phase)
                    if capture_start <= phase <= capture_end and phase not in saved_event_phases[index]:
                        _source_exact_capture(output, task_id, state_cpu, phase=phase, label="event")
                        saved_event_phases[index].append(phase)

                if phase % config.checkpoint_every == 0 or phase == config.source_horizon:
                    summary, _ = multiscale_summary(state_cpu)
                    multiscale[index].append({"phase": phase, **summary})

            previous_states = states.clone()
            interval_reactions.zero_()
            interval_delta_m.zero_()

        if phase % config.checkpoint_every == 0 or phase == config.source_horizon:
            _save_source_checkpoint(
                checkpoint,
                plan_hash=plan["payload_sha256"],
                task_ids=task_ids,
                phase=phase,
                states=states,
                initial_states=initial_states,
                series=series,
                flags=flags,
                multiscale=multiscale,
                branch_points=branch_points,
                rings=rings,
                saved_event_phases=saved_event_phases,
                previous_states=previous_states,
                interval_reactions=interval_reactions,
                interval_delta_m=interval_delta_m,
            )
            _status(
                output,
                {
                    "state": "running_sources",
                    "batch_task_ids": task_ids,
                    "phase": phase,
                    "target_phase": config.source_horizon,
                },
            )
            control = _check_controls(output)
            if control:
                raise ManagedPause(f"{control} requested after safe source checkpoint")
            if managed and resource_policy.enabled:
                pressure = sample_pressure(resource_policy)
                append_resource_log(output / "resource_log.csv", pressure)
                if pressure.reasons:
                    raise ManagedPause("resource pressure: " + ", ".join(pressure.reasons))

    for index, task in enumerate(batch):
        task_id = task["task_id"]
        passage = time_to_event(
            [int(row["phase"]) for row in series[index]],
            flags[index],
            horizon=config.source_horizon,
            persistence_followups=config.persistence_followups,
        )
        final_capture = _source_exact_capture(
            output, task_id, states[index], phase=config.source_horizon, label="final"
        )
        payload = {
            "schema": "zws-reversible-null-phase-v032-source-result-v1",
            "task": task,
            "frozen_plan_payload_sha256": plan["payload_sha256"],
            "winner_sha256": rule.sha256(),
            "calibration": calibrations[index].to_dict(),
            "initial_state": branch_points[index]["initial"],
            "final_state": final_capture,
            "series": series[index],
            "strong_domain_flags": flags[index],
            "time_to_event": passage,
            "branch_points": branch_points[index],
            "multiscale_series": multiscale[index],
            "event_exact_phases": sorted(set(saved_event_phases[index])),
            "completed_utc": _utc(),
        }
        atomic_write_json(_source_result_path(output, task_id), payload)


def run_history_sources(
    config_path: str | Path,
    experiment_path: str | Path,
    output_path: str | Path,
    *,
    managed: bool = False,
) -> dict[str, Any]:
    config, plan, manifest, experiment = _load_protocol(config_path, experiment_path)
    output = Path(output_path).resolve()
    device = resolve_device(config.device)
    rule = ReversibleRule.load(config.rule_path)
    rule.assert_valid()
    if rule.sha256() != config.rule_sha256_uint8:
        raise RuntimeError("loaded winner differs from the frozen raw uint8 hash")
    _initialize_output(output, config, plan, manifest, experiment, device)
    sources = list(plan["source_tasks"])
    pending = [task for task in sources if not _verified_sidecar(_source_result_path(output, task["task_id"]))]
    completed_before = len(sources) - len(pending)
    _status(
        output,
        {
            "state": "starting_sources",
            "source_count": len(sources),
            "completed_sources": completed_before,
            "pending_sources": len(pending),
        },
    )
    resource_policy = ResourceConfig.from_mapping(config.resource)
    try:
        completed_now = 0
        for batch in _chunks(pending, SOURCE_BATCH_SIZE):
            wait_for_quiet(resource_policy, log_path=output / "resource_log.csv", stop_file=output / "STOP.request")
            _run_source_batch(config, plan, output, batch, rule, device, managed=managed)
            completed_now += len(batch)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _status(
                output,
                {
                    "state": "running_sources",
                    "source_count": len(sources),
                    "completed_sources": completed_before + completed_now,
                    "pending_sources": len(sources) - completed_before - completed_now,
                },
            )
    except ManagedPause as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _status(
            output,
            {
                "state": "paused_sources",
                "reason": str(exc),
                "source_count": len(sources),
                "completed_sources": sum(
                    _verified_sidecar(_source_result_path(output, task["task_id"])) for task in sources
                ),
            },
        )
        raise
    _status(
        output,
        {"state": "sources_complete", "source_count": len(sources), "completed_sources": len(sources)},
    )
    return {"source_count": len(sources), "completed": len(sources), "output": str(output)}


def _load_source_result(output: Path, task_id: str) -> dict[str, Any]:
    path = _source_result_path(output, task_id)
    if not _verified_sidecar(path):
        raise RuntimeError(f"source result is missing or invalid: {task_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _untouched_reference(source: dict[str, Any], branch_phase: int, branch_horizon: int) -> dict[str, Any]:
    event = source["time_to_event"]
    hit = event["T_hit"]
    persistent = event["T_persistent"]
    available = max(0, min(branch_horizon, int(source["task"]["horizon_phases"]) - branch_phase))
    return {
        "R_hit": None if hit is None else int(hit) - branch_phase,
        "hit_observed": hit is not None and int(hit) >= branch_phase,
        "R_persistent": None if persistent is None else int(persistent) - branch_phase,
        "persistent_observed": persistent is not None and int(persistent) >= branch_phase,
        "censor_elapsed": available,
        "source_T_hit": hit,
        "source_T_persistent": persistent,
    }


def _alter_state(source_state: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    intervention = spec["intervention"]
    if intervention == "global_shuffle":
        order = global_permutation(int(spec["intervention_seed"]), source_state.size)
        actual_digest = permutation_digest(order)
        if actual_digest != spec["permutation_sha256"]:
            raise RuntimeError(f"global permutation digest mismatch for {spec['task_id']}")
        altered = apply_global_shuffle(source_state, order)
        generator = {"permutation_sha256": actual_digest}
    elif intervention in {"tile_32", "tile_128"}:
        order = np.asarray(spec["tile_permutation"], dtype=np.int64)
        actual_digest = permutation_digest(order)
        if actual_digest != spec["tile_permutation_sha256"]:
            raise RuntimeError(f"tile permutation digest mismatch for {spec['task_id']}")
        altered = apply_tile_permutation(
            source_state,
            tile_size=int(spec["tile_size"]),
            permutation=order,
            temporary_even_offset_yx=spec["temporary_even_offset_yx"],
        )
        generator = {
            "tile_permutation_sha256": actual_digest,
            "temporary_even_offset_yx": spec["temporary_even_offset_yx"],
        }
    elif intervention == "even_translation_null":
        if any(int(v) % 2 for v in spec["translation_offset_yx"]):
            raise RuntimeError("translation null offset is not even")
        altered = translate_state(source_state, spec["translation_offset_yx"])
        generator = {"translation_offset_yx": spec["translation_offset_yx"]}
    else:
        raise ValueError(f"unknown intervention {intervention}")
    return altered, generator


def _load_reusable_realized_plan(
    output: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    path = output / "REALIZED_BRANCH_PLAN.json"
    if not _verified_sidecar(path):
        return None
    realized = json.loads(path.read_text(encoding="utf-8"))
    if payload_sha256(realized) != realized.get("payload_sha256"):
        raise RuntimeError("existing realized branch-plan payload hash mismatch")
    if realized.get("frozen_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("existing realized plan belongs to another frozen plan")
    if realized.get("intervention_manifest_payload_sha256") != manifest["payload_sha256"]:
        raise RuntimeError("existing realized plan belongs to another intervention manifest")
    prepared = list(realized.get("prepared_entries", []))
    ineligible = list(realized.get("ineligible_entries", []))
    realized_ids = [str(task["task_id"]) for task in [*prepared, *ineligible]]
    manifest_ids = [str(task["task_id"]) for task in manifest["entries"]]
    if (
        len(realized_ids) != len(set(realized_ids))
        or set(realized_ids) != set(manifest_ids)
        or int(realized.get("prepared_count", -1)) != len(prepared)
        or int(realized.get("ineligible_count", -1)) != len(ineligible)
    ):
        raise RuntimeError("existing realized plan task partition is inconsistent")
    for task in prepared:
        initial = Path(task["initial_state_path"])
        audit = Path(task["intervention_audit_path"])
        if not _verified_sidecar(initial) or not _verified_sidecar(audit):
            raise RuntimeError(f"existing prepared branch artifacts failed verification: {task['task_id']}")
        if sha256_file(audit) != task["intervention_audit_sha256"]:
            raise RuntimeError(f"existing intervention audit hash changed: {task['task_id']}")
    return realized


def prepare_history_branches(
    config_path: str | Path,
    experiment_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config, plan, manifest, experiment = _load_protocol(config_path, experiment_path)
    output = Path(output_path).resolve()
    existing = _load_reusable_realized_plan(output, plan, manifest)
    if existing is not None:
        return {
            "prepared": int(existing["prepared_count"]),
            "ineligible": int(existing["ineligible_count"]),
            "source_gate": existing["source_gate"],
            "output": str(output),
            "payload_sha256": existing["payload_sha256"],
            "reused": True,
        }
    sources = {task["task_id"]: _load_source_result(output, task["task_id"]) for task in plan["source_tasks"]}
    b_eligible_by_density = {str(density): 0 for density in config.initial_densities}
    source_branch_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for task_id, source in sources.items():
        for point in ("A", "B"):
            branch = source["branch_points"].get(point)
            if branch and branch.get("eligible"):
                source_branch_lookup[(task_id, point)] = branch
                if point == "B":
                    b_eligible_by_density[str(float(source["task"]["density"]))] += 1
    gate_passed = all(value >= 6 for value in b_eligible_by_density.values())
    gate = {
        "schema": "zws-reversible-null-phase-v032-source-gate-v1",
        "minimum_B_eligible_per_density": 6,
        "B_eligible_by_density": b_eligible_by_density,
        "passed": gate_passed,
        "failure_action": None if gate_passed else "stop intervention campaign without changing densities or seeds",
        "evaluated_utc": _utc(),
    }
    atomic_write_json(output / "SOURCE_GATE.json", gate)
    if not gate_passed:
        _status(output, {"state": "source_gate_failed", "source_gate": gate})
        return {"prepared": 0, "source_gate": gate, "output": str(output)}

    prepared: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    source_metric_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for spec in manifest["entries"]:
        key = (str(spec["source_task_id"]), str(spec["branch_point"]))
        branch = source_branch_lookup.get(key)
        if branch is None:
            ineligible.append(
                {
                    "task_id": spec["task_id"],
                    "source_task_id": spec["source_task_id"],
                    "branch_point": spec["branch_point"],
                    "intervention": spec["intervention"],
                    "reason": "source branch point ineligible",
                }
            )
            continue
        source = sources[str(spec["source_task_id"])]
        source_path = Path(branch["path"])
        source_state = _load_state_npz(source_path)
        if state_sha256(source_state) != branch["state_sha256"]:
            raise RuntimeError(f"source branch-state hash mismatch: {source_path}")
        altered, generator = _alter_state(source_state, spec)
        exact = intervention_audit(
            source_state,
            altered,
            source_phase=int(branch["phase"]),
            altered_phase=int(branch["phase"]),
        )
        branch_dir = _branch_dir(output, str(spec["task_id"]))
        branch_dir.mkdir(parents=True, exist_ok=True)
        altered_capture = _save_exact_state(
            branch_dir / "altered_initial.npz",
            altered,
            phase=int(branch["phase"]),
            elapsed=0,
        )

        if key not in source_metric_cache:
            source_summary, source_power = multiscale_summary(source_state)
            source_spectrum = source_path.with_name(source_path.stem + "_structure_factor.npz")
            if not _verified_sidecar(source_spectrum):
                _save_npz(source_spectrum, power=source_power)
            source_metric_cache[key] = (
                source_summary,
                {"path": str(source_spectrum), "sha256": sha256_file(source_spectrum)},
            )
        source_summary, source_spectrum_record = source_metric_cache[key]
        altered_summary, altered_power = multiscale_summary(altered)
        altered_spectrum = branch_dir / "altered_initial_structure_factor.npz"
        _save_npz(altered_spectrum, power=altered_power)
        audit_payload = {
            "schema": "zws-reversible-null-phase-v032-intervention-audit-v1",
            "task_id": spec["task_id"],
            "source_task_id": spec["source_task_id"],
            "branch_point": spec["branch_point"],
            "intervention": spec["intervention"],
            "intervention_seed": spec["intervention_seed"],
            "generator": generator,
            "exact_audit": exact,
            "source_multiscale": source_summary,
            "altered_multiscale": altered_summary,
            "source_structure_factor": source_spectrum_record,
            "altered_structure_factor": {
                "path": str(altered_spectrum),
                "sha256": sha256_file(altered_spectrum),
            },
            "created_utc": _utc(),
        }
        audit_path = branch_dir / "intervention_audit.json"
        atomic_write_json(audit_path, audit_payload)
        prepared.append(
            {
                "task_id": spec["task_id"],
                "source_task_id": spec["source_task_id"],
                "density": spec["density"],
                "source_seed": spec["source_seed"],
                "branch_point": spec["branch_point"],
                "branch_phase": int(branch["phase"]),
                "intervention": spec["intervention"],
                "realization": spec["realization"],
                "intervention_seed": spec["intervention_seed"],
                "source_state_path": str(source_path),
                "source_state_sha256": branch["state_sha256"],
                "initial_state_path": str(branch_dir / "altered_initial.npz"),
                "initial_state_sha256": altered_capture["state_sha256"],
                "intervention_audit_path": str(audit_path),
                "intervention_audit_sha256": sha256_file(audit_path),
                "untouched_reference": _untouched_reference(
                    source, int(branch["phase"]), config.branch_horizon
                ),
            }
        )
    realized = {
        "schema": "zws-reversible-null-phase-v032-realized-branch-plan-v1",
        "frozen_plan_payload_sha256": plan["payload_sha256"],
        "intervention_manifest_payload_sha256": manifest["payload_sha256"],
        "source_gate": gate,
        "prepared_entries": prepared,
        "ineligible_entries": ineligible,
        "prepared_count": len(prepared),
        "ineligible_count": len(ineligible),
        "maximum_manifest_count": len(manifest["entries"]),
        "total_scientific_trajectory_count": len(plan["source_tasks"]) + len(prepared),
        "created_utc": _utc(),
    }
    realized["payload_sha256"] = payload_sha256(realized)
    atomic_write_json(output / "REALIZED_BRANCH_PLAN.json", realized)
    _status(
        output,
        {
            "state": "branches_prepared",
            "prepared_branches": len(prepared),
            "ineligible_branches": len(ineligible),
        },
    )
    return {
        "prepared": len(prepared),
        "ineligible": len(ineligible),
        "source_gate": gate,
        "output": str(output),
    }


def _branch_checkpoint_path(output: Path, batch: list[dict[str, Any]]) -> Path:
    joined = "_".join(task["task_id"] for task in batch)
    digest = __import__("hashlib").sha256(joined.encode("ascii")).hexdigest()[:20]
    return output / "checkpoints" / f"branch_{digest}.pt"


def _resume_replay_audit_path(output: Path, batch: list[dict[str, Any]]) -> Path:
    checkpoint = _branch_checkpoint_path(output, batch)
    return checkpoint.with_name(f"{checkpoint.stem}.resume_replay_audit.json")


def _audit_branch_checkpoint_replay(
    config: HistoryConfig,
    realized: dict[str, Any],
    output: Path,
    batch: list[dict[str, Any]],
    rule: ReversibleRule,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = _branch_checkpoint_path(output, batch)
    if not _checkpoint_available(checkpoint):
        raise RuntimeError(f"no branch checkpoint is available to audit: {checkpoint}")
    saved = _load_verified_torch(checkpoint)
    task_ids = [str(task["task_id"]) for task in batch]
    if saved.get("task_ids") != task_ids:
        raise RuntimeError("checkpoint task IDs do not match the current frozen batch")
    elapsed = int(saved["elapsed"])
    if elapsed < 0 or elapsed > config.branch_horizon:
        raise RuntimeError("checkpoint elapsed phase is outside the frozen branch horizon")
    active = saved["active"].to(dtype=torch.bool, device="cpu")
    if not bool(active.all()):
        raise RuntimeError("automatic checkpoint replay audit requires every batch member to remain active")

    arrays = [_load_state_npz(Path(task["initial_state_path"])) for task in batch]
    for array, task in zip(arrays, batch, strict=True):
        if state_sha256(array) != task["initial_state_sha256"]:
            raise RuntimeError(f"current prepared initial state hash mismatch: {task['task_id']}")
    states = torch.stack([torch.from_numpy(array) for array in arrays]).to(device=device)
    reference_arrays = [
        _load_state_npz(Path(task["source_state_path"]))
        if task["intervention"] == "even_translation_null"
        else np.zeros_like(arrays[index])
        for index, task in enumerate(batch)
    ]
    reference_states = torch.stack([torch.from_numpy(array) for array in reference_arrays]).to(
        device=device
    )
    reference_active = torch.tensor(
        [task["intervention"] == "even_translation_null" for task in batch],
        dtype=torch.bool,
        device=device,
    )
    table = torch.stack([rule.table for _ in batch]).to(device=device)
    started = time.monotonic()
    with torch.inference_mode():
        for phase in range(elapsed):
            states = step_batched(states, table, phase)
            if bool(reference_active.any()):
                candidate = step_batched(reference_states, table, phase)
                reference_states = torch.where(
                    reference_active[:, None, None], candidate, reference_states
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    replayed = states.detach().cpu().numpy()
    checkpoint_states = saved["states"].detach().cpu().numpy()
    mismatches = [
        int(np.count_nonzero(replayed[index] != checkpoint_states[index]))
        for index in range(len(batch))
    ]
    reference_mismatches: list[int] = []
    replayed_references = reference_states.detach().cpu().numpy()
    checkpoint_references = saved["reference_states"].detach().cpu().numpy()
    for index, is_reference in enumerate(reference_active.detach().cpu().tolist()):
        reference_mismatches.append(
            int(np.count_nonzero(replayed_references[index] != checkpoint_references[index]))
            if is_reference
            else 0
        )
    payload = {
        "schema": "zws-reversible-null-phase-v032-resume-replay-audit-v1",
        "reason": "realized plan metadata was regenerated before idempotent preparation was enforced",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_elapsed": elapsed,
        "task_ids": task_ids,
        "prior_realized_plan_payload_sha256": saved.get("realized_plan_payload_sha256"),
        "current_realized_plan_payload_sha256": realized["payload_sha256"],
        "replayed_state_sha256": [state_sha256(state) for state in replayed],
        "checkpoint_state_sha256": [state_sha256(state) for state in checkpoint_states],
        "mismatch_counts": mismatches,
        "reference_mismatch_counts": reference_mismatches,
        "all_checkpoint_members_active": bool(active.all()),
        "wall_seconds": time.monotonic() - started,
        "passed": not any(mismatches) and not any(reference_mismatches),
        "created_utc": _utc(),
    }
    audit_path = _resume_replay_audit_path(output, batch)
    atomic_write_json(audit_path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"checkpoint replay audit failed; evidence saved to {audit_path}")
    return {**payload, "audit_path": str(audit_path.resolve()), "audit_sha256": sha256_file(audit_path)}


def audit_pending_branch_resume(
    config_path: str | Path,
    experiment_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config, plan, manifest, _ = _load_protocol(config_path, experiment_path)
    output = Path(output_path).resolve()
    realized = _load_reusable_realized_plan(output, plan, manifest)
    if realized is None:
        raise RuntimeError("a verified realized branch plan is required for resume audit")
    pending = [
        task
        for task in realized["prepared_entries"]
        if not _verified_sidecar(_branch_result_path(output, task["task_id"]))
    ]
    if not pending:
        return {"required": False, "reason": "no pending branch batch", "output": str(output)}
    batch = pending[:BRANCH_BATCH_SIZE]
    checkpoint = _branch_checkpoint_path(output, batch)
    if not _checkpoint_available(checkpoint):
        return {
            "required": False,
            "reason": "the next pending batch has no checkpoint",
            "task_ids": [task["task_id"] for task in batch],
            "output": str(output),
        }
    saved = _load_verified_torch(checkpoint)
    if saved.get("realized_plan_payload_sha256") == realized["payload_sha256"]:
        return {
            "required": False,
            "reason": "checkpoint already matches the current realized plan",
            "task_ids": [task["task_id"] for task in batch],
            "output": str(output),
        }
    rule = ReversibleRule.load(config.rule_path)
    rule.assert_valid()
    if rule.sha256() != plan["winner_raw_uint8_sha256"]:
        raise RuntimeError("winner changed before checkpoint replay audit")
    result = _audit_branch_checkpoint_replay(
        config, realized, output, batch, rule, resolve_device(config.device)
    )
    return {"required": True, **result}


def _require_resume_replay_audit(
    output: Path,
    batch: list[dict[str, Any]],
    saved: dict[str, Any],
    current_plan_hash: str,
) -> None:
    audit_path = _resume_replay_audit_path(output, batch)
    if not _verified_sidecar(audit_path):
        raise RuntimeError(
            "branch checkpoint plan lineage changed; run the exact checkpoint replay audit before resuming"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checkpoint = _branch_checkpoint_path(output, batch)
    task_ids = [str(task["task_id"]) for task in batch]
    passed = (
        audit.get("passed") is True
        and audit.get("checkpoint_path") == str(checkpoint.resolve())
        and audit.get("checkpoint_sha256") == sha256_file(checkpoint)
        and int(audit.get("checkpoint_elapsed", -1)) == int(saved["elapsed"])
        and audit.get("task_ids") == task_ids
        and audit.get("prior_realized_plan_payload_sha256")
        == saved.get("realized_plan_payload_sha256")
        and audit.get("current_realized_plan_payload_sha256") == current_plan_hash
        and not any(int(value) for value in audit.get("mismatch_counts", [1]))
        and not any(int(value) for value in audit.get("reference_mismatch_counts", [1]))
    )
    if not passed:
        raise RuntimeError("branch checkpoint replay audit does not match the pending batch")


def _save_branch_checkpoint(
    path: Path,
    *,
    realized_plan_hash: str,
    task_ids: list[str],
    elapsed: int,
    states: torch.Tensor,
    reference_states: torch.Tensor,
    reference_active: torch.Tensor,
    active: torch.Tensor,
    series: list[list[dict[str, Any]]],
    flags: list[list[bool]],
    multiscale: list[list[dict[str, Any]]],
    rings: list[list[dict[str, Any]]],
    saved_state_elapsed: list[list[int]],
    previous_states: torch.Tensor,
    interval_reactions: torch.Tensor,
    interval_delta_m: torch.Tensor,
    null_audits: list[list[dict[str, Any]]],
    completion_reasons: list[str | None],
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema": "zws-reversible-null-phase-v032-branch-checkpoint-v1",
            "realized_plan_payload_sha256": realized_plan_hash,
            "task_ids": task_ids,
            "elapsed": int(elapsed),
            "states": states.detach().cpu(),
            "reference_states": reference_states.detach().cpu(),
            "reference_active": reference_active.detach().cpu(),
            "active": active.detach().cpu(),
            "series": series,
            "flags": flags,
            "multiscale": multiscale,
            "rings": rings,
            "saved_state_elapsed": saved_state_elapsed,
            "previous_states": previous_states.detach().cpu(),
            "interval_reactions": interval_reactions.detach().cpu(),
            "interval_delta_m": interval_delta_m.detach().cpu(),
            "null_audits": null_audits,
            "completion_reasons": completion_reasons,
            "saved_utc": _utc(),
        },
    )


def _branch_capture(
    output: Path,
    task_id: str,
    state: torch.Tensor | np.ndarray,
    *,
    branch_phase: int,
    elapsed: int,
    label: str,
) -> dict[str, Any]:
    absolute = branch_phase + elapsed
    path = _branch_dir(output, task_id) / "states" / f"{label}_elapsed_{elapsed:06d}_phase_{absolute:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return _save_exact_state(path, state, phase=absolute, elapsed=elapsed)


def _load_audit(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not _verified_sidecar(target):
        raise RuntimeError(f"intervention audit verification failed: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _run_branch_batch(
    config: HistoryConfig,
    realized: dict[str, Any],
    output: Path,
    batch: list[dict[str, Any]],
    rule: ReversibleRule,
    device: torch.device,
    *,
    managed: bool,
) -> None:
    checkpoint = _branch_checkpoint_path(output, batch)
    task_ids = [task["task_id"] for task in batch]
    source_results = [_load_source_result(output, task["source_task_id"]) for task in batch]
    source_prefixes = [
        [row for row in source["series"] if int(row["phase"]) < int(task["branch_phase"])]
        for source, task in zip(source_results, batch, strict=True)
    ]
    calibrations = [SourceCalibration(**source["calibration"]) for source in source_results]
    offsets = [
        _load_audit(task["intervention_audit_path"])["generator"].get("translation_offset_yx")
        for task in batch
    ]
    if any(int(task["branch_phase"]) % 2 for task in batch):
        raise RuntimeError("a frozen branch point is not even-phase compatible")

    if _checkpoint_available(checkpoint):
        saved = _load_verified_torch(checkpoint)
        if saved.get("task_ids") != task_ids:
            raise RuntimeError("branch checkpoint does not match realized plan/batch")
        if saved.get("realized_plan_payload_sha256") != realized["payload_sha256"]:
            _require_resume_replay_audit(
                output, batch, saved, str(realized["payload_sha256"])
            )
        elapsed = int(saved["elapsed"])
        states = saved["states"].to(device=device)
        reference_states = saved["reference_states"].to(device=device)
        reference_active = saved["reference_active"].to(device=device)
        active = saved["active"].to(device=device)
        series = saved["series"]
        flags = saved["flags"]
        multiscale = saved["multiscale"]
        rings = saved["rings"]
        saved_state_elapsed = saved["saved_state_elapsed"]
        previous_states = saved["previous_states"].to(device=device)
        interval_reactions = saved["interval_reactions"].to(device=device)
        interval_delta_m = saved["interval_delta_m"].to(device=device)
        null_audits = saved["null_audits"]
        completion_reasons = saved["completion_reasons"]
    else:
        elapsed = 0
        arrays = [_load_state_npz(Path(task["initial_state_path"])) for task in batch]
        for array, task in zip(arrays, batch, strict=True):
            if state_sha256(array) != task["initial_state_sha256"]:
                raise RuntimeError(f"prepared branch initial hash mismatch: {task['task_id']}")
        states = torch.stack([torch.from_numpy(array) for array in arrays]).to(device=device)
        reference_arrays = [
            _load_state_npz(Path(task["source_state_path"]))
            if task["intervention"] == "even_translation_null"
            else np.zeros_like(arrays[index])
            for index, task in enumerate(batch)
        ]
        reference_states = torch.stack([torch.from_numpy(array) for array in reference_arrays]).to(device=device)
        reference_active = torch.tensor(
            [task["intervention"] == "even_translation_null" for task in batch],
            dtype=torch.bool,
            device=device,
        )
        active = torch.ones(len(batch), dtype=torch.bool, device=device)
        series = []
        flags = []
        multiscale = []
        rings = []
        saved_state_elapsed = []
        null_audits = []
        completion_reasons = [None for _ in batch]
        for index, task in enumerate(batch):
            absolute = int(task["branch_phase"])
            metric = _metric_row(states[index], phase=absolute)
            classification = strong_domain_prefix(source_prefixes[index] + [metric], calibrations[index])
            metric.update(classification)
            metric["elapsed"] = 0
            metric["absolute_phase"] = absolute
            series.append([metric])
            flags.append([bool(classification["strong_domain"])])
            audit = _load_audit(task["intervention_audit_path"])
            multiscale.append([{"elapsed": 0, "absolute_phase": absolute, **audit["altered_multiscale"]}])
            rings.append([{"elapsed": 0, "state": states[index].detach().cpu()}])
            _branch_capture(output, task["task_id"], states[index], branch_phase=absolute, elapsed=0, label="fixed")
            saved_state_elapsed.append([0])
            audit_rows: list[dict[str, Any]] = []
            if bool(reference_active[index]):
                offset = tuple(int(value) for value in offsets[index])
                audit_rows.append(
                    _translation_null_check(
                        states[index],
                        reference_states[index],
                        offset_yx=offset,
                        elapsed=0,
                        absolute_phase=absolute,
                        metric=metric,
                    )
                )
            null_audits.append(audit_rows)
        previous_states = states.clone()
        interval_reactions = torch.zeros(len(batch), dtype=torch.int64, device=device)
        interval_delta_m = torch.zeros(len(batch), dtype=torch.int64, device=device)

    table = torch.stack([rule.table for _ in batch]).to(device=device)
    aux = _rule_tensors(rule, device)
    reaction_mask = torch.stack([aux["reaction"] for _ in batch])
    delta_m_lookup = torch.stack([aux["delta_m"] for _ in batch])
    blocks_per_phase = (config.lattice_size // 2) ** 2
    resource_policy = ResourceConfig.from_mapping(config.resource)

    while elapsed < config.branch_horizon and bool(active.any()):
        evolved = step_batched(states, table, elapsed, return_codes=True)
        assert isinstance(evolved, tuple)
        candidate_states, input_codes, _ = evolved
        active_mask = active[:, None, None]
        states = torch.where(active_mask, candidate_states, states)
        flat_codes = input_codes.reshape(len(batch), -1)
        interval_reactions += reaction_mask.gather(1, flat_codes).sum(dim=1) * active.to(torch.int64)
        interval_delta_m += delta_m_lookup.gather(1, flat_codes).sum(dim=1) * active.to(torch.int64)
        reference_candidate = step_batched(reference_states, table, elapsed)
        assert isinstance(reference_candidate, torch.Tensor)
        reference_mask = (active & reference_active)[:, None, None]
        reference_states = torch.where(reference_mask, reference_candidate, reference_states)
        elapsed += 1

        if elapsed % config.observation_every == 0 or elapsed == config.branch_horizon:
            for index, task in enumerate(batch):
                if not bool(active[index]):
                    continue
                absolute = int(task["branch_phase"]) + elapsed
                metric = _metric_row(states[index], phase=absolute)
                metric["activity_from_previous"] = float(
                    (states[index] != previous_states[index]).float().mean()
                )
                denominator = config.observation_every * blocks_per_phase
                metric["reaction_fraction"] = float(interval_reactions[index]) / denominator
                metric["delta_m_flux"] = int(interval_delta_m[index])
                classification = strong_domain_prefix(
                    source_prefixes[index] + series[index] + [metric], calibrations[index]
                )
                metric.update(classification)
                metric["elapsed"] = elapsed
                metric["absolute_phase"] = absolute
                series[index].append(metric)
                flags[index].append(bool(classification["strong_domain"]))
                if int(metric["charge"]) != 0:
                    raise RuntimeError(f"exact Q violation in branch {task['task_id']} at elapsed {elapsed}")

                if bool(reference_active[index]):
                    offset = tuple(int(v) for v in offsets[index])
                    null_audits[index].append(
                        _translation_null_check(
                            states[index],
                            reference_states[index],
                            offset_yx=offset,
                            elapsed=elapsed,
                            absolute_phase=absolute,
                            metric=metric,
                        )
                    )

                state_cpu = states[index].detach().cpu()
                rings[index].append({"elapsed": elapsed, "state": state_cpu})
                rings[index] = [
                    row for row in rings[index] if elapsed - int(row["elapsed"]) <= 16384
                ]
                passage = time_to_event(
                    [int(row["elapsed"]) for row in series[index]],
                    flags[index],
                    horizon=config.branch_horizon,
                    persistence_followups=config.persistence_followups,
                )
                hit = passage["T_hit"]
                if hit is not None:
                    for row in rings[index]:
                        row_elapsed = int(row["elapsed"])
                        if (
                            int(hit) - 16384 <= row_elapsed <= int(hit) + 16384
                            and row_elapsed not in saved_state_elapsed[index]
                        ):
                            _branch_capture(
                                output,
                                task["task_id"],
                                row["state"],
                                branch_phase=int(task["branch_phase"]),
                                elapsed=row_elapsed,
                                label="event",
                            )
                            saved_state_elapsed[index].append(row_elapsed)

                capture_fixed = elapsed in config.fixed_elapsed_phases
                capture_hit = hit is not None and abs(elapsed - int(hit)) <= 16384
                if (capture_fixed or capture_hit) and elapsed not in saved_state_elapsed[index]:
                    _branch_capture(
                        output,
                        task["task_id"],
                        state_cpu,
                        branch_phase=int(task["branch_phase"]),
                        elapsed=elapsed,
                        label="fixed" if capture_fixed else "event",
                    )
                    saved_state_elapsed[index].append(elapsed)
                if capture_fixed or (hit is not None and elapsed == int(hit)):
                    summary, _ = multiscale_summary(state_cpu)
                    multiscale[index].append(
                        {"elapsed": elapsed, "absolute_phase": absolute, **summary}
                    )
                persistent = passage["T_persistent"]
                if persistent is not None and elapsed >= int(persistent) + config.post_persistent_capture:
                    active[index] = False
                    completion_reasons[index] = "post_persistent_capture_complete"

            previous_states = states.clone()
            interval_reactions.zero_()
            interval_delta_m.zero_()

        if elapsed % config.checkpoint_every == 0 or elapsed == config.branch_horizon or not bool(active.any()):
            _save_branch_checkpoint(
                checkpoint,
                realized_plan_hash=realized["payload_sha256"],
                task_ids=task_ids,
                elapsed=elapsed,
                states=states,
                reference_states=reference_states,
                reference_active=reference_active,
                active=active,
                series=series,
                flags=flags,
                multiscale=multiscale,
                rings=rings,
                saved_state_elapsed=saved_state_elapsed,
                previous_states=previous_states,
                interval_reactions=interval_reactions,
                interval_delta_m=interval_delta_m,
                null_audits=null_audits,
                completion_reasons=completion_reasons,
            )
            _status(
                output,
                {
                    "state": "running_branches",
                    "batch_task_ids": task_ids,
                    "elapsed": elapsed,
                    "target_elapsed": config.branch_horizon,
                    "active_in_batch": int(active.sum()),
                },
            )
            control = _check_controls(output)
            if control:
                raise ManagedPause(f"{control} requested after safe branch checkpoint")
            if managed and resource_policy.enabled:
                pressure = sample_pressure(resource_policy)
                append_resource_log(output / "resource_log.csv", pressure)
                if pressure.reasons:
                    raise ManagedPause("resource pressure: " + ", ".join(pressure.reasons))

    for index, task in enumerate(batch):
        if completion_reasons[index] is None:
            completion_reasons[index] = "right_censored_at_horizon"
        final_elapsed = int(series[index][-1]["elapsed"])
        if final_elapsed not in saved_state_elapsed[index]:
            _branch_capture(
                output,
                task["task_id"],
                states[index],
                branch_phase=int(task["branch_phase"]),
                elapsed=final_elapsed,
                label="final",
            )
            saved_state_elapsed[index].append(final_elapsed)
        if not multiscale[index] or int(multiscale[index][-1]["elapsed"]) != final_elapsed:
            summary, _ = multiscale_summary(states[index])
            multiscale[index].append(
                {
                    "elapsed": final_elapsed,
                    "absolute_phase": int(task["branch_phase"]) + final_elapsed,
                    **summary,
                }
            )
        passage = time_to_event(
            [int(row["elapsed"]) for row in series[index]],
            flags[index],
            horizon=config.branch_horizon,
            persistence_followups=config.persistence_followups,
        )
        residual = {
            "R_hit": passage["T_hit"],
            "hit_observed": passage["hit_observed"],
            "R_persistent": passage["T_persistent"],
            "persistent_observed": passage["persistent_observed"],
            "censor_elapsed": config.branch_horizon,
            "right_censored_R_hit": passage["right_censored_T_hit"],
            "right_censored_R_persistent": passage["right_censored_T_persistent"],
        }
        payload = {
            "schema": "zws-reversible-null-phase-v032-branch-result-v1",
            "task": task,
            "realized_plan_payload_sha256": realized["payload_sha256"],
            "winner_sha256": rule.sha256(),
            "calibration": calibrations[index].to_dict(),
            "calibration_inherited_from_source": True,
            "denominator_reset_at_branch": False,
            "untouched_reference": task["untouched_reference"],
            "residual_time_to_event": residual,
            "series": series[index],
            "strong_domain_flags": flags[index],
            "multiscale_series": multiscale[index],
            "saved_exact_elapsed": sorted(set(saved_state_elapsed[index])),
            "completion_reason": completion_reasons[index],
            "translation_null_audits": null_audits[index],
            "translation_null_passed": (
                all(
                    row["mismatch_count"] == 0 and row.get("metric_match") is True
                    for row in null_audits[index]
                )
                if task["intervention"] == "even_translation_null"
                else None
            ),
            "final_state_sha256": state_sha256(states[index]),
            "completed_utc": _utc(),
        }
        atomic_write_json(_branch_result_path(output, task["task_id"]), payload)


def run_history_branches(
    config_path: str | Path,
    experiment_path: str | Path,
    output_path: str | Path,
    *,
    managed: bool = False,
) -> dict[str, Any]:
    config, plan, manifest, _ = _load_protocol(config_path, experiment_path)
    output = Path(output_path).resolve()
    realized_path = output / "REALIZED_BRANCH_PLAN.json"
    if not _verified_sidecar(realized_path):
        raise RuntimeError("realized branch plan is missing; run preparation first")
    realized = json.loads(realized_path.read_text(encoding="utf-8"))
    if payload_sha256(realized) != realized["payload_sha256"]:
        raise RuntimeError("realized branch-plan payload hash mismatch")
    if not realized["source_gate"]["passed"]:
        return {"branch_count": 0, "completed": 0, "source_gate": realized["source_gate"], "output": str(output)}
    device = resolve_device(config.device)
    rule = ReversibleRule.load(config.rule_path)
    rule.assert_valid()
    if rule.sha256() != plan["winner_raw_uint8_sha256"]:
        raise RuntimeError("winner changed after source run")
    tasks = list(realized["prepared_entries"])
    pending = [task for task in tasks if not _verified_sidecar(_branch_result_path(output, task["task_id"]))]
    completed_before = len(tasks) - len(pending)
    _status(
        output,
        {
            "state": "starting_branches",
            "branch_count": len(tasks),
            "completed_branches": completed_before,
            "pending_branches": len(pending),
        },
    )
    resource_policy = ResourceConfig.from_mapping(config.resource)
    try:
        completed_now = 0
        for batch in _chunks(pending, BRANCH_BATCH_SIZE):
            wait_for_quiet(resource_policy, log_path=output / "resource_log.csv", stop_file=output / "STOP.request")
            _run_branch_batch(config, realized, output, batch, rule, device, managed=managed)
            completed_now += len(batch)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _status(
                output,
                {
                    "state": "running_branches",
                    "branch_count": len(tasks),
                    "completed_branches": completed_before + completed_now,
                    "pending_branches": len(tasks) - completed_before - completed_now,
                },
            )
    except ManagedPause as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _status(
            output,
            {
                "state": "paused_branches",
                "reason": str(exc),
                "branch_count": len(tasks),
                "completed_branches": sum(
                    _verified_sidecar(_branch_result_path(output, task["task_id"])) for task in tasks
                ),
            },
        )
        raise
    _status(
        output,
        {"state": "branches_complete", "branch_count": len(tasks), "completed_branches": len(tasks)},
    )
    return {"branch_count": len(tasks), "completed": len(tasks), "output": str(output)}


def run_history_full(
    config_path: str | Path,
    experiment_path: str | Path,
    output_path: str | Path,
    *,
    managed: bool = False,
) -> dict[str, Any]:
    source_result = run_history_sources(
        config_path, experiment_path, output_path, managed=managed
    )
    preparation = prepare_history_branches(config_path, experiment_path, output_path)
    if not preparation["source_gate"]["passed"]:
        return {"sources": source_result, "preparation": preparation, "branches": None}
    branches = run_history_branches(
        config_path, experiment_path, output_path, managed=managed
    )
    return {"sources": source_result, "preparation": preparation, "branches": branches}


def run_history_smoke(
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config = HistoryConfig.load(config_path)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    rule = ReversibleRule.load(config.rule_path)
    rule.assert_valid()
    if rule.sha256() != config.rule_sha256_uint8:
        raise RuntimeError("smoke loaded a non-frozen winner")
    size = 512
    inventory = inventory_for_density(size, size, 0.20)
    source = make_state(
        size,
        size,
        inventory,
        seed=93201,
        arrangement="homogeneous",
        device=device,
    )
    initial_hash = state_sha256(source)
    for phase in range(16):
        source = step_batched(source.unsqueeze(0), rule.table.unsqueeze(0).to(device), phase)[0]
    branch_source = as_uint8_state(source)
    global_order = global_permutation(93202, branch_source.size)
    tile32_order = global_permutation(93203, (size // 32) ** 2)
    tile128_order = global_permutation(93204, (size // 128) ** 2)
    altered = {
        "global_shuffle": apply_global_shuffle(branch_source, global_order),
        "tile_32": apply_tile_permutation(
            branch_source,
            tile_size=32,
            permutation=tile32_order,
            temporary_even_offset_yx=(10, 14),
        ),
        "tile_128": apply_tile_permutation(
            branch_source,
            tile_size=128,
            permutation=tile128_order,
            temporary_even_offset_yx=(18, 22),
        ),
        "even_translation_null": translate_state(branch_source, (26, 30)),
    }
    audits: dict[str, Any] = {}
    multiscale: dict[str, Any] = {}
    for name, state in altered.items():
        audits[name] = intervention_audit(branch_source, state, source_phase=16, altered_phase=16)
        summary, _ = multiscale_summary(state)
        multiscale[name] = summary
        _save_exact_state(output / f"{name}_initial.npz", state, phase=16, elapsed=0)

    stacked = torch.stack([torch.from_numpy(state) for state in altered.values()]).to(device=device)
    tables = torch.stack([rule.table for _ in altered]).to(device=device)
    reference = torch.from_numpy(branch_source).to(device=device)
    for elapsed in range(32):
        stacked = step_batched(stacked, tables, elapsed)
        assert isinstance(stacked, torch.Tensor)
        reference = step_batched(reference.unsqueeze(0), rule.table.unsqueeze(0).to(device), elapsed)[0]
    translated_final = stacked[list(altered).index("even_translation_null")]
    recovered = translate_state(translated_final, (-26, -30))
    reference_array = as_uint8_state(reference)
    translation_mismatch = int(np.count_nonzero(recovered != reference_array))
    if translation_mismatch:
        raise RuntimeError("smoke translation null failed")

    resume_initial = make_state(
        size,
        size,
        inventory,
        seed=93205,
        arrangement="homogeneous",
        device=device,
    )
    uninterrupted = resume_initial.clone()
    for phase in range(32):
        uninterrupted = step_batched(
            uninterrupted.unsqueeze(0), rule.table.unsqueeze(0).to(device), phase
        )[0]
    interrupted = resume_initial.clone()
    for phase in range(16):
        interrupted = step_batched(
            interrupted.unsqueeze(0), rule.table.unsqueeze(0).to(device), phase
        )[0]
    checkpoint = output / "smoke_resume_checkpoint.pt"
    _atomic_torch_save(
        checkpoint,
        {"schema": "zws-v032-smoke-resume-v1", "phase": 16, "state": interrupted.cpu()},
    )
    del interrupted
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loaded = _load_verified_torch(checkpoint)
    resumed = loaded["state"].to(device=device)
    for phase in range(int(loaded["phase"]), 32):
        resumed = step_batched(resumed.unsqueeze(0), rule.table.unsqueeze(0).to(device), phase)[0]
    resume_exact = state_sha256(resumed) == state_sha256(uninterrupted)
    if not resume_exact:
        raise RuntimeError("smoke interrupted/resumed state differs from uninterrupted state")

    payload = {
        "schema": "zws-reversible-null-phase-v032-smoke-v1",
        "winner_sha256": rule.sha256(),
        "device": str(device),
        "size": size,
        "source_seed": 93201,
        "initial_state_sha256": initial_hash,
        "branch_source_sha256": state_sha256(branch_source),
        "audits": audits,
        "multiscale": multiscale,
        "translation_comparison_count": 1,
        "translation_mismatch_count": translation_mismatch,
        "translation_null_passed": translation_mismatch == 0,
        "checkpoint_verified": _verified_sidecar(checkpoint),
        "worker_unloaded_before_resume": True,
        "resume_exact": resume_exact,
        "uninterrupted_final_sha256": state_sha256(uninterrupted),
        "resumed_final_sha256": state_sha256(resumed),
        "completed_utc": _utc(),
    }
    atomic_write_json(output / "SMOKE_REPORT.json", payload)
    return payload
