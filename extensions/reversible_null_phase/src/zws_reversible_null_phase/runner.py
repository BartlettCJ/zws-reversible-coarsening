from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import torch

from .alphabet import DEFAULT_ALPHABET
from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .config import StudyConfig
from .initial import inventory_for_density, make_state, state_sha256
from .metrics import StrongDomainCriterion, snapshot_metrics
from .nucleation import finalize_time_to_event, new_tracker, save_event_window, update_tracker
from .reaction_network import summarize_flux
from .resources import append_resource_log, sample_pressure, wait_for_quiet
from .rule import ReversibleRule, rotate_table
from .simulator import step_batched
from .spectra import fit_log_growth_rate, shell_power
from .tasks import Task, generate_tasks
from .visualize import save_state_npz, save_state_png


class ManagedPause(RuntimeError):
    pass


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _checkpoint_generation(path: Path, generation: int) -> Path:
    if generation == 0:
        return path
    return path.with_name(f"{path.stem}.prev{generation}{path.suffix}")


def _replace_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 20,
    initial_delay: float = 0.05,
) -> None:
    """Replace a file despite short-lived Windows sharing/antivirus locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            delay = min(initial_delay * (2 ** min(attempt, 4)), 0.5)
            time.sleep(delay)


def _atomic_torch_save(path: Path, payload: Any, *, generations: int = 3) -> None:
    """Write a checksummed checkpoint while retaining verified predecessors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rotate oldest to newest. A crash after rotation still leaves a verified predecessor.
    for generation in reversed(range(1, generations)):
        source = _checkpoint_generation(path, generation - 1)
        source_sidecar = source.with_suffix(source.suffix + ".sha256")
        target = _checkpoint_generation(path, generation)
        target_sidecar = target.with_suffix(target.suffix + ".sha256")
        if source.exists() and source_sidecar.exists():
            _replace_with_retry(source, target)
            _replace_with_retry(source_sidecar, target_sidecar)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        torch.save(payload, temp)
        # Windows' CRT rejects fsync on a read-only descriptor (EBADF).  Open
        # the completed checkpoint read/write so the durability barrier works
        # on both Windows and POSIX before the atomic replace.
        with temp.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temp, path)
        atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n")
    finally:
        temp.unlink(missing_ok=True)


def _load_verified_torch(path: Path, *, generations: int = 3) -> Any:
    errors: list[str] = []
    for generation in range(generations):
        candidate = _checkpoint_generation(path, generation)
        sidecar = candidate.with_suffix(candidate.suffix + ".sha256")
        if not candidate.exists() or not sidecar.exists():
            continue
        try:
            expected = sidecar.read_text(encoding="utf-8").split()[0]
            actual = sha256_file(candidate)
            if expected != actual:
                raise RuntimeError("checksum mismatch")
            return torch.load(candidate, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, IndexError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    if errors:
        raise RuntimeError("no valid checkpoint generation: " + "; ".join(errors))
    raise FileNotFoundError(path)


def _task_result_path(output: Path, task: Task) -> Path:
    return output / "tasks" / task.stage / f"{task.id}.json"


def _task_result_valid(path: Path) -> bool:
    if not path.exists():
        return False
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return False
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        return expected == sha256_file(path)
    except (OSError, IndexError):
        return False


def _variant_rules(winner: ReversibleRule) -> dict[str, ReversibleRule]:
    rotation = ReversibleRule(torch.tensor(rotate_table(1)), name="pure_rotation")
    result = {
        "winner": winner,
        "rotation": rotation,
        "qm_sibling": winner.closest_qm_preserving_sibling(),
    }
    pairs = winner.reactive_pairs_relative_to_rotation()
    for index in range(len(pairs)):
        result[f"ablate_{index:02d}"] = winner.ablate_pairs([index], name=f"winner_ablate_{index:02d}")
    return result


def _rule_tensors(rule: ReversibleRule, device: torch.device) -> dict[str, torch.Tensor]:
    digits = DEFAULT_ALPHABET.block_digits_numpy()
    out_digits = digits[rule.table.numpy()]
    hist_in = np.stack([(digits == state).sum(axis=1) for state in range(4)], axis=1)
    hist_out = np.stack([(out_digits == state).sum(axis=1) for state in range(4)], axis=1)
    reaction = np.any(hist_in != hist_out, axis=1)
    deltas = rule.transition_deltas()
    pair_index = np.full(256, -1, dtype=np.int64)
    try:
        for idx, (a, b) in enumerate(rule.reactive_pairs_relative_to_rotation()):
            pair_index[a] = idx
            pair_index[b] = idx
    except ValueError:
        pass
    return {
        "reaction": torch.tensor(reaction, dtype=torch.bool, device=device),
        "delta_m": torch.tensor(deltas["delta_m"], dtype=torch.int64, device=device),
        "pair_index": torch.tensor(pair_index, dtype=torch.int64, device=device),
    }


def write_plan(config: StudyConfig, output: Path) -> list[Task]:
    output.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(config)
    payload = {
        "schema": "zws-reversible-null-phase-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.sha256(),
        "config_hash_scheme": config.HASH_SCHEME,
        "config": config.canonical_payload(),
        "task_count": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }
    atomic_write_json(output / "PLAN.json", payload)
    return tasks


def _batch_key(task: Task) -> tuple[Any, ...]:
    return (
        task.stage,
        task.size,
        task.steps,
        task.sample_every,
        task.topology_every,
        task.checkpoint_every,
    )


def _chunks(items: list[Task], size: int) -> Iterable[list[Task]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _checkpoint_path(output: Path, batch: list[Task]) -> Path:
    digest = hashlib.sha256("|".join(task.id for task in batch).encode()).hexdigest()[:20]
    return output / "checkpoints" / f"batch_{digest}.pt"


def _save_batch_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    tasks: list[Task],
    phase: int,
    states: torch.Tensor,
    initial_states: torch.Tensor,
    topological_series: list[list[dict[str, Any]]],
    temporal_series: list[list[dict[str, Any]]],
    cumulative_input_hist: torch.Tensor,
    interval_reaction_counts: torch.Tensor,
    interval_delta_m: torch.Tensor,
    previous_sample_states: torch.Tensor,
    peak_scales: list[float],
    peak_states: list[torch.Tensor],
    nucleation_tracking: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": "zws-reversible-null-phase-checkpoint-v1",
        "config_sha256": config_sha256,
        "task_ids": [task.id for task in tasks],
        "phase": phase,
        "states": states.detach().cpu(),
        "initial_states": initial_states.detach().cpu(),
        "topological_series": topological_series,
        "temporal_series": temporal_series,
        "cumulative_input_hist": cumulative_input_hist.detach().cpu(),
        "interval_reaction_counts": interval_reaction_counts.detach().cpu(),
        "interval_delta_m": interval_delta_m.detach().cpu(),
        "previous_sample_states": previous_sample_states.detach().cpu(),
        "peak_scales": peak_scales,
        "peak_states": [state.detach().cpu() for state in peak_states],
        "nucleation_tracking": nucleation_tracking,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_torch_save(path, payload)


def _status(output: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(output / "status.json", payload)


def _write_file_sidecar(path: Path) -> None:
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"), f"{sha256_file(path)}  {path.name}\n"
    )


def _check_controls(output: Path) -> str | None:
    for name, kind in (
        ("STOP.request", "stop"),
        ("HIBERNATE.request", "pause"),
        ("PAUSE.request", "pause"),
        ("RESOURCE_PAUSE.request", "pause"),
    ):
        if (output / name).exists():
            return kind
    return None


def _run_batch(
    config: StudyConfig,
    output: Path,
    batch: list[Task],
    variants: dict[str, ReversibleRule],
    device: torch.device,
    criterion: StrongDomainCriterion,
    *,
    managed: bool,
) -> None:
    checkpoint = _checkpoint_path(output, batch)
    task_ids = [task.id for task in batch]
    rule_list = [variants[task.rule_variant] for task in batch]
    for rule in rule_list:
        rule.assert_valid()
    inventory_list = [inventory_for_density(task.size, task.size, task.density) for task in batch]

    if checkpoint.exists():
        saved = _load_verified_torch(checkpoint)
        # Checkpoints written by package <= 0.3.1 carry scheme-1 hashes; accept either.
        if saved.get("config_sha256") not in {config.sha256(), config.legacy_sha256()} or saved.get("task_ids") != task_ids:
            raise RuntimeError("checkpoint does not match current task batch/configuration")
        phase = int(saved["phase"])
        states = saved["states"].to(device=device)
        initial_states = saved["initial_states"].to(device=device)
        topological_series = saved["topological_series"]
        temporal_series = saved["temporal_series"]
        cumulative_input_hist = saved["cumulative_input_hist"].to(device=device)
        interval_reaction_counts = saved["interval_reaction_counts"].to(device=device)
        interval_delta_m = saved["interval_delta_m"].to(device=device)
        previous_sample_states = saved["previous_sample_states"].to(device=device)
        peak_scales = [float(x) for x in saved["peak_scales"]]
        peak_states = [state.to(device=device) for state in saved["peak_states"]]
        nucleation_tracking = saved.get("nucleation_tracking", [])
    else:
        phase = 0
        initial = [
            make_state(
                task.size,
                task.size,
                inventory,
                seed=task.seed,
                arrangement=task.arrangement,  # type: ignore[arg-type]
                concentration=0.65,
                device=device,
            )
            for task, inventory in zip(batch, inventory_list, strict=True)
        ]
        states = torch.stack(initial, dim=0)
        initial_states = states.clone()
        topological_series = []
        temporal_series = []
        peak_scales = []
        peak_states = []
        for index, task in enumerate(batch):
            initial_metric = snapshot_metrics(
                states[index], phase=0, low_k_max=config.low_k_max, topology=True
            ).to_dict()
            topological_series.append([initial_metric])
            shells = shell_power(states[index], task.shells)
            temporal_series.append(
                [
                    {
                        "phase": 0,
                        "shell_power": shells.tolist(),
                        "activity": None,
                        "reaction_fraction": 0.0,
                        "delta_m_flux": 0,
                        "occupancy_fraction": initial_metric["occupancy_fraction"],
                        "charge": initial_metric["charge"],
                    }
                ]
            )
            peak_scales.append(float(initial_metric["finite_empty_scale"]))
            peak_states.append(states[index].clone())
        cumulative_input_hist = torch.zeros((len(batch), 256), dtype=torch.int64, device=device)
        interval_reaction_counts = torch.zeros(len(batch), dtype=torch.int64, device=device)
        interval_delta_m = torch.zeros(len(batch), dtype=torch.int64, device=device)
        previous_sample_states = states.clone()
        nucleation_tracking = (
            [new_tracker(states[index], topological_series[index][0]) for index in range(len(batch))]
            if config.nucleation
            else []
        )

    tables = torch.stack([rule.table for rule in rule_list], dim=0).to(device=device)
    aux = [_rule_tensors(rule, device) for rule in rule_list]
    reaction_masks = torch.stack([item["reaction"] for item in aux], dim=0)
    delta_m_lookup = torch.stack([item["delta_m"] for item in aux], dim=0)
    collect_hist = any(task.stage.startswith("ablation") or task.stage.startswith("mechanism") for task in batch)
    total_blocks_per_phase = (batch[0].size // 2) ** 2
    resource_log = output / "resource_log.csv"

    target_steps = batch[0].steps
    while phase < target_steps:
        result = step_batched(states, tables, phase, return_codes=True)
        assert isinstance(result, tuple)
        states, in_codes, _ = result
        flat_codes = in_codes.reshape(len(batch), -1)
        interval_reaction_counts += reaction_masks.gather(1, flat_codes).sum(dim=1)
        interval_delta_m += delta_m_lookup.gather(1, flat_codes).sum(dim=1)
        if collect_hist:
            for index in range(len(batch)):
                cumulative_input_hist[index] += torch.bincount(
                    flat_codes[index], minlength=256
                )
        phase += 1

        if phase % batch[0].sample_every == 0 or phase == target_steps:
            interval_phases = batch[0].sample_every if phase % batch[0].sample_every == 0 else phase % batch[0].sample_every
            denominator = max(1, interval_phases * total_blocks_per_phase)
            for index, task in enumerate(batch):
                shells = shell_power(states[index], task.shells)
                temporal_series[index].append(
                    {
                        "phase": phase,
                        "shell_power": shells.tolist(),
                        "activity": float((states[index] != previous_sample_states[index]).float().mean()),
                        "reaction_fraction": float(interval_reaction_counts[index]) / denominator,
                        "delta_m_flux": int(interval_delta_m[index]),
                        "occupancy_fraction": float((states[index] != 0).float().mean()),
                        "charge": int(DEFAULT_ALPHABET.total_charge(states[index])),
                    }
                )
            previous_sample_states = states.clone()
            interval_reaction_counts.zero_()
            interval_delta_m.zero_()

        if phase % batch[0].topology_every == 0 or phase == target_steps:
            for index, task in enumerate(batch):
                metric = snapshot_metrics(
                    states[index], phase=phase, low_k_max=config.low_k_max, topology=True
                )
                topological_series[index].append(metric.to_dict())
                if metric.finite_empty_scale > peak_scales[index]:
                    peak_scales[index] = metric.finite_empty_scale
                    peak_states[index] = states[index].clone()
                if metric.charge != topological_series[index][0]["charge"]:
                    raise RuntimeError(f"exact charge violation in task {task.id} at phase {phase}")
                if config.nucleation:
                    update_tracker(
                        nucleation_tracking[index],
                        state=states[index],
                        series=topological_series[index],
                        criterion=criterion,
                        settings=config.nucleation,
                    )

        if phase % batch[0].checkpoint_every == 0 or phase == target_steps:
            _save_batch_checkpoint(
                checkpoint,
                config_sha256=config.sha256(),
                tasks=batch,
                phase=phase,
                states=states,
                initial_states=initial_states,
                topological_series=topological_series,
                temporal_series=temporal_series,
                cumulative_input_hist=cumulative_input_hist,
                interval_reaction_counts=interval_reaction_counts,
                interval_delta_m=interval_delta_m,
                previous_sample_states=previous_sample_states,
                peak_scales=peak_scales,
                peak_states=peak_states,
                nucleation_tracking=nucleation_tracking,
            )
            _status(
                output,
                {
                    "state": "running",
                    "batch_tasks": task_ids,
                    "phase": phase,
                    "target_steps": target_steps,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            control = _check_controls(output)
            if control == "stop":
                raise ManagedPause("stop requested after safe checkpoint")
            if control == "pause":
                raise ManagedPause("pause requested after safe checkpoint")
            if managed and config.resource.enabled:
                sample = sample_pressure(config.resource)
                append_resource_log(resource_log, sample)
                if sample.reasons:
                    raise ManagedPause("resource pressure: " + ", ".join(sample.reasons))

    # Finalize each task into an immutable compact result.
    for index, task in enumerate(batch):
        metrics_objects = []
        from .metrics import SnapshotMetrics

        for payload in topological_series[index]:
            metrics_objects.append(SnapshotMetrics(**payload))
        classification = criterion.classify(metrics_objects)
        temporal = temporal_series[index]
        times = np.asarray([row["phase"] for row in temporal], dtype=np.float64)
        shell_matrix = np.asarray([row["shell_power"] for row in temporal], dtype=np.float64)
        fit_mask = (times >= task.early_fit_start) & (times <= task.early_fit_end)
        shell_growth = []
        for shell_index, shell in enumerate(task.shells):
            if int(fit_mask.sum()) >= 3:
                rate, r2 = fit_log_growth_rate(times[fit_mask], shell_matrix[fit_mask, shell_index])
            else:
                rate, r2 = float("nan"), float("nan")
            shell_growth.append({"shell": list(shell), "rate_per_phase": rate, "r2": r2})

        initial_low = max(float(metrics_objects[0].low_k_power), 1e-30)
        low_ratios = [float(metric.low_k_power) / initial_low for metric in metrics_objects]
        first_passage = None
        initial_scale = max(metrics_objects[0].finite_empty_scale, criterion.initial_scale_floor)
        for metric in metrics_objects[1:]:
            if (
                metric.finite_empty_scale / initial_scale >= criterion.min_peak_scale_ratio
                and metric.largest_finite_empty_fraction
                >= criterion.min_peak_largest_finite_empty_fraction
            ):
                first_passage = metric.phase
                break

        rule = rule_list[index]
        input_hist = cumulative_input_hist[index].detach().cpu().numpy()
        time_to_event = (
            finalize_time_to_event(
                nucleation_tracking[index],
                series=topological_series[index],
                settings=config.nucleation,
                horizon=task.steps,
            )
            if config.nucleation
            else None
        )
        if time_to_event is not None:
            first_passage = time_to_event["T_hit"]
        result_payload = {
            "schema": "zws-reversible-null-phase-task-result-v1",
            "task": task.to_dict(),
            "config_sha256": config.sha256(),
            "rule": {
                "name": rule.name,
                "sha256": rule.sha256(),
                "audit": asdict(rule.audit()),
            },
            "initial": {
                "inventory": asdict(inventory_list[index]),
                "state_sha256": state_sha256(initial_states[index]),
            },
            "final": {
                "state_sha256": state_sha256(states[index]),
            },
            "classification": {
                **classification,
                "first_passage_phase": first_passage,
                "peak_low_k_ratio": float(max(low_ratios)),
                "final_low_k_ratio": float(low_ratios[-1]),
            },
            "shell_growth": shell_growth,
            "topological_series": topological_series[index],
            "temporal_series": temporal_series[index],
            "reaction_flux": summarize_flux(rule, input_hist) if collect_hist else None,
            "time_to_event": time_to_event,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        task_dir = output / "tasks" / task.stage / task.id
        task_dir.mkdir(parents=True, exist_ok=True)
        if config.nucleation:
            result_payload["event_window"] = save_event_window(
                task_dir / "event_window.npz", nucleation_tracking[index]
            )
        save_state_npz(initial_states[index], task_dir / "initial.npz")
        save_state_npz(peak_states[index], task_dir / "peak.npz")
        save_state_npz(states[index], task_dir / "final.npz")
        save_state_png(initial_states[index], task_dir / "initial.png")
        save_state_png(peak_states[index], task_dir / "peak.png")
        save_state_png(states[index], task_dir / "final.png")
        for artifact in (
            task_dir / "initial.npz",
            task_dir / "peak.npz",
            task_dir / "final.npz",
            task_dir / "initial.png",
            task_dir / "peak.png",
            task_dir / "final.png",
        ):
            _write_file_sidecar(artifact)
        result_path = _task_result_path(output, task)
        atomic_write_json(result_path, result_payload)

    # A completed v0.3.1 batch checkpoint is the exact continuation capsule
    # required by the preregistered conditional Stage 2.  Keep only its newest
    # verified generation; legacy studies retain their prior cleanup behavior.
    first_generation = 1 if config.nucleation else 0
    for generation in range(first_generation, 3):
        candidate = _checkpoint_generation(checkpoint, generation)
        candidate.unlink(missing_ok=True)
        candidate.with_suffix(candidate.suffix + ".sha256").unlink(missing_ok=True)


def run_study(
    config: StudyConfig,
    output: str | Path,
    *,
    managed: bool = False,
    only_stages: set[str] | None = None,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    all_tasks = write_plan(config, output_path)
    if only_stages:
        unknown = sorted(only_stages - set(config.stages))
        if unknown:
            raise ValueError(f"unknown requested stages: {unknown}")
        tasks = [task for task in all_tasks if task.stage in only_stages]
    else:
        tasks = all_tasks
    winner = ReversibleRule.load(config.rule_path)
    winner.assert_valid()
    variants = _variant_rules(winner)
    missing = sorted({task.rule_variant for task in tasks} - variants.keys())
    if missing:
        raise ValueError(f"unknown rule variants in plan: {missing}")
    criterion = StrongDomainCriterion(**config.criterion)
    device = resolve_device(config.device)
    atomic_write_json(
        output_path / "environment.json",
        {
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "config_sha256": config.sha256(),
            "winner_sha256": winner.sha256(),
            "selected_stages": sorted(only_stages) if only_stages else list(config.stages),
        },
    )

    pending = [task for task in tasks if not _task_result_valid(_task_result_path(output_path, task))]
    grouped: dict[tuple[Any, ...], list[Task]] = {}
    for task in pending:
        grouped.setdefault(_batch_key(task), []).append(task)

    completed_before = len(tasks) - len(pending)
    completed_now = 0
    _status(
        output_path,
        {
            "state": "starting",
            "task_count": len(tasks),
            "completed": completed_before,
            "pending": len(pending),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "selected_stages": sorted(only_stages) if only_stages else list(config.stages),
        },
    )
    try:
        for key in sorted(grouped, key=str):
            stage_name = key[0]
            batch_size = config.stages[stage_name].batch_size
            for batch in _chunks(grouped[key], batch_size):
                wait_for_quiet(
                    config.resource,
                    log_path=output_path / "resource_log.csv",
                    stop_file=output_path / "STOP.request",
                )
                _run_batch(
                    config,
                    output_path,
                    batch,
                    variants,
                    device,
                    criterion,
                    managed=managed,
                )
                completed_now += len(batch)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _status(
                    output_path,
                    {
                        "state": "running",
                        "task_count": len(tasks),
                        "completed": completed_before + completed_now,
                        "pending": len(tasks) - completed_before - completed_now,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
    except ManagedPause as exc:
        _status(
            output_path,
            {
                "state": "paused",
                "reason": str(exc),
                "task_count": len(tasks),
                "completed": completed_before + completed_now,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    _status(
        output_path,
        {
            "state": "complete",
            "task_count": len(tasks),
            "completed": len(tasks),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "task_count": len(tasks),
        "completed": len(tasks),
        "output": str(output_path),
        "selected_stages": sorted(only_stages) if only_stages else list(config.stages),
    }
