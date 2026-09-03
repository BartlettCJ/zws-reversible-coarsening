from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .atomic import (
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from .control import WorkloadPaused
from .events import EventCandidate
from .explorer import export_explorer_bundle
from .initial import SeedSpec, balanced_random
from .margolus import step_batched
from .rules import ReversibleRule
from .visualize import save_trajectory_gif


SCHEMA = "zwslab-sparse-observation-v1"


def _rule_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(bytes(int(value) for value in rule.table.tolist())).hexdigest()


def _configuration_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(normalized)


def _checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _save_verified_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    atomic_torch_save(path, payload)
    digest = sha256_file(path)
    atomic_write_text(_checkpoint_sidecar(path), f"{digest}  {path.name}\n")
    return path


def _load_verified_checkpoint(path: Path) -> dict[str, Any]:
    sidecar = _checkpoint_sidecar(path)
    if not sidecar.is_file():
        raise ValueError(f"checkpoint checksum is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported sparse checkpoint: {path}")
    return payload


def _archive_path(root: Path, tick: int) -> Path:
    return root / "checkpoints" / "states" / f"state_t{tick:012d}.pt"


def _save_archive(
    root: Path,
    *,
    tick: int,
    state_cpu: torch.Tensor,
    rule_sha256: str,
    config_sha256: str,
) -> Path:
    return _save_verified_checkpoint(
        _archive_path(root, tick),
        {
            "schema": SCHEMA,
            "kind": "state",
            "tick": tick,
            "state": state_cpu,
            "rule_sha256": rule_sha256,
            "config_sha256": config_sha256,
        },
    )


def _save_resume(
    root: Path,
    *,
    tick: int,
    state_cpu: torch.Tensor,
    rows: list[dict[str, Any]],
    sample_history: list[tuple[int, torch.Tensor]],
    rule_sha256: str,
    config_sha256: str,
) -> Path:
    return _save_verified_checkpoint(
        root / "checkpoints" / "RESUME.pt",
        {
            "schema": SCHEMA,
            "kind": "resume",
            "tick": tick,
            "state": state_cpu,
            "rows": rows,
            "sample_history": [
                {"tick": sample_tick, "state": sample_state.detach().cpu()}
                for sample_tick, sample_state in sample_history
            ],
            "rule_sha256": rule_sha256,
            "config_sha256": config_sha256,
        },
    )


def _entropy(state: torch.Tensor) -> float:
    counts = torch.bincount(state.reshape(-1).long(), minlength=4).double()
    probabilities = counts / counts.sum().clamp_min(1)
    valid = probabilities > 0
    value = -(probabilities[valid] * probabilities[valid].log()).sum() / math.log(4.0)
    return float(value)


def _boundary_density(state: torch.Tensor) -> float:
    vertical = (state != torch.roll(state, -1, dims=-2)).float().mean()
    horizontal = (state != torch.roll(state, -1, dims=-1)).float().mean()
    return float(0.5 * (vertical + horizontal))


def _hotspot(changes: torch.Tensor, tile_size: int) -> tuple[int, int, float]:
    height, width = changes.shape[-2:]
    tile = max(2, min(int(tile_size), height, width))
    pooled = functional.avg_pool2d(
        changes.float().unsqueeze(0).unsqueeze(0), kernel_size=tile, stride=tile
    )[0, 0]
    index = int(pooled.argmax())
    tiles_wide = pooled.shape[1]
    tile_y, tile_x = divmod(index, tiles_wide)
    center_y = min(height - 1, tile_y * tile + tile // 2)
    center_x = min(width - 1, tile_x * tile + tile // 2)
    return center_x, center_y, float(pooled.flatten()[index])


def _sample_row(
    *,
    tick: int,
    previous: torch.Tensor,
    following: torch.Tensor,
    reaction: torch.Tensor,
    history: list[tuple[int, torch.Tensor]],
    tile_size: int,
) -> dict[str, Any]:
    changes = following[0] != previous[0]
    center_x, center_y, local_activity = _hotspot(changes, tile_size)
    recurrence = 0.0
    for _, prior in history:
        recurrence = max(recurrence, float((following == prior).float().mean()))
    return {
        "tick": tick,
        "activity": float(changes.float().mean()),
        "reaction": float(reaction[0]),
        "entropy": _entropy(following),
        "boundary_density": _boundary_density(following),
        "recurrence": recurrence,
        "hotspot_x": center_x,
        "hotspot_y": center_y,
        "hotspot_activity": local_activity,
    }


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    return (values - median) / max(1e-9, 1.4826 * deviation)


def _rank_events(
    rows: list[dict[str, Any]], *, count: int, min_separation: int
) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        return []
    activity = np.asarray([row["activity"] for row in rows], dtype=np.float64)
    reaction = np.asarray([row["reaction"] for row in rows], dtype=np.float64)
    entropy = np.asarray([row["entropy"] for row in rows], dtype=np.float64)
    boundary = np.asarray([row["boundary_density"] for row in rows], dtype=np.float64)
    recurrence = np.asarray([row["recurrence"] for row in rows], dtype=np.float64)
    hotspot = np.asarray([row["hotspot_activity"] for row in rows], dtype=np.float64)
    delta_entropy = np.abs(np.diff(entropy, prepend=entropy[0]))
    delta_boundary = np.abs(np.diff(boundary, prepend=boundary[0]))
    features = {
        "activity": activity,
        "reaction": reaction,
        "entropy_transition": delta_entropy,
        "boundary_transition": delta_boundary,
        "recurrence": recurrence,
        "local_activity": hotspot,
    }
    z = {name: np.clip(_robust_z(values), -3.0, 8.0) for name, values in features.items()}
    score = (
        0.16 * np.maximum(z["activity"], 0.0)
        + 0.19 * np.maximum(z["reaction"], 0.0)
        + 0.14 * np.maximum(z["entropy_transition"], 0.0)
        + 0.11 * np.maximum(z["boundary_transition"], 0.0)
        + 0.12 * np.maximum(z["recurrence"], 0.0)
        + 0.28 * np.maximum(z["local_activity"], 0.0)
    )
    order = sorted(range(len(rows)), key=lambda index: (float(score[index]), index), reverse=True)
    chosen: list[int] = []
    for index in order:
        tick = int(rows[index]["tick"])
        if all(abs(tick - int(rows[previous]["tick"])) >= min_separation for previous in chosen):
            chosen.append(index)
        if len(chosen) >= count:
            break
    labels = {
        "activity": "global activity spike",
        "reaction": "reaction-density spike",
        "entropy_transition": "rapid entropy transition",
        "boundary_transition": "rapid boundary transition",
        "recurrence": "sampled recurrence pulse",
        "local_activity": "local activity hotspot",
    }
    events: list[dict[str, Any]] = []
    for index in sorted(chosen, key=lambda item: int(rows[item]["tick"])):
        contributions = sorted(
            ((float(z[name][index]), name) for name in features), reverse=True
        )
        reasons = [labels[name] for value, name in contributions[:3] if value > 0.5]
        events.append(
            {
                **rows[index],
                "score": float(score[index]),
                "reasons": reasons or ["combined streaming anomaly"],
            }
        )
    return events


def _torus_crop(state: torch.Tensor, center_x: int, center_y: int, size: int) -> torch.Tensor:
    if state.ndim == 3:
        state = state[0]
    height, width = state.shape
    ys = (torch.arange(size, device=state.device) + center_y - size // 2) % height
    xs = (torch.arange(size, device=state.device) + center_x - size // 2) % width
    return state.index_select(0, ys).index_select(1, xs)


def _available_archives(root: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in (root / "checkpoints" / "states").glob("state_t*.pt"):
        tick = int(path.stem.removeprefix("state_t"))
        if _checkpoint_sidecar(path).is_file():
            result.append((tick, path))
    return sorted(result)


def _reconstruct_clip(
    root: Path,
    rule: ReversibleRule,
    *,
    event_tick: int,
    center_x: int,
    center_y: int,
    before: int,
    after: int,
    crop_size: int,
    final_tick: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    start = max(0, event_tick - before)
    end = min(final_tick, event_tick + after)
    archives = _available_archives(root)
    earlier = [(tick, path) for tick, path in archives if tick <= start]
    later = [(tick, path) for tick, path in archives if tick >= end]
    forward = earlier[-1] if earlier else None
    backward = later[0] if later else None
    forward_cost = end - forward[0] if forward else math.inf
    backward_cost = backward[0] - start if backward else math.inf
    clips: list[torch.Tensor] = []
    ticks: list[int] = []

    if forward_cost <= backward_cost:
        assert forward is not None
        checkpoint_tick, checkpoint_path = forward
        payload = _load_verified_checkpoint(checkpoint_path)
        state = payload["state"].to(device)
        table = rule.table.to(device).unsqueeze(0)
        current = checkpoint_tick
        if current >= start:
            clips.append(_torus_crop(state, center_x, center_y, crop_size).cpu())
            ticks.append(current)
        while current < end:
            state = step_batched(state, table, current)  # type: ignore[assignment]
            current += 1
            if current >= start:
                clips.append(_torus_crop(state, center_x, center_y, crop_size).cpu())
                ticks.append(current)
        mode = "forward"
    else:
        assert backward is not None
        checkpoint_tick, checkpoint_path = backward
        payload = _load_verified_checkpoint(checkpoint_path)
        state = payload["state"].to(device)
        inverse = rule.inverse_table.to(device).unsqueeze(0)
        current = checkpoint_tick
        reverse_clips: list[torch.Tensor] = []
        reverse_ticks: list[int] = []
        while current >= start:
            if current <= end:
                reverse_clips.append(_torus_crop(state, center_x, center_y, crop_size).cpu())
                reverse_ticks.append(current)
            if current == start:
                break
            state = step_batched(state, inverse, current - 1)  # type: ignore[assignment]
            current -= 1
        clips = list(reversed(reverse_clips))
        ticks = list(reversed(reverse_ticks))
        mode = "backward"

    return (
        torch.stack(clips).to(torch.uint8),
        ticks,
        {
            "mode": mode,
            "checkpoint_tick": checkpoint_tick,
            "checkpoint": str(checkpoint_path),
            "start_tick": start,
            "end_tick": end,
        },
    )


def _save_event_outputs(
    root: Path,
    rule: ReversibleRule,
    events: list[dict[str, Any]],
    *,
    before: int,
    after: int,
    crop_size: int,
    final_tick: int,
    world_height: int,
    world_width: int,
    device: torch.device,
) -> tuple[
    list[EventCandidate],
    torch.Tensor,
    list[int],
    list[list[int]],
    list[dict[str, Any]],
]:
    event_root = root / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    candidates: list[EventCandidate] = []
    preview_clips: list[torch.Tensor] = []
    preview_ticks: list[int] = []
    preview_origins: list[list[int]] = []
    records: list[dict[str, Any]] = []
    preview_offset = 0
    for event_id, row in enumerate(events):
        clip, ticks, reconstruction = _reconstruct_clip(
            root,
            rule,
            event_tick=int(row["tick"]),
            center_x=int(row["hotspot_x"]),
            center_y=int(row["hotspot_y"]),
            before=before,
            after=after,
            crop_size=crop_size,
            final_tick=final_tick,
            device=device,
        )
        directory = event_root / f"event_{event_id:03d}_t{int(row['tick']):012d}"
        directory.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, clip.numpy(), allow_pickle=False)
        states_path = directory / "local_clip.npy"
        atomic_write_bytes(states_path, buffer.getvalue())
        gif_path = save_trajectory_gif(
            clip,
            directory / "local_clip.gif",
            stride=1,
            max_frames=clip.shape[0],
            scale=max(1, min(4, 1024 // crop_size)),
            fps=20,
        )
        event_frame = ticks.index(int(row["tick"]))
        origin_x = (int(row["hotspot_x"]) - crop_size // 2) % world_width
        origin_y = (int(row["hotspot_y"]) - crop_size // 2) % world_height
        candidate = EventCandidate(
            event_id=event_id,
            frame=preview_offset + event_frame,
            tick=int(row["tick"]),
            score=float(row["score"]),
            reasons=tuple(str(value) for value in row["reasons"]),
            activity=float(row["activity"]),
            reaction_fraction=float(row["reaction"]),
            entropy=float(row["entropy"]),
            boundary_density=float(row["boundary_density"]),
            recurrence=float(row["recurrence"]),
            occupied_fraction=0.0,
            bound_fraction=0.0,
            crop=(origin_x, origin_y, crop_size, crop_size),
        )
        record = {
            **row,
            "event_id": event_id,
            "clip_ticks": ticks,
            "crop_origin": [origin_x, origin_y],
            "crop_size": crop_size,
            "states_npy": str(states_path),
            "gif": str(gif_path),
            "states_sha256": sha256_file(states_path),
            "reconstruction": reconstruction,
        }
        atomic_write_json(directory / "event.json", record)
        candidates.append(candidate)
        records.append(record)
        preview_clips.append(clip)
        preview_ticks.extend(ticks)
        preview_origins.extend([[origin_x, origin_y] for _ in ticks])
        preview_offset += clip.shape[0]
    atomic_write_json(event_root / "events.json", records)
    if not preview_clips:
        raise RuntimeError("no event clips were selected; reduce --sample-every or increase steps")
    return candidates, torch.cat(preview_clips), preview_ticks, preview_origins, records


def observe_rule(
    rule: ReversibleRule,
    output: str | Path,
    *,
    height: int,
    width: int,
    steps: int,
    device: str = "auto",
    seed: int = 260824,
    sample_every: int = 8,
    checkpoint_every: int = 2048,
    event_count: int = 24,
    event_min_separation: int = 256,
    event_before: int = 64,
    event_after: int = 64,
    crop_size: int = 256,
    tile_size: int = 128,
    positive_fraction: float = 0.12,
    bound_fraction: float = 0.08,
    localized: bool = False,
    resume: bool = False,
    stop_after_tick: int | None = None,
) -> dict[str, Any]:
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("observation dimensions must be even and at least 2")
    if steps < 1 or sample_every < 1 or checkpoint_every < 1:
        raise ValueError("steps, sample_every, and checkpoint_every must be positive")
    if sample_every > steps:
        raise ValueError("sample_every must not exceed steps")
    if event_count < 1:
        raise ValueError("event_count must be at least 1")
    if crop_size < 2 or crop_size > min(height, width):
        raise ValueError("crop_size must be between 2 and the smaller world dimension")
    root = Path(output).resolve()
    completed = root / "observation.json"
    if resume and completed.is_file():
        return json.loads(completed.read_text(encoding="utf-8"))
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"observation output already exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    selected_rule = rule.save(root / "selected_rule.json")
    requested = {
        "schema": SCHEMA,
        "height": int(height),
        "width": int(width),
        "steps": int(steps),
        "device": device,
        "seed": int(seed),
        "sample_every": int(sample_every),
        "checkpoint_every": int(checkpoint_every),
        "event_count": int(event_count),
        "event_min_separation": int(event_min_separation),
        "event_before": int(event_before),
        "event_after": int(event_after),
        "crop_size": int(crop_size),
        "tile_size": int(tile_size),
        "positive_fraction": float(positive_fraction),
        "bound_fraction": float(bound_fraction),
        "localized": bool(localized),
    }
    rule_sha256 = _rule_digest(rule)
    config_sha256 = _configuration_digest({**requested, "rule_sha256": rule_sha256})
    atomic_write_json(root / "observation_config.json", requested)
    actual_device = _resolve_device(device)
    resume_path = root / "checkpoints" / "RESUME.pt"
    rows: list[dict[str, Any]] = []
    sample_history: list[tuple[int, torch.Tensor]] = []

    if resume:
        payload = _load_verified_checkpoint(resume_path)
        if payload.get("rule_sha256") != rule_sha256:
            raise ValueError("resume rule hash does not match")
        if payload.get("config_sha256") != config_sha256:
            raise ValueError("resume configuration hash does not match")
        tick = int(payload["tick"])
        state = payload["state"].to(actual_device)
        rows = list(payload.get("rows", []))
        sample_history = [
            (int(item["tick"]), item["state"].to(actual_device))
            for item in payload.get("sample_history", [])
        ]
    else:
        tick = 0
        state = balanced_random(
            1,
            height,
            width,
            spec=SeedSpec(
                positive_fraction=positive_fraction,
                bound_fraction=bound_fraction,
                localized=localized,
            ),
            seed=seed,
            device=actual_device,
        )
        state_cpu = state.detach().cpu().contiguous()
        _save_archive(
            root,
            tick=0,
            state_cpu=state_cpu,
            rule_sha256=rule_sha256,
            config_sha256=config_sha256,
        )
        _save_resume(
            root,
            tick=0,
            state_cpu=state_cpu,
            rows=rows,
            sample_history=sample_history,
            rule_sha256=rule_sha256,
            config_sha256=config_sha256,
        )

    table = rule.table.to(actual_device).unsqueeze(0)
    segment_start_tick = tick
    segment_started = time.perf_counter()
    if actual_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual_device)
    try:
        while tick < steps:
            next_tick = tick + 1
            sampled = next_tick % sample_every == 0
            if sampled:
                following, reaction = step_batched(
                    state, table, tick, return_reaction=True
                )  # type: ignore[misc]
                rows.append(
                    _sample_row(
                        tick=next_tick,
                        previous=state,
                        following=following,
                        reaction=reaction,
                        history=sample_history,
                        tile_size=tile_size,
                    )
                )
                sample_history.append((next_tick, following.detach().clone()))
                if len(sample_history) > 4:
                    sample_history.pop(0)
            else:
                following = step_batched(state, table, tick)  # type: ignore[assignment]
            state = following
            tick = next_tick
            should_checkpoint = tick % checkpoint_every == 0 or tick == steps
            should_pause = stop_after_tick is not None and tick >= stop_after_tick
            if should_checkpoint or should_pause:
                state_cpu = state.detach().cpu().contiguous()
                _save_archive(
                    root,
                    tick=tick,
                    state_cpu=state_cpu,
                    rule_sha256=rule_sha256,
                    config_sha256=config_sha256,
                )
                _save_resume(
                    root,
                    tick=tick,
                    state_cpu=state_cpu,
                    rows=rows,
                    sample_history=sample_history,
                    rule_sha256=rule_sha256,
                    config_sha256=config_sha256,
                )
                atomic_write_json(
                    root / "status.json",
                    {
                        "schema": SCHEMA,
                        "status": "running" if tick < steps else "reconstructing",
                        "tick": tick,
                        "steps": steps,
                        "progress": tick / steps,
                        "device": str(actual_device),
                    },
                )
                elapsed = max(1e-9, time.perf_counter() - segment_started)
                print(
                    f"[ZWSLab observe] tick {tick:,}/{steps:,} "
                    f"({100.0 * tick / steps:.1f}%) · "
                    f"{(tick - segment_start_tick) / elapsed:,.1f} ticks/s"
                )
            if should_pause:
                raise WorkloadPaused(f"sparse observation paused at exact tick {tick}")
    except KeyboardInterrupt as exc:
        state_cpu = state.detach().cpu().contiguous()
        _save_archive(
            root,
            tick=tick,
            state_cpu=state_cpu,
            rule_sha256=rule_sha256,
            config_sha256=config_sha256,
        )
        _save_resume(
            root,
            tick=tick,
            state_cpu=state_cpu,
            rows=rows,
            sample_history=sample_history,
            rule_sha256=rule_sha256,
            config_sha256=config_sha256,
        )
        raise WorkloadPaused(f"sparse observation paused at exact tick {tick}") from exc

    simulation_seconds = time.perf_counter() - segment_started
    simulated_ticks = tick - segment_start_tick
    peak_cuda_gib = (
        float(torch.cuda.max_memory_allocated(actual_device) / 2**30)
        if actual_device.type == "cuda"
        else None
    )
    ranked = _rank_events(rows, count=event_count, min_separation=event_min_separation)
    candidates, preview, preview_ticks, preview_origins, event_records = _save_event_outputs(
        root,
        rule,
        ranked,
        before=event_before,
        after=event_after,
        crop_size=crop_size,
        final_tick=steps,
        world_height=height,
        world_width=width,
        device=actual_device,
    )
    explorer_paths = export_explorer_bundle(
        preview.unsqueeze(1),
        root / "explorer",
        rule_name=rule.name,
        rule_sha256=rule_sha256,
        events=candidates,
        metric_series={},
        ticks=preview_ticks,
        extra_metadata={
            "sparse_observation": True,
            "world_height": height,
            "world_width": width,
            "frame_origins": preview_origins,
            "note": "Only exact local clips around ranked events are stored.",
        },
    )
    atomic_write_json(root / "metrics.json", rows)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "rule": str(selected_rule),
        "rule_sha256": rule_sha256,
        "config_sha256": config_sha256,
        "device": str(actual_device),
        "world": [height, width],
        "steps": steps,
        "samples": len(rows),
        "checkpoints": len(_available_archives(root)),
        "performance": {
            "segment_start_tick": segment_start_tick,
            "segment_ticks": simulated_ticks,
            "simulation_seconds": simulation_seconds,
            "ticks_per_second": simulated_ticks / max(1e-9, simulation_seconds),
            "peak_cuda_allocated_gib": peak_cuda_gib,
        },
        "events": event_records,
        "explorer": explorer_paths,
        "storage_note": (
            "Full history was not stored. Initial/final/periodic exact states plus scalar "
            "metrics and reversibly reconstructed local event clips are retained."
        ),
    }
    atomic_write_json(completed, result)
    atomic_write_json(
        root / "status.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "tick": steps,
            "steps": steps,
            "progress": 1.0,
            "events": len(event_records),
        },
    )
    return result
