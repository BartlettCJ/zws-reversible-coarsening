from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .atomic import atomic_write_text, sha256_file
from .history_erasure import state_histogram, state_sha256
from .history_figures import (
    INTERVENTION_COLORS,
    _branch_entry,
    _load_discovery_state,
    _read_json,
    _realized,
    _record_state,
    _representatives,
    _source_result,
)
from .history_media import index_task_states, load_exact_npz
from .history_publication import RULE_SHA256, STATE_HEX, STATE_NAMES, aligned_densest_crop


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 12


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


PALETTE_RGB = np.asarray([_hex_rgb(value) for value in STATE_HEX], dtype=np.uint8)


def _state_image(
    state: np.ndarray,
    size: tuple[int, int],
    *,
    view: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    array = np.asarray(state, dtype=np.uint8)
    image = Image.fromarray(PALETTE_RGB[array], mode="RGB")
    if view is not None:
        x, y, width, height = view
        left = max(0, min(array.shape[1] - 1, int(round(x))))
        top = max(0, min(array.shape[0] - 1, int(round(y))))
        right = max(left + 1, min(array.shape[1], int(round(x + width))))
        bottom = max(top + 1, min(array.shape[0], int(round(y + height))))
        image = image.crop((left, top, right, bottom))
    return image.resize(size, resample=Image.Resampling.NEAREST)


def _camera_view(
    shape: tuple[int, int],
    crop: tuple[int, int, int, int],
    fraction: float,
) -> tuple[float, float, float, float]:
    height, width = shape
    crop_x, crop_y, crop_width, crop_height = crop
    eased = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, fraction)))
    return (
        crop_x * eased,
        crop_y * eased,
        width + (crop_width - width) * eased,
        height + (crop_height - height) * eased,
    )


def _state_label(state: np.ndarray) -> tuple[int, int, list[int]]:
    histogram = state_histogram(state).tolist()
    return int(histogram[1] - histogram[2]), int(sum(histogram[1:])), histogram


def render_history_frame(
    states: dict[str, np.ndarray],
    *,
    density: float,
    elapsed: int,
    absolute_phases: dict[str, int],
    state_hashes: dict[str, str],
    crop: tuple[int, int, int, int],
    camera_fraction: float,
    event_times: dict[str, int | None],
) -> Image.Image:
    order = ("untouched", "global_shuffle", "tile_32", "tile_128")
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#071018")
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 34), f"Same ingredients, different history · initial density {density:.2f}", font=_font(36, bold=True), fill="#f6f8fa")
    draw.text((60, 84), f"elapsed {elapsed:,} phases · exact saved states only", font=_font(24), fill="#b7c7d3")
    panel_width, panel_height = 430, 720
    gap, left, top = 28, 45, 145
    for index, intervention in enumerate(order):
        state = states[intervention]
        x = left + index * (panel_width + gap)
        view = _camera_view(state.shape, crop, camera_fraction)
        panel = _state_image(state, (panel_width, panel_height), view=view)
        canvas.paste(panel, (x, top))
        color = INTERVENTION_COLORS[intervention]
        draw.rectangle((x, top, x + panel_width, top + panel_height), outline=color, width=4)
        q_value, m_value, histogram = _state_label(state)
        draw.text((x, top + panel_height + 18), intervention.replace("_", " "), font=_font(24, bold=True), fill=color)
        draw.text((x, top + panel_height + 55), f"Q={q_value:,}   M={m_value:,}", font=_font(21), fill="#f6f8fa")
        draw.text((x, top + panel_height + 86), f"phase {absolute_phases[intervention]:,}", font=_font(18), fill="#b7c7d3")
        draw.text((x, top + panel_height + 113), f"counts {histogram}", font=_font(16), fill="#9fb0bf")
        hit = event_times.get(intervention)
        draw.text((x, top + panel_height + 139), "T_hit " + (f"{hit:,}" if hit is not None else "censored/not yet"), font=_font(16), fill="#9fb0bf")
        draw.text((x, top + panel_height + 164), state_hashes[intervention][:16] + "…", font=_font(14), fill="#728697")
    legend_y = 1040
    legend_x = 60
    for name, color in zip(STATE_NAMES, STATE_HEX, strict=True):
        draw.rectangle((legend_x, legend_y - 14, legend_x + 22, legend_y + 8), fill=color, outline="#ffffff66")
        draw.text((legend_x + 30, legend_y - 18), name, font=_font(16), fill="#d7e9f6")
        legend_x += 210
    draw.text((1100, legend_y - 18), "Camera motion only; automaton states are never interpolated.", font=_font(16), fill="#cc79a7")
    return canvas


def _ffmpeg_path() -> str:
    candidate = shutil.which("ffmpeg")
    if candidate:
        return candidate
    pinned = Path("C:/Tools/ffmpeg/bin/ffmpeg.exe")
    if pinned.is_file():
        return str(pinned)
    raise RuntimeError("ffmpeg executable is unavailable")


def _encode_mp4(frame_dir: Path, output: Path, *, fps: int = VIDEO_FPS) -> dict[str, Any]:
    command = [
        _ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"ffmpeg failed ({completed.returncode}): {completed.stderr[-2000:]}")
    atomic_write_text(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256_file(output)}  {output.name}\n",
    )
    return {"command": command, "stderr": completed.stderr, "returncode": completed.returncode}


def _history_exact_records(
    run: Path,
    experiment: Path,
    representative: dict[str, Any],
) -> tuple[
    dict[str, dict[int, tuple[np.ndarray, dict[str, Any]]]],
    tuple[int, int, int, int],
    dict[str, int | None],
]:
    source_id = str(representative["source_task_id"])
    realized = _realized(run)
    source = _source_result(run, source_id)
    source_state, _, _ = load_exact_npz(Path(source["branch_points"]["B"]["path"]))
    crop = aligned_densest_crop(source_state, 256)
    states: dict[str, dict[int, tuple[np.ndarray, dict[str, Any]]]] = {}
    event_times: dict[str, int | None] = {"untouched": 32768}
    for intervention in ("untouched", "global_shuffle", "tile_32", "tile_128"):
        if intervention == "untouched":
            state_dir = run / "media_states" / "untouched" / "branch_B" / source_id
        else:
            entry = _branch_entry(realized, source_id, "B", intervention)
            state_dir = run / "branches" / entry["task_id"]
            result = _read_json(state_dir / "branch_result.json")
            event_times[intervention] = result["residual_time_to_event"]["R_hit"]
        indexed = index_task_states(state_dir)
        states[intervention] = {}
        for record in indexed:
            if record["elapsed"] is None:
                continue
            state, _, _, _ = _record_state(record)
            states[intervention][int(record["elapsed"])] = (state, record)
    common = set.intersection(*(set(records) for records in states.values()))
    if len(common) < 2 or 0 not in common:
        raise RuntimeError(f"history video lacks a synchronized exact-state intersection: {source_id}")
    return states, crop, event_times


def create_history_video(
    run_path: str | Path,
    experiment_path: str | Path,
    representative: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    run, experiment, output = Path(run_path).resolve(), Path(experiment_path).resolve(), Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    by_branch, crop, event_times = _history_exact_records(run, experiment, representative)
    common = sorted(set.intersection(*(set(records) for records in by_branch.values())))
    exact_frames: list[dict[str, Any]] = []
    subframes = 6
    with tempfile.TemporaryDirectory(prefix="zws_history_video_") as temporary:
        frame_dir = Path(temporary)
        frame_index = 0
        total = max(1, len(common) * subframes - 1)
        for exact_index, elapsed in enumerate(common):
            states = {name: by_branch[name][elapsed][0] for name in by_branch}
            records = {name: by_branch[name][elapsed][1] for name in by_branch}
            exact_frames.append(
                {
                    "elapsed": elapsed,
                    "states": {
                        name: {
                            "path": record["path"],
                            "phase": record["phase"],
                            "state_sha256": record["state_sha256"],
                        }
                        for name, record in records.items()
                    },
                }
            )
            for subframe in range(subframes):
                progress = (exact_index * subframes + subframe) / total
                camera = 0.5 - 0.5 * math.cos(2 * math.pi * progress)
                image = render_history_frame(
                    states,
                    density=float(representative["density"]),
                    elapsed=elapsed,
                    absolute_phases={name: int(record["phase"]) for name, record in records.items()},
                    state_hashes={name: str(record["state_sha256"]) for name, record in records.items()},
                    crop=crop,
                    camera_fraction=camera,
                    event_times=event_times,
                )
                image.save(frame_dir / f"frame_{frame_index:06d}.png", format="PNG", optimize=True)
                frame_index += 1
        encoder = _encode_mp4(frame_dir, output)
    return {
        "name": output.name,
        "path": str(output),
        "sha256": sha256_file(output),
        "rule_hash": RULE_SHA256,
        "density": float(representative["density"]),
        "source_task_id": representative["source_task_id"],
        "representative_rule": representative["selection_rule"],
        "camera": {"crop": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]}, "motion_only": True},
        "exact_state_frames": exact_frames,
        "encoded_frame_count": frame_index,
        "fps": VIDEO_FPS,
        "encoder": encoder,
        "intermediate_automaton_states_synthesized": False,
    }


def render_hero_frame(
    state: np.ndarray,
    *,
    direction: str,
    traversal_phase: int,
    microscopic_phase: int,
    crop: tuple[int, int, int, int],
    camera_fraction: float,
    final_mismatch: int | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#071018")
    draw = ImageDraw.Draw(canvas)
    view = _camera_view(state.shape, crop, camera_fraction)
    panel = _state_image(state, (900, 900), view=view)
    canvas.paste(panel, (55, 120))
    draw.rectangle((55, 120, 955, 1020), outline="#f0e442", width=4)
    draw.text((1020, 90), "A reversible zero-balanced world", font=_font(42, bold=True), fill="#f6f8fa")
    draw.text((1020, 175), direction.upper(), font=_font(32, bold=True), fill="#56b4e9" if direction == "forward" else "#cc79a7")
    draw.text((1020, 240), f"traversal phase {traversal_phase:,}", font=_font(25), fill="#b7c7d3")
    draw.text((1020, 280), f"microscopic phase {microscopic_phase:,}", font=_font(25), fill="#b7c7d3")
    q_value, m_value, histogram = _state_label(state)
    draw.text((1020, 370), f"Q = {q_value:,} exactly", font=_font(34, bold=True), fill="#f0e442")
    draw.text((1020, 425), f"M = {m_value:,}", font=_font(34, bold=True), fill="#f6f8fa")
    draw.text((1020, 480), f"counts {histogram}", font=_font(20), fill="#9fb0bf")
    draw.text((1020, 560), state_sha256(state), font=_font(15), fill="#728697")
    if final_mismatch is not None:
        draw.text((1020, 680), f"FINAL MISMATCH = {final_mismatch}", font=_font(38, bold=True), fill="#70d6a1" if final_mismatch == 0 else "#ff6b6b")
    draw.text((1020, 820), "Exact saved states only.\nCamera movement is visual only.", font=_font(22), fill="#c8d7e3")
    return canvas


def create_hero_video(repo_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    repo, output = Path(repo_root).resolve(), Path(output_path).resolve()
    artifact = repo / "experiments/domain_mechanism_20260825/artifacts/discovery_macro_arrow"
    macro = _read_json(artifact / "REVERSIBLE_MACRO_ARROW.json", require_sidecar=False)
    integrity = _read_json(artifact / "DISCOVERY_CHECKPOINT_INPUT_INTEGRITY.json", require_sidecar=False)
    if not macro["exact_reverse_verified_at_all_retained_boundaries"]:
        raise RuntimeError("hero reverse audit did not pass")
    ticks = [int(row["tick"]) for row in integrity["checkpoints"]]
    states = {tick: _load_discovery_state(repo, integrity, tick) for tick in ticks}
    peak = int(macro["capture_ticks"]["peak_organisation"])
    crop = aligned_densest_crop(states[peak][0], 512)
    sequence = [("forward", tick, tick) for tick in ticks] + [
        ("reverse", ticks[-1] + (ticks[-1] - tick), tick) for tick in reversed(ticks[:-1])
    ]
    exact_frames: list[dict[str, Any]] = []
    subframes = 3
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zws_hero_video_") as temporary:
        frame_dir = Path(temporary)
        frame_index = 0
        for direction, traversal, microscopic in sequence:
            state, path = states[microscopic]
            exact_frames.append({"direction": direction, "traversal_phase": traversal, "microscopic_phase": microscopic, "path": str(path), "state_sha256": state_sha256(state)})
            camera = max(0.0, 1.0 - abs(microscopic - peak) / max(peak, ticks[-1] - peak))
            for _ in range(subframes):
                image = render_hero_frame(
                    state,
                    direction=direction,
                    traversal_phase=traversal,
                    microscopic_phase=microscopic,
                    crop=crop,
                    camera_fraction=camera,
                    final_mismatch=0 if direction == "reverse" and microscopic == 0 else None,
                )
                image.save(frame_dir / f"frame_{frame_index:06d}.png", format="PNG", optimize=True)
                frame_index += 1
        encoder = _encode_mp4(frame_dir, output)
    return {
        "name": output.name,
        "path": str(output),
        "sha256": sha256_file(output),
        "rule_hash": RULE_SHA256,
        "exact_reverse_verified": True,
        "final_mismatch_count": 0,
        "exact_state_frames": exact_frames,
        "encoded_frame_count": frame_index,
        "fps": VIDEO_FPS,
        "encoder": encoder,
        "intermediate_automaton_states_synthesized": False,
    }


def create_preview_gif(
    run_path: str | Path,
    experiment_path: str | Path,
    representative: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    run, experiment, output = Path(run_path).resolve(), Path(experiment_path).resolve(), Path(output_path).resolve()
    by_branch, crop, event_times = _history_exact_records(run, experiment, representative)
    common = sorted(set.intersection(*(set(records) for records in by_branch.values())))
    images: list[Image.Image] = []
    exact_frames: list[dict[str, Any]] = []
    for index, elapsed in enumerate(common):
        states = {name: by_branch[name][elapsed][0] for name in by_branch}
        records = {name: by_branch[name][elapsed][1] for name in by_branch}
        images.append(
            render_history_frame(
                states,
                density=float(representative["density"]),
                elapsed=elapsed,
                absolute_phases={name: int(record["phase"]) for name, record in records.items()},
                state_hashes={name: str(record["state_sha256"]) for name, record in records.items()},
                crop=crop,
                camera_fraction=(index / max(1, len(common) - 1)),
                event_times=event_times,
            ).resize((960, 540), resample=Image.Resampling.LANCZOS)
        )
        exact_frames.append({"elapsed": elapsed, "state_hashes": {name: record["state_sha256"] for name, record in records.items()}})
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=700, loop=0, optimize=True, disposal=2)
    atomic_write_text(output.with_suffix(output.suffix + ".sha256"), f"{sha256_file(output)}  {output.name}\n")
    return {
        "name": output.name,
        "path": str(output),
        "sha256": sha256_file(output),
        "rule_hash": RULE_SHA256,
        "exact_state_frames": exact_frames,
        "intermediate_automaton_states_synthesized": False,
    }


def build_required_videos(
    repo_root: str | Path,
    run_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    representatives = _representatives(Path(experiment_path).resolve())
    entries = [create_hero_video(repo_root, output / "hero_forward_reverse_exact.mp4")]
    for representative in representatives:
        suffix = "020" if float(representative["density"]) == 0.20 else "024"
        entries.append(
            create_history_video(
                run_path,
                experiment_path,
                representative,
                output / f"history_erasure_density{suffix}.mp4",
            )
        )
    entries.append(
        create_preview_gif(
            run_path,
            experiment_path,
            representatives[0],
            output / "history_erasure_short_preview.gif",
        )
    )
    return entries
