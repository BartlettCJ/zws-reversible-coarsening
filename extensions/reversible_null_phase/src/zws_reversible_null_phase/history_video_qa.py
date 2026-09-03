from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_media import verify_sidecar


VIDEO_NAMES = (
    "hero_forward_reverse_exact.mp4",
    "history_erasure_density020.mp4",
    "history_erasure_density024.mp4",
    "history_erasure_short_preview.gif",
)


def expected_video_paths(paper_root: str | Path) -> list[Path]:
    media = Path(paper_root).resolve() / "media"
    return [media / name for name in VIDEO_NAMES]


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout


def _probe(path: Path, executable: str) -> dict[str, Any]:
    raw = _run(
        [
            executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(raw)
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    duration_text = stream.get("duration") or payload.get("format", {}).get("duration")
    duration = float(duration_text or 0.0)
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    minimum = (960, 540) if path.suffix.lower() == ".gif" else (1280, 720)
    if duration <= 0 or width < minimum[0] or height < minimum[1]:
        raise RuntimeError(f"invalid video geometry/duration for {path}: {width}x{height}, {duration}s")
    return {
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "average_frame_rate": stream.get("avg_frame_rate"),
        "declared_frames": stream.get("nb_frames"),
    }


def _review_timestamps(probe: dict[str, Any]) -> tuple[float, ...]:
    """Choose five seek times that remain inside the final coded frame.

    Container duration is the end boundary, not necessarily a frame timestamp.
    Short exact-state videos can therefore have no frame at 96% of the declared
    duration.  When frame count/rate metadata is available, cap every seek at a
    half-frame margin before the last nominal frame; otherwise use 90%.
    """
    duration = float(probe["duration_seconds"])
    safe_end = duration * 0.90
    try:
        frame_rate = float(Fraction(str(probe["average_frame_rate"])))
        frame_count = int(probe["declared_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        frame_rate, frame_count = 0.0, 0
    if frame_rate > 0 and frame_count > 1:
        safe_end = min(safe_end, max(0.0, (frame_count - 1.5) / frame_rate))
    first = duration * 0.02
    safe_end = max(first, safe_end)
    return tuple(
        min(duration * fraction, safe_end)
        for fraction in (0.02, 0.25, 0.50, 0.75, 1.00)
    )


def _extract_review_frames(
    path: Path,
    output: Path,
    executable: str,
    probe: dict[str, Any],
) -> list[Path]:
    directory = output / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, timestamp in enumerate(_review_timestamps(probe), start=1):
        target = directory / f"frame_{index:02d}.png"
        _run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=1280:-2:flags=lanczos",
                "-y",
                str(target),
            ]
        )
        if not target.is_file():
            raise RuntimeError(f"FFmpeg did not create review frame: {target}")
        frames.append(target)
    return frames


def _contact_sheet(frames: list[Path], target: Path, label: str) -> None:
    cards: list[Image.Image] = []
    try:
        for index, frame in enumerate(frames, start=1):
            with Image.open(frame) as raw:
                image = raw.convert("RGB")
                image.thumbnail((960, 540), Image.Resampling.LANCZOS)
            card = Image.new("RGB", (1000, 600), "#151b23")
            card.paste(image, ((card.width - image.width) // 2, 42))
            ImageDraw.Draw(card).text((20, 15), f"{label} · sample {index}", fill="white")
            cards.append(card)
        sheet = Image.new("RGB", (2000, 1800), "#0d1117")
        positions = ((0, 0), (1000, 0), (0, 600), (1000, 600), (500, 1200))
        for card, position in zip(cards, positions, strict=True):
            sheet.paste(card, position)
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target, format="PNG", optimize=True)
    finally:
        for card in cards:
            card.close()


def render_video_qa(
    *,
    paper_root: str | Path,
    output_path: str | Path,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    encoder = ffmpeg or shutil.which("ffmpeg")
    inspector = ffprobe or shutil.which("ffprobe")
    if not encoder or not inspector:
        raise RuntimeError("FFmpeg and FFprobe executables are required")
    records: list[dict[str, Any]] = []
    for path in expected_video_paths(paper_root):
        verify_sidecar(path)
        probe = _probe(path, inspector)
        frames = _extract_review_frames(path, output, encoder, probe)
        contact = output / "contact_sheets" / f"{path.stem}.png"
        _contact_sheet(frames, contact, path.name)
        atomic_write_text(
            contact.with_suffix(contact.suffix + ".sha256"),
            f"{sha256_file(contact)}  {contact.name}\n",
        )
        records.append(
            {
                "name": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "probe": probe,
                "review_frames": [str(frame) for frame in frames],
                "contact_sheet": str(contact),
                "contact_sheet_sha256": sha256_file(contact),
            }
        )
    result = {
        "schema": "zws-reversible-null-phase-v032-video-technical-qa-v1",
        "video_count": len(records),
        "all_sidecars_and_decoding_passed": len(records) == 4,
        "visual_review_required": True,
        "records": records,
    }
    atomic_write_json(output / "VIDEO_TECHNICAL_QA.json", result)
    return result
