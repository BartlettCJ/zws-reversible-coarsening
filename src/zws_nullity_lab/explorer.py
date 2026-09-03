from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.resources
import json
import mimetypes
from pathlib import Path
import threading
import webbrowser
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch

from .atomic import atomic_write_bytes, atomic_write_json, atomic_write_text, sha256_file
from .events import EventCandidate
from .visualize import state_rgb


mimetypes.add_type("application/octet-stream", ".bin")


def export_explorer_bundle(
    trajectory: torch.Tensor,
    directory: str | Path,
    *,
    rule_name: str,
    rule_sha256: str,
    events: list[EventCandidate] | None = None,
    metric_series: dict[str, list[float]] | None = None,
    fps: int = 20,
    ticks: list[int] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    states = trajectory[:, 0] if trajectory.ndim == 4 else trajectory
    if states.ndim != 3:
        raise ValueError("trajectory must be [T,H,W] or [T,B,H,W]")
    array = states.detach().cpu().to(torch.uint8).contiguous().numpy()
    frames, height, width = array.shape
    if ticks is not None and len(ticks) != frames:
        raise ValueError("ticks must contain one exact global tick per explorer frame")
    binary = root / "trajectory.bin"
    atomic_write_bytes(binary, array.tobytes(order="C"))
    series_path = root / "metric_series.json"
    atomic_write_json(series_path, metric_series or {})
    metadata = {
        "schema": "zws-trajectory-bundle-v1",
        "binary": binary.name,
        "binary_sha256": sha256_file(binary),
        "dtype": "uint8",
        "layout": "frames-height-width",
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "fps": int(fps),
        "rule_name": rule_name,
        "rule_sha256": rule_sha256,
        "states": ["empty", "positive", "negative", "bound"],
        "charges": [0, 1, -1, 0],
        "palette_rgb": [[15, 18, 24], [239, 82, 80], [72, 149, 239], [236, 200, 95]],
        "events": [event.to_dict() for event in (events or [])],
        "metric_series": series_path.name,
    }
    if ticks is not None:
        metadata["ticks"] = [int(tick) for tick in ticks]
    if extra_metadata:
        metadata["context"] = extra_metadata
    metadata_path = root / "trajectory.json"
    atomic_write_json(metadata_path, metadata)
    template = importlib.resources.files("zws_nullity_lab.web").joinpath("explorer.html")
    html_path = root / "explorer.html"
    atomic_write_text(html_path, template.read_text(encoding="utf-8"))
    return {
        "directory": str(root),
        "html": str(html_path),
        "metadata": str(metadata_path),
        "binary": str(binary),
        "metric_series": str(series_path),
    }


def find_explorer_directory(path: str | Path) -> Path:
    source = Path(path).resolve()
    candidates = [
        source,
        source / "explorer",
        source / "analysis" / "explorer",
    ]
    if source.is_file():
        candidates.insert(0, source.parent)
    for candidate in candidates:
        if (candidate / "explorer.html").is_file() and (candidate / "trajectory.json").is_file():
            return candidate
    matches = list(source.glob("**/explorer.html")) if source.is_dir() else []
    for match in matches:
        if (match.parent / "trajectory.json").is_file():
            return match.parent
    raise FileNotFoundError(f"no trajectory explorer bundle found under {source}")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[explorer] {fmt % args}")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()


def serve_explorer(
    path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> None:
    directory = find_explorer_directory(path)
    handler = partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, int(port)), handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/explorer.html"
    print(f"ZWSLab trajectory explorer: {url}")
    print("Press Ctrl+C to close the local-only server.")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def load_trajectory_bundle(path: str | Path) -> tuple[np.ndarray, dict[str, Any], Path]:
    directory = find_explorer_directory(path)
    metadata = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    binary = directory / metadata["binary"]
    if sha256_file(binary) != metadata["binary_sha256"]:
        raise ValueError("trajectory binary checksum does not match metadata")
    data = np.fromfile(binary, dtype=np.uint8)
    expected = int(metadata["frames"]) * int(metadata["height"]) * int(metadata["width"])
    if data.size != expected:
        raise ValueError(f"trajectory binary has {data.size} cells; expected {expected}")
    return data.reshape(metadata["frames"], metadata["height"], metadata["width"]), metadata, directory


def capture_bundle(
    path: str | Path,
    *,
    frame: int,
    output: str | Path,
    before: int = 24,
    after: int = 48,
    crop: tuple[int, int, int, int] | None = None,
    scale: int = 6,
    fps: int = 20,
) -> dict[str, str]:
    trajectory, metadata, _ = load_trajectory_bundle(path)
    index = max(0, min(int(frame), trajectory.shape[0] - 1))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = trajectory[index]
    if crop is None:
        crop = (0, 0, state.shape[1], state.shape[0])
    x, y, width, height = crop
    x = max(0, min(x, state.shape[1] - 1))
    y = max(0, min(y, state.shape[0] - 1))
    width = max(1, min(width, state.shape[1] - x))
    height = max(1, min(height, state.shape[0] - y))
    stem = f"capture_t{index:06d}"
    full_path = root / f"{stem}.png"
    crop_path = root / f"{stem}_crop.png"
    state_path = root / f"{stem}_state.npy"
    gif_path = root / f"{stem}_clip.gif"
    metadata_path = root / f"{stem}.json"
    Image.fromarray(state_rgb(torch.from_numpy(state.copy()), scale=scale), mode="RGB").save(full_path)
    cropped = state[y : y + height, x : x + width]
    Image.fromarray(
        state_rgb(torch.from_numpy(cropped.copy()), scale=max(scale, 8)), mode="RGB"
    ).save(crop_path)
    np.save(state_path, state, allow_pickle=False)
    first = max(0, index - max(0, before))
    last = min(trajectory.shape[0], index + max(0, after) + 1)
    frames = [state_rgb(torch.from_numpy(item.copy()), scale=scale) for item in trajectory[first:last]]
    iio.imwrite(gif_path, frames, duration=1000 / max(1, fps), loop=0)
    capture_metadata = {
        "schema": "zws-capture-bundle-v1",
        "source_rule": metadata.get("rule_name"),
        "source_rule_sha256": metadata.get("rule_sha256"),
        "frame": index,
        "tick": index,
        "clip_range": [first, last - 1],
        "crop": [x, y, width, height],
        "state_sha256": sha256_file(state_path),
        "paths": {
            "full_png": str(full_path),
            "crop_png": str(crop_path),
            "state_npy": str(state_path),
            "clip_gif": str(gif_path),
        },
    }
    atomic_write_json(metadata_path, capture_metadata)
    return {**capture_metadata["paths"], "metadata_json": str(metadata_path)}
