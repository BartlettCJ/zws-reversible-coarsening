from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import zlib

import torch


@dataclass(frozen=True)
class WindowMeasurement:
    sequence: str
    point: int
    time: int
    height: int
    width: int
    event_count: int
    active_density: float
    conditional_entropy: float
    compression_ratio: float


@dataclass(frozen=True)
class ConvergenceSummary:
    sequence: str
    candidate: str
    last_value: float
    tail_relative_range: float
    stabilized: bool


@dataclass(frozen=True)
class MeasureStressReport:
    measurements: list[WindowMeasurement]
    convergence: list[ConvergenceSummary]
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "measurements": [asdict(item) for item in self.measurements],
            "convergence": [asdict(item) for item in self.convergence],
            "verdict": self.verdict,
        }


def _center_crop(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    full_h, full_w = frames.shape[-2:]
    top = (full_h - height) // 2
    left = (full_w - width) // 2
    return frames[..., top : top + height, left : left + width]


def _conditional_entropy(frames: torch.Tensor) -> float:
    before = frames[:-1].flatten().long()
    after = frames[1:].flatten().long()
    if before.numel() == 0:
        return 0.0
    pair = before * 4 + after
    joint = torch.bincount(pair, minlength=16).double().reshape(4, 4)
    total = joint.sum()
    if total == 0:
        return 0.0
    px = joint.sum(dim=1, keepdim=True)
    value = 0.0
    for x in range(4):
        for y in range(4):
            count = float(joint[x, y])
            if count > 0:
                pxy = count / float(total)
                pyx = count / float(px[x, 0])
                value -= pxy * math.log(pyx)
    return value / math.log(4.0)


def _measure(frames: torch.Tensor, sequence: str, point: int) -> WindowMeasurement:
    changes = frames[1:] != frames[:-1]
    event_count = int(changes.sum())
    denominator = max(1, changes.numel())
    raw = frames.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()
    compressed = zlib.compress(raw, level=9)
    return WindowMeasurement(
        sequence=sequence,
        point=point,
        time=frames.shape[0],
        height=frames.shape[-2],
        width=frames.shape[-1],
        event_count=event_count,
        active_density=event_count / denominator,
        conditional_entropy=_conditional_entropy(frames.cpu()),
        compression_ratio=len(compressed) / max(1, len(raw)),
    )




def trajectory_candidate_values(trajectory: torch.Tensor) -> dict[str, float]:
    """Evaluate the three declared process-measure candidates on one trace.

    This helper is used by the redescription audit as well as the finite-window
    audit. The compression ratio is intentionally retained as a coding-sensitive
    foil; it is not expected to be invariant under every harmless recoding.
    """
    if trajectory.ndim == 4:
        if trajectory.shape[1] != 1:
            raise ValueError("candidate measures accept one trajectory batch item")
        frames = trajectory[:, 0]
    elif trajectory.ndim == 3:
        frames = trajectory
    else:
        raise ValueError("trajectory must be [T,H,W] or [T,1,H,W]")
    row = _measure(frames.detach().cpu(), "audit", 0)
    return {
        "active_density": row.active_density,
        "conditional_entropy": row.conditional_entropy,
        "compression_ratio": row.compression_ratio,
    }


def _even_values(low: int, high: int, points: int) -> list[int]:
    if high < low:
        return [high]
    values = torch.linspace(low, high, steps=max(2, points)).round().long().tolist()
    result = sorted({max(2, int(value) - int(value) % 2) for value in values})
    return [value for value in result if value <= high]


def _summaries(measurements: list[WindowMeasurement]) -> list[ConvergenceSummary]:
    summaries: list[ConvergenceSummary] = []
    for sequence in sorted({m.sequence for m in measurements}):
        rows = [m for m in measurements if m.sequence == sequence]
        for candidate in ("active_density", "conditional_entropy", "compression_ratio"):
            values = [float(getattr(row, candidate)) for row in rows]
            tail = values[-min(3, len(values)) :]
            scale = max(1e-9, abs(tail[-1]), sum(abs(v) for v in tail) / len(tail))
            relative_range = (max(tail) - min(tail)) / scale
            summaries.append(
                ConvergenceSummary(
                    sequence=sequence,
                    candidate=candidate,
                    last_value=values[-1],
                    tail_relative_range=relative_range,
                    stabilized=relative_range < 0.05,
                )
            )
    return summaries


def stress_finite_windows(trajectory: torch.Tensor, points: int = 8) -> MeasureStressReport:
    """Stress three candidate weights against inequivalent finite exhaustions.

    ``event_count`` is extensive and therefore intentionally not compared for
    stabilization. ``active_density`` is a simple process-rate candidate;
    conditional entropy is vocabulary-dependent; compression ratio is included
    as a coding-sensitive foil. No candidate is declared *the* process measure.
    """
    if trajectory.ndim == 4:
        if trajectory.shape[1] != 1:
            raise ValueError("measure stress accepts one trajectory batch item")
        frames = trajectory[:, 0]
    elif trajectory.ndim == 3:
        frames = trajectory
    else:
        raise ValueError("trajectory must be [T,H,W] or [T,1,H,W]")
    frames = frames.detach().cpu()
    total_t, full_h, full_w = frames.shape
    min_space = min(8, full_h, full_w)
    min_space -= min_space % 2
    min_time = min(9, total_t)
    times = sorted({max(2, int(x)) for x in torch.linspace(min_time, total_t, points).round()})
    spaces = _even_values(min_space, min(full_h, full_w), points)
    count = min(len(times), len(spaces))
    measurements: list[WindowMeasurement] = []

    # Balanced growth of spatial and temporal extent.
    for i in range(count):
        window = _center_crop(frames[: times[i]], spaces[i], spaces[i])
        measurements.append(_measure(window, "balanced", i))

    # Time grows first while spatial support is narrow, then space grows at full time.
    point = 0
    for time in times:
        measurements.append(
            _measure(_center_crop(frames[:time], spaces[0], spaces[0]), "time-first", point)
        )
        point += 1
    for space in spaces[1:]:
        measurements.append(
            _measure(_center_crop(frames, space, space), "time-first", point)
        )
        point += 1

    # Space grows first at a short history, then temporal extent grows.
    point = 0
    for space in spaces:
        measurements.append(
            _measure(_center_crop(frames[: times[0]], space, space), "space-first", point)
        )
        point += 1
    for time in times[1:]:
        measurements.append(
            _measure(_center_crop(frames[:time], spaces[-1], spaces[-1]), "space-first", point)
        )
        point += 1

    # Rectangular exhaustions expose shape/order dependence that a sequence of
    # centered squares can hide.
    point = 0
    for width in spaces:
        measurements.append(
            _measure(_center_crop(frames, spaces[0], width), "width-first", point)
        )
        point += 1
    for height in spaces[1:]:
        measurements.append(
            _measure(_center_crop(frames, height, spaces[-1]), "width-first", point)
        )
        point += 1

    point = 0
    for height in spaces:
        measurements.append(
            _measure(_center_crop(frames, height, spaces[0]), "height-first", point)
        )
        point += 1
    for width in spaces[1:]:
        measurements.append(
            _measure(_center_crop(frames, spaces[-1], width), "height-first", point)
        )
        point += 1

    # A toroidal coordinate shift changes which finite region is centered while
    # leaving the complete trace structurally conjugate.
    shifted = torch.roll(
        frames,
        shifts=(max(1, full_h // 4), -max(1, full_w // 4)),
        dims=(-2, -1),
    )
    for i in range(count):
        window = _center_crop(shifted[: times[i]], spaces[i], spaces[i])
        measurements.append(_measure(window, "shifted-balanced", i))

    convergence = _summaries(measurements)
    stable_all = all(item.stabilized for item in convergence if item.candidate == "active_density")
    verdict = (
        "The activity-density candidate was robust across the tested tails. This is a finite "
        "empirical result, not uniqueness of measure."
        if stable_all
        else "At least one exhaustion sequence failed the 5% tail-stability criterion; the "
        "candidate process weight is window-order dependent on this trace."
    )
    return MeasureStressReport(measurements, convergence, verdict)
