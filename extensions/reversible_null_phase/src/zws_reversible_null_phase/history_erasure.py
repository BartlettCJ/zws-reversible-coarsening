from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .initial import state_sha256


BLOCK_SIZES = (2, 4, 8, 16, 32, 64, 128, 256)
MI_OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def payload_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def load_frozen_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    claimed = payload.get("payload_sha256")
    if claimed is not None and payload_sha256(payload) != claimed:
        raise RuntimeError(f"frozen payload hash mismatch: {target}")
    return payload


@dataclass(frozen=True)
class HistoryConfig:
    name: str
    rule_path: Path
    rule_sha256_uint8: str
    device: str
    output_root: Path
    lattice_size: int
    initial_densities: tuple[float, ...]
    source_seeds: tuple[int, ...]
    source_horizon: int
    branch_horizon: int
    observation_every: int
    topology_every: int
    checkpoint_every: int
    persistence_followups: int
    post_persistent_capture: int
    fixed_elapsed_phases: tuple[int, ...]
    event_relative_phases: tuple[int, ...]
    block_sizes: tuple[int, ...]
    mutual_information_offsets: tuple[int, ...]
    angular_bins: int
    anisotropy_k_min: int
    anisotropy_k_max: int
    branch_a_occupancy_fraction: float
    branch_b_lead_phases: int
    branch_b_minimum_hit_phase: int
    equivalence_margin_phases: int
    bootstrap_replicates: int
    extra_intervention_subset_seeds: tuple[int, ...]
    resource: dict[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "HistoryConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if raw.get("schema") != "zws-reversible-null-phase-history-erasure-v1":
            raise ValueError("unsupported history-erasure configuration schema")
        rule_path = Path(raw["rule_path"])
        if not rule_path.is_absolute():
            rule_path = (source.parent / rule_path).resolve()
        output_root = Path(raw["output_root"])
        if not output_root.is_absolute():
            output_root = (source.parent / output_root).resolve()
        config = cls(
            name=str(raw["name"]),
            rule_path=rule_path,
            rule_sha256_uint8=str(raw["rule_sha256_uint8"]),
            device=str(raw.get("device", "auto")),
            output_root=output_root,
            lattice_size=int(raw["lattice_size"]),
            initial_densities=tuple(float(v) for v in raw["initial_densities"]),
            source_seeds=tuple(int(v) for v in raw["source_seeds"]),
            source_horizon=int(raw["source_horizon"]),
            branch_horizon=int(raw["branch_horizon"]),
            observation_every=int(raw["observation_every"]),
            topology_every=int(raw["topology_every"]),
            checkpoint_every=int(raw["checkpoint_every"]),
            persistence_followups=int(raw["persistence_followups"]),
            post_persistent_capture=int(raw["post_persistent_capture"]),
            fixed_elapsed_phases=tuple(int(v) for v in raw["fixed_elapsed_phases"]),
            event_relative_phases=tuple(int(v) for v in raw["event_relative_phases"]),
            block_sizes=tuple(int(v) for v in raw["block_sizes"]),
            mutual_information_offsets=tuple(int(v) for v in raw["mutual_information_offsets"]),
            angular_bins=int(raw["angular_bins"]),
            anisotropy_k_min=int(raw["anisotropy_k_min"]),
            anisotropy_k_max=int(raw["anisotropy_k_max"]),
            branch_a_occupancy_fraction=float(raw["branch_a_occupancy_fraction"]),
            branch_b_lead_phases=int(raw["branch_b_lead_phases"]),
            branch_b_minimum_hit_phase=int(raw["branch_b_minimum_hit_phase"]),
            equivalence_margin_phases=int(raw["equivalence_margin_phases"]),
            bootstrap_replicates=int(raw["bootstrap_replicates"]),
            extra_intervention_subset_seeds=tuple(int(v) for v in raw["extra_intervention_subset_seeds"]),
            resource=dict(raw["resource"]),
            source_path=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.lattice_size <= 0 or self.lattice_size % 256:
            raise ValueError("lattice size must be a positive multiple of 256")
        if self.source_horizon % self.observation_every or self.branch_horizon % self.observation_every:
            raise ValueError("horizons must lie on the observation grid")
        if self.checkpoint_every % self.observation_every:
            raise ValueError("checkpoint interval must be an observation multiple")
        if self.topology_every != self.observation_every:
            raise ValueError("topology and observation intervals must match")
        if self.block_sizes != BLOCK_SIZES or self.mutual_information_offsets != MI_OFFSETS:
            raise ValueError("multiscale schedule differs from the frozen plan")
        if self.persistence_followups != 2:
            raise ValueError("T_persistent requires exactly two followups")
        if self.branch_b_lead_phases != 32768 or self.equivalence_margin_phases != 8192:
            raise ValueError("frozen branch-B or equivalence interval changed")

    def file_sha256(self) -> str:
        return hashlib.sha256(self.source_path.read_bytes()).hexdigest()


def as_uint8_state(state: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        array = state.detach().cpu().numpy()
    else:
        array = np.asarray(state)
    if array.ndim != 2:
        raise ValueError("state must be a two-dimensional lattice")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 3):
        raise ValueError("state values must lie in 0..3")
    return np.ascontiguousarray(array, dtype=np.uint8)


def state_histogram(state: torch.Tensor | np.ndarray) -> np.ndarray:
    return np.bincount(as_uint8_state(state).reshape(-1), minlength=4).astype(np.int64)


def q_m_from_histogram(histogram: Iterable[int]) -> tuple[int, int]:
    counts = tuple(int(v) for v in histogram)
    if len(counts) != 4:
        raise ValueError("histogram must contain four counts")
    return counts[1] - counts[2], counts[1] + counts[2] + counts[3]


def single_site_entropy(histogram: Iterable[int]) -> float:
    counts = np.asarray(tuple(histogram), dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(4.0))


def apply_global_shuffle(state: torch.Tensor | np.ndarray, permutation: Iterable[int]) -> np.ndarray:
    source = as_uint8_state(state)
    order = np.asarray(tuple(permutation) if not isinstance(permutation, np.ndarray) else permutation, dtype=np.int64)
    if order.shape != (source.size,) or not np.array_equal(np.sort(order), np.arange(source.size)):
        raise ValueError("global permutation is not a complete index permutation")
    return np.ascontiguousarray(source.reshape(-1)[order].reshape(source.shape), dtype=np.uint8)


def global_permutation(seed: int, site_count: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(int(seed))).permutation(site_count).astype("<i8", copy=False)


def permutation_digest(permutation: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(permutation, dtype="<i8"))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def translate_state(state: torch.Tensor | np.ndarray, offset_yx: Iterable[int]) -> np.ndarray:
    source = as_uint8_state(state)
    offset = tuple(int(v) for v in offset_yx)
    if len(offset) != 2:
        raise ValueError("translation offset must contain y and x")
    return np.ascontiguousarray(np.roll(source, shift=offset, axis=(0, 1)), dtype=np.uint8)


def apply_tile_permutation(
    state: torch.Tensor | np.ndarray,
    *,
    tile_size: int,
    permutation: Iterable[int],
    temporary_even_offset_yx: Iterable[int],
) -> np.ndarray:
    source = as_uint8_state(state)
    height, width = source.shape
    if height != width or height % tile_size:
        raise ValueError("tile size must divide the square lattice")
    offset = tuple(int(v) for v in temporary_even_offset_yx)
    if len(offset) != 2 or any(value % 2 for value in offset) or tile_size % 2:
        raise ValueError("tile dimensions and temporary offsets must be even")
    grid = height // tile_size
    order = np.asarray(tuple(permutation), dtype=np.int64)
    tile_count = grid * grid
    if order.shape != (tile_count,) or not np.array_equal(np.sort(order), np.arange(tile_count)):
        raise ValueError("tile permutation is not complete")
    shifted = np.roll(source, shift=offset, axis=(0, 1))
    tiles = shifted.reshape(grid, tile_size, grid, tile_size).transpose(0, 2, 1, 3)
    moved = tiles.reshape(tile_count, tile_size, tile_size)[order]
    rebuilt = moved.reshape(grid, grid, tile_size, tile_size).transpose(0, 2, 1, 3).reshape(height, width)
    return np.ascontiguousarray(np.roll(rebuilt, shift=(-offset[0], -offset[1]), axis=(0, 1)), dtype=np.uint8)


def intervention_audit(
    source: torch.Tensor | np.ndarray,
    altered: torch.Tensor | np.ndarray,
    *,
    source_phase: int,
    altered_phase: int,
) -> dict[str, Any]:
    before = as_uint8_state(source)
    after = as_uint8_state(altered)
    before_hist = state_histogram(before)
    after_hist = state_histogram(after)
    before_q, before_m = q_m_from_histogram(before_hist)
    after_q, after_m = q_m_from_histogram(after_hist)
    result = {
        "source_state_sha256": state_sha256(before),
        "altered_state_sha256": state_sha256(after),
        "source_histogram": before_hist.tolist(),
        "altered_histogram": after_hist.tolist(),
        "source_Q": before_q,
        "altered_Q": after_q,
        "source_M": before_m,
        "altered_M": after_m,
        "source_single_site_entropy": single_site_entropy(before_hist),
        "altered_single_site_entropy": single_site_entropy(after_hist),
        "source_phase": int(source_phase),
        "altered_phase": int(altered_phase),
        "shape_preserved": before.shape == after.shape,
        "histogram_preserved": bool(np.array_equal(before_hist, after_hist)),
        "Q_preserved": before_q == after_q,
        "M_preserved": before_m == after_m,
        "single_site_entropy_preserved": single_site_entropy(before_hist) == single_site_entropy(after_hist),
        "partition_phase_preserved": int(source_phase) % 2 == int(altered_phase) % 2,
    }
    result["valid"] = bool(
        result["shape_preserved"]
        and result["histogram_preserved"]
        and result["Q_preserved"]
        and result["M_preserved"]
        and result["single_site_entropy_preserved"]
        and result["partition_phase_preserved"]
    )
    if not result["valid"]:
        raise RuntimeError("intervention failed the exact inventory/phase audit")
    return result


def block_entropy_hierarchy(
    state: torch.Tensor | np.ndarray,
    block_sizes: Iterable[int] = BLOCK_SIZES,
) -> dict[str, float]:
    source = as_uint8_state(state)
    height, width = source.shape
    requested = tuple(int(v) for v in block_sizes)
    if any(v <= 0 or v & (v - 1) or height % v or width % v for v in requested):
        raise ValueError("block sizes must be powers of two dividing the lattice")
    counts = np.eye(4, dtype=np.int32)[source]
    result: dict[str, float] = {}
    block = 1
    while block < max(requested):
        counts = counts.reshape(counts.shape[0] // 2, 2, counts.shape[1] // 2, 2, 4).sum(axis=(1, 3))
        block *= 2
        if block not in requested:
            continue
        probabilities = counts.astype(np.float64) / float(block * block)
        terms = np.zeros_like(probabilities)
        np.log(probabilities, out=terms, where=probabilities > 0)
        entropies = -(probabilities * terms).sum(axis=-1) / math.log(4.0)
        result[str(block)] = float(entropies.mean())
    return result


def categorical_mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.uint8).reshape(-1)
    right = np.asarray(second, dtype=np.uint8).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("mutual-information arrays must have equal shape")
    joint = np.bincount((left.astype(np.int64) * 4 + right).reshape(-1), minlength=16).reshape(4, 4)
    probabilities = joint.astype(np.float64) / max(1, left.size)
    p_left = probabilities.sum(axis=1, keepdims=True)
    p_right = probabilities.sum(axis=0, keepdims=True)
    expected = p_left @ p_right
    mask = probabilities > 0
    return float((probabilities[mask] * np.log(probabilities[mask] / expected[mask])).sum())


def spatial_mutual_information(
    state: torch.Tensor | np.ndarray,
    offsets: Iterable[int] = MI_OFFSETS,
) -> dict[str, dict[str, float]]:
    source = as_uint8_state(state)
    result: dict[str, dict[str, float]] = {}
    for offset in (int(v) for v in offsets):
        if offset <= 0 or offset >= min(source.shape):
            raise ValueError("mutual-information offsets must be within the torus")
        mi_y = categorical_mutual_information(source, np.roll(source, shift=-offset, axis=0))
        mi_x = categorical_mutual_information(source, np.roll(source, shift=-offset, axis=1))
        result[str(offset)] = {"x": mi_x, "y": mi_y, "mean": (mi_x + mi_y) / 2.0}
    return result


def structure_factor_2d(state: torch.Tensor | np.ndarray) -> np.ndarray:
    source = as_uint8_state(state)
    occupancy = (source != 0).astype(np.float64)
    occupancy -= occupancy.mean()
    transformed = np.fft.fft2(occupancy)
    return np.ascontiguousarray((transformed.real**2 + transformed.imag**2) / source.size, dtype=np.float32)


def spectral_summary(
    state: torch.Tensor | np.ndarray,
    *,
    low_k_max: int = 8,
    angular_bins: int = 16,
    anisotropy_k_min: int = 1,
    anisotropy_k_max: int = 32,
) -> tuple[dict[str, Any], np.ndarray]:
    source = as_uint8_state(state)
    if source.shape[0] != source.shape[1]:
        raise ValueError("spectral summary requires a square lattice")
    size = source.shape[0]
    power = structure_factor_2d(source)
    frequency = np.fft.fftfreq(size) * size
    ky, kx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(kx * kx + ky * ky)
    nonzero = radius > 0
    total = float(power[nonzero].sum(dtype=np.float64))
    low_mask = nonzero & (radius <= low_k_max)
    low_power = float(power[low_mask].sum(dtype=np.float64))
    second_denominator = float((power[nonzero] * (radius[nonzero] ** 2)).sum(dtype=np.float64))
    correlation_length = (
        math.sqrt(total / second_denominator) * size / (2.0 * math.pi)
        if total > 0 and second_denominator > 0
        else 0.0
    )
    angle = np.mod(np.arctan2(ky, kx), math.pi)
    angular_mask = (radius >= anisotropy_k_min) & (radius <= anisotropy_k_max)
    indices = np.floor(angle / math.pi * angular_bins).astype(np.int64) % angular_bins
    angular_power = np.zeros(angular_bins, dtype=np.float64)
    angular_counts = np.zeros(angular_bins, dtype=np.int64)
    np.add.at(angular_power, indices[angular_mask], power[angular_mask].astype(np.float64))
    np.add.at(angular_counts, indices[angular_mask], 1)
    angular_power = np.divide(
        angular_power,
        angular_counts,
        out=np.zeros_like(angular_power),
        where=angular_counts > 0,
    )
    maximum = float(angular_power.max()) if angular_power.size else 0.0
    minimum = float(angular_power.min()) if angular_power.size else 0.0
    anisotropy = (maximum - minimum) / (maximum + minimum) if maximum + minimum > 0 else 0.0
    centers = (np.arange(angular_bins) + 0.5) * math.pi / angular_bins

    def nearest(target: float) -> int:
        distance = np.abs(((centers - target + math.pi / 2) % math.pi) - math.pi / 2)
        return int(np.argmin(distance))

    axis = float(np.mean(angular_power[[nearest(0.0), nearest(math.pi / 2)]]))
    diagonal = float(np.mean(angular_power[[nearest(math.pi / 4), nearest(3 * math.pi / 4)]]))
    contrast = (diagonal - axis) / (diagonal + axis) if diagonal + axis > 0 else 0.0
    summary = {
        "low_k_max": int(low_k_max),
        "low_k_power": low_power,
        "low_k_fraction": low_power / total if total > 0 else 0.0,
        "total_nonzero_power": total,
        "correlation_length": correlation_length,
        "angular_bins": int(angular_bins),
        "anisotropy_k_min": int(anisotropy_k_min),
        "anisotropy_k_max": int(anisotropy_k_max),
        "angular_power": angular_power.tolist(),
        "anisotropy": float(anisotropy),
        "diagonal_axis_contrast": float(contrast),
    }
    return summary, power


def interface_density(state: torch.Tensor | np.ndarray) -> float:
    occupancy = as_uint8_state(state) != 0
    horizontal = np.mean(occupancy != np.roll(occupancy, shift=-1, axis=1))
    vertical = np.mean(occupancy != np.roll(occupancy, shift=-1, axis=0))
    return float((horizontal + vertical) / 2.0)


def multiscale_summary(state: torch.Tensor | np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    source = as_uint8_state(state)
    histogram = state_histogram(source)
    q_value, m_value = q_m_from_histogram(histogram)
    spectrum, power = spectral_summary(source)
    return (
        {
            "state_sha256": state_sha256(source),
            "histogram": histogram.tolist(),
            "Q": q_value,
            "M": m_value,
            "single_site_entropy": single_site_entropy(histogram),
            "block_entropy": block_entropy_hierarchy(source),
            "mutual_information": spatial_mutual_information(source),
            "interface_density": interface_density(source),
            **spectrum,
        },
        power,
    )


@dataclass(frozen=True)
class SourceCalibration:
    initial_finite_empty_scale: float
    initial_charge: int
    min_peak_scale_ratio: float = 4.0
    min_current_scale_ratio: float = 2.0
    min_peak_largest_finite_empty_fraction: float = 0.001
    initial_scale_floor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_initial_metric(cls, metric: dict[str, Any]) -> "SourceCalibration":
        return cls(
            initial_finite_empty_scale=max(float(metric["finite_empty_scale"]), 1.0),
            initial_charge=int(metric["charge"]),
        )


def strong_domain_prefix(
    series: list[dict[str, Any]],
    calibration: SourceCalibration,
) -> dict[str, Any]:
    if not series:
        raise ValueError("strong-domain classification requires at least one metric")
    scales = np.asarray([float(row["finite_empty_scale"]) for row in series], dtype=np.float64)
    ratios = scales / calibration.initial_finite_empty_scale
    peak_index = int(np.argmax(ratios))
    charge_exact = all(int(row["charge"]) == calibration.initial_charge for row in series)
    strong = bool(
        ratios[peak_index] >= calibration.min_peak_scale_ratio
        and ratios[-1] >= calibration.min_current_scale_ratio
        and float(series[peak_index]["largest_finite_empty_fraction"])
        >= calibration.min_peak_largest_finite_empty_fraction
        and charge_exact
    )
    return {
        "strong_domain": strong,
        "initial_finite_empty_scale": calibration.initial_finite_empty_scale,
        "peak_scale_ratio": float(ratios[peak_index]),
        "current_scale_ratio": float(ratios[-1]),
        "peak_phase": int(series[peak_index]["phase"]),
        "peak_largest_finite_empty_fraction": float(series[peak_index]["largest_finite_empty_fraction"]),
        "charge_exact": charge_exact,
    }


def time_to_event(
    phases: Iterable[int],
    flags: Iterable[bool],
    *,
    horizon: int,
    persistence_followups: int = 2,
) -> dict[str, Any]:
    phase_list = [int(v) for v in phases]
    flag_list = [bool(v) for v in flags]
    if len(phase_list) != len(flag_list) or phase_list != sorted(set(phase_list)):
        raise ValueError("event phases and flags must be equal-length and strictly increasing")
    hit = next((phase for phase, flag in zip(phase_list, flag_list, strict=True) if flag), None)
    persistent = None
    width = persistence_followups + 1
    for index in range(len(flag_list) - width + 1):
        if all(flag_list[index : index + width]):
            persistent = phase_list[index]
            break
    return {
        "T_hit": hit,
        "hit_observed": hit is not None,
        "T_persistent": persistent,
        "persistent_observed": persistent is not None,
        "censor_phase": int(horizon),
        "right_censored_T_hit": hit is None,
        "right_censored_T_persistent": persistent is None,
        "persistence_followups": int(persistence_followups),
    }


def residual_record(
    *,
    branch_phase: int,
    absolute_hit: int | None,
    absolute_persistent: int | None,
    branch_horizon: int,
) -> dict[str, Any]:
    if absolute_hit is not None and absolute_hit < branch_phase:
        raise ValueError("hit predates branch")
    if absolute_persistent is not None and absolute_persistent < branch_phase:
        raise ValueError("persistent event predates branch")
    return {
        "R_hit": None if absolute_hit is None else int(absolute_hit - branch_phase),
        "hit_observed": absolute_hit is not None,
        "R_persistent": None if absolute_persistent is None else int(absolute_persistent - branch_phase),
        "persistent_observed": absolute_persistent is not None,
        "censor_elapsed": int(branch_horizon),
        "right_censored_R_hit": absolute_hit is None,
        "right_censored_R_persistent": absolute_persistent is None,
    }
