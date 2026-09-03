from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from .initial import SeedSpec, balanced_random


DESIGN_NAMESPACE = "ZWSLab-homogeneous-ic-v1:2026082601"
ARM_LOCALIZED = "A_localized"
ARM_HOMOGENEOUS = "B_homogeneous"
ARM_GRADIENT = "C_weak_gradient"
ARM_HIGH_DENSITY = "D_homogeneous_32pct"
MATCHED_ARMS = (ARM_LOCALIZED, ARM_HOMOGENEOUS, ARM_GRADIENT)
ALL_ARMS = (*MATCHED_ARMS, ARM_HIGH_DENSITY)


def deterministic_seed(label: str, *, namespace: str = DESIGN_NAMESPACE) -> int:
    payload = f"{namespace}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 31) - 1)


def tensor_digest(state: torch.Tensor) -> str:
    array = state.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def state_inventory(state: torch.Tensor) -> dict[str, Any]:
    if state.ndim != 2:
        raise ValueError("initial state must have shape [height,width]")
    counts = torch.bincount(state.detach().cpu().long().flatten(), minlength=4)
    values = [int(value) for value in counts.tolist()]
    charge = values[1] - values[2]
    mass = sum(values[1:])
    return {
        "state_counts": values,
        "empty_count": values[0],
        "positive_count": values[1],
        "negative_count": values[2],
        "bound_count": values[3],
        "occupancy_mass": mass,
        "occupied_fraction": mass / state.numel(),
        "total_charge": charge,
    }


def localized_reference(
    height: int,
    width: int,
    *,
    inventory_seed: int,
    localized_fraction: float = 0.45,
    positive_fraction: float = 0.12,
    bound_fraction: float = 0.08,
) -> torch.Tensor:
    return balanced_random(
        1,
        height,
        width,
        spec=SeedSpec(
            positive_fraction=positive_fraction,
            bound_fraction=bound_fraction,
            localized=True,
            localized_fraction=localized_fraction,
        ),
        seed=inventory_seed,
        device="cpu",
    )[0]


def uniformly_rearranged(reference: torch.Tensor, *, arrangement_seed: int) -> torch.Tensor:
    if reference.ndim != 2:
        raise ValueError("reference must have shape [height,width]")
    generator = torch.Generator(device="cpu").manual_seed(arrangement_seed)
    flat = reference.detach().cpu().contiguous().flatten()
    permutation = torch.randperm(flat.numel(), generator=generator)
    return flat[permutation].reshape(reference.shape)


def gradient_weights(
    height: int,
    width: int,
    *,
    modulation_amplitude: float = 0.25,
) -> np.ndarray:
    if not 0.0 < modulation_amplitude < 1.0:
        raise ValueError("modulation_amplitude must be in (0,1)")
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    half = modulation_amplitude / 2.0
    values = 1.0 + half * np.cos(2.0 * np.pi * x)[None, :]
    values = values + half * np.cos(2.0 * np.pi * y)[:, None]
    return values


def weak_gradient_rearranged(
    reference: torch.Tensor,
    *,
    arrangement_seed: int,
    modulation_amplitude: float = 0.25,
) -> torch.Tensor:
    if reference.ndim != 2:
        raise ValueError("reference must have shape [height,width]")
    reference_cpu = reference.detach().cpu().contiguous()
    flat = reference_cpu.numpy().reshape(-1)
    occupied_values = flat[flat != 0].copy()
    height, width = reference_cpu.shape
    weights = gradient_weights(
        height,
        width,
        modulation_amplitude=modulation_amplitude,
    ).reshape(-1)
    rng = np.random.default_rng(arrangement_seed)
    positions = rng.choice(
        flat.size,
        size=occupied_values.size,
        replace=False,
        p=weights / weights.sum(),
    )
    rng.shuffle(occupied_values)
    output = np.zeros(flat.size, dtype=np.uint8)
    output[positions] = occupied_values
    return torch.from_numpy(output.reshape(height, width))


def high_density_homogeneous(
    height: int,
    width: int,
    *,
    arrangement_seed: int,
    positive_fraction: float = 0.12,
    bound_fraction: float = 0.08,
) -> torch.Tensor:
    return balanced_random(
        1,
        height,
        width,
        spec=SeedSpec(
            positive_fraction=positive_fraction,
            bound_fraction=bound_fraction,
            localized=False,
            localized_fraction=1.0,
        ),
        seed=arrangement_seed,
        device="cpu",
    )[0]


def construct_initial_state(
    arm_id: str,
    *,
    height: int,
    width: int,
    inventory_seed: int,
    arrangement_seed: int,
    localized_fraction: float = 0.45,
    positive_fraction: float = 0.12,
    bound_fraction: float = 0.08,
    modulation_amplitude: float = 0.25,
) -> torch.Tensor:
    reference = localized_reference(
        height,
        width,
        inventory_seed=inventory_seed,
        localized_fraction=localized_fraction,
        positive_fraction=positive_fraction,
        bound_fraction=bound_fraction,
    )
    if arm_id == ARM_LOCALIZED:
        state = reference
    elif arm_id == ARM_HOMOGENEOUS:
        state = uniformly_rearranged(reference, arrangement_seed=arrangement_seed)
    elif arm_id == ARM_GRADIENT:
        state = weak_gradient_rearranged(
            reference,
            arrangement_seed=arrangement_seed,
            modulation_amplitude=modulation_amplitude,
        )
    elif arm_id == ARM_HIGH_DENSITY:
        state = high_density_homogeneous(
            height,
            width,
            arrangement_seed=arrangement_seed,
            positive_fraction=positive_fraction,
            bound_fraction=bound_fraction,
        )
    else:
        raise ValueError(f"unsupported initial-condition arm: {arm_id}")
    if state_inventory(state)["total_charge"] != 0:
        raise AssertionError("initial-condition construction changed exact charge")
    return state


def assert_matched_inventories(states: dict[str, torch.Tensor]) -> dict[str, Any]:
    missing = sorted(set(MATCHED_ARMS) - set(states))
    if missing:
        raise ValueError(f"matched arm states are missing: {missing}")
    inventories = {arm: state_inventory(states[arm]) for arm in MATCHED_ARMS}
    reference = inventories[ARM_LOCALIZED]["state_counts"]
    for arm_id, inventory in inventories.items():
        if inventory["state_counts"] != reference:
            raise AssertionError(f"paired inventory differs for {arm_id}")
        if inventory["total_charge"] != 0:
            raise AssertionError(f"paired charge is not zero for {arm_id}")
    return {
        "matched": True,
        "state_counts": reference,
        "occupancy_mass": inventories[ARM_LOCALIZED]["occupancy_mass"],
        "total_charge": 0,
    }


def _mode_pair_power(power: torch.Tensor, ky: int, kx: int) -> torch.Tensor:
    height, width = power.shape[-2:]
    return power[..., ky % height, kx % width] + power[..., (-ky) % height, (-kx) % width]


def low_wavenumber_diagnostics(state: torch.Tensor) -> list[dict[str, Any]]:
    if state.ndim != 3:
        raise ValueError("state must have shape [batch,height,width]")
    batch, height, width = state.shape
    field = (state != 0).to(torch.float32)
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(field, norm="ortho")
    power = spectrum.abs().square().to(torch.float64)
    power[..., 0, 0] = 0.0
    total = power.sum(dim=(-2, -1))

    fy = torch.fft.fftfreq(height, device=state.device) * height
    fx = torch.fft.fftfreq(width, device=state.device) * width
    radii = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    bands = {
        "k01_02": (1.0, 2.0),
        "k01_04": (1.0, 4.0),
        "k03_04": (3.0, 4.0),
        "k05_08": (5.0, 8.0),
        "k09_16": (9.0, 16.0),
    }
    band_values: dict[str, dict[str, torch.Tensor | int]] = {}
    for name, (lower, upper) in bands.items():
        mask = (radii >= lower) & (radii <= upper)
        integrated = power[..., mask].sum(dim=-1)
        band_values[name] = {
            "mode_count": int(mask.sum().item()),
            "integrated_power": integrated,
            "mean_power": integrated / max(int(mask.sum().item()), 1),
            "fraction_nonzero_power": integrated / total.clamp_min(torch.finfo(torch.float64).tiny),
        }

    covariance = torch.fft.ifft2(power.to(spectrum.dtype), norm="ortho").real
    zero = covariance[..., 0, 0]
    spatial_lags = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    directional_pairs = {
        "kx1_ky0": (0, 1),
        "kx0_ky1": (1, 0),
        "kx1_ky1": (1, 1),
        "kx1_ky_minus1": (-1, 1),
        "kx2_ky0": (0, 2),
        "kx0_ky2": (2, 0),
    }
    mode_powers = {
        name: _mode_pair_power(power, ky, kx)
        for name, (ky, kx) in directional_pairs.items()
    }
    flat_power = power.flatten(1)
    strongest_single = flat_power.max(dim=1).values

    maximum_radius = min(height, width) // 2
    rounded_radii = radii.round().long()
    radial_valid = (rounded_radii > 0) & (rounded_radii <= maximum_radius)
    radial_indices = rounded_radii[radial_valid]
    radial_peak_values = []
    for index in range(batch):
        shell_power = torch.bincount(
            radial_indices,
            weights=power[index][radial_valid],
            minlength=maximum_radius + 1,
        )
        radial_peak_values.append(shell_power[1:].max())
    radial_peak = torch.stack(radial_peak_values)

    output: list[dict[str, Any]] = []
    for index in range(batch):
        denominator = float(total[index].item())
        autocorrelation: dict[str, float | None] = {}
        variance_scale = float(zero[index].item())
        for lag in spatial_lags:
            if lag >= min(height, width):
                continue
            value = 0.25 * (
                covariance[index, lag, 0]
                + covariance[index, (-lag) % height, 0]
                + covariance[index, 0, lag]
                + covariance[index, 0, (-lag) % width]
            )
            autocorrelation[str(lag)] = (
                float(value.item() / variance_scale) if variance_scale > 0.0 else None
            )
        output.append(
            {
                "total_nonzero_power": denominator,
                "bands": {
                    name: {
                        "mode_count": int(values["mode_count"]),
                        "integrated_power": float(values["integrated_power"][index].item()),
                        "mean_power": float(values["mean_power"][index].item()),
                        "fraction_nonzero_power": float(
                            values["fraction_nonzero_power"][index].item()
                        ),
                    }
                    for name, values in band_values.items()
                },
                "directional_mode_pair_power": {
                    name: float(values[index].item()) for name, values in mode_powers.items()
                },
                "strongest_mode_pair_fraction_upper_bound": (
                    float(2.0 * strongest_single[index].item() / denominator)
                    if denominator > 0.0
                    else None
                ),
                "radial_peak_shell_fraction": (
                    float(radial_peak[index].item() / denominator)
                    if denominator > 0.0
                    else None
                ),
                "spatial_autocorrelation": autocorrelation,
            }
        )
    return output


def low_k_growth_ratio(
    initial: dict[str, Any],
    current: dict[str, Any],
    *,
    band: str = "k01_04",
    floor: float = 1e-30,
) -> dict[str, float]:
    initial_band = initial["bands"][band]
    current_band = current["bands"][band]
    return {
        "mean_power_ratio": float(current_band["mean_power"])
        / max(float(initial_band["mean_power"]), floor),
        "spectral_fraction_ratio": float(current_band["fraction_nonzero_power"])
        / max(float(initial_band["fraction_nonzero_power"]), floor),
    }


def simple_regular_spectral_signature(diagnostic: dict[str, Any]) -> bool:
    strongest = diagnostic.get("strongest_mode_pair_fraction_upper_bound")
    radial = diagnostic.get("radial_peak_shell_fraction")
    return bool(
        strongest is not None
        and radial is not None
        and float(strongest) >= 0.15
        and float(radial) >= 0.30
    )
