from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .alphabet import Alphabet, DEFAULT_ALPHABET
from .margolus import encode_blocks, step_batched, _to_blocks
from .motifs import patch_repetition


def _histogram(codes: torch.Tensor, bins: int) -> torch.Tensor:
    if codes.ndim < 2:
        raise ValueError("codes need a batch dimension")
    batch = codes.shape[0]
    flat = codes.long().reshape(batch, -1)
    counts = torch.zeros((batch, bins), dtype=torch.float32, device=codes.device)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
    return counts


def _entropy_counts(counts: torch.Tensor, normalizer: float | None = None) -> torch.Tensor:
    total = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    p = counts / total
    entropy = -(torch.where(p > 0, p * torch.log(p.clamp_min(1e-30)), torch.zeros_like(p))).sum(
        dim=-1
    )
    return entropy if normalizer is None else entropy / normalizer


def state_entropy(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 2:
        state = state.unsqueeze(0)
    return _entropy_counts(_histogram(state, 4), math.log(4.0))


def block_entropy(state: torch.Tensor, phase: int = 0) -> torch.Tensor:
    if state.ndim == 2:
        state = state.unsqueeze(0)
    blocks, _ = _to_blocks(state, phase)
    codes = encode_blocks(blocks)
    return _entropy_counts(_histogram(codes, 256), math.log(256.0))


def mutual_information(a: torch.Tensor, b: torch.Tensor, alphabet_size: int = 4) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("a and b must have equal shape")
    if a.ndim == 2:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    pair = a.long() * alphabet_size + b.long()
    joint = _histogram(pair, alphabet_size * alphabet_size)
    total = joint.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pxy = (joint / total).reshape(-1, alphabet_size, alphabet_size)
    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)
    denom = (px * py).clamp_min(1e-30)
    terms = torch.where(pxy > 0, pxy * torch.log(pxy.clamp_min(1e-30) / denom), 0.0)
    return terms.sum(dim=(1, 2)) / math.log(float(alphabet_size))


def spatial_mutual_information(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 2:
        state = state.unsqueeze(0)
    right = torch.roll(state, -1, dims=-1)
    down = torch.roll(state, -1, dims=-2)
    return 0.5 * (mutual_information(state, right) + mutual_information(state, down))


def temporal_mutual_information(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return mutual_information(before, after)


def _target_band(x: torch.Tensor, low: float, high: float, softness: float = 0.08) -> torch.Tensor:
    """Smooth reward for being nontrivially inside a declared band."""
    left = torch.sigmoid((x - low) / softness)
    right = torch.sigmoid((high - x) / softness)
    return left * right


@dataclass
class PopulationMetrics:
    quality: torch.Tensor
    activity: torch.Tensor
    reaction: torch.Tensor
    state_entropy: torch.Tensor
    block_entropy: torch.Tensor
    spatial_mi: torch.Tensor
    temporal_mi: torch.Tensor
    recurrence: torch.Tensor
    perturbation_distance: torch.Tensor
    patch_repetition: torch.Tensor
    descriptor: torch.Tensor
    charge_preserved: torch.Tensor
    final_state: torch.Tensor
    final_perturbed: torch.Tensor

    def cpu_dict(self) -> dict[str, list[float] | list[bool]]:
        fields = (
            "quality",
            "activity",
            "reaction",
            "state_entropy",
            "block_entropy",
            "spatial_mi",
            "temporal_mi",
            "recurrence",
            "perturbation_distance",
            "patch_repetition",
            "charge_preserved",
        )
        result: dict[str, list[float] | list[bool]] = {}
        for name in fields:
            value = getattr(self, name).detach().cpu()
            if value.dtype == torch.bool:
                result[name] = [bool(x) for x in value.tolist()]
            else:
                result[name] = [float(x) for x in value.tolist()]
        return result


def evaluate_population(
    initial: torch.Tensor,
    perturbed: torch.Tensor,
    rule_tables: torch.Tensor,
    *,
    steps: int = 128,
    burn_in: int = 32,
    sample_every: int = 4,
    start_phase: int = 0,
    alphabet: Alphabet = DEFAULT_ALPHABET,
) -> PopulationMetrics:
    """Evaluate one exact rule per batch element on GPU or CPU.

    The score is only a search heuristic. Every component is retained
    separately so a scientific report never mistakes the optimization target
    for a derived process measure.
    """
    if initial.shape != perturbed.shape or initial.ndim != 3:
        raise ValueError("initial and perturbed must be equal [B,H,W] tensors")
    batch = initial.shape[0]
    if rule_tables.ndim == 1:
        rule_tables = rule_tables.unsqueeze(0).expand(batch, -1)
    if rule_tables.shape != (batch, 256):
        raise ValueError("one [256] rule table is required per batch item")
    if not 0 <= burn_in < steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < steps")

    state = initial.clone()
    alt = perturbed.clone()
    q0 = alphabet.total_charge(state)
    q0_alt = alphabet.total_charge(alt)

    sums = {
        name: torch.zeros(batch, device=state.device, dtype=torch.float32)
        for name in (
            "activity",
            "reaction",
            "state_entropy",
            "block_entropy",
            "spatial_mi",
            "temporal_mi",
            "recurrence",
            "perturbation_distance",
        )
    }
    samples = 0
    history: list[torch.Tensor] = []

    for t in range(steps):
        next_state, reaction = step_batched(
            state, rule_tables, start_phase + t, return_reaction=True, alphabet=alphabet
        )
        next_alt = step_batched(alt, rule_tables, start_phase + t)
        if t >= burn_in and ((t - burn_in) % sample_every == 0):
            sums["activity"] += (next_state != state).float().mean(dim=(-2, -1))
            sums["reaction"] += reaction
            sums["state_entropy"] += state_entropy(next_state)
            sums["block_entropy"] += block_entropy(next_state, phase=start_phase + t + 1)
            sums["spatial_mi"] += spatial_mutual_information(next_state)
            sums["temporal_mi"] += temporal_mutual_information(state, next_state)
            sums["perturbation_distance"] += (next_state != next_alt).float().mean(
                dim=(-2, -1)
            )
            if len(history) >= 2:
                sums["recurrence"] += (next_state == history[-2]).float().mean(dim=(-2, -1))
            history.append(next_state.clone())
            if len(history) > 3:
                history.pop(0)
            samples += 1
        state, alt = next_state, next_alt  # type: ignore[assignment]

    if samples == 0:
        raise RuntimeError("no metric samples were collected")
    mean = {name: value / samples for name, value in sums.items()}
    patches = patch_repetition(state, patch=4, root_stride=2).repeated_fraction
    q1 = alphabet.total_charge(state)
    q1_alt = alphabet.total_charge(alt)
    charge_ok = (q0 == q1) & (q0_alt == q1_alt)

    # A deliberately declared search heuristic, shaped to reject dead rules,
    # full randomization, and perturbations that either vanish or saturate.
    quality = (
        0.14 * _target_band(mean["activity"], 0.015, 0.48, softness=0.055)
        + 0.11 * _target_band(mean["reaction"], 0.003, 0.42, softness=0.050)
        + 0.11 * _target_band(mean["state_entropy"], 0.16, 0.90, softness=0.055)
        + 0.11 * _target_band(mean["block_entropy"], 0.08, 0.82, softness=0.055)
        + 0.14 * _target_band(mean["spatial_mi"], 0.018, 0.62, softness=0.045)
        + 0.14 * _target_band(mean["temporal_mi"], 0.10, 0.92, softness=0.055)
        + 0.10 * _target_band(mean["recurrence"], 0.08, 0.92, softness=0.060)
        + 0.09 * _target_band(mean["perturbation_distance"], 0.001, 0.38, softness=0.045)
        + 0.06 * _target_band(patches, 0.005, 0.92, softness=0.040)
    )
    dynamics_gate = torch.sigmoid((mean["activity"] - 0.012) / 0.004) * torch.sigmoid(
        (mean["reaction"] - 0.002) / 0.0015
    )
    quality = quality * (0.08 + 0.92 * dynamics_gate)
    quality = torch.where(charge_ok, quality, torch.full_like(quality, -1e9))
    descriptor = torch.stack(
        [
            mean["activity"],
            mean["reaction"],
            mean["state_entropy"],
            mean["block_entropy"],
            mean["spatial_mi"],
            mean["temporal_mi"],
            mean["recurrence"],
            mean["perturbation_distance"],
            patches,
        ],
        dim=1,
    )
    return PopulationMetrics(
        quality=quality,
        activity=mean["activity"],
        reaction=mean["reaction"],
        state_entropy=mean["state_entropy"],
        block_entropy=mean["block_entropy"],
        spatial_mi=mean["spatial_mi"],
        temporal_mi=mean["temporal_mi"],
        recurrence=mean["recurrence"],
        perturbation_distance=mean["perturbation_distance"],
        patch_repetition=patches,
        descriptor=descriptor,
        charge_preserved=charge_ok,
        final_state=state,
        final_perturbed=alt,
    )


def novelty_scores(descriptor: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Mean distance to k nearest behavioural neighbours."""
    if descriptor.shape[0] < 2:
        return torch.zeros(descriptor.shape[0], device=descriptor.device)
    std = descriptor.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (descriptor - descriptor.mean(dim=0)) / std
    distances = torch.cdist(z, z)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.topk(distances, k=min(k, descriptor.shape[0] - 1), largest=False).values
    return nearest.mean(dim=1)
