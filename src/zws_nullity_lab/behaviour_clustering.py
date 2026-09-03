from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from statistics import median
from typing import Any, Iterable

import numpy as np


RAW_FEATURES = (
    "log10_peak_scale_ratio",
    "log10_final_scale_ratio",
    "peak_phase_fraction",
    "occupied_fraction_change",
    "log10_final_dispersion_block64",
    "mean_reaction_density",
    "final_boundary_density",
    "final_positive_fraction",
    "final_negative_fraction",
    "final_bound_fraction",
    "log10_final_structure_peak_to_median",
    "final_coarse_recurrence_lag128",
    "stasis_rate",
    "collapse_rate",
    "noise_like_rate",
)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _log10_positive(value: object) -> float | None:
    observed = _finite(value)
    return math.log10(observed) if observed is not None and observed > 0.0 else None


def build_behaviour_features(
    coordinates: Iterable[dict[str, Any]],
    conditions: Iterable[dict[str, Any]],
    *,
    phases: int,
) -> list[dict[str, Any]]:
    """Reduce four paired trajectories to separate per-rule behavioural coordinates."""
    roles = {
        str(condition["condition_id"]): str(condition.get("family_role", "unknown"))
        for condition in conditions
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coordinates:
        grouped[str(row["condition_id"])].append(row)
    output: list[dict[str, Any]] = []
    for condition_id in sorted(roles):
        members = grouped.get(condition_id, [])
        if not members:
            raise ValueError(f"condition has no behavioural coordinates: {condition_id}")
        if len({int(row["initial_seed"]) for row in members}) != len(members):
            raise ValueError(f"condition seed block contains duplicates: {condition_id}")

        def median_available(key: str) -> float | None:
            values = [value for value in (_finite(row.get(key)) for row in members) if value is not None]
            return median(values) if values else None

        peak_ratio = median_available("peak_scale_ratio")
        final_ratio = median_available("final_scale_ratio")
        dispersion = median_available("final_dispersion_block64")
        structure = median_available("final_structure_peak_to_median")
        output.append(
            {
                "condition_id": condition_id,
                "family_role": roles[condition_id],
                "trajectory_count": len(members),
                "log10_peak_scale_ratio": _log10_positive(peak_ratio),
                "log10_final_scale_ratio": _log10_positive(final_ratio),
                "peak_phase_fraction": median_available("peak_tick") / phases,
                "occupied_fraction_change": median_available("occupied_fraction_change"),
                "log10_final_dispersion_block64": _log10_positive(dispersion),
                "mean_reaction_density": median_available("mean_reaction_density"),
                "final_boundary_density": median_available("final_boundary_density"),
                "final_positive_fraction": median_available("final_positive_fraction"),
                "final_negative_fraction": median_available("final_negative_fraction"),
                "final_bound_fraction": median_available("final_bound_fraction"),
                "log10_final_structure_peak_to_median": _log10_positive(structure),
                "final_coarse_recurrence_lag128": median_available(
                    "final_coarse_recurrence_lag128"
                ),
                "stasis_rate": sum(bool(row["stasis_flag"]) for row in members) / len(members),
                "collapse_rate": sum(bool(row["collapse_flag"]) for row in members)
                / len(members),
                "noise_like_rate": sum(bool(row["noise_like_flag"]) for row in members)
                / len(members),
            }
        )
    return output


def _matrix(feature_rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    columns = list(RAW_FEATURES)
    values = np.array(
        [
            [float(row[column]) if row[column] is not None else np.nan for column in columns]
            for row in feature_rows
        ],
        dtype=np.float64,
    )
    imputed: dict[str, int] = {}
    for index, column in enumerate(columns):
        missing = ~np.isfinite(values[:, index])
        imputed[column] = int(missing.sum())
        observed = values[~missing, index]
        if not observed.size:
            raise ValueError(f"behaviour feature is entirely missing: {column}")
        values[missing, index] = np.median(observed)
    standard_deviation = values.std(axis=0)
    quartile_span = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    retained = (standard_deviation > 1e-12) & (quartile_span > 1e-12)
    if int(retained.sum()) < 2:
        raise ValueError("fewer than two nonconstant behavioural features remain")
    return values[:, retained], [column for column, keep in zip(columns, retained, strict=True) if keep], imputed


def _standardize(values: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        center = values.mean(axis=0)
        scale = values.std(axis=0)
    elif method == "robust_iqr":
        center = np.median(values, axis=0)
        scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    else:
        raise ValueError("standardization must be zscore or robust_iqr")
    if np.any(scale <= 1e-12):
        raise ValueError("retained behaviour feature became constant")
    return (values - center) / scale


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join(("ZWSLab-behaviour-clustering-v1", str(base_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _kmeans_plus_plus(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = [values[int(rng.integers(0, values.shape[0]))]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.square(values - center).sum(axis=1) for center in centers]),
            axis=0,
        )
        total = float(distances.sum())
        if total <= 0.0:
            candidates = [
                index
                for index in range(values.shape[0])
                if not any(np.array_equal(values[index], center) for center in centers)
            ]
            if not candidates:
                raise ValueError("not enough distinct points for requested cluster count")
            centers.append(values[candidates[0]])
        else:
            centers.append(values[int(rng.choice(values.shape[0], p=distances / total))])
    return np.stack(centers)


def deterministic_kmeans(
    values: np.ndarray,
    k: int,
    *,
    seed: int,
    restarts: int = 40,
    maximum_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    if values.ndim != 2 or not 2 <= k < values.shape[0]:
        raise ValueError("k-means requires a 2-D matrix and 2 <= k < row count")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(_derived_seed(seed, k, restart))
        centers = _kmeans_plus_plus(values, k, rng)
        labels = np.zeros(values.shape[0], dtype=np.int64)
        for _ in range(maximum_iterations):
            distances = np.stack(
                [np.square(values - center).sum(axis=1) for center in centers], axis=1
            )
            updated_labels = np.argmin(distances, axis=1)
            if np.array_equal(updated_labels, labels) and _:
                break
            labels = updated_labels
            updated_centers = []
            for cluster in range(k):
                members = values[labels == cluster]
                if not members.size:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    updated_centers.append(values[farthest])
                else:
                    updated_centers.append(members.mean(axis=0))
            centers = np.stack(updated_centers)
        inertia = float(np.square(values - centers[labels]).sum())
        if best is None or inertia < best[2] - 1e-12:
            best = (labels.copy(), centers.copy(), inertia)
    if best is None:
        raise AssertionError("k-means produced no fit")
    return best


def silhouette_values(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.sqrt(np.square(values[:, None, :] - values[None, :, :]).sum(axis=2))
    output = np.zeros(values.shape[0], dtype=np.float64)
    unique = np.unique(labels)
    for index in range(values.shape[0]):
        same = labels == labels[index]
        same[index] = False
        if not same.any():
            output[index] = 0.0
            continue
        within = float(distances[index, same].mean())
        between = min(
            float(distances[index, labels == cluster].mean())
            for cluster in unique
            if cluster != labels[index]
        )
        output[index] = (between - within) / max(within, between) if max(within, between) else 0.0
    return output


def stable_cluster_analysis(
    feature_rows: list[dict[str, Any]],
    *,
    winner_condition_id: str = "winner",
    seed: int,
) -> dict[str, Any]:
    condition_ids = [str(row["condition_id"]) for row in feature_rows]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("behaviour feature identifiers are not unique")
    if winner_condition_id not in condition_ids:
        raise ValueError("frozen winner is absent from behaviour features")
    winner_index = condition_ids.index(winner_condition_id)
    raw_values, retained_features, imputed = _matrix(feature_rows)
    standardized = {
        method: _standardize(raw_values, method) for method in ("zscore", "robust_iqr")
    }
    fits: dict[tuple[str, int], dict[str, Any]] = {}
    for method, values in standardized.items():
        for k in range(2, 9):
            labels, _, inertia = deterministic_kmeans(values, k, seed=seed)
            silhouettes = silhouette_values(values, labels)
            winner_label = int(labels[winner_index])
            members = [
                condition_ids[index]
                for index in range(len(condition_ids))
                if int(labels[index]) == winner_label
            ]
            fits[(method, k)] = {
                "standardization": method,
                "k": k,
                "inertia": inertia,
                "overall_mean_silhouette": float(silhouettes.mean()),
                "winner_silhouette": float(silhouettes[winner_index]),
                "winner_cluster_mean_silhouette": float(
                    silhouettes[labels == winner_label].mean()
                ),
                "winner_cluster_members": members,
                "winner_cluster_size": len(members),
            }

    stability = []
    passing_k = []
    for k in range(2, 9):
        first = fits[("zscore", k)]
        second = fits[("robust_iqr", k)]
        first_members = set(first["winner_cluster_members"])
        second_members = set(second["winner_cluster_members"])
        jaccard = len(first_members & second_members) / len(first_members | second_members)
        separated = (
            min(first["winner_silhouette"], second["winner_silhouette"]) >= 0.25
            and min(
                first["winner_cluster_mean_silhouette"],
                second["winner_cluster_mean_silhouette"],
            )
            >= 0.20
            and min(first["winner_cluster_size"], second["winner_cluster_size"]) >= 3
        )
        passed = jaccard >= 0.75 and separated
        if passed:
            passing_k.append(k)
        stability.append(
            {
                "k": k,
                "winner_cluster_membership_jaccard": jaccard,
                "separation_thresholds_pass": separated,
                "stable_separated_pass": passed,
            }
        )

    robust_values = standardized["robust_iqr"]
    centered = robust_values - robust_values.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scores = left[:, :2] * singular[:2]
    total_variance = float(np.square(singular).sum())
    explained = [
        float(singular[index] ** 2 / total_variance) if total_variance else 0.0
        for index in range(min(2, singular.size))
    ]
    coordinates = [
        {
            **row,
            "behaviour_pc1": float(scores[index, 0]),
            "behaviour_pc2": float(scores[index, 1]) if scores.shape[1] > 1 else 0.0,
        }
        for index, row in enumerate(feature_rows)
    ]
    distinct = len(passing_k) >= 4
    return {
        "schema": "zwslab-behaviour-cluster-stability-v1",
        "retained_features": retained_features,
        "median_imputation_counts": imputed,
        "standardizations": ["zscore", "robust_iqr"],
        "cluster_counts": list(range(2, 9)),
        "fit_summaries": [fits[key] for key in sorted(fits)],
        "stability_by_cluster_count": stability,
        "passing_cluster_counts": passing_k,
        "formal_inference": {
            "distinct_cluster": distinct,
            "criterion": (
                "at least four of k=2..8 pass cross-standardization Jaccard >= 0.75, "
                "winner silhouette >= 0.25, winner-cluster mean silhouette >= 0.20, "
                "and winner-cluster size >= 3 under both standardizations"
            ),
        },
        "pca_projection": {
            "standardization": "robust_iqr",
            "explained_variance_fraction": explained,
            "coordinates": coordinates,
        },
    }
