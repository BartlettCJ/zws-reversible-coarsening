from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable


COMPOSITION_FIELDS = (
    "positive_fraction",
    "negative_fraction",
    "bound_neutral_fraction",
    "charged_fraction",
    "occupied_fraction",
    "normalized_state_entropy",
)
STATE_NAMES = ("empty", "positive", "negative", "bound")


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    roles = [str(value) for value in record["roles"]]
    if "finite_domain_scale_peak" in roles:
        role_group = "peak"
    elif "initial" in roles:
        role_group = "initial"
    elif "final" in roles:
        role_group = "final"
    else:
        role_group = "intermediate"
    return {
        "stage_id": record["stage_id"],
        "seed_index": record.get("seed_index"),
        "initial_seed": record["initial_seed"],
        "tick": int(record["tick"]),
        "roles": ";".join(roles),
        "role_group": role_group,
        "exact_state_sha256_uint8": record["exact_state_sha256_uint8"],
    }


def _threshold_identity(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item["metadata"]
    return {
        "box_width": int(metadata["box_width"]),
        "lower_quantile": float(metadata["lower_quantile"]),
        "upper_quantile": float(metadata["upper_quantile"]),
        "lower_threshold": float(metadata["lower_threshold"]),
        "upper_threshold": float(metadata["upper_threshold"]),
        "phase_threshold_method": metadata["phase_threshold_method"],
        "phase_threshold": float(metadata["phase_threshold"]),
        "threshold_scheme": item["threshold_scheme"],
    }


def _composition_row(composition: dict[str, Any]) -> dict[str, Any]:
    return {
        "site_count": int(composition["site_count"]),
        "positive_fraction": composition["positive_fraction"],
        "negative_fraction": composition["negative_fraction"],
        "bound_neutral_fraction": composition["bound_neutral_fraction"],
        "charged_fraction": composition["charged_fraction"],
        "occupied_fraction": composition["occupied_fraction"],
        "normalized_state_entropy": composition["normalized_state_entropy"],
        "signed_charge_per_site": (
            float(composition["total_signed_charge"]) / int(composition["site_count"])
            if int(composition["site_count"])
            else None
        ),
    }


def environment_composition_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if int(record.get("diagnostic_revision", 0)) < 2:
            raise ValueError("spatial reduction requires diagnostic revision 2")
        identity = _identity(record)
        for family, items in (
            ("density_robustness", record["density_environment_robustness"]),
            ("interface_threshold_robustness", record["interface_threshold_robustness"]),
        ):
            for item in items:
                threshold = _threshold_identity(item)
                for environment, composition in item["composition"].items():
                    output.append(
                        {
                            **identity,
                            "analysis_family": family,
                            **threshold,
                            "environment": environment,
                            **_composition_row(composition),
                        }
                    )
    return output


def composition_contrast_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return dense-minus-sparse contrasts for all nine preregistered definitions."""
    output: list[dict[str, Any]] = []
    for record in records:
        if int(record.get("diagnostic_revision", 0)) < 2:
            raise ValueError("spatial reduction requires diagnostic revision 2")
        identity = _identity(record)
        for item in record["density_environment_robustness"]:
            dense = item["composition"]["dense_interior"]
            sparse = item["composition"]["sparse_interior"]
            interface = item["composition"]["interface"]
            row: dict[str, Any] = {**identity, **_threshold_identity(item)}
            for field in COMPOSITION_FIELDS:
                dense_value = dense[field]
                sparse_value = sparse[field]
                interface_value = interface[field]
                row[f"dense_{field}"] = dense_value
                row[f"sparse_{field}"] = sparse_value
                row[f"interface_{field}"] = interface_value
                row[f"dense_minus_sparse_{field}"] = (
                    float(dense_value) - float(sparse_value)
                    if dense_value is not None and sparse_value is not None
                    else None
                )
                row[f"interface_minus_interior_midpoint_{field}"] = (
                    float(interface_value) - 0.5 * (float(dense_value) + float(sparse_value))
                    if interface_value is not None
                    and dense_value is not None
                    and sparse_value is not None
                    else None
                )
            row["dense_signed_charge_per_site"] = _composition_row(dense)[
                "signed_charge_per_site"
            ]
            row["sparse_signed_charge_per_site"] = _composition_row(sparse)[
                "signed_charge_per_site"
            ]
            row["interface_signed_charge_per_site"] = _composition_row(interface)[
                "signed_charge_per_site"
            ]
            geometry = item["geometry"]
            row.update(
                {
                    "interface_crossing_edge_density": geometry["crossing_edge_density"],
                    "interface_site_fraction": geometry["interface_site_fraction"],
                    "interface_thickness_area_over_edge": geometry[
                        "area_over_edge_thickness_cells"
                    ],
                    "interface_roughness_proxy": geometry["isoperimetric_roughness_proxy"],
                }
            )
            output.append(row)
    return output


def primary_dynamics_rows(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reduce reaction/motif/neighbourhood diagnostics without particle-tracking claims."""
    environment_rows: list[dict[str, Any]] = []
    motif_rows: list[dict[str, Any]] = []
    neighbourhood_rows: list[dict[str, Any]] = []
    for record in records:
        identity = _identity(record)
        dynamics = record["primary_environment_dynamics"]
        if not dynamics["estimable"]:
            environment_rows.append(
                {
                    **identity,
                    "environment": None,
                    "estimable": False,
                    "reason": dynamics["reason"],
                }
            )
            continue
        for environment, values in dynamics["environments"].items():
            composition = values["composition"]
            matrix = [[int(value) for value in row] for row in values["eulerian_state_transition_matrix"]]
            row: dict[str, Any] = {
                **identity,
                "environment": environment,
                "estimable": True,
                "outgoing_phase": int(dynamics["outgoing_phase"]),
                "site_count": int(composition["site_count"]),
                "reaction_cell_fraction": values["reaction_cell_fraction"],
            }
            for state_code, state_name in enumerate(STATE_NAMES):
                start_count = sum(matrix[state_code])
                end_count = sum(matrix[source][state_code] for source in range(4))
                row[f"{state_name}_eulerian_start_count"] = start_count
                row[f"{state_name}_eulerian_end_count"] = end_count
                row[f"{state_name}_eulerian_net_count_change"] = end_count - start_count
            environment_rows.append(row)

            motif_counts = [int(value) for value in values["motif_2x2_histogram"]]
            motif_total = sum(motif_counts)
            for motif_code, count in enumerate(motif_counts):
                motif_rows.append(
                    {
                        **identity,
                        "environment": environment,
                        "motif_code": motif_code,
                        "count": count,
                        "fraction": count / motif_total if motif_total else None,
                    }
                )
            conditioned = values["central_state_conditioned_neighbourhoods_3x3"]
            for central_name, summary in conditioned.items():
                observed = int(summary["observed_count"])
                for rank, pattern in enumerate(summary["top"], start=1):
                    neighbourhood_rows.append(
                        {
                            **identity,
                            "environment": environment,
                            "central_state": central_name,
                            "central_state_code": int(summary["central_state_code"]),
                            "observed_central_state_count": observed,
                            "rank_within_central_state": rank,
                            "neighbourhood_code": int(pattern["code"]),
                            "count": int(pattern["count"]),
                            "conditional_fraction": (
                                int(pattern["count"]) / observed if observed else None
                            ),
                        }
                    )
    return environment_rows, motif_rows, neighbourhood_rows


def local_block_distribution_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        identity = _identity(record)
        for environment, values in record["primary_local_block_distributions"].items():
            block_count = int(values["block_count"])
            if not block_count:
                output.append(
                    {**identity, "environment": environment, "block_count": 0}
                )
                continue
            charge = values["signed_charge"]
            output.append(
                {
                    **identity,
                    "environment": environment,
                    "block_count": block_count,
                    "block_width": int(values["block_width"]),
                    "mean_signed_charge": charge["mean"],
                    "mean_absolute_signed_charge": charge["mean_absolute"],
                    "signed_charge_standard_deviation": charge["standard_deviation"],
                    "minimum_signed_charge": charge["minimum"],
                    "maximum_signed_charge": charge["maximum"],
                    "median_occupancy_mass": values["occupancy_mass_quantiles"]["0.5"],
                    "median_normalized_state_entropy": values[
                        "normalized_state_entropy_quantiles"
                    ]["0.5"],
                }
            )
    return output


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def contrast_robustness_summary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for field in COMPOSITION_FIELDS:
            value = row[f"dense_minus_sparse_{field}"]
            if value is not None:
                grouped[(str(row["stage_id"]), str(row["role_group"]), field)].append(
                    float(value)
                )
    output = []
    for (stage_id, role_group, field), values in sorted(grouped.items()):
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        same_sign_fraction = max(positive, negative) / len(values)
        if same_sign_fraction == 1.0:
            sign_label = "same_sign_all_definitions"
        elif same_sign_fraction >= 0.9:
            sign_label = "same_sign_at_least_90_percent"
        else:
            sign_label = "threshold_or_trajectory_sensitive"
        output.append(
            {
                "stage_id": stage_id,
                "role_group": role_group,
                "field": field,
                "contrast_count": len(values),
                "median_dense_minus_sparse": median(values),
                "q10_dense_minus_sparse": _quantile(values, 0.1),
                "q90_dense_minus_sparse": _quantile(values, 0.9),
                "positive_fraction": positive / len(values),
                "negative_fraction": negative / len(values),
                "sign_robustness": sign_label,
            }
        )
    return output
