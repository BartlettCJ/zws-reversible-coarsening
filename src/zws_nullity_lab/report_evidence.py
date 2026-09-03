from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .atomic import sha256_file


REPORT_QUESTIONS = (
    "What exact local states make up the dense domains?",
    "What exact local states make up the sparse domains?",
    "What quantity/order parameter is actually phase-separating?",
    "Why does occupied fraction increase while Q remains exactly zero?",
    "What drives the enormous ~98x transient peak in domain scale?",
    "Why does the scale subsequently relax?",
    "Is there evidence of a stable characteristic macro-scale?",
    "Is there evidence of a slower macrocycle or recurrence?",
    "Does domain organisation survive a second exact local conservation law?",
    "How common is winner-like behaviour within the matched rule family?",
    "What did coarse-density dispersion get wrong as a proxy?",
    "Which observations are genuinely new relative to ordinary reversible cellular-automaton behaviour?",
    "Does anything observed justify investigating higher-order identity, self-maintenance, heredity or replication?",
    "What is the strongest scientific interpretation supported by the data?",
    "What speculative implications, including possible relevance to Zero Without Stasis, are worth considering?",
)


QUESTION_SOURCES = (
    ("state_semantics", "spatial_composition", "spatial_dynamics"),
    ("state_semantics", "spatial_composition", "spatial_dynamics"),
    ("spatial_composition", "spatial_local_charge", "existing_dynamics"),
    ("state_semantics", "existing_dynamics", "double_conservation_design"),
    ("existing_dynamics", "spatial_dynamics", "perturbation_response"),
    ("existing_dynamics", "spatial_dynamics", "perturbation_response"),
    ("existing_dynamics", "double_conservation_screen"),
    ("existing_dynamics", "macro_arrow"),
    ("double_conservation_screen", "perturbation_response"),
    ("prevalence_screen", "behaviour_clustering"),
    ("confirmation", "existing_dynamics"),
    ("confirmation", "macro_arrow", "double_conservation_screen"),
    ("all_measurements", "interpretation_boundary"),
    ("all_measurements", "formal_classifications", "interpretation_boundary"),
    ("all_measurements", "interpretation_boundary"),
)


def question_evidence_map() -> list[dict[str, Any]]:
    if len(REPORT_QUESTIONS) != 15 or len(QUESTION_SOURCES) != 15:
        raise AssertionError("the final report must retain exactly fifteen frozen questions")
    return [
        {
            "question_number": index,
            "question": question,
            "required_evidence_groups": list(sources),
            "answer_status": "requires_post_run_synthesis",
        }
        for index, (question, sources) in enumerate(
            zip(REPORT_QUESTIONS, QUESTION_SOURCES, strict=True), start=1
        )
    ]


def _read_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"unexpected report-evidence schema in {path}: {value.get('schema')!r}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required report evidence is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required report evidence table is empty: {path}")
    return rows


def spatial_robustness_evidence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, str, str]] = set()
    required_fields = {
        "stage_id",
        "role_group",
        "field",
        "contrast_count",
        "median_dense_minus_sparse",
        "q10_dense_minus_sparse",
        "q90_dense_minus_sparse",
        "sign_robustness",
    }
    for source in rows:
        if required_fields - set(source):
            raise ValueError("spatial robustness row lacks a required field")
        key = (str(source["stage_id"]), str(source["role_group"]), str(source["field"]))
        if key in seen:
            raise ValueError(f"duplicate spatial robustness coordinate: {key}")
        seen.add(key)
        output.append(
            {
                "stage_id": key[0],
                "role_group": key[1],
                "field": key[2],
                "contrast_count": int(source["contrast_count"]),
                "median_dense_minus_sparse": float(source["median_dense_minus_sparse"]),
                "q10_dense_minus_sparse": float(source["q10_dense_minus_sparse"]),
                "q90_dense_minus_sparse": float(source["q90_dense_minus_sparse"]),
                "sign_robustness": str(source["sign_robustness"]),
            }
        )
    if not output:
        raise ValueError("spatial robustness evidence is empty")
    return sorted(output, key=lambda row: (row["stage_id"], row["role_group"], row["field"]))


def selected_perturbation_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prefixes = (
        "occupancy_fraction_delta_",
        "occupancy_mass_delta_",
        "finite_domain_scale_relative_delta_",
        "structure_spectrum_total_variation_distance_",
        "state_boundary_density_delta_",
        "reaction_density_delta_",
        "state_fraction_positive_delta_",
        "state_fraction_negative_delta_",
        "state_fraction_bound_delta_",
        "exact_recurrence_lag4_delta_",
        "exact_recurrence_lag8_delta_",
        "coarse_recurrence_lag64_delta_",
    )
    identity = (
        "condition_id",
        "intervention_tick",
        "intervention_kind",
        "evaluation_tick",
        "response_horizon_phases",
        "paired_seed_count",
    )
    output = []
    for source in rows:
        row = {field: source[field] for field in identity}
        for key, value in source.items():
            if key.startswith(prefixes) and key.endswith(("_median", "_q25", "_q75", "_n")):
                row[key] = value
        if any("resilience_score" in key for key in source):
            raise ValueError("a scalar resilience score is prohibited")
        output.append(row)
    if not output:
        raise ValueError("perturbation summary evidence is empty")
    return output


def _optional_float(value: object) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def _aggregate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    identity_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in identity_fields)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(grouped.items()):
        summary: dict[str, Any] = dict(zip(identity_fields, key, strict=True))
        summary["row_count"] = len(members)
        for field in value_fields:
            values = [
                value
                for row in members
                if (value := _optional_float(row.get(field))) is not None
            ]
            summary[f"median_{field}"] = median(values) if values else None
            summary[f"{field}_n"] = len(values)
        output.append(summary)
    return output


def compact_spatial_evidence(
    compositions: Iterable[dict[str, Any]],
    dynamics: Iterable[dict[str, Any]],
    local_blocks: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    composition_fields = (
        "positive_fraction",
        "negative_fraction",
        "bound_neutral_fraction",
        "charged_fraction",
        "occupied_fraction",
        "normalized_state_entropy",
        "signed_charge_per_site",
    )
    dynamics_fields = (
        "reaction_cell_fraction",
        "empty_eulerian_net_count_change",
        "positive_eulerian_net_count_change",
        "negative_eulerian_net_count_change",
        "bound_eulerian_net_count_change",
    )
    local_fields = (
        "mean_signed_charge",
        "mean_absolute_signed_charge",
        "signed_charge_standard_deviation",
        "median_occupancy_mass",
        "median_normalized_state_entropy",
    )
    return {
        "environment_composition_medians": _aggregate_rows(
            compositions,
            identity_fields=(
                "stage_id",
                "role_group",
                "analysis_family",
                "environment",
            ),
            value_fields=composition_fields,
        ),
        "primary_environment_dynamics_medians": _aggregate_rows(
            (row for row in dynamics if str(row.get("estimable", "")).lower() == "true"),
            identity_fields=("stage_id", "role_group", "environment"),
            value_fields=dynamics_fields,
        ),
        "local_block_distribution_medians": _aggregate_rows(
            (row for row in local_blocks if int(row.get("block_count") or 0) > 0),
            identity_fields=("stage_id", "role_group", "environment"),
            value_fields=local_fields,
        ),
    }


def _compact_winner_analysis(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id",
        "stage_id",
        "seed_index",
        "initial_seed",
        "initial_scale",
        "peak_scale",
        "peak_scale_ratio",
        "peak_tick",
        "final_scale",
        "final_scale_ratio",
        "final_to_peak_ratio",
        "cross_correlations",
        "post_peak_models",
    )
    return {field: value[field] for field in fields}


def _source_record(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(resolved)
    return {"path": relative, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def collect_report_evidence(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    experiment = root / "experiments" / "domain_mechanism_20260825"
    confirmation_dir = root / "experiments" / "phase_separation_confirmation_20260825"
    existing_dir = experiment / "artifacts" / "existing_metrics"
    spatial_dir = experiment / "artifacts" / "spatial_reduction"
    double_dir = experiment / "artifacts" / "double_conservation_screen"
    prevalence_dir = experiment / "artifacts" / "prevalence_screen"
    macro_dir = experiment / "artifacts" / "discovery_macro_arrow"
    perturbation_dir = experiment / "artifacts" / "perturbation_response"

    paths = {
        "confirmation_summary": confirmation_dir / "CONFIRMATION_SUMMARY.json",
        "confirmation_integrity": confirmation_dir / "CONFIRMATION_INTEGRITY.json",
        "existing_dynamics": existing_dir / "EXISTING_MACRODYNAMICS.json",
        "spatial_robustness": spatial_dir / "CONTRAST_ROBUSTNESS.csv",
        "spatial_compositions": spatial_dir / "ENVIRONMENT_COMPOSITIONS.csv",
        "spatial_dynamics": spatial_dir / "PRIMARY_ENVIRONMENT_DYNAMICS.csv",
        "spatial_local_blocks": spatial_dir / "LOCAL_BLOCK_DISTRIBUTIONS.csv",
        "double_classification": double_dir / "DOUBLE_CONSERVATION_CLASSIFICATION.json",
        "double_conditions": double_dir / "CONDITION_SUMMARIES.csv",
        "prevalence_classification": prevalence_dir / "PREVALENCE_CLASSIFICATION.json",
        "prevalence_conditions": prevalence_dir / "CONDITION_SUMMARIES.csv",
        "behaviour_clustering": prevalence_dir / "BEHAVIOUR_CLUSTER_STABILITY.json",
        "macro_arrow": macro_dir / "REVERSIBLE_MACRO_ARROW.json",
        "perturbation_measurement": perturbation_dir / "PERTURBATION_RESPONSE_MEASUREMENT.json",
        "perturbation_summaries": perturbation_dir / "PERTURBATION_RESPONSE_SUMMARIES.csv",
        "double_exact_audit": experiment / "DOUBLE_CONSERVATION_INDEPENDENT_AUDIT.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required report evidence is missing: {path}")

    confirmation = _read_json(
        paths["confirmation_summary"], "zwslab-overnight-confirmation-summary-v1"
    )
    integrity = _read_json(paths["confirmation_integrity"])
    existing = _read_json(paths["existing_dynamics"], "zwslab-existing-macrodynamics-v1")
    spatial_robustness = spatial_robustness_evidence(
        _read_csv(paths["spatial_robustness"])
    )
    spatial_compositions = _read_csv(paths["spatial_compositions"])
    spatial_dynamics = _read_csv(paths["spatial_dynamics"])
    spatial_local_blocks = _read_csv(paths["spatial_local_blocks"])
    compact_spatial = compact_spatial_evidence(
        spatial_compositions, spatial_dynamics, spatial_local_blocks
    )
    double_classification = _read_json(
        paths["double_classification"], "zwslab-double-conservation-classification-v1"
    )
    double_conditions = _read_csv(paths["double_conditions"])
    prevalence_classification = _read_json(
        paths["prevalence_classification"], "zwslab-prevalence-classification-v1"
    )
    prevalence_conditions = _read_csv(paths["prevalence_conditions"])
    clustering = _read_json(
        paths["behaviour_clustering"], "zwslab-behaviour-cluster-stability-v1"
    )
    macro_arrow = _read_json(paths["macro_arrow"], "zwslab-reversible-macro-arrow-v1")
    perturbation_measurement = _read_json(
        paths["perturbation_measurement"], "zwslab-perturbation-response-measurement-v1"
    )
    perturbation_summaries = selected_perturbation_summaries(
        _read_csv(paths["perturbation_summaries"])
    )
    double_audit = _read_json(
        paths["double_exact_audit"], "zwslab-independent-double-conservation-audit-v1"
    )

    if confirmation.get("classification") != "D":
        raise ValueError("the report evidence must preserve strict confirmation classification D")
    if not integrity.get("valid") or int(integrity.get("error_count", -1)) != 0:
        raise ValueError("confirmation integrity did not pass")
    if int(existing.get("winner_trajectory_count", -1)) != 12:
        raise ValueError("existing dynamics does not cover all 12 winner trajectories")
    if not double_audit.get("valid") or int(double_audit.get("valid_rule_count", -1)) != 9:
        raise ValueError("independent exact-Q+M audit did not pass all nine frozen rules")
    if not macro_arrow.get("exact_reverse_verified_at_all_retained_boundaries"):
        raise ValueError("macro-arrow lacks exact reverse verification")
    if perturbation_measurement.get("formal_inference", {}).get("scalar_resilience_score") is not None:
        raise ValueError("perturbation evidence defines a prohibited scalar resilience score")

    stage7b_milestones = existing.get("stage7b_population_milestones", {})
    source_integrity = [_source_record(root, path) for path in paths.values()]
    return {
        "schema": "zwslab-domain-mechanism-report-evidence-v1",
        "source_integrity": source_integrity,
        "state_semantics": {
            "states": [
                {"code": 0, "name": "empty", "Q": 0, "M": 0},
                {"code": 1, "name": "positive", "Q": 1, "M": 1},
                {"code": 2, "name": "negative", "Q": -1, "M": 1},
                {"code": 3, "name": "bound_neutral", "Q": 0, "M": 1},
            ],
            "Q_definition": "sum of per-site values [0,+1,-1,0]",
            "M_definition": "number of nonempty sites; per-site values [0,1,1,1]",
        },
        "confirmation": {
            "classification": confirmation["classification"],
            "classification_reason": confirmation["classification_reason"],
            "trajectory_count": integrity["observed_trajectory_count"],
            "failed_runs": confirmation["failed_runs"],
            "paired_effects": confirmation["paired_effects"],
            "questions": confirmation["questions"],
        },
        "existing_dynamics": {
            "winner_trajectory_count": existing["winner_trajectory_count"],
            "stage7b_winner_trajectory_count": existing["stage7b_winner_trajectory_count"],
            "stage7b_population_milestones": stage7b_milestones,
            "stage7b_post_peak_model_preferences": existing[
                "stage7b_post_peak_model_preferences"
            ],
            "stage7b_slow_macroperiod_test": existing["stage7b_slow_macroperiod_test"],
            "winner_analyses": [
                _compact_winner_analysis(value) for value in existing["winner_analyses"]
            ],
            "interpretation_boundary": existing["interpretation_boundary"],
        },
        "spatial_analysis": {
            "contrast_robustness": spatial_robustness,
            **compact_spatial,
        },
        "double_conservation": {
            "classification": double_classification,
            "condition_summaries": double_conditions,
            "independent_exact_audit": {
                "valid": double_audit["valid"],
                "valid_rule_count": double_audit["valid_rule_count"],
                "manifest_sha256": double_audit["manifest_sha256"],
            },
        },
        "prevalence": {
            "classification": prevalence_classification,
            "condition_summaries": prevalence_conditions,
            "cluster_stability": clustering,
        },
        "macro_arrow": macro_arrow,
        "perturbation_response": {
            "measurement": perturbation_measurement,
            "coordinate_summaries": perturbation_summaries,
        },
        "question_evidence_map": question_evidence_map(),
        "interpretation_boundary": {
            "measurement": "Only exact values and declared statistics from the hashed sources above.",
            "formal_inference": "Only exact conservation/reversal proofs and frozen decision rules.",
            "interpretation_speculation": (
                "Mechanistic synthesis and Zero Without Stasis relevance must be labelled and "
                "must not be presented as uniquely entailed by morphology."
            ),
            "prohibited_unless_new_evidence": [
                "organismality",
                "heredity",
                "replication",
                "adaptive evolution",
                "agency",
                "self-repair",
            ],
        },
    }
