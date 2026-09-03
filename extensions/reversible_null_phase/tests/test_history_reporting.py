from __future__ import annotations

import pytest

import zws_reversible_null_phase.history_reporting as reporting


def paired_rows(delay: int, *, count: int = 16) -> list[dict]:
    return [
        {
            "cluster": f"density_{0.20 if index < count // 2 else 0.24:.2f}_seed_{index}",
            "density": 0.20 if index < count // 2 else 0.24,
            "R_hit": 32768 + delay,
            "hit_observed": True,
            "censor_elapsed": reporting.HORIZON,
            "untouched_R_hit": 32768,
            "untouched_hit_observed": True,
            "untouched_censor_elapsed": reporting.HORIZON,
        }
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def compact_bootstrap(monkeypatch) -> None:
    monkeypatch.setattr(reporting, "BOOTSTRAP_REPLICATES", 200)


def test_frozen_classification_inventory_equivalent() -> None:
    result = reporting._classify("B", paired_rows(0))
    assert result["classification"] == "INVENTORY-EQUIVALENT"
    assert result["pooled"]["support_counts"]["equivalent"] == 16


def test_frozen_classification_spatial_history_delay() -> None:
    result = reporting._classify("B", paired_rows(32768))
    assert result["classification"] == "SPATIAL-HISTORY DELAY"
    assert result["pooled"]["paired_rmst_delay"]["ci95"][0] > reporting.MARGIN


def test_frozen_classification_spatial_history_inhibition() -> None:
    result = reporting._classify("B", paired_rows(-16384))
    assert result["classification"] == "SPATIAL-HISTORY INHIBITION"
    assert result["pooled"]["paired_rmst_delay"]["ci95"][1] < -reporting.MARGIN


def test_censored_branch_is_not_treated_as_horizon_event() -> None:
    row = paired_rows(0, count=1)[0]
    row["R_hit"] = None
    row["hit_observed"] = False
    row["censor_elapsed"] = reporting.HORIZON
    record = reporting._analysis_record(row)
    assert record["altered_time"] == reporting.HORIZON
    assert not record["altered_observed"]


def test_tile_interpretation_uses_frozen_order() -> None:
    near = {"paired_rmst_delay": {"ci90": [-100, 100], "ci95": [-200, 200]}}
    delayed = {
        "paired_rmst_delay": {
            "ci90": [10000, 20000],
            "ci95": [9000, 21000],
        }
    }
    assert (
        reporting._tile_interpretation(near, delayed, "SPATIAL-HISTORY DELAY")
        == "causally useful information is largely retained below about 32 cells"
    )


def test_density_direction_agreement_is_evaluated_within_branch() -> None:
    result = {
        "by_density": {
            "0.20": {"paired_rmst_delay": {"point_estimate": 100.0}},
            "0.24": {"paired_rmst_delay": {"point_estimate": 200.0}},
        }
    }
    assert reporting._density_direction_agreement(result)
    result["by_density"]["0.24"]["paired_rmst_delay"]["point_estimate"] = -200.0
    assert not reporting._density_direction_agreement(result)


def test_density_direction_agreement_requires_both_frozen_densities() -> None:
    result = {
        "by_density": {
            "0.20": {"paired_rmst_delay": {"point_estimate": 100.0}},
        }
    }
    assert not reporting._density_direction_agreement(result)


def test_translation_null_requires_exact_state_metrics_and_event_clock() -> None:
    branch = {
        "translation_null_passed": True,
        "residual_time_to_event": {
            "R_hit": 32768,
            "hit_observed": True,
            "R_persistent": 32768,
            "persistent_observed": True,
        },
        "untouched_reference": {
            "R_hit": 32768,
            "hit_observed": True,
            "R_persistent": 32768,
            "persistent_observed": True,
        },
    }
    assert reporting._translation_null_exact(branch)
    branch["residual_time_to_event"]["R_hit"] = 22528
    assert not reporting._translation_null_exact(branch)


def test_report_answers_branch_direction_in_plain_language() -> None:
    answers = reporting._answer_text(
        {"classification": "INVENTORY-EQUIVALENT"},
        {"classification": "SPATIAL-HISTORY DELAY"},
        "important organisation lies approximately between 32 and 128 cells",
        "B. LATE-EMERGING SPATIAL MEMORY",
        {"A": True, "B": False},
    )
    assert "unchanged within" in answers[0]
    assert "delayed the remaining" in answers[1]
    assert "agreed at Branch A" in answers[2]
    assert "did not agree at Branch B" in answers[2]


def test_integrity_accepts_only_exactly_audited_plan_lineage() -> None:
    source_tasks = [{"task_id": f"source-{index}"} for index in range(16)]
    current_branch_tasks = [
        {
            "task_id": f"branch-{index}",
            "intervention": "global_shuffle",
            "intervention_audit_sha256": "current-audit",
        }
        for index in range(72)
    ]
    historical_branch_tasks = [
        {**task, "intervention_audit_sha256": "historical-audit"}
        for task in current_branch_tasks
    ]
    sources = [
        {
            "task": task,
            "winner_sha256": reporting.RULE_SHA256,
            "series": [{"charge": 0}],
        }
        for task in source_tasks
    ]
    branches = [
        {
            "task": task,
            "winner_sha256": reporting.RULE_SHA256,
            "realized_plan_payload_sha256": "historical-plan",
            "series": [{"charge": 0}],
            "translation_null_passed": None,
            "calibration_inherited_from_source": True,
            "denominator_reset_at_branch": False,
        }
        for task in historical_branch_tasks
    ]
    audits = [
        {
            "valid": True,
            "source_histogram": [1, 1, 1, 1],
            "altered_histogram": [1, 1, 1, 1],
            "source_Q": 0,
            "altered_Q": 0,
            "source_M": 3,
            "altered_M": 3,
        }
        for _ in current_branch_tasks
    ]
    resume_audit = {
        "passed": True,
        "prior_realized_plan_payload_sha256": "historical-plan",
        "current_realized_plan_payload_sha256": "current-plan",
        "mismatch_counts": [0, 0],
        "reference_mismatch_counts": [0, 0],
    }
    result = reporting._integrity(
        {"source_tasks": source_tasks},
        {"entries": current_branch_tasks, "entry_count": 72},
        {
            "prepared_entries": current_branch_tasks,
            "prepared_count": 72,
            "ineligible_count": 0,
            "payload_sha256": "current-plan",
        },
        sources,
        branches,
        audits,
        resume_audit,
    )
    assert result["passed"] is True
    assert result["checks"]["branch_tasks_match_realized_scientific_core"] is True
    assert result["checks"]["realized_plan_lineage_valid"] is True
    assert result["resume_replay_audit_used"] is True

    resume_audit["mismatch_counts"] = [1, 0]
    failed = reporting._integrity(
        {"source_tasks": source_tasks},
        {"entries": current_branch_tasks, "entry_count": 72},
        {
            "prepared_entries": current_branch_tasks,
            "prepared_count": 72,
            "ineligible_count": 0,
            "payload_sha256": "current-plan",
        },
        sources,
        branches,
        audits,
        resume_audit,
    )
    assert failed["checks"]["realized_plan_lineage_valid"] is False
    assert failed["passed"] is False
