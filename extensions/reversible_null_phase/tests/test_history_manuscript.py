from __future__ import annotations

from pathlib import Path

import pytest

from zws_reversible_null_phase.atomic import atomic_write_json, atomic_write_text, sha256_file
from zws_reversible_null_phase.history_manuscript import (
    _main_tex,
    build_manuscript_sources,
    result_language,
)


def _decision(a: str, b: str, combined: str) -> dict[str, object]:
    def branch(label: str) -> dict[str, object]:
        return {
            "classification": label,
            "pooled": {
                "n": 16,
                "paired_rmst_delay": {
                    "point_estimate": 1000.0,
                    "ci90": [-2000.0, 3000.0],
                    "ci95": [-3000.0, 4000.0],
                },
            },
        }

    return {
        "branch_A": branch(a),
        "branch_B": branch(b),
        "combined_mechanism": combined,
        "tile_interpretation": "not_resolved",
    }


def test_legacy_result_language_is_correction_safe() -> None:
    decision = _decision(
        "INVENTORY-EQUIVALENT",
        "SPATIAL-HISTORY DELAY",
        "B. LATE-EMERGING SPATIAL MEMORY",
    )
    language = result_language(decision)
    assert language["title"].startswith("Delayed Distributed Coarsening")
    assert language["combined"] == "CLASSIFIER FAILURE; NO CAUSAL HISTORY CONCLUSION"
    assert "withdrawn" in language["result_sentence"]
    assert "LATE-EMERGING SPATIAL MEMORY" not in " ".join(language.values())
    with pytest.raises(RuntimeError, match="legacy v0.3.2 manuscript generator is retired"):
        _main_tex(decision, language)


def _write_checked(path: Path, text: str) -> None:
    atomic_write_text(path, text)
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def test_legacy_manuscript_generator_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    experiment = repo / "experiment"
    paper = repo / "paper"
    (repo / "paper/prior_art").mkdir(parents=True)
    experiment.mkdir(parents=True)
    _write_checked(repo / "paper/prior_art/references.bib", "@article{test,title={Test}}\n")
    decision = _decision(
        "INVENTORY-EQUIVALENT",
        "SPATIAL-HISTORY DELAY",
        "B. LATE-EMERGING SPATIAL MEMORY",
    )
    atomic_write_json(experiment / "SCIENTIFIC_DECISION_V032.json", decision)
    atomic_write_json(experiment / "INTEGRITY_AUDIT.json", {"passed": True})
    _write_checked(experiment / "HISTORY_ERASURE_REPORT.md", "# result\n")
    _write_checked(
        experiment / "SOURCE_TRAJECTORIES.csv",
        "source_task_id,density,seed\ns1,0.20,6201\n",
    )
    with pytest.raises(RuntimeError, match="tracked companion-paper v0.2 sources"):
        build_manuscript_sources(
            repo_root=repo,
            experiment_path=experiment,
            paper_path=paper,
        )
    assert not (paper / "main.tex").exists()
    assert not (repo / "FINAL_RESEARCH_STOP_DECISION.md").exists()
