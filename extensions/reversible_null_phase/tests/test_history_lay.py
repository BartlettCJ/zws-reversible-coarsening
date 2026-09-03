from __future__ import annotations

import numpy as np

from zws_reversible_null_phase.history_lay import lay_copy, render_one_page_lay_summary


def _decision() -> dict[str, object]:
    branch = {
        "classification": "INVENTORY-EQUIVALENT",
        "pooled": {
            "n": 16,
            "paired_rmst_delay": {
                "point_estimate": 0.0,
                "ci90": [-1000.0, 1000.0],
                "ci95": [-2000.0, 2000.0],
            },
        },
    }
    return {
        "branch_A": branch,
        "branch_B": branch,
        "combined_mechanism": "A. INVENTORY-DOMINATED THROUGHOUT",
        "tile_interpretation": "inventory or very local statistics dominate",
    }


def test_lay_copy_answers_required_questions_and_renders() -> None:
    decision = _decision()
    copy = lay_copy(decision)
    assert set(copy) >= {
        "world",
        "zero",
        "changes",
        "surprise",
        "test",
        "finding",
        "not_prove",
    }
    assert "Q = N+ - N-" in copy["zero"]
    assert "M = N+ + N- + Nbound" in copy["changes"]
    assert "withdrawn" in copy["finding"]
    assert "hidden spatial memory" not in copy["finding"].lower()
    hero = np.zeros((900, 1600, 3), dtype=np.uint8)
    ledger = np.zeros((900, 1600, 3), dtype=np.uint8)
    fig = render_one_page_lay_summary(
        hero_image=hero,
        ledger_image=ledger,
        decision=decision,
    )
    assert tuple(round(value, 1) for value in fig.get_size_inches()) == (8.5, 11.0)
    fig.clear()
