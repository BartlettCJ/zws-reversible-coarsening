from __future__ import annotations

from pathlib import Path

from zws_reversible_null_phase.history_pdf_qa import FIGURE_IDS, expected_pdf_paths


def test_pdf_inventory_is_exact_and_stable(tmp_path: Path) -> None:
    paths = expected_pdf_paths(tmp_path)
    assert len(paths) == 7
    assert {path.stem for path in paths[:3]} == set(FIGURE_IDS)
    assert paths[3] == tmp_path / "build/manuscript.pdf"
    assert paths[4] == tmp_path / "build/supplement.pdf"
    assert paths[5] == tmp_path / "ONE_PAGE_LAY_SUMMARY.pdf"
    assert paths[6] == tmp_path / "graphical_abstract.pdf"
