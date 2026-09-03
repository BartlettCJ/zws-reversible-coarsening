from __future__ import annotations

from pathlib import Path

from zws_nullity_lab.eshkol_export import export_eshkol_oracle
from zws_nullity_lab.rules import ReversibleRule


def test_rule_roundtrip(tmp_path: Path) -> None:
    rule = ReversibleRule.reactive_rotation(12, name="roundtrip")
    path = rule.save(tmp_path / "rule.json")
    loaded = ReversibleRule.load(path)
    assert loaded.name == rule.name
    assert loaded.is_valid()
    assert loaded.table.tolist() == rule.table.tolist()


def test_eshkol_oracle_contains_all_entries(tmp_path: Path) -> None:
    rule = ReversibleRule.reactive_rotation(12, name="oracle-test")
    path = export_eshkol_oracle(rule, tmp_path / "oracle.esk")
    text = path.read_text(encoding="utf-8")
    assert "PASS -- bijective and exactly charge-preserving" in text
    vector_text = text.split("(define rule '#(", 1)[1].split("))", 1)[0]
    assert len(vector_text.split()) == 256
