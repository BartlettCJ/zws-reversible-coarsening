from __future__ import annotations

import json
from pathlib import Path

from zws_nullity_lab.analysis import analyze_rule
from zws_nullity_lab.config import load_config
from zws_nullity_lab.rules import ReversibleRule
from zws_nullity_lab.search import run_search


def test_small_search_and_analysis(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "smoke.yaml")
    result = run_search(config, tmp_path / "run")
    assert Path(result.best_rule_path).exists()
    rule = ReversibleRule.load(result.best_rule_path)
    assert rule.is_valid()
    analysis = analyze_rule(rule, config, tmp_path / "run" / "analysis")
    assert analysis["local_rule_audit"]["valid"]
    assert analysis["trace_audit"]["state_equal_after_reverse"]
    assert analysis["redescription"]["passed"]
    assert analysis["coarse_graining"]["passed_additive_composition"]
    parsed = json.loads((tmp_path / "run" / "analysis" / "analysis.json").read_text())
    assert parsed["schema"] == "zws-nullity-analysis-v2"
    assert Path(parsed["paths"]["explorer_html"]).exists()
    assert Path(parsed["paths"]["trajectory_binary"]).exists()
    assert Path(parsed["paths"]["events_json"]).exists()

    # Force stage-resume rather than the completed-analysis fast path.
    (tmp_path / "run" / "analysis" / "analysis.json").unlink()
    resumed = analyze_rule(rule, config, tmp_path / "run" / "analysis", resume=True)
    assert resumed["schema"] == "zws-nullity-analysis-v2"
    assert Path(resumed["paths"]["explorer_html"]).exists()
