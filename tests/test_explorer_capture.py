from __future__ import annotations

from pathlib import Path


from zws_nullity_lab.events import scan_interesting_events
from zws_nullity_lab.explorer import capture_bundle, export_explorer_bundle, load_trajectory_bundle
from zws_nullity_lab.initial import paired_impulse
from zws_nullity_lab.margolus import run_trajectory
from zws_nullity_lab.rules import ReversibleRule


def test_explorer_bundle_and_capture(tmp_path: Path) -> None:
    rule = ReversibleRule.reactive_rotation(12)
    trajectory = run_trajectory(paired_impulse(16, 16).unsqueeze(0), rule, 24)
    events, series = scan_interesting_events(
        trajectory, rule, max_events=4, min_separation=2, padding=2
    )
    paths = export_explorer_bundle(
        trajectory,
        tmp_path / "explorer",
        rule_name=rule.name,
        rule_sha256="abc123",
        events=events,
        metric_series=series,
    )
    assert Path(paths["html"]).exists()
    loaded, metadata, _ = load_trajectory_bundle(tmp_path / "explorer")
    assert loaded.shape == (25, 16, 16)
    assert metadata["rule_name"] == rule.name
    capture = capture_bundle(
        tmp_path / "explorer", frame=8, output=tmp_path / "capture", before=2, after=3
    )
    for path in capture.values():
        assert Path(path).exists()
