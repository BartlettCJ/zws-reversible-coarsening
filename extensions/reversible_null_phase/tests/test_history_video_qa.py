from __future__ import annotations

from pathlib import Path

from zws_reversible_null_phase.history_video_qa import (
    VIDEO_NAMES,
    _review_timestamps,
    expected_video_paths,
)


def test_video_inventory_is_exact_and_stable(tmp_path: Path) -> None:
    paths = expected_video_paths(tmp_path)
    assert len(paths) == 4
    assert tuple(path.name for path in paths) == VIDEO_NAMES
    assert all(path.parent == tmp_path / "media" for path in paths)


def test_review_timestamps_stay_before_short_video_eof() -> None:
    probe = {
        "duration_seconds": 1.5,
        "average_frame_rate": "12/1",
        "declared_frames": "18",
    }
    timestamps = _review_timestamps(probe)
    assert len(timestamps) == 5
    assert timestamps == tuple(sorted(timestamps))
    assert timestamps[-1] == 1.35
    assert timestamps[-1] < probe["duration_seconds"]
