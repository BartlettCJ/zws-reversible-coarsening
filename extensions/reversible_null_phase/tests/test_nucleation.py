from __future__ import annotations

from pathlib import Path

import numpy as np

from zws_reversible_null_phase.nucleation import (
    _event_crop_geometry,
    first_passage_from_flags,
    periodic_crop,
    save_event_window,
    topology_first_times,
)


def test_first_passage_persistence_and_censoring() -> None:
    direct = first_passage_from_flags(
        [0, 2, 4, 6], [False, True, True, True], horizon=6, persistence_followups=2
    )
    assert direct["T_hit"] == 2
    assert direct["T_persistent"] == 2
    delayed = first_passage_from_flags(
        [0, 2, 4, 6, 8, 10],
        [False, True, False, True, True, True],
        horizon=10,
        persistence_followups=2,
    )
    assert delayed["T_hit"] == 2
    assert delayed["T_persistent"] == 6
    censored = first_passage_from_flags(
        [0, 2, 4], [False, False, False], horizon=4, persistence_followups=2
    )
    assert censored["T_hit"] is None
    assert censored["right_censored_T_hit"] is True
    assert censored["censor_phase"] == 4


def test_topology_first_times() -> None:
    series = [
        {"phase": 0, "occupied_wrap_x": False, "occupied_wrap_y": True, "empty_wrap_x": True, "empty_wrap_y": False},
        {"phase": 8, "occupied_wrap_x": True, "occupied_wrap_y": True, "empty_wrap_x": True, "empty_wrap_y": False},
    ]
    assert topology_first_times(series) == {
        "occupied_wrap_x": 8,
        "occupied_wrap_y": 0,
        "empty_wrap_x": 0,
        "empty_wrap_y": None,
    }


def test_periodic_event_crop_and_hash(tmp_path: Path) -> None:
    before = np.zeros((8, 8), dtype=np.uint8)
    hit = before.copy()
    hit[7, 7] = 1
    hit[0, 0] = 1
    center, origin, crop = _event_crop_geometry(before, hit, 0.5)
    assert crop == 4
    extracted = periodic_crop(hit, origin, crop)
    assert int((extracted != 0).sum()) == 2
    tracker = {
        "event_window_states": [periodic_crop(before, origin, crop), extracted],
        "event_window_phases": [0, 8],
        "event_crop_center": list(center),
        "event_crop_origin": list(origin),
        "event_crop_size": crop,
        "event_change_total": 2,
        "event_change_in_crop": 2,
        "event_change_localization_fraction": 1.0,
    }
    payload = save_event_window(tmp_path / "event_window.npz", tracker)
    assert payload is not None
    assert (tmp_path / "event_window.npz.sha256").exists()
    loaded = np.load(tmp_path / "event_window.npz")
    assert loaded["states"].shape == (2, 4, 4)
