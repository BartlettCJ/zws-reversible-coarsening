from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from zws_reversible_null_phase.history_erasure import state_sha256
from zws_reversible_null_phase import history_figures
from zws_reversible_null_phase.history_figures import figure_output_paths
from zws_reversible_null_phase.history_media import reconstruct_untouched_media_states
from zws_reversible_null_phase.history_package import _require_completed_run
from zws_reversible_null_phase.history_publication import (
    STATE_RGB,
    aligned_densest_crop,
    build_history_dashboard,
    build_history_explorer,
    explorer_frame,
    make_panel_record,
    state_png_bytes,
    validate_panel_record,
)
from zws_reversible_null_phase.history_video import render_history_frame


def _panel(state: np.ndarray) -> dict[str, object]:
    return make_panel_record(
        source_task="source-test",
        seed=6201,
        density=0.2,
        branch_point="B",
        intervention="global_shuffle",
        phase=32768,
        elapsed=0,
        requested_elapsed=0,
        state_hash=state_sha256(state),
        crop=(0, 0, state.shape[1], state.shape[0]),
        source_path=None,
    )


def test_aligned_densest_crop_is_deterministic_and_tie_top_left() -> None:
    state = np.zeros((8, 8), dtype=np.uint8)
    state[0:4, 4:8] = 1
    state[4:8, 0:4] = 2
    assert aligned_densest_crop(state, 4) == (4, 0, 4, 4)
    state[4:8, 4:8] = 3
    assert aligned_densest_crop(state, 4, bounds=(4, 0, 4, 8)) == (4, 0, 4, 4)


def test_state_png_is_lossless_frozen_palette() -> None:
    state = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    rgb = np.asarray(Image.open(BytesIO(state_png_bytes(state))).convert("RGB"))
    assert np.array_equal(rgb, STATE_RGB[state])


def test_panel_metadata_enforces_rule_and_nearest_sampling() -> None:
    state = np.zeros((2, 2), dtype=np.uint8)
    record = _panel(state)
    validate_panel_record(record)
    record["rule_hash"] = "bad"
    with pytest.raises(RuntimeError, match="frozen winner"):
        validate_panel_record(record)


def test_dashboard_and_explorer_are_self_contained_local_files(tmp_path: Path) -> None:
    state = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    panel = _panel(state)
    frame = explorer_frame(
        state,
        panel,
        metrics={"low_k_power": 1.5, "finite_domain_scale": 2.0},
        events=[{"label": "T_hit", "elapsed": 32768}],
    )
    explorer = tmp_path / "history_erasure_explorer.html"
    dashboard = tmp_path / "dashboard.html"
    build_history_explorer(
        explorer,
        frames_by_density={
            "0.20": {
                "untouched": [frame],
                "global shuffle": [frame],
                "tile 32": [frame],
                "tile 128": [frame],
            }
        },
        fixed_crops={"0.20": {"x": 0, "y": 0, "width": 2, "height": 2}},
        title="test explorer",
    )
    build_history_dashboard(
        dashboard,
        decision={
            "combined_mechanism": "F. INCONCLUSIVE",
            "branch_A": {"classification": "MIXED / INCONCLUSIVE"},
            "branch_B": {"classification": "MIXED / INCONCLUSIVE"},
        },
        results_rows=[
            {
                "comparison": "Branch A",
                "rmst_delay": "0",
                "ci95_low": "-1",
                "ci95_high": "1",
            }
        ],
        survival_rows=[
            {
                "density": "pooled",
                "branch_point": "A",
                "intervention": "global_shuffle",
                "time": "0",
                "survival": "1",
            }
        ],
        integrity={"checks": {"winner_hash_exact": True}},
    )
    for path in (explorer, dashboard):
        text = path.read_text(encoding="utf-8").lower()
        assert "<script src=" not in text
        assert "fetch(" not in text
        assert "http://" not in text
        assert "https://" not in text
        assert path.with_suffix(path.suffix + ".sha256").is_file()
    explorer_text = explorer.read_text(encoding="utf-8").lower()
    assert "data:image/png;base64," in explorer_text
    assert 'id="branchtoggles"' in explorer_text
    assert "data-branch" in explorer_text
    assert "visible_branches" in explorer_text
    assert all(
        branch in explorer_text
        for branch in ("untouched", "global shuffle", "tile 32", "tile 128")
    )


def test_figure_contract_has_eight_pdf_svg_png_triplets() -> None:
    paths = figure_output_paths("figures")
    assert len(paths) == 24
    assert len({path.stem for path in paths}) == 8
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}


def test_discovery_loader_verifies_frozen_raw_uint8_digest(tmp_path: Path) -> None:
    relative = Path("results/observe_4096/checkpoints/states/state_t000000000000.pt")
    checkpoint = tmp_path / relative
    checkpoint.parent.mkdir(parents=True)
    state = (torch.arange(16, dtype=torch.uint8) % 4).reshape(1, 4, 4)
    torch.save({"state": state}, checkpoint)
    file_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".pt.sha256").write_text(
        f"{file_digest}  {checkpoint.name}\n", encoding="ascii"
    )
    raw_digest = hashlib.sha256(state.numpy().tobytes(order="C")).hexdigest()
    integrity = {
        "checkpoints": [
            {
                "tick": 0,
                "path": relative.as_posix(),
                "sha256": file_digest,
                "state_sha256_uint8": raw_digest,
            }
        ]
    }

    loaded, loaded_path = history_figures._load_discovery_state(tmp_path, integrity, 0)

    assert loaded.shape == (4, 4)
    assert np.array_equal(loaded, state.numpy()[0])
    assert loaded_path == checkpoint


def test_mean_curves_accepts_directional_mutual_information_records() -> None:
    records = [
        {
            "intervention": "global_shuffle",
            "mutual_information": {
                "1": {"x": 1.0, "y": 3.0, "mean": 2.0},
                "2": {"x": 3.0, "y": 5.0, "mean": 4.0},
            },
        },
        {
            "intervention": "global_shuffle",
            "mutual_information": {
                "1": {"x": 3.0, "y": 5.0, "mean": 4.0},
                "2": {"x": 5.0, "y": 7.0, "mean": 6.0},
            },
        },
    ]

    assert history_figures._mean_curves(records, "mutual_information") == {
        "global_shuffle": {1: 3.0, 2: 5.0}
    }


def test_mechanism_ledger_metadata_without_writing_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_save(fig: object, output_dir: Path, figure_id: str, **kwargs: object) -> dict[str, object]:
        captured.update({"figure_id": figure_id, **kwargs})
        history_figures.plt.close(fig)
        return {"figure_id": figure_id}

    monkeypatch.setattr(history_figures, "_save_triplet", fake_save)
    assert history_figures.figure_07_mechanism_ledger(tmp_path)["figure_id"].startswith("FIGURE_07")
    panels = captured["panels"]
    assert isinstance(panels, list) and len(panels) == 2
    for panel in panels:
        validate_panel_record(panel)
        assert panel["evidence_type"] == "MODEL SCHEMATIC"


def test_history_video_frame_is_exact_state_montage() -> None:
    base = np.asarray(
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]],
        dtype=np.uint8,
    )
    names = ("untouched", "global_shuffle", "tile_32", "tile_128")
    image = render_history_frame(
        {name: np.roll(base, index, axis=0) for index, name in enumerate(names)},
        density=0.2,
        elapsed=32768,
        absolute_phases={name: 65536 for name in names},
        state_hashes={name: state_sha256(np.roll(base, index, axis=0)) for index, name in enumerate(names)},
        crop=(0, 0, 4, 4),
        camera_fraction=0.5,
        event_times={name: 32768 for name in names},
    )
    assert image.size == (1920, 1080)
    assert image.mode == "RGB"


def test_publication_refuses_partial_campaign(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text('{"state":"running_sources"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed"):
        _require_completed_run(tmp_path)


def test_media_replay_rejects_unfrozen_branch_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="A or B"):
        reconstruct_untouched_media_states(
            tmp_path / "missing.yaml",
            tmp_path,
            [],
            branch_point="C",
        )
