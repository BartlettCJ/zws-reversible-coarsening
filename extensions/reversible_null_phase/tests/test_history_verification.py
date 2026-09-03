from __future__ import annotations

from pathlib import Path

from zws_reversible_null_phase.atomic import atomic_write_text, sha256_file
from zws_reversible_null_phase.history_media import load_verified_json
from zws_reversible_null_phase.history_verification import (
    _verification_python_paths,
    record_manual_qa,
    resource_governor_gate,
    verify_sidecar_tree,
)


def _checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def test_sidecar_tree_verifies_all_present_sidecars(tmp_path: Path) -> None:
    _checked(tmp_path / "a.json", "{}\n")
    _checked(tmp_path / "nested/b.csv", "x\n1\n")
    result = verify_sidecar_tree((tmp_path,))
    assert result == {"passed": True, "checked": 2, "failures": []}


def test_final_verifier_uses_distinct_root_and_extension_environments(tmp_path: Path) -> None:
    root_python = tmp_path / ".venv/Scripts/python.exe"
    extension_python = tmp_path / ".venv-v030/Scripts/python.exe"
    for path in (root_python, extension_python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test executable placeholder")
    assert _verification_python_paths(tmp_path) == (root_python, extension_python)


def test_manual_qa_is_checksummed_and_hashes_review_artifacts(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"not a real image; hash provenance only")
    output = tmp_path / "BROWSER_QA.json"
    result = record_manual_qa(
        output_path=output,
        kind="browser",
        summary="Explorer controls and exports passed.",
        details=("No console errors.",),
        artifacts=(screenshot,),
        passed=True,
    )
    assert result["passed"] is True
    assert result["artifacts"][0]["sha256"] == sha256_file(screenshot)
    assert load_verified_json(output)["summary"].startswith("Explorer")


def test_video_visual_qa_kind_is_supported(tmp_path: Path) -> None:
    frame = tmp_path / "contact.png"
    frame.write_bytes(b"contact sheet provenance")
    result = record_manual_qa(
        output_path=tmp_path / "VIDEO_VISUAL_QA.json",
        kind="video_visual",
        summary="All sampled video frames are legible and synchronized.",
        details=("Four contact sheets reviewed.",),
        artifacts=(frame,),
        passed=True,
    )
    assert result["schema"].endswith("video_visual-qa-v1")


def test_resource_governor_accepts_empty_stderr(tmp_path: Path) -> None:
    stderr = tmp_path / "supervisor.stderr.log"
    stderr.write_text("", encoding="utf-8")

    result = resource_governor_gate({"state": "branches_complete"}, stderr)

    assert result["passed"] is True
    assert result["managed_pause_count"] == 0
    assert result["unexpected_stderr_lines"] == []


def test_resource_governor_accepts_documented_managed_pauses(tmp_path: Path) -> None:
    stderr = tmp_path / "supervisor.stderr.log"
    stderr.write_text(
        "managed pause: resource pressure: cpu=91.1%\n"
        "managed pause: resource pressure: cpu=100.0%\n",
        encoding="utf-8",
    )

    result = resource_governor_gate({"state": "branches_complete"}, stderr)

    assert result["passed"] is True
    assert result["managed_pause_count"] == 2
    assert result["unexpected_stderr_lines"] == []


def test_resource_governor_rejects_unexpected_stderr_or_nonterminal_state(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "supervisor.stderr.log"
    stderr.write_text("Traceback: worker failed\n", encoding="utf-8")

    unexpected = resource_governor_gate({"state": "branches_complete"}, stderr)
    incomplete = resource_governor_gate({"state": "branches_running"}, stderr)

    assert unexpected["passed"] is False
    assert unexpected["unexpected_stderr_lines"] == ["Traceback: worker failed"]
    assert incomplete["passed"] is False
