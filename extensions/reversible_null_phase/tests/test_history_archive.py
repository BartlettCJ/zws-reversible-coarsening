from __future__ import annotations

from pathlib import Path

import pytest

from zws_reversible_null_phase.atomic import atomic_write_json, atomic_write_text, sha256_file
from zws_reversible_null_phase.history_archive import (
    _allowed_file,
    _archive_citation,
    _data_volume_note,
    assert_no_staged_credentials,
    collect_archive_candidates,
)


def _write_checked(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def test_archive_excludes_large_representative_checkpoint_and_secrets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run = tmp_path / "governed_run"
    experiment = repo / "experiment"
    paper = repo / "paper"
    state = run / "checkpoints" / "states" / "state_t00000000.pt"
    _write_checked(state, b"representative exact state")
    paper.mkdir(parents=True)
    atomic_write_text(paper / "manuscript.md", "# Manuscript\n")
    atomic_write_text(paper / ".env", "SECRET=never archive\n")
    atomic_write_json(
        experiment / "FIGURE_MANIFEST.json",
        {
            "figures": [
                {
                    "panels": [
                        {"source_path": str(state)},
                    ]
                }
            ]
        },
    )

    candidates = collect_archive_candidates(
        repo_root=repo,
        run_path=run,
        experiment_path=experiment,
        paper_path=paper,
    )
    destinations = {destination.as_posix() for _, destination in candidates}
    assert not any(path.startswith("representative_states/") for path in destinations)
    assert "paper/manuscript.md" in destinations
    assert "paper/.env" not in destinations


def test_archive_sanitizes_local_path_and_blocks_credential_content(tmp_path: Path) -> None:
    note = _data_volume_note(Path("C:/example/raw/governed_run"))
    assert "governed_run" in note
    assert "C:/private" not in note

    safe = tmp_path / "safe"
    safe.mkdir()
    atomic_write_text(safe / "README.md", "No credentials here.\n")
    assert_no_staged_credentials(safe)
    fake_token = "github_" + "pat_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    atomic_write_text(safe / "leak.txt", fake_token + "\n")
    with pytest.raises(RuntimeError, match="credential-like content"):
        assert_no_staged_credentials(safe)


def test_archive_citation_targets_companion_dataset() -> None:
    citation = _archive_citation("A companion title")
    assert 'title: "A companion title"' in citation
    assert "type: dataset" in citation
    assert "version: 0.2.0" in citation
    assert "Bartlett" in citation
    assert "artificial life" not in citation.lower()


@pytest.mark.parametrize(
    "relative",
    [
        ".git/config",
        ".venv/pyvenv.cfg",
        "cache/data.json",
        "logs/supervisor.log",
        "credentials.json",
        "private_access_token.txt",
        "nested/archive.zip",
    ],
)
def test_archive_rejects_forbidden_file_classes(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert not _allowed_file(path)
