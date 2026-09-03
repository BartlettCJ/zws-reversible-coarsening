from __future__ import annotations

from pathlib import Path

import pytest

from zws_reversible_null_phase.rule import ReversibleRule


@pytest.fixture(scope="session")
def extension_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def winner(extension_root: Path) -> ReversibleRule:
    return ReversibleRule.load(extension_root / "rules" / "winner_d1fe99.json")
