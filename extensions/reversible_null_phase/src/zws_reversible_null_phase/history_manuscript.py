"""Correction-safe compatibility layer for the retired v0.3.2 paper generator.

The historical generator derived public claims directly from a strong-domain
endpoint that was later shown to fail under shuffled null geometry.  Its outputs
are preserved as historical artifacts, but this module now fails closed instead
of regenerating them.  The canonical v0.2 publication sources are tracked under
``paper/`` and compiled directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


CANONICAL_TITLE = (
    "Delayed Distributed Coarsening in an Exactly Reversible "
    "Signed-Charge-Conserving Block Cellular Automaton"
)


def result_language(decision: dict[str, Any]) -> dict[str, str]:
    """Return correction-safe public language for legacy lay/closure callers."""
    del decision
    return {
        "title": CANONICAL_TITLE,
        "headline": (
            "A perfectly reversible artificial world can reorganise random material "
            "into enormous irregular structures and retrace the entire history exactly."
        ),
        "result_sentence": (
            "The robust result is finite-time delayed distributed coarsening-like "
            "organisation; the v0.3.2 causal history-erasure inference is withdrawn "
            "because its endpoint misclassified shuffled null geometry."
        ),
        "claim": (
            "One frozen rule exhibits exact reversal, exact signed-charge conservation, "
            "changing occupancy, density-dependent waiting, and control-sensitive "
            "macroscopic organisation at the tested finite sizes and horizons."
        ),
        "branch_a": "WITHDRAWN AS A CAUSAL CLASSIFICATION",
        "branch_b": "WITHDRAWN AS A CAUSAL CLASSIFICATION",
        "combined": "CLASSIFIER FAILURE; NO CAUSAL HISTORY CONCLUSION",
        "tile": "diagnostic only",
        "stop_decision": "A. REPAIR COMPLETE; NO NEW SIMULATION IS WARRANTED.",
    }


def _withdrawn_public_generator_error() -> RuntimeError:
    return RuntimeError(
        "The legacy v0.3.2 manuscript generator is retired because its endpoint "
        "failed under shuffled null geometry. Use the tracked companion-paper v0.2 "
        "sources in paper/; historical v0.3.2 artifacts remain immutable."
    )


def _main_tex(decision: dict[str, Any], language: dict[str, str]) -> str:
    """Fail closed for callers that attempt to recreate the invalid manuscript."""
    del decision, language
    raise _withdrawn_public_generator_error()


def build_manuscript_sources(
    *,
    repo_root: str | Path,
    experiment_path: str | Path,
    paper_path: str | Path,
) -> dict[str, Any]:
    """Fail closed; canonical corrected sources are not generated from v0.3.2 labels."""
    del repo_root, experiment_path, paper_path
    raise _withdrawn_public_generator_error()
