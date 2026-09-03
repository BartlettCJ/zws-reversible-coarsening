from __future__ import annotations

import os
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_manuscript import result_language
from .history_media import load_verified_json, verify_sidecar
from .history_publication import RULE_SHA256, STATE_HEX, STATE_NAMES


def _sidecar(path: Path) -> None:
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def lay_copy(decision: dict[str, Any]) -> dict[str, str]:
    language = result_language(decision)
    return {
        "headline": language["headline"],
        "world": (
            "A square artificial world of empty, positive, negative, and bound-neutral cells. "
            "A bijective local rule updates 2x2 blocks, so every microscopic step can be undone."
        ),
        "zero": "Signed charge Q = N+ - N- remains exactly zero at every audited phase.",
        "changes": (
            "Occupancy M = N+ + N- + Nbound may grow or shrink. Arrangement, interfaces, "
            "entropy, and domain scale can change enormously."
        ),
        "surprise": (
            "The world forms irregular excitation-rich and near-vacuum regions without becoming "
            "a simple crystal. Reversing the rule returns every retained cell exactly."
        ),
        "test": (
            "Density, prepared geometry, an active exact-Q+M sibling, pure rotation, "
            "and reaction-pair removals test nearby explanations."
        ),
        "finding": (
            "The robust finding is delayed, distributed, density-dependent organisation "
            "with exact cell-for-cell reversal. The later history-erasure headline is "
            "withdrawn because its endpoint failed on shuffled null states."
        ),
        "not_prove": (
            "It does not prove a physical cosmology, a natural invariant of our universe, life, "
            "consciousness, an equilibrium phase transition, or universality across automata."
        ),
        "combined": language["combined"],
        "title": language["title"],
    }


def _image_array(path: Path) -> np.ndarray:
    verify_sidecar(path)
    return np.asarray(Image.open(path).convert("RGB"))


def render_one_page_lay_summary(
    *,
    hero_image: np.ndarray,
    ledger_image: np.ndarray,
    decision: dict[str, Any],
) -> plt.Figure:
    copy = lay_copy(decision)
    fig = plt.figure(figsize=(8.5, 11), facecolor="#f7f4ed")
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    headline = textwrap.fill(copy["headline"], width=44, break_long_words=False)
    canvas.text(
        0.05,
        0.972,
        headline,
        fontsize=20,
        linespacing=0.95,
        weight="bold",
        color="#071018",
        va="top",
    )
    canvas.text(
        0.05,
        0.850,
        "ZWSLab · formal computational result",
        fontsize=10,
        color="#557083",
        va="top",
    )

    hero_axis = fig.add_axes((0.05, 0.60, 0.90, 0.23))
    hero_axis.imshow(hero_image)
    hero_axis.axis("off")
    for spine in hero_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#36566e")

    left = fig.add_axes((0.05, 0.365, 0.43, 0.225))
    right = fig.add_axes((0.52, 0.365, 0.43, 0.225))
    for axis in (left, right):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
    left.add_patch(Rectangle((0, 0), 1, 1, facecolor="#071018", edgecolor="#36566e", linewidth=1.5))
    left.text(0.06, 0.88, "The ledger", fontsize=14, weight="bold", color="#f0e442")
    left.text(0.06, 0.70, "Q = N+ - N-", fontsize=16, weight="bold", color="white")
    left.text(0.06, 0.58, "exactly conserved · runs use Q = 0", fontsize=9.3, color="#b7c7d3")
    left.text(0.06, 0.38, "M = N+ + N- + Nbound", fontsize=14, weight="bold", color="white")
    left.text(0.06, 0.25, "allowed to change", fontsize=10, color="#cc79a7")
    legend_x = 0.06
    for name, color in zip(STATE_NAMES, STATE_HEX, strict=True):
        left.add_patch(Rectangle((legend_x, 0.08), 0.035, 0.06, facecolor=color, edgecolor="white", linewidth=0.4))
        left.text(legend_x + 0.045, 0.11, name, fontsize=6.8, va="center", color="#d7e9f6")
        legend_x += 0.235
    right.imshow(ledger_image)
    right.set_xlim(0, ledger_image.shape[1])
    right.set_ylim(ledger_image.shape[0], 0)

    items = [
        ("What is the world?", copy["world"]),
        ("What stays zero?", copy["zero"]),
        ("What can change?", copy["changes"]),
        ("What is surprising?", copy["surprise"]),
        ("What did the controls test?", copy["test"]),
        ("What did it find?", copy["finding"]),
        ("What does it not prove?", copy["not_prove"]),
    ]
    start_y = 0.325
    for index, (heading, body) in enumerate(items):
        column = index % 2
        row = index // 2
        x = 0.05 + column * 0.47
        y = start_y - row * 0.073
        width = 0.43 if index < 6 else 0.90
        if index == 6:
            x = 0.05
        canvas.add_patch(Rectangle((x, y - 0.058), width, 0.062, facecolor="#ffffff", edgecolor="#d8d2c5", linewidth=0.8))
        canvas.text(x + 0.012, y - 0.008, heading, fontsize=8.5, weight="bold", color="#0072b2", va="top")
        wrapped_body = textwrap.fill(
            body,
            width=58 if index < 6 else 120,
            break_long_words=False,
            break_on_hyphens=False,
        )
        canvas.text(
            x + 0.012,
            y - 0.026,
            wrapped_body,
            fontsize=6.5,
            linespacing=1.08,
            color="#24323d",
            va="top",
        )
    canvas.text(0.05, 0.018, f"Companion v0.2 repair · robust finite-time result · Rule SHA-256 {RULE_SHA256[:16]}… · historical causal headline withdrawn.", fontsize=7, color="#557083")
    return fig


def _save_figure(fig: plt.Figure, path: Path, *, format_name: str, dpi: int | None = None) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    options: dict[str, Any] = {"format": format_name, "bbox_inches": None, "facecolor": fig.get_facecolor()}
    if dpi is not None:
        options["dpi"] = dpi
    fig.savefig(temporary, **options)
    os.replace(temporary, path)
    _sidecar(path)


def build_lay_artifacts(
    *,
    experiment_path: str | Path,
    paper_path: str | Path,
) -> dict[str, Any]:
    experiment, paper = Path(experiment_path).resolve(), Path(paper_path).resolve()
    decision = load_verified_json(experiment / "SCIENTIFIC_DECISION_V032.json")
    integrity = load_verified_json(experiment / "INTEGRITY_AUDIT.json")
    if not integrity["passed"]:
        raise RuntimeError("lay publication stopped because the final integrity audit failed")
    hero_path = paper / "figures/FIGURE_01_ZERO_BALANCE_REVERSIBLE_HERO.png"
    ledger_path = paper / "figures/FIGURE_07_MECHANISM_LEDGER.png"
    hero = _image_array(hero_path)
    ledger = _image_array(ledger_path)
    fig = render_one_page_lay_summary(hero_image=hero, ledger_image=ledger, decision=decision)
    png = paper / "ONE_PAGE_LAY_SUMMARY.png"
    pdf = paper / "ONE_PAGE_LAY_SUMMARY.pdf"
    _save_figure(fig, png, format_name="png", dpi=300)
    _save_figure(fig, pdf, format_name="pdf")
    plt.close(fig)

    graphical = paper / "graphical_abstract.png"
    graphical_pdf = paper / "graphical_abstract.pdf"
    graph_fig = plt.figure(figsize=(16, 9), facecolor="#071018")
    axis = graph_fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    copy = lay_copy(decision)
    graph_headline = textwrap.fill(copy["headline"], width=76, break_long_words=False)
    axis.text(0.04, 0.95, graph_headline, color="#f6f8fa", fontsize=22, linespacing=0.98, weight="bold", va="top")
    hero_axis = graph_fig.add_axes((0.04, 0.24, 0.58, 0.50))
    hero_axis.imshow(hero)
    hero_axis.axis("off")
    axis.add_patch(FancyArrowPatch((0.64, 0.49), (0.70, 0.49), arrowstyle="-|>", mutation_scale=25, color="#f0e442", linewidth=2))
    axis.add_patch(Rectangle((0.70, 0.24), 0.26, 0.50, facecolor="#102331", edgecolor="#36566e", linewidth=2))
    axis.text(0.73, 0.68, "Exact ledger", fontsize=18, weight="bold", color="#f0e442")
    axis.text(0.73, 0.62, "Q exact · M can change\n16,777,216 cells reversed", fontsize=12.5, color="#d7e9f6")
    axis.text(0.73, 0.52, "Finite-time kinetics", fontsize=18, weight="bold", color="#56b4e9")
    axis.text(0.73, 0.46, "density shifts waiting time\ndistributed event morphology", fontsize=12.5, color="#d7e9f6")
    axis.text(0.73, 0.36, "Mechanism controls", fontsize=18, weight="bold", color="#cc79a7")
    axis.text(0.73, 0.30, "Q+M sibling · rotation\nprepared shapes · ablations", fontsize=12, color="#f6f8fa")
    axis.text(0.04, 0.11, "Robust result: delayed irregular organisation coexists with exact reversal and signed balance.", fontsize=16, color="#d7e9f6", wrap=True)
    axis.text(0.04, 0.035, "FORMAL COMPUTATIONAL RESULT · exact reversible block automaton · not a cosmological or biological claim", fontsize=11, color="#9fb0bf")
    _save_figure(graph_fig, graphical, format_name="png", dpi=300)
    _save_figure(graph_fig, graphical_pdf, format_name="pdf")
    plt.close(graph_fig)

    metadata = {
        "schema": "zws-reversible-null-phase-companion-v02-lay-artifact-metadata-v1",
        "rule_hash": RULE_SHA256,
        "correction_status": copy["combined"],
        "headline": copy["headline"],
        "inputs": {
            "hero": {"path": str(hero_path), "sha256": sha256_file(hero_path)},
            "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
            "decision": {
                "path": str((experiment / "SCIENTIFIC_DECISION_V032.json").resolve()),
                "sha256": sha256_file(experiment / "SCIENTIFIC_DECISION_V032.json"),
            },
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (png, pdf, graphical, graphical_pdf)
        ],
    }
    atomic_write_json(paper / "LAY_ARTIFACT_METADATA.json", metadata)
    return metadata
