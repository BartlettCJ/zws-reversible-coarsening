"""Publication figures for the corrected ZWSLab companion paper v0.2.

This module reads only compact, already-frozen result tables.  It does not run
the simulator and does not overwrite the checksummed v0.3.2 figure set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


RULE_SHA256 = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
VERSION = "0.2.0"

BG = "#07111b"
PANEL = "#0c1824"
TEXT = "#f2f4f5"
MUTED = "#a8bac8"
GRID = "#355064"
YELLOW = "#F0E442"
BLUE = "#56B4E9"
DEEP_BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_sidecar(path: Path) -> None:
    _write_text(path.with_name(path.name + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.32, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_triplet(fig: plt.Figure, output_root: Path, stem: str, title: str) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = output_root / f"{stem}.{suffix}"
        if suffix == "png":
            fig.savefig(
                path,
                format="png",
                dpi=220,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                metadata={"Title": title, "Author": "ZWSLab companion paper v0.2"},
            )
        elif suffix == "pdf":
            fig.savefig(
                path,
                format="pdf",
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                metadata={
                    "Title": title,
                    "Author": "ZWSLab companion paper v0.2",
                    "Subject": "Existing-data scientific figure",
                    "Creator": "zws_reversible_null_phase.companion_v02_figures",
                    "CreationDate": datetime(2026, 8, 31, tzinfo=timezone.utc),
                },
            )
        else:
            fig.savefig(
                path,
                format="svg",
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                metadata={"Title": title, "Creator": "ZWSLab companion paper v0.2", "Date": "2026-08-31"},
            )
        write_sidecar(path)
        outputs[path.name] = sha256_file(path)
    return outputs


def _density_rows(repo: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], Path, Path]:
    root = repo / "experiments" / "reversible_null_phase_v031_nucleation_20260827"
    survival_path = root / "SURVIVAL_SUMMARY.csv"
    first_passage_path = root / "FIRST_PASSAGE_RESULTS.csv"
    survival = [
        row
        for row in read_csv(survival_path)
        if row["stage"] == "primary_homogeneous_1024"
        and row["arrangement"] == "homogeneous"
        and row["endpoint"] == "T_hit"
    ]
    first_passage = [
        row
        for row in read_csv(first_passage_path)
        if row["stage"] == "primary_homogeneous_1024" and row["arrangement"] == "homogeneous"
    ]
    return survival, first_passage, survival_path, first_passage_path


def _km_median(rows: Iterable[dict[str, str]]) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row["phase"]))
    for row in ordered:
        if float(row["survival"]) <= 0.5:
            return float(row["phase"])
    return None


def figure_density_first_passage(repo: Path, output_root: Path) -> dict[str, Any]:
    survival, first_passage, survival_path, first_passage_path = _density_rows(repo)
    densities = sorted({float(row["density"]) for row in first_passage})
    density_colors = {
        0.16: MUTED,
        0.18: BLUE,
        0.20: GREEN,
        0.22: ORANGE,
        0.24: PURPLE,
        0.25: YELLOW,
        0.26: VERMILLION,
    }

    grouped_survival: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in survival:
        grouped_survival[float(row["density"])].append(row)

    fig = plt.figure(figsize=(16, 9.5), facecolor=BG, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    ax_survival = fig.add_subplot(grid[0, :])
    ax_median = fig.add_subplot(grid[1, 0])
    ax_fixed = fig.add_subplot(grid[1, 1])
    for ax in (ax_survival, ax_median, ax_fixed):
        _style_axes(ax)

    for density in densities:
        rows = sorted(grouped_survival[density], key=lambda row: float(row["phase"]))
        ax_survival.step(
            [float(row["phase"]) for row in rows],
            [float(row["survival"]) for row in rows],
            where="post",
            linewidth=2.6,
            color=density_colors[density],
            label=f"{density:.2f}",
        )
    ax_survival.set_title(
        "A  Confirmatory Kaplan-Meier curves: time before the frozen strong-domain crossing",
        loc="left",
        fontweight="bold",
        color=TEXT,
    )
    ax_survival.set_xlabel("Microscopic phases")
    ax_survival.set_ylabel("Fraction not yet crossing")
    ax_survival.set_ylim(-0.03, 1.04)
    survival_legend = ax_survival.legend(
        title="Initial occupancy", ncol=4, frameon=False, labelcolor=TEXT, title_fontsize=10
    )
    survival_legend.get_title().set_color(TEXT)
    ax_survival.text(
        0.995,
        0.03,
        "12 independent seeds per density; censoring retained",
        transform=ax_survival.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=10,
    )

    medians = [_km_median(grouped_survival[density]) for density in densities]
    for density, median in zip(densities, medians, strict=True):
        color = density_colors[density]
        if median is None:
            bound = max(float(row["phase"]) for row in grouped_survival[density])
            ax_median.scatter([density], [bound], marker="^", s=110, color=color, zorder=4)
            ax_median.annotate(
                f">{bound/1000:.0f}k",
                (density, bound),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=TEXT,
                fontsize=9,
            )
        else:
            ax_median.scatter([density], [median], s=100, color=color, edgecolor=TEXT, linewidth=0.7, zorder=4)
            ax_median.annotate(
                f"{median/1000:.1f}k",
                (density, median),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=TEXT,
                fontsize=9,
            )
    observed_x = [density for density, median in zip(densities, medians, strict=True) if median is not None]
    observed_y = [median for median in medians if median is not None]
    ax_median.plot(observed_x, observed_y, color=BLUE, linewidth=1.7, alpha=0.8)
    ax_median.set_title("B  Median first-passage time", loc="left", fontweight="bold", color=TEXT)
    ax_median.set_xlabel("Initial occupancy")
    ax_median.set_ylabel("Microscopic phases")
    ax_median.set_xlim(0.155, 0.265)
    ax_median.set_ylim(0, 560000)
    ax_median.text(
        0.02,
        0.04,
        "0.16: median not identified\nwithin its 262,144-phase horizon",
        transform=ax_median.transAxes,
        color=MUTED,
        fontsize=9.5,
    )

    grouped_fp: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in first_passage:
        grouped_fp[float(row["density"])].append(row)
    fixed_horizons = (131072, 262144)
    width = 0.0035
    offsets = (-width / 2, width / 2)
    fixed_colors = (DEEP_BLUE, ORANGE)
    for offset, horizon, color in zip(offsets, fixed_horizons, fixed_colors, strict=True):
        fractions: list[float] = []
        for density in densities:
            rows = grouped_fp[density]
            hits = sum(
                row["hit_observed"].lower() == "true"
                and row["T_hit"] != ""
                and float(row["T_hit"]) <= horizon
                for row in rows
            )
            fractions.append(hits / len(rows))
        ax_fixed.bar(
            np.asarray(densities) + offset,
            fractions,
            width=width * 0.86,
            color=color,
            label=f"by {horizon:,} phases",
        )
    ax_fixed.set_title(
        "C  The apparent onset moves with the observation horizon",
        loc="left",
        fontweight="bold",
        color=TEXT,
    )
    ax_fixed.set_xlabel("Initial occupancy")
    ax_fixed.set_ylabel("Fraction crossed")
    ax_fixed.set_xlim(0.155, 0.265)
    ax_fixed.set_ylim(0, 1.08)
    ax_fixed.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    ax_fixed.text(
        0.98,
        0.04,
        "Finite-time crossover, not a demonstrated\ntime-independent critical density",
        transform=ax_fixed.transAxes,
        ha="right",
        color=YELLOW,
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        "Density controls a steep waiting-time spectrum for delayed distributed coarsening",
        color=TEXT,
        fontsize=21,
        fontweight="bold",
    )
    stem = "COMPANION_V02_FIGURE_03_DENSITY_FIRST_PASSAGE"
    outputs = _save_triplet(fig, output_root, stem, "Density-sensitive first-passage kinetics")
    plt.close(fig)
    return {
        "figure_id": stem,
        "main_text_number": 3,
        "evidence_class": "preregistered-confirmatory",
        "source_files": {
            str(survival_path.relative_to(repo)).replace("\\", "/"): sha256_file(survival_path),
            str(first_passage_path.relative_to(repo)).replace("\\", "/"): sha256_file(first_passage_path),
        },
        "outputs": outputs,
        "warranted_inference": "Higher initial occupancy shortened first-passage times over the tested finite sizes and horizons; longer follow-up weakened the earlier apparent fixed threshold.",
        "limitation": "The curves do not establish an equilibrium critical point, infinite-time threshold, or universal scaling law.",
    }


def _control_rows(repo: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], Path, Path]:
    control_path = (
        repo
        / "experiments"
        / "domain_mechanism_20260825"
        / "artifacts"
        / "double_conservation_screen"
        / "CONDITION_SUMMARIES.csv"
    )
    ablation_path = (
        repo
        / "experiments"
        / "reversible_null_phase_v030_20260827"
        / "balanced"
        / "report"
        / "SUMMARY.csv"
    )
    controls = [row for row in read_csv(control_path) if row["condition_id"] in {"winner", "qm_sibling", "rotation"}]
    ablations = [row for row in read_csv(ablation_path) if row["stage"] == "ablation_512"]
    return controls, ablations, control_path, ablation_path


def figure_controls_ablations(repo: Path, output_root: Path) -> dict[str, Any]:
    controls, ablations, control_path, ablation_path = _control_rows(repo)
    by_id = {row["condition_id"]: row for row in controls}
    control_order = ("winner", "qm_sibling", "rotation")
    control_labels = ("Winner\nQ fixed, M free", "Reactive sibling\nQ and M fixed", "Pure rotation")
    control_colors = (YELLOW, BLUE, GREEN)

    pooled: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "passes": 0.0})
    for row in ablations:
        pooled[row["rule_variant"]]["n"] += float(row["trajectories"])
        pooled[row["rule_variant"]]["passes"] += float(row["strong_passes"])
    ablation_order = (
        "winner",
        "ablate_01",
        "ablate_11",
        "ablate_05",
        "ablate_08",
        "ablate_09",
        "ablate_06",
        "ablate_00",
        "ablate_02",
        "ablate_03",
        "ablate_04",
        "ablate_07",
        "ablate_10",
        "qm_sibling",
        "rotation",
    )

    fig = plt.figure(figsize=(16, 8.8), facecolor=BG, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.55])
    ax_controls = fig.add_subplot(grid[0, 0])
    ax_ablation = fig.add_subplot(grid[0, 1])
    for ax in (ax_controls, ax_ablation):
        _style_axes(ax)

    x = np.arange(len(control_order))
    width = 0.34
    peak = [float(by_id[name]["median_peak_scale_ratio"]) for name in control_order]
    final = [float(by_id[name]["median_final_scale_ratio"]) for name in control_order]
    bars_peak = ax_controls.bar(x - width / 2, peak, width, label="Median peak scale ratio", color=control_colors, alpha=0.95)
    bars_final = ax_controls.bar(
        x + width / 2,
        final,
        width,
        label="Median final scale ratio",
        color=control_colors,
        alpha=0.48,
        edgecolor=TEXT,
        linewidth=0.8,
    )
    ax_controls.set_yscale("log")
    ax_controls.set_ylim(0.8, 45)
    ax_controls.set_xticks(x, control_labels)
    ax_controls.set_ylabel("Scale relative to initial")
    ax_controls.set_title(
        "A  Matched 1024x1024 mechanism controls", loc="left", fontweight="bold", color=TEXT
    )
    ax_controls.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper right")
    for index, name in enumerate(control_order):
        row = by_id[name]
        ax_controls.text(
            index,
            max(peak[index], final[index]) * 1.18,
            f"strong-domain: {row['strong_domain_count']}/{row['trajectory_count']}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
        )
    for bars in (bars_peak, bars_final):
        for bar in bars:
            value = bar.get_height()
            ax_controls.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.05,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=8,
            )
    ax_controls.text(
        0.02,
        0.03,
        "The Q+M sibling remains reactive\nbut did not enter the same regime.",
        transform=ax_controls.transAxes,
        color=MUTED,
        fontsize=10,
    )

    fractions = [pooled[name]["passes"] / pooled[name]["n"] for name in ablation_order]
    colors = []
    for name in ablation_order:
        if name == "winner":
            colors.append(YELLOW)
        elif name == "qm_sibling":
            colors.append(BLUE)
        elif name == "rotation":
            colors.append(GREEN)
        elif pooled[name]["passes"] == 0:
            colors.append(VERMILLION)
        else:
            colors.append(PURPLE)
    bars = ax_ablation.bar(np.arange(len(ablation_order)), fractions, color=colors)
    ax_ablation.set_xticks(
        np.arange(len(ablation_order)),
        [name.replace("ablate_", "pair ") for name in ablation_order],
        rotation=57,
        ha="right",
    )
    ax_ablation.set_ylim(0, 0.72)
    ax_ablation.set_ylabel("Strong-domain pass fraction")
    ax_ablation.set_title(
        "B  Single-reaction-pair ablations at 512x512", loc="left", fontweight="bold", color=TEXT
    )
    for bar, name in zip(bars, ablation_order, strict=True):
        passes = int(pooled[name]["passes"])
        n = int(pooled[name]["n"])
        ax_ablation.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 0.018, 0.018),
            f"{passes}/{n}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=8,
            rotation=90 if bar.get_height() == 0 else 0,
        )
    ax_ablation.text(
        0.99,
        0.94,
        "Six distinct single-pair removals gave 0/16 passes.\nOther removals weakened rather than always abolished the regime.",
        transform=ax_ablation.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.5,
    )

    fig.suptitle(
        "The large-domain regime depends on occupancy-changing, multi-reaction dynamics",
        color=TEXT,
        fontsize=21,
        fontweight="bold",
    )
    stem = "COMPANION_V02_FIGURE_04_CONTROLS_AND_ABLATIONS"
    outputs = _save_triplet(fig, output_root, stem, "Mechanism controls and single-pair ablations")
    plt.close(fig)
    return {
        "figure_id": stem,
        "main_text_number": 4,
        "evidence_class": "matched-control and preregistered ablation",
        "source_files": {
            str(control_path.relative_to(repo)).replace("\\", "/"): sha256_file(control_path),
            str(ablation_path.relative_to(repo)).replace("\\", "/"): sha256_file(ablation_path),
        },
        "outputs": outputs,
        "warranted_inference": "The closest tested reactive exact-Q+M sibling and pure rotation did not reproduce the winner regime; several single-pair removals also suppressed it over the tested protocols.",
        "limitation": "These controls do not prove that double conservation forbids organisation in general or that every reaction pair is individually necessary.",
    }


def figure_research_story(output_root: Path) -> dict[str, Any]:
    steps = [
        ("1", "Discovery", "A search-selected trajectory showed dramatic, reversible structure.", DEEP_BLUE),
        ("2", "Confirmation", "Frozen independent seeds reproduced large-domain formation.", GREEN),
        ("3", "Interpretation narrowed", "The peak was transient; late sustained coarsening was not supported.", ORANGE),
        ("4", "Composition", "Excitation-rich and near-vacuum regions remained locally high-entropy.", PURPLE),
        ("5", "Q versus M", "The winner conserved Q while M changed; the exact-Q+M sibling did not match it.", BLUE),
        ("6", "Density and arrangement", "Homogeneous first-passage times depended steeply on density and preparation.", YELLOW),
        ("7", "Longer horizons", "The apparent fixed 27.5% boundary moved downward with time.", ORANGE),
        ("8", "Distributed coarsening", "Event windows favoured distributed/simultaneous change, not one nucleus.", GREEN),
        ("9", "Transparent correction", "State shuffling exposed a non-invariant endpoint; the causal history question remains open.", VERMILLION),
    ]
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    positions = [
        (0, 2), (1, 2), (2, 2),
        (2, 1), (1, 1), (0, 1),
        (0, 0), (1, 0), (2, 0),
    ]
    for index, ((number, title, body, color), (column, row)) in enumerate(zip(steps, positions, strict=True)):
        left = column + 0.08
        bottom = row + 0.12
        ax.add_patch(
            plt.Rectangle(
                (left, bottom),
                0.84,
                0.72,
                facecolor=PANEL,
                edgecolor=color,
                linewidth=2.4 if index == 8 else 1.5,
            )
        )
        ax.text(left + 0.06, bottom + 0.61, number, color=color, fontsize=15, fontweight="bold", va="top")
        ax.text(left + 0.17, bottom + 0.61, title, color=TEXT, fontsize=14, fontweight="bold", va="top")
        ax.text(
            left + 0.06,
            bottom + 0.42,
            textwrap.fill(body, width=39),
            color=MUTED,
            fontsize=9.3,
            va="top",
            linespacing=1.22,
        )
        if index < len(positions) - 1:
            next_column, next_row = positions[index + 1]
            if row == next_row and next_column > column:
                start = (column + 0.94, row + 0.48)
                end = (next_column + 0.06, next_row + 0.48)
            elif row == next_row:
                start = (column + 0.06, row + 0.48)
                end = (next_column + 0.94, next_row + 0.48)
            else:
                start = (column + 0.5, row + 0.08)
                end = (next_column + 0.5, next_row + 0.94)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.5},
            )
    ax.text(
        1.5,
        3.06,
        "A corrected research story: each campaign narrowed the claim",
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.02,
        "Correction is part of the result: the final intervention was technically valid, but its inherited endpoint was not.",
        ha="center",
        va="bottom",
        color=YELLOW,
        fontsize=11.5,
    )
    stem = "COMPANION_V02_FIGURE_08_CORRECTED_RESEARCH_STORY"
    outputs = _save_triplet(fig, output_root, stem, "Corrected ZWSLab research story")
    plt.close(fig)
    return {
        "figure_id": stem,
        "main_text_number": None,
        "supplement_role": "research-programme timeline",
        "evidence_class": "explanatory audit timeline",
        "source_files": {},
        "outputs": outputs,
        "warranted_inference": "The programme progressively narrowed its claims and openly records the final endpoint failure.",
        "limitation": "This explanatory figure is not an additional scientific measurement.",
    }


def _existing_figure_record(repo: Path, filename: str, main_number: int, inference: str, limitation: str) -> dict[str, Any]:
    path = repo / "paper" / "figures" / filename
    stem = path.stem
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        candidate = path.with_suffix(f".{suffix}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        outputs[candidate.name] = sha256_file(candidate)
    return {
        "figure_id": stem,
        "main_text_number": main_number,
        "evidence_class": "audited existing evidence",
        "source_files": {"historical_figure_set": "preserved byte-for-byte"},
        "outputs": outputs,
        "warranted_inference": inference,
        "limitation": limitation,
    }


def build_all(repo: Path) -> Path:
    repo = repo.resolve()
    output_root = repo / "paper" / "figures"
    records = [
        _existing_figure_record(
            repo,
            "FIGURE_01_ZERO_BALANCE_REVERSIBLE_HERO.png",
            1,
            "The audited implementation exactly retraced a 4096x4096 trajectory after 65,536 phases with zero cell mismatches while Q remained zero and M changed.",
            "Exact reversal is expected from a correctly implemented permutation rule and is an implementation/coexistence audit, not a novelty claim about reversibility.",
        ),
        _existing_figure_record(
            repo,
            "FIGURE_06_MICRO_DISORDER_MACRO_ORDER.png",
            2,
            "Large-scale excitation-rich and near-vacuum regions coexist with high local four-state entropy and no simple tested crystal, stripe, or checkerboard signature.",
            "Morphology is finite-size and finite-time and does not establish equilibrium phase separation.",
        ),
        figure_density_first_passage(repo, output_root),
        figure_controls_ablations(repo, output_root),
        _existing_figure_record(
            repo,
            "FIGURE_07_MECHANISM_LEDGER.png",
            5,
            "Q constrains one signed direction while M remains an independent degree of freedom in the winner; the matched sibling conserves both.",
            "The comparison does not show that Q causes organisation or that Q+M conservation forbids it in all models.",
        ),
        figure_research_story(output_root),
    ]
    manifest = {
        "schema": "zwslab-companion-v02-figure-manifest-v1",
        "version": VERSION,
        "date": "2026-08-31",
        "rule_sha256_raw_uint8": RULE_SHA256,
        "simulation_run": False,
        "rendering": {
            "lattice_sampling": "nearest-neighbour in preserved state figures",
            "palette": "colour-blind-conscious Okabe-Ito-derived palette",
            "raster_dpi": 220,
            "vector_formats": ["pdf", "svg"],
        },
        "main_text_order": [1, 2, 3, 4, 5],
        "figures": records,
    }
    manifest_path = repo / "paper" / "COMPANION_V02_FIGURE_MANIFEST.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_sidecar(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_all(args.repo)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
