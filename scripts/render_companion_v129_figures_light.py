"""Render the companion paper's five data figures in a light, print-friendly palette (v1.29).

Same data, same code paths, same numbers as the dark v0.2.1 renders and the v0.3.2 history
figures; only the canvas, text and accent colours change, and the outputs get new file stems
so that the earlier renders and their manifests remain intact as provenance.

  COMPANION_V129_FIGURE_03_DENSITY_FIRST_PASSAGE      (from COMPANION_V021_FIGURE_03, frozen CSVs)
  COMPANION_V129_FIGURE_04_CONTROLS_AND_ABLATIONS     (from COMPANION_V021_FIGURE_04, frozen CSVs)
  COMPANION_V129_FIGURE_08_CORRECTED_RESEARCH_STORY   (from COMPANION_V021_FIGURE_08, text only)
  FIGURE_06_MICRO_DISORDER_MACRO_ORDER_V129           (from history_figures.figure_06_micro_macro)
  FIGURE_07_MECHANISM_LEDGER_V129                     (from history_figures.figure_07_mechanism_ledger)

No simulator is run. State panels keep the exact four-state palette (they are pictures of data).

Usage (from the worktree root, with the extension package importable and torch available):
  python scripts/render_companion_v129_figures_light.py --repo . --run <v0.3.2 run dir> --out paper/figures
"""
from __future__ import annotations
import argparse, csv, hashlib, json, textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

# ---------------------------------------------------------------- light palette (Okabe-Ito accents)
BG = "#ffffff"; PANEL = "#f6f8fa"; TEXT = "#233240"; MUTED = "#5a6b7a"; GRID = "#cfd8e0"
COLORS = ["#6e7f8d", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#B8A400", "#D55E00"]  # densities 16..26%
WINNER = "#B8A400"   # darker Okabe-Ito yellow for white paper
RULE = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
TODAY = datetime(2026, 9, 5, tzinfo=timezone.utc)

plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                     "legend.labelcolor": TEXT, "axes.edgecolor": MUTED})

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))

def style(ax):
    ax.set_facecolor(PANEL); ax.tick_params(colors=TEXT, labelsize=9)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, alpha=.9, linewidth=.6); ax.set_axisbelow(True)

def save(fig, out: Path, stem: str, sources: list[Path], title: str, evidence: Path):
    out.mkdir(parents=True, exist_ok=True); outputs = {}
    for ext in ["pdf", "svg", "png"]:
        p = out / (stem + "." + ext)
        if ext == "pdf": meta = {"Title": title, "Author": "Chris Bartlett / ZWSLab", "CreationDate": TODAY}
        elif ext == "svg": meta = {"Title": title, "Creator": "ZWSLab publication rendering v1.29 (light palette)", "Date": "2026-09-05"}
        else: meta = {"Title": title, "Software": "ZWSLab existing-data rendering v1.29"}
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=BG, metadata=meta)
        outputs[p.name] = sha(p); p.with_name(p.name + ".sha256").write_text(sha(p) + "  " + p.name + "\n")
    plt.close(fig)
    return {"figure_id": stem, "source_files": {str(p.relative_to(evidence)).replace("\\", "/"): sha(p) for p in sources},
            "outputs": outputs, "new_simulation": False}

# ---------------------------------------------------------------- the three v0.2.1 charts, light
def render_v021_set(evidence: Path, out: Path) -> list[dict]:
    root = evidence / "experiments/reversible_null_phase_v031_nucleation_20260827"
    sp = root / "SURVIVAL_SUMMARY.csv"; fp = root / "FIRST_PASSAGE_RESULTS.csv"
    s = [r for r in read(sp) if r["stage"] == "primary_homogeneous_1024" and r["arrangement"] == "homogeneous" and r["endpoint"] == "T_hit"]
    f = [r for r in read(fp) if r["stage"] == "primary_homogeneous_1024" and r["arrangement"] == "homogeneous"]
    assert len(f) == 84
    ds = sorted({float(r["density"]) for r in f})
    fig = plt.figure(figsize=(13.8, 8.0), facecolor=BG)
    g = fig.add_gridspec(2, 2, left=.08, right=.975, top=.86, bottom=.12, hspace=.47, wspace=.28, height_ratios=[1.05, 1])
    ax = fig.add_subplot(g[0, :]); am = fig.add_subplot(g[1, 0]); af = fig.add_subplot(g[1, 1])
    for a in [ax, am, af]: style(a)
    medians = []
    for d, c in zip(ds, COLORS):
        sr = sorted([r for r in s if float(r["density"]) == d], key=lambda r: float(r["phase"]))
        ax.step([float(r["phase"]) / 1000 for r in sr], [float(r["survival"]) for r in sr], where="post", color=c, label=f"{d:.0%}", lw=2.2)
        m = next((float(r["phase"]) for r in sr if float(r["survival"]) <= .5), None); medians.append(m)
        y = (m if m is not None else 262144) / 1000
        am.scatter([100 * d], [y], color=c, s=60, marker="o" if m is not None else "^", edgecolor=TEXT, lw=.5, zorder=4)
        am.annotate(f"{y:.1f}" if m is not None else "not reached", (100 * d, y), xytext=(0, 10), textcoords="offset points", ha="center", color=TEXT, fontsize=8)
    assert medians == [None, 495616., 321536., 206848., 108544., 83968., 65536.]
    ax.set_title("A   Fraction not yet crossing the frozen geometrical endpoint", color=TEXT, loc="left", weight="bold", fontsize=11)
    ax.set_xlabel("Elapsed block phases (thousands)"); ax.set_ylabel("Kaplan–Meier survival"); ax.set_ylim(-.02, 1.03)
    leg = ax.legend(ncol=7, frameon=False, labelcolor=TEXT, title="Initial occupancy", loc="upper right", fontsize=9); leg.get_title().set_color(TEXT)
    am.plot([100 * d for d, m in zip(ds, medians) if m is not None], [m / 1000 for m in medians if m is not None], color="#56B4E9", lw=1)
    am.set_title("B   Median first-passage phase", color=TEXT, loc="left", weight="bold", fontsize=11)
    am.set_xlabel("Initial occupancy (%)"); am.set_ylabel("Phases (thousands)"); am.set_ylim(0, 555); am.set_xlim(15.3, 26.7)
    for h, offset, c in [(131072, -.17, "#0072B2"), (262144, .17, "#E69F00")]:
        vals = []
        for d in ds:
            rr = [r for r in f if float(r["density"]) == d]
            vals.append(sum(r["hit_observed"].lower() == "true" and float(r["T_hit"]) <= h for r in rr) / len(rr))
        af.bar(np.array(ds) * 100 + offset, vals, width=.32, color=c, label=f"by {h:,} phases")
    af.set_title("C   Same endpoint; different observation windows", color=TEXT, loc="left", weight="bold", fontsize=11)
    af.set_xlabel("Initial occupancy (%)"); af.set_ylabel("Fraction crossed"); af.set_ylim(0, 1.14); af.set_xlim(15.3, 26.7)
    af.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.suptitle("Density-sensitive first passage in homogeneous 1024 × 1024 worlds", color=TEXT, fontsize=17, weight="bold", y=.967)
    fig.text(.5, .913, "12 seeds per density • spectral diagnostics are separate from the event predicate", ha="center", color=MUTED, fontsize=10)
    fig.text(.5, .035, "Right censoring and actual cohort follow-up are retained. These curves do not locate a thermodynamic critical point.", ha="center", color=MUTED, fontsize=9)
    records = [save(fig, out, "COMPANION_V129_FIGURE_03_DENSITY_FIRST_PASSAGE", [sp, fp], "Density-sensitive first passage", evidence)]

    cp = evidence / "experiments/domain_mechanism_20260825/artifacts/double_conservation_screen/CONDITION_SUMMARIES.csv"
    ap = evidence / "experiments/reversible_null_phase_v030_20260827/balanced/report/SUMMARY.csv"
    controls = {r["condition_id"]: r for r in read(cp)}
    pooled = defaultdict(lambda: [0, 0])
    for r in read(ap):
        if r["stage"] == "ablation_512":
            pooled[r["rule_variant"]][0] += int(r["strong_passes"]); pooled[r["rule_variant"]][1] += int(r["trajectories"])
    assert pooled["winner"] == [10, 16] and pooled["qm_sibling"] == [0, 16] and pooled["rotation"] == [0, 16]
    order = ["winner"] + [f"ablate_{i:02}" for i in range(12)] + ["qm_sibling", "rotation"]
    fig = plt.figure(figsize=(14.4, 7.1), facecolor=BG)
    g = fig.add_gridspec(1, 2, left=.065, right=.985, top=.81, bottom=.27, wspace=.25, width_ratios=[1, 1.8])
    a = fig.add_subplot(g[0, 0]); b = fig.add_subplot(g[0, 1]); style(a); style(b)
    keys = ["winner", "qm_sibling", "rotation"]; x = np.arange(3); cc = [WINNER, "#56B4E9", "#009E73"]
    for offset, field, label, alpha in [(-.18, "median_peak_scale_ratio", "Peak", 1), (.18, "median_final_scale_ratio", "Final", .48)]:
        v = [float(controls[k][field]) for k in keys]; bars = a.bar(x + offset, v, .33, color=cc, alpha=alpha, label=label)
        for rect, val in zip(bars, v): a.text(rect.get_x() + rect.get_width() / 2, val * 1.08, f"{val:.2f}×", ha="center", color=TEXT, fontsize=8)
    a.set_yscale("log"); a.set_ylim(.7, 65); a.set_xticks(x, ["Winner\nQ fixed; M free", "Reactive sibling\nQ and M fixed", "Rotation"]); a.set_ylabel("Scale relative to each initial state")
    a.set_title("A   Focused 1024² control comparison", color=TEXT, loc="left", fontsize=11, weight="bold"); a.legend(frameon=False, labelcolor=TEXT)
    for i, k in enumerate(keys):
        r = controls[k]; a.text(i, 48, f"{r['strong_domain_count']}/{r['trajectory_count']} pass", ha="center", color=TEXT, fontsize=8)
    colors = [WINNER] + ["#D55E00" if pooled[k][0] == 0 else "#CC79A7" for k in order[1:-2]] + ["#56B4E9", "#009E73"]
    vals = [pooled[k][0] / pooled[k][1] for k in order]
    b.bar(range(len(order)), vals, color=colors)
    for i, k in enumerate(order): b.text(i, vals[i] + .025, f"{pooled[k][0]}/{pooled[k][1]}", ha="center", color=TEXT, fontsize=8, rotation=90 if vals[i] == 0 else 0)
    b.set_ylim(0, .82); b.set_ylabel("End-of-window strong-domain pass fraction"); b.set_xticks(range(len(order)), ["Winner"] + [f"Pair {i:02}" for i in range(12)] + ["Q+M", "Rotation"], rotation=55, ha="right")
    b.set_title("B   Single-pair ablations at 512² (pooled protocol)", color=TEXT, loc="left", fontsize=11, weight="bold")
    fig.suptitle("Matched controls constrain the mechanism in the tested rule family", color=TEXT, fontsize=17, weight="bold", y=.97)
    fig.text(.5, .90, "The reactive double-conserving sibling and rotation do not reproduce the same endpoint; several removals suppress it.", ha="center", color=MUTED, fontsize=10)
    fig.text(.5, .065, "Separate protocols are shown in A and B. These comparisons do not prove that double conservation forbids organisation in general.", ha="center", color=MUTED, fontsize=9)
    records.append(save(fig, out, "COMPANION_V129_FIGURE_04_CONTROLS_AND_ABLATIONS", [cp, ap], "Controls and ablations", evidence))

    # the nine step texts of the v0.2.1 figure, unchanged except British spelling
    steps = [("Discovery", "Search-selected rule; the large initial demonstration used a localised random patch."),
             ("Confirmation", "Frozen seeds reproduced irregular, nonmonotone large-domain growth."),
             ("Composition", "Dense regions are mixed excitations; sparse regions are near-vacuum."),
             ("Controls", "Pure rotation and the active Q+M sibling did not reproduce the same regime."),
             ("Homogeneous starts", "Separate whole-world random ensembles established density dependence."),
             ("First passage", "A geometrical endpoint was followed through longer observation windows."),
             ("Prepared shapes", "Large shapes persisted; relative-growth counts are not absolute formation tests."),
             ("Intervention audit", "Shuffling destroyed measured structure while crossing the legacy detector."),
             ("Current conclusion", "The causal history claim is withdrawn. The observed reversible domain phenomenon remains.")]
    fig, ax = plt.subplots(figsize=(14.4, 8.8), facecolor=BG); ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(0, 3.2)
    for i, (title, body) in enumerate(steps):
        col = i % 3; row = 2 - i // 3; x0 = col + .04; y0 = row + .16
        ax.add_patch(plt.Rectangle((x0, y0), .91, .77, facecolor=PANEL, edgecolor=COLORS[i % 7], lw=1.8))
        ax.text(x0 + .04, y0 + .67, f"{i + 1}. {title}", color=TEXT, fontsize=11.3, weight="bold", va="top")
        ax.text(x0 + .04, y0 + .48, textwrap.fill(body, 37), color=MUTED, fontsize=9.5, va="top", linespacing=1.28)
    ax.text(1.5, 3.16, "A bounded research programme, with its corrections retained", ha="center", color=TEXT, fontsize=17, weight="bold")
    ax.text(1.5, .025, "Explanatory timeline of existing studies • no new scientific measurement", ha="center", color=MUTED, fontsize=10)
    records.append(save(fig, out, "COMPANION_V129_FIGURE_08_CORRECTED_RESEARCH_STORY", [], "Corrected research chronology", evidence))
    return records

# ---------------------------------------------------------------- figures 06 and 07, light variants
def render_history_set(run: Path, experiment: Path, out: Path) -> list[dict]:
    import zws_reversible_null_phase.history_figures as hf
    from zws_reversible_null_phase.history_figures import (
        _representatives, _source_result, index_task_states, _record_state, aligned_densest_crop, crop_state,
        state_histogram, STATE_HEX, STATE_NAMES, CMAP, _state_title, _save_triplet, _state_panel_from_record, make_panel_record)
    # the module sets a dark rcParams theme at import; restore the light one
    plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "legend.labelcolor": TEXT})

    def show_state(axis, state, *, title, phase, crop=None):
        shown = crop_state(state, crop) if crop is not None else state
        axis.imshow(shown, cmap=CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest", rasterized=True)
        axis.set_title(_state_title(state, title, phase), fontsize=10, loc="left", color=TEXT)
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_color(MUTED)

    def chart_style(axis, title):
        axis.set_facecolor(PANEL); axis.set_title(title, color=TEXT, loc="left", fontsize=13)
        axis.tick_params(colors=TEXT)
        for spine in axis.spines.values(): spine.set_color(MUTED)
        axis.grid(color=GRID, linewidth=0.7)

    def state_legend(fig, *, y=0.01):
        handles = [Patch(facecolor=c, edgecolor="#00000055", label=n) for n, c in zip(STATE_NAMES, STATE_HEX, strict=True)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, frameon=False, labelcolor=TEXT)

    # ---- figure 06 (same selection logic as history_figures.figure_06_micro_macro)
    representative = _representatives(experiment)[0]
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    source = _source_result(run, source_id)
    target_phase = int(source["time_to_event"]["T_hit"])
    records = index_task_states(run / "sources" / source_id)
    exact = [r for r in records if int(r["phase"]) == target_phase]
    if exact:
        record = min(exact, key=lambda row: row["path"])
    else:
        before = [r for r in records if int(r["phase"]) <= target_phase]
        record = max(before, key=lambda row: int(row["phase"]))
    state, _, _, _ = _record_state(record)
    crop256 = aligned_densest_crop(state, 256); crop64 = aligned_densest_crop(state, 64, bounds=crop256)
    local = crop_state(state, crop64); histogram = state_histogram(local)
    probabilities = histogram / histogram.sum()
    entropy = float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])) / np.log(4.0))
    occupied = (state != 0).astype(np.float32); block = 32
    coarse = occupied.reshape(state.shape[0] // block, block, state.shape[1] // block, block).mean(axis=(1, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG)
    fig.suptitle("Microscopic disorder can coexist with macroscopic order", fontsize=21, color=TEXT)
    show_state(axes[0, 0], state, title="Full 1024×1024 world", phase=int(record["phase"]))
    show_state(axes[0, 1], state, title=f"256×256 crop · ({crop256[0]},{crop256[1]})", phase=int(record["phase"]), crop=crop256)
    show_state(axes[0, 2], state, title=f"64×64 crop · ({crop64[0]},{crop64[1]})", phase=int(record["phase"]), crop=crop64)
    chart_style(axes[1, 0], "Local four-state histogram")
    axes[1, 0].bar(STATE_NAMES, histogram, color=STATE_HEX, edgecolor=MUTED, linewidth=0.6)
    axes[1, 0].tick_params(axis="x", rotation=20); axes[1, 0].set_ylabel("cells in 64×64 crop")
    chart_style(axes[1, 1], "Coarse occupancy field · 32×32 blocks")
    image = axes[1, 1].imshow(coarse, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="occupied fraction")
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
    axes[1, 2].set_facecolor(PANEL); axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.82, f"Normalised local entropy\n{entropy:.3f} / 1.000", color="#8a7a00", fontsize=22, weight="bold")
    axes[1, 2].text(0.05, 0.55, "Large-scale domains are visible\nwithout a simple microscopic\ncheckerboard, stripe, or crystal.", color=TEXT, fontsize=16)
    axes[1, 2].text(0.05, 0.25, "This panel describes morphology;\nit does not claim equilibrium\nor a universal phase transition.", color=MUTED, fontsize=12)
    state_legend(fig, y=0.005)
    base = dict(source_task=source_id, seed=seed, density=density, branch_point="source_event", intervention="untouched", requested_elapsed=None)
    full = (0, 0, state.shape[1], state.shape[0])
    panels = [
        _state_panel_from_record(record, crop=full, extra={"panel": "full_world"}, **base),
        _state_panel_from_record(record, crop=crop256, extra={"panel": "crop_256", "crop_selection": "frozen aligned maximum occupancy"}, **base),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "crop_64", "crop_selection": "frozen nested aligned maximum occupancy"}, **base),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "local_histogram", "histogram": histogram.tolist()}, **base),
        _state_panel_from_record(record, crop=full, extra={"panel": "coarse_density", "coarse_block_size": 32}, **base),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "normalized_local_entropy", "normalized_entropy": entropy}, **base),
    ]
    rec06 = _save_triplet(fig, out, "FIGURE_06_MICRO_DISORDER_MACRO_ORDER_V129", panels=panels,
                          caption="A frozen event-centred representative shown at full-world, 256-cell, and 64-cell scales (light palette re-render, v1.29).",
                          major_morphology=True)
    rec06["entropy"] = entropy; rec06["histogram"] = histogram.tolist(); rec06["phase"] = int(record["phase"])

    # ---- figure 07 (model schematic), light
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=BG)
    fig.suptitle("The mechanism ledger: exact charge balance does not freeze occupancy", fontsize=22, color=TEXT)
    for axis in axes:
        axis.set_facecolor(PANEL); axis.set_xlim(0, 10); axis.set_ylim(0, 10); axis.axis("off")
    left, right = axes
    left.set_title("Frozen winner: exact Q, variable M", color="#8a7a00", fontsize=16)
    left.text(5, 8.7, r"$Q=N_{+}-N_{-}=0$ exactly", ha="center", color=TEXT, fontsize=22)
    left.text(5, 7.7, r"$M=N_{+}+N_{-}+N_{bound}$ may change", ha="center", color=TEXT, fontsize=19)
    for x, color, label in ((3.8, STATE_HEX[1], "+"), (6.2, STATE_HEX[2], "-")):
        left.add_patch(Rectangle((x - 0.65, 4.8), 1.3, 1.3, facecolor=color, edgecolor=TEXT, linewidth=2))
        left.text(x, 5.45, label, ha="center", va="center", color="#05070a" if label == "+" else "white", fontsize=24, weight="bold")
    left.add_patch(FancyArrowPatch((5, 7.1), (5, 6.2), arrowstyle="-|>", mutation_scale=18, color=MUTED))
    left.text(5, 4.1, "Balanced pair creation changes M by +2\nwithout changing Q", ha="center", color=TEXT, fontsize=15)
    left.text(5, 2.1, "Trajectory evidence measures when and where\nthis changing inventory becomes organised.", ha="center", color=MUTED, fontsize=13)
    right.set_title("Exact-Q+M sibling control", color="#0072B2", fontsize=16)
    right.text(5, 8.7, r"$Q=0$ and $M=M_{initial}>0$", ha="center", color=TEXT, fontsize=22)
    right.text(5, 7.7, "The same number of occupied cells is retained", ha="center", color=TEXT, fontsize=16)
    for i, (color, label) in enumerate(((STATE_HEX[1], "+"), (STATE_HEX[2], "-"), (STATE_HEX[3], "b"))):
        x = 3.1 + i * 1.9
        right.add_patch(Rectangle((x - 0.6, 4.8), 1.2, 1.2, facecolor=color, edgecolor=TEXT, linewidth=2))
        right.text(x, 5.4, label, ha="center", va="center", color="white" if i else "#05070a", fontsize=22, weight="bold")
    right.text(5, 4.0, "Rearrangements may occur, but total M is fixed", ha="center", color=TEXT, fontsize=15)
    right.text(5, 2.1, "This control does not reproduce the winner’s\nstrong-domain regime under the tested protocol.", ha="center", color=MUTED, fontsize=13)
    fig.text(0.5, 0.03, "MODEL SCHEMATIC - not a trajectory snapshot", ha="center", color="#a23b72", fontsize=13, weight="bold")
    model_source = Path(hf.__file__).resolve()
    panels = [make_panel_record(source_task="model_schematic", seed=None, density=None, branch_point=None, intervention="frozen_winner_ledger",
                                phase="not_applicable", state_hash="model_schematic_no_state", crop=(0, 0, 0, 0), source_path=model_source,
                                extra={"panel": "variable_M_winner", "evidence_type": "MODEL SCHEMATIC"}),
              make_panel_record(source_task="model_schematic", seed=None, density=None, branch_point=None, intervention="exact_Q_M_sibling_control",
                                phase="not_applicable", state_hash="model_schematic_no_state", crop=(0, 0, 0, 0), source_path=model_source,
                                extra={"panel": "exact_Q_M_control", "evidence_type": "MODEL SCHEMATIC"})]
    rec07 = _save_triplet(fig, out, "FIGURE_07_MECHANISM_LEDGER_V129", panels=panels,
                          caption="Model schematic (light palette re-render, v1.29). The winner keeps signed charge Q exactly zero but permits occupancy M to change; the exact-Q+M sibling instead fixes a positive, nonzero occupancy.",
                          major_morphology=False)
    return [rec06, rec07]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="repository root (the release folder); supplies the compact CSV tables")
    p.add_argument("--out", type=Path, required=True, help="output folder for the rendered figures, e.g. paper/figures")
    p.add_argument("--run", type=Path, default=None,
                   help="OPTIONAL: the author-retained v0.3.2 run directory (archived state arrays). Without it only the three "
                        "compact-evidence figures (03, 04, 08) are rendered; the two state-based figures (06, 07) need the arrays.")
    p.add_argument("--manifest", type=Path, default=None, help="where to write the render manifest (default: <out>/../V129_LIGHT_RENDER_MANIFEST.json)")
    a = p.parse_args()
    repo = a.repo.resolve(); out = a.out.resolve()
    recs = render_v021_set(repo, out)
    if a.run is not None:
        recs += render_history_set(a.run.resolve(), (repo / "experiments" / "reversible_null_phase_v032_history_erasure_20260829").resolve(), out)
        scope = "all five data figures"
    else:
        scope = "compact-evidence figures 03, 04 and 08 only (no --run directory given; figures 06 and 07 need the retained state arrays)"
    manifest = {"version": "1.29", "scope": scope,
                "purpose": "light-palette re-render of the data figures; same data and code paths, colours only",
                "derived_from": ["V021_RENDER_MANIFEST.json (figures 03, 04, 08)", "history_figures.figure_06_micro_macro", "history_figures.figure_07_mechanism_ledger"],
                "rule_sha256": RULE, "new_simulations": False, "rendered_utc": TODAY.isoformat(), "figures": recs}
    mp = a.manifest.resolve() if a.manifest else out.parent / "V129_LIGHT_RENDER_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(mp, "|", scope)
