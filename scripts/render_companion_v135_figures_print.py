"""Render the companion paper's code-generated figures to one print typography (v1.35).

Every figure is drawn on a canvas 12.44 inches wide, twice the paper's 15.8 cm text width, with its
type at twice the printed size, and the LaTeX sources include each figure at scale 0.5. On the page
every figure therefore prints with the same type sizes by construction:

  figure title 10 pt bold, panel title 8.5 pt bold, axis label 7.5 pt, tick and annotation 7 pt,
  state-panel header 6.5 pt; the two 24-panel branch grids use 6.5 pt headers, six columns having
  to share the width.

Palette: the v1.29 light palette (white canvas, slate text, Okabe-Ito accents); state panels keep the
exact four-state colours because they are pictures of data. Same stored data and measurements as the
earlier renders, and no simulator is run. Commentary drawn inside the earlier figures
(footers, side notes) is not carried over: a figure carries its picture, its labels and its key, and
the argument belongs to the text and the caption. State-panel headers give the phase actually stored
in each panel and its own elapsed offset, so a panel whose nearest stored state lies before the
requested offset is labelled with the offset it really has.

Outputs (new stems; the earlier renders and their manifests stay in place as provenance):
  COMPANION_V135_FIGURE_02_HOW_THE_RULE_WORKS         vector redraw of the v0.2.3 raster, from the frozen rule table
  COMPANION_V135_FIGURE_03_DENSITY_FIRST_PASSAGE      frozen CSV tables
  COMPANION_V135_FIGURE_04_CONTROLS_AND_ABLATIONS     frozen CSV tables
  COMPANION_V135_FIGURE_08_CORRECTED_RESEARCH_STORY   text only
  FIGURE_07_MECHANISM_LEDGER_V135                     model schematic
  FIGURE_05_MULTISCALE_MEMORY_V135                    v0.3.2 experiment tables
  with --run (the author-retained v0.3.2 run directory holding the archived state arrays):
  FIGURE_06_MICRO_DISORDER_MACRO_ORDER_V135
  FIGURE_02A_HISTORY_ERASURE_BRANCHES_V135, FIGURE_02B_HISTORY_ERASURE_BRANCHES_V135   (one density per page)
  FIGURE_03_FIXED_OCCUPANCY_HISTORY_ERASURE_V135
  FIGURE_04_EVENT_PROXIMAL_HISTORY_ERASURE_V135

Usage (from the repository root, with the extension package importable and torch available):
  python scripts/render_companion_v135_figures_print.py --repo . --out <output folder> [--run <run dir>] [--manifest <path>]

Without --run the six table- and text-driven figures are rendered from the release's rule file and CSV tables, and the
manifest goes to <output folder>/V135_PRINT_RENDER_MANIFEST_compact.json, so the shipped eleven-figure manifest in
paper/ is never replaced by a partial one; use a fresh output folder. With --run (the author-retained v0.3.2 run
directory: REALIZED_BRANCH_PLAN.json, sources/<id>/source_result.json, media_states/ and branches/ state arrays) the five
state-based figures are rendered too; they also verify, through the campaign's own readers, the experiment folder's
INTEGRITY_AUDIT.json, REPRESENTATIVE_STATE_INDEX.csv, RESIDUAL_FIRST_PASSAGE.csv, HISTORY_ERASURE_RESULTS.csv,
INTERVENTION_AUDIT.csv and FROZEN_PLAN.json with their .sha256 sidecars, all of which the release includes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, struct, textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter

KFMT = FuncFormatter(lambda v, p: "0" if v == 0 else f"{v / 1000:g}k")   # phase counts in thousands on tick labels

import zws_reversible_null_phase.history_figures as hf   # data access helpers of the v0.3.2 campaign; sets a dark theme at import

VERSION = "1.35"
TODAY = datetime(2026, 9, 6, tzinfo=timezone.utc)
RULE = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"

# ---- print typography: canvas twice the text width, type twice its printed size, included at scale 0.5
CANVAS_W = 12.44                                          # inches, 2 x 15.8 cm: the widest a cropped canvas may be
FIG_W = 12.30                                             # width handed to matplotlib; the tight crop plus its pad stays under CANVAS_W
F_TITLE, F_PANEL, F_LABEL, F_TICK, F_ANNOT, F_HEADER, F_GRID = 19, 15, 14, 13, 13, 12, 13
PNG_DPI = 340                                             # about 4000 px across a cropped canvas, above the 3840 px floor of the v0.3.2 figure code
PDF_DPI = 300                                             # resolution of the rasterised state panels inside the vector PDFs

# ---- light palette (v1.29), Okabe-Ito accents, green kept off the boxes and labels
BG = "#ffffff"; PANEL = "#f6f8fa"; TEXT = "#233240"; MUTED = "#5a6b7a"; GRID = "#cfd8e0"; EDGE = "#b9c4cd"
COLORS = ["#6e7f8d", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#B8A400", "#D55E00"]   # densities 16..26 %
WINNER = "#B8A400"; BLUE = "#0072B2"; VERMILION = "#D55E00"; SKY = "#56B4E9"; PINK = "#CC79A7"; ORANGE = "#E69F00"; GREEN = "#009E73"
STATE_HEX = ("#05070a", "#f0e442", "#0072b2", "#cc79a7")  # pictures of data keep the exact state colours
STATE_NAMES = ("empty", "positive", "negative", "bound-neutral")
INTERVENTION = {"untouched": WINNER, "global_shuffle": SKY, "tile_32": PINK, "tile_128": ORANGE}
SIBLING_NAME = "Exact-Q+M sibling"
CMAP = ListedColormap(STATE_HEX)
assert tuple(hf.STATE_HEX) == STATE_HEX and tuple(hf.STATE_NAMES) == STATE_NAMES


def apply_rc() -> None:
    """The one typography, applied before every figure (history_figures installs a dark theme at import)."""
    plt.rcParams.update({
        "font.size": F_TICK, "axes.titlesize": F_PANEL, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": F_LABEL, "xtick.labelsize": F_TICK, "ytick.labelsize": F_TICK,
        "legend.fontsize": F_ANNOT, "legend.title_fontsize": F_ANNOT, "legend.frameon": False,
        "figure.titlesize": F_TITLE, "figure.titleweight": "bold",
        "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "legend.labelcolor": TEXT,
        "axes.edgecolor": MUTED, "axes.facecolor": PANEL, "figure.facecolor": BG, "savefig.facecolor": BG,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "svg.hashsalt": "zwslab-companion",     # deterministic element ids, so a re-render leaves identical SVGs
    })


# ---------------------------------------------------------------- small helpers
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def style(ax) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def chart(ax, title: str) -> None:
    style(ax)
    ax.set_title(title)


_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def pdf_size_in(p: Path) -> tuple[float, float]:
    m = _MEDIABOX.search(p.read_bytes())
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def png_size(p: Path) -> tuple[int, int]:
    with p.open("rb") as fh:
        fh.read(16)
        return struct.unpack(">II", fh.read(8))


def save_set(fig, out: Path, stem: str, title: str) -> tuple[dict[str, str], tuple[float, float]]:
    """PDF, SVG and PNG with sha256 sidecars; returns the hashes and the PDF canvas size in inches."""
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer(); limit = fig.get_figwidth() * fig.dpi
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bb = text.get_window_extent(renderer)
        if bb.width and (bb.x1 > limit * 1.004 or bb.x0 < -limit * 0.004) and text.get_text().strip()[0] not in "-−0123456789$":   # tick labels beyond the limits are not drawn
            raise RuntimeError(f"{stem}: text runs past the canvas edge ({bb.x0 / fig.dpi:.2f}..{bb.x1 / fig.dpi:.2f} in): {text.get_text()[:60]!r}")
    outputs: dict[str, str] = {}
    for ext in ("pdf", "svg", "png"):
        p = out / f"{stem}.{ext}"
        if ext == "pdf":
            meta = {"Title": title, "Author": "Chris Bartlett / ZWSLab", "CreationDate": TODAY}
        elif ext == "svg":
            meta = {"Title": title, "Creator": f"ZWSLab publication rendering v{VERSION} (print typography, light palette)", "Date": "2026-09-06"}
        else:
            meta = {"Title": title, "Software": f"ZWSLab existing-data rendering v{VERSION}"}
        fig.savefig(p, dpi=PNG_DPI if ext == "png" else PDF_DPI, bbox_inches="tight", pad_inches=0.04, facecolor=BG, metadata=meta)
        outputs[p.name] = sha(p)
        (out / (p.name + ".sha256")).write_text(f"{outputs[p.name]}  {p.name}\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    w, h = pdf_size_in(out / f"{stem}.pdf")
    if w > CANVAS_W:
        raise RuntimeError(f"{stem}: canvas {w:.2f} in exceeds the text width at scale 0.5")
    print(f"  {stem}: canvas {w:.2f} x {h:.2f} in, prints {w / 2:.2f} x {h / 2:.2f} in")
    return outputs, (w, h)


def compact_record(stem: str, sources: list[Path], repo: Path, outputs: dict[str, str], size: tuple[float, float]) -> dict:
    return {"figure_id": stem, "source_files": {str(p.relative_to(repo)).replace("\\", "/"): sha(p) for p in sources},
            "outputs": outputs, "canvas_inches": [round(size[0], 3), round(size[1], 3)], "new_simulation": False}


def state_record(stem: str, panels: list[dict], caption: str, outputs: dict[str, str], size: tuple[float, float],
                 *, major_morphology: bool, out: Path) -> dict:
    for panel in panels:
        hf.validate_panel_record(panel)
    records = []
    for name, digest in outputs.items():
        rec = {"path": str((out / name).resolve()), "sha256": digest, "format": name.rsplit(".", 1)[1]}
        if name.endswith(".png"):
            w, h = png_size(out / name)
            rec["pixels"] = [w, h]
            if major_morphology and (w < 3840 or h < 2160):
                raise RuntimeError(f"major morphology figure is below 3840x2160: {name} ({w}x{h})")
        records.append(rec)
    return {"figure_id": stem, "caption": caption, "panels": panels, "outputs": records, "major_morphology": major_morphology,
            "canvas_inches": [round(size[0], 3), round(size[1], 3)], "new_simulation": False}


def q_m(state: np.ndarray) -> tuple[int, int]:
    h = hf.state_histogram(state)
    return int(h[1] - h[2]), int(h[1:].sum())


def show_state(axis, state: np.ndarray, lines: list[str], *, crop=None, header: int = F_HEADER) -> None:
    shown = hf.crop_state(state, crop) if crop is not None else state
    axis.imshow(shown, cmap=CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest", rasterized=True)
    axis.set_title("\n".join(lines), fontsize=header, loc="left", color=TEXT, weight="normal")
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(MUTED)


def phase_line(phase: int, state: np.ndarray) -> str:
    """Stored phase and occupancy; the signed charge is printed only if it is not the zero every panel shares."""
    q, m = q_m(state)
    return f"phase {phase:,} · M={m:,}" if q == 0 else f"phase {phase:,} · Q={q:,} · M={m:,}"


def header_lines(name: str, phase: int, state: np.ndarray) -> list[str]:
    return [name, phase_line(phase, state).replace("M=", "global M=")]


def state_legend(fig, *, y: float, blend_note: bool = False) -> None:
    handles = [Patch(facecolor=c, edgecolor="#00000055", label=n) for n, c in zip(STATE_NAMES, STATE_HEX, strict=True)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, frameon=False, fontsize=F_ANNOT)
    if blend_note:
        fig.text(.5, y - .018, "at this size the occupied states blend to grey; black is empty",
                 ha="center", va="top", fontsize=F_ANNOT, color=MUTED)


# ---------------------------------------------------------------- Figure 2: how one block update works (vector)
CHARGE = {0: 0, 1: 1, 2: -1, 3: 0}
OCCUPANCY = {0: 0, 1: 1, 2: 1, 3: 1}


def decode(code: int) -> list[int]:
    return [(code // (4 ** index)) % 4 for index in range(4)]      # top-left, top-right, bottom-left, bottom-right


def invariants(states: list[int]) -> tuple[int, int]:
    return sum(CHARGE[s] for s in states), sum(OCCUPANCY[s] for s in states)


def render_rule_figure(repo: Path, out: Path) -> dict:
    apply_rc()
    rule_path = repo / "extensions/reversible_null_phase/rules/winner_d1fe99.json"
    table = [int(v) for v in json.loads(rule_path.read_text(encoding="utf-8"))["table"]]
    if sorted(table) != list(range(256)):
        raise ValueError("frozen rule is not a permutation of 0..255")
    if hashlib.sha256(bytes(table)).hexdigest() != RULE:
        raise ValueError("frozen rule hash changed")
    for code_in, code_out in enumerate(table):
        if invariants(decode(code_in))[0] != invariants(decode(code_out))[0]:
            raise ValueError(f"rule violates signed charge at input {code_in}")

    W, H = 3600, 2500                                     # the v0.2.3 raster's coordinate frame, reused for the layout
    fig = plt.figure(figsize=(FIG_W, FIG_W * H / W), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")

    def rbox(x0, y0, x1, y1, r, **kw):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", **kw))

    def chip(x, y, size, state, symbol_size):
        rbox(x, y, x + size, y + size, size * 0.15, facecolor=STATE_HEX[state], edgecolor=TEXT if state else EDGE, linewidth=1.2)
        ax.text(x + size / 2, y + size / 2, {0: "0", 1: "+", 2: "−", 3: "b"}[state], ha="center", va="center",
                fontsize=symbol_size, weight="bold", color="#ffffff" if state in (0, 2, 3) else "#101010")

    def arrow(x0, y0, x1, y1, color):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color=color))

    ax.text(180, 95, "How one reversible block update works", fontsize=F_TITLE, weight="bold", color=TEXT, va="top")
    ax.text(180, 232, "Exact entries from the frozen 256-entry permutation", fontsize=F_LABEL, color=MUTED, va="top")
    for x0, y0, x1, y1 in [(150, 360, 1735, 1260), (1865, 360, 3450, 1260), (150, 1370, 1735, 2350), (1865, 1370, 3450, 2350)]:
        rbox(x0, y0, x1, y1, 40, facecolor=PANEL, edgecolor=EDGE, linewidth=1.0)

    # A: the four cell states
    ax.text(220, 410, "A   Four cell states", fontsize=F_PANEL, weight="bold", color=TEXT, va="top")
    for idx, (state, name, values) in enumerate([(0, "empty", "Q = 0,  M = 0"), (1, "positive", "Q = +1,  M = 1"),
                                                  (2, "negative", "Q = −1,  M = 1"), (3, "bound-neutral", "Q = 0,  M = 1")]):
        row, col = divmod(idx, 2); x = 240 + col * 735; y = 555 + row * 290
        chip(x, y, 190, state, 26)
        ax.text(x + 225, y + 28, name, fontsize=F_LABEL, weight="bold", color=TEXT, va="top")
        ax.text(x + 225, y + 105, values, fontsize=F_ANNOT, color=MUTED, va="top")

    # B: alternating Margolus partitions
    ax.text(1935, 410, "B   Alternating partitions", fontsize=F_PANEL, weight="bold", color=TEXT, va="top")

    def partition(x0, y0, shifted, color):
        cell = 102
        for r in range(6):
            for c in range(6):
                ax.add_patch(Rectangle((x0 + c * cell, y0 + r * cell), cell, cell, facecolor="#ffffff", edgecolor=GRID, linewidth=0.8))
        # Continuous boundaries cover every cell. Odd-cut edge pieces join on the torus.
        off = 1 if shifted else 0
        for boundary in range(off, 7, 2):
            ax.plot([x0 + boundary * cell] * 2, [y0, y0 + 6 * cell], color=color, lw=3)
            ax.plot([x0, x0 + 6 * cell], [y0 + boundary * cell] * 2, color=color, lw=3)
        if shifted:
            for x, y, marker in [(x0 + 12, y0 + 3 * cell, "<"), (x0 + 6 * cell - 12, y0 + 3 * cell, ">"),
                                 (x0 + 3 * cell, y0 + 12, "^"), (x0 + 3 * cell, y0 + 6 * cell - 12, "v")]:
                ax.scatter([x], [y], marker=marker, s=28, color=color, zorder=4)

    partition(1970, 555, False, BLUE); partition(2755, 555, True, VERMILION)
    ax.text(2660, 500, "one-cell diagonal shift; edges wrap", ha="center", fontsize=F_ANNOT, color=MUTED, va="top")
    ax.text(1970 + 306, 1195, "phase 0", ha="center", fontsize=F_ANNOT, weight="bold", color=BLUE, va="top")
    ax.text(2755 + 306, 1195, "phase 1", ha="center", fontsize=F_ANNOT, weight="bold", color=VERMILION, va="top")
    arrow(2610, 861, 2725, 861, MUTED)

    # C and D: two exact table entries
    examples = []
    expected = {1: (1, 1, 1, 1), 130: (-2, 2, -2, 4)}
    for k, (code_in, title, accent, note) in enumerate([(1, "C   Transport-like lookup", BLUE, "transport: Q and M unchanged"),
                                                       (130, "D   Balanced reaction lookup", VERMILION, "balanced pair creation: Q unchanged, M + 2")]):
        code_out = table[code_in]; s_in = decode(code_in); s_out = decode(code_out)
        q0, m0 = invariants(s_in); q1, m1 = invariants(s_out)
        assert (q0, m0, q1, m1) == expected[code_in], (code_in, q0, m0, q1, m1)
        left = 150 if k == 0 else 1865
        ax.text(left + 70, 1420, title, fontsize=F_PANEL, weight="bold", color=TEXT, va="top")
        ax.text(left + 75, 1530, f"code {code_in}", fontsize=F_ANNOT, weight="bold", color=accent, va="top")
        ax.text(left + 890, 1530, f"code {code_out}", fontsize=F_ANNOT, weight="bold", color=accent, va="top")
        for gx, states in ((left + 90, s_in), (left + 915, s_out)):
            for i, s in enumerate(states):
                r, c = divmod(i, 2)
                chip(gx + c * 220 + 5, 1615 + r * 220 + 5, 210, s, 22)
        arrow(left + 610, 1835, left + 870, 1835, accent)
        ax.text(left + 75, 2110, f"Q: {q0:+d} → {q1:+d}      M: {m0} → {m1}".replace("-", "−"), fontsize=F_LABEL, weight="bold", color=TEXT, va="top")
        ax.text(left + 75, 2205, note, fontsize=F_ANNOT, color=MUTED, va="top")
        examples.append({"input_code": code_in, "output_code": code_out, "input_states_tl_tr_bl_br": s_in, "output_states_tl_tr_bl_br": s_out,
                         "input_Q_M": [q0, m0], "output_Q_M": [q1, m1]})
    stem = "COMPANION_V135_FIGURE_02_HOW_THE_RULE_WORKS"
    outputs, size = save_set(fig, out, stem, "How one reversible block update works")
    rec = compact_record(stem, [], repo, outputs, size)
    rec["worked_rule"] = {"rule_file": "extensions/reversible_null_phase/rules/winner_d1fe99.json",
                          "rule_raw_uint8_sha256": RULE,
                          "rule_file_digest": "not recorded here: the rule is identified by the raw 256-byte table digest above; the JSON file's "
                                              "own digest is in the release MANIFEST.csv (its path field is rewritten for release, see REDACTIONS.md)",
                          "state_order": "top-left, top-right, bottom-left, bottom-right", "block_code": "s_TL + 4*s_TR + 16*s_BL + 64*s_BR",
                          "examples": examples, "derived_from": "scripts/render_companion_v023_figures.py (raster, v0.2.3)"}
    return rec


# ---------------------------------------------------------------- Figures 3, 4 and the supplement's timeline (frozen tables)
def render_density(repo: Path, out: Path) -> dict:
    apply_rc()
    root = repo / "experiments/reversible_null_phase_v031_nucleation_20260827"
    sp = root / "SURVIVAL_SUMMARY.csv"; fp = root / "FIRST_PASSAGE_RESULTS.csv"
    s = [r for r in read_csv(sp) if r["stage"] == "primary_homogeneous_1024" and r["arrangement"] == "homogeneous" and r["endpoint"] == "T_hit"]
    f = [r for r in read_csv(fp) if r["stage"] == "primary_homogeneous_1024" and r["arrangement"] == "homogeneous"]
    assert len(f) == 84
    ds = sorted({float(r["density"]) for r in f})
    fig = plt.figure(figsize=(FIG_W, 7.8), facecolor=BG)
    g = fig.add_gridspec(2, 2, left=.075, right=.985, top=.905, bottom=.09, hspace=.72, wspace=.26, height_ratios=[1.05, 1])
    ax = fig.add_subplot(g[0, :]); am = fig.add_subplot(g[1, 0]); af = fig.add_subplot(g[1, 1])
    for a in (ax, am, af):
        style(a)
    medians = []
    for d, c in zip(ds, COLORS):
        sr = sorted([r for r in s if float(r["density"]) == d], key=lambda r: float(r["phase"]))
        ax.step([float(r["phase"]) / 1000 for r in sr], [float(r["survival"]) for r in sr], where="post", color=c, label=f"{d:.0%}", lw=2.2,
                zorder=10 if d == .16 else 2)
        m = next((float(r["phase"]) for r in sr if float(r["survival"]) <= .5), None); medians.append(m)
        if m is not None:
            y = m / 1000
            am.scatter([100 * d], [y], color=c, s=70, marker="o", edgecolor=TEXT, lw=.6, zorder=4)
            am.annotate(f"{y:.1f}", (100 * d, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=TEXT, fontsize=F_ANNOT)
        elif d == .16:
            ax.plot([262.144], [1], marker="+", markersize=10, markeredgewidth=2, color=c, zorder=11)
            ax.text(365, .24, "16%: none crossed\nby 262,144 phases", fontsize=F_ANNOT, color=TEXT, zorder=12)
            am.text(.04, .13, "16%: median not reached;\nfollow-up 262,144 phases", transform=am.transAxes,
                    fontsize=F_ANNOT, color=TEXT, va="bottom")
    assert medians == [None, 495616., 321536., 206848., 108544., 83968., 65536.]
    ax.set_title("A   Runs not yet across the frozen endpoint (12 seeds per density)")
    ax.set_xlabel("Elapsed block phases (thousands)"); ax.set_ylabel("Kaplan–Meier survival"); ax.set_ylim(-.02, 1.03)
    ax.legend(ncol=7, title="Initial occupancy", loc="upper center", bbox_to_anchor=(0.5, -0.2))   # below the panel: every region inside it carries a curve
    am.plot([100 * d for d, m in zip(ds, medians) if m is not None], [m / 1000 for m in medians if m is not None], color=SKY, lw=1.2)
    am.set_title("B   Median first-passage phase")
    am.set_xlabel("Initial occupancy (%)"); am.set_ylabel("Phases (thousands)"); am.set_ylim(0, 555); am.set_xlim(15.3, 26.7)
    for h, offset, c in [(131072, -.17, BLUE), (262144, .17, ORANGE)]:
        vals = []
        for d in ds:
            rr = [r for r in f if float(r["density"]) == d]
            vals.append(sum(r["hit_observed"].lower() == "true" and float(r["T_hit"]) <= h for r in rr) / len(rr))
        af.bar(np.array(ds) * 100 + offset, vals, width=.32, color=c, label=f"by {h:,} phases")
    af.set_title("C   Two observation windows")
    af.set_xlabel("Initial occupancy (%)"); af.set_ylabel("Fraction crossed"); af.set_ylim(0, 1.14); af.set_xlim(15.3, 26.7)
    af.legend(loc="upper left")
    fig.suptitle("Density-sensitive first passage in homogeneous 1024 × 1024 worlds", y=.985)
    stem = "COMPANION_V135_FIGURE_03_DENSITY_FIRST_PASSAGE"
    outputs, size = save_set(fig, out, stem, "Density-sensitive first passage")
    return compact_record(stem, [sp, fp], repo, outputs, size)


def render_controls(repo: Path, out: Path) -> dict:
    apply_rc()
    cp = repo / "experiments/domain_mechanism_20260825/artifacts/double_conservation_screen/CONDITION_SUMMARIES.csv"
    ap = repo / "experiments/reversible_null_phase_v030_20260827/balanced/report/SUMMARY.csv"
    controls = {r["condition_id"]: r for r in read_csv(cp)}
    pooled = defaultdict(lambda: [0, 0])
    for r in read_csv(ap):
        if r["stage"] == "ablation_512":
            pooled[r["rule_variant"]][0] += int(r["strong_passes"]); pooled[r["rule_variant"]][1] += int(r["trajectories"])
    assert pooled["winner"] == [10, 16] and pooled["qm_sibling"] == [0, 16] and pooled["rotation"] == [0, 16]
    order = ["winner"] + [f"ablate_{i:02}" for i in range(12)] + ["qm_sibling", "rotation"]
    fig = plt.figure(figsize=(FIG_W,6.6), facecolor=BG)
    g = fig.add_gridspec(1, 2, left=.07, right=.985, top=.87, bottom=.29, wspace=.26, width_ratios=[1.15, 1.65])
    a = fig.add_subplot(g[0, 0]); b = fig.add_subplot(g[0, 1]); style(a); style(b)
    keys = ["winner", "qm_sibling", "rotation"]; x = np.arange(3); cc = [WINNER, SKY, GREEN]
    for offset, field, label, alpha in [(-.18, "median_peak_scale_ratio", "Peak", 1), (.18, "median_final_scale_ratio", "Final", .48)]:
        v = [float(controls[k][field]) for k in keys]
        bars = a.bar(x + offset, v, .33, color=cc, alpha=alpha, label=label)
        for index, (rect, val) in enumerate(zip(bars, v)):
            a.annotate(f"{val:.2f}×", (rect.get_x() + rect.get_width() / 2, val),
                       xytext=(0, 21 if index == 2 and label == "Final" else 4), textcoords="offset points",
                       ha="center", color=TEXT, fontsize=F_ANNOT)
    a.set_yscale("log"); a.set_ylim(.7, 65)
    a.set_xticks(x, ["Winner\n(M free)", SIBLING_NAME.replace(" sibling", "\nsibling"), "Rotation"]); a.set_ylabel("Scale relative to each initial state")
    a.set_title("A   Focused 1024² controls"); a.legend()
    for i, k in enumerate(keys):
        r = controls[k]; a.text(i, 48, f"{r['strong_domain_count']}/{r['trajectory_count']} pass", ha="center", color=TEXT, fontsize=F_ANNOT)
    colors = [WINNER] + [VERMILION if pooled[k][0] == 0 else PINK for k in order[1:-2]] + [SKY, GREEN]
    vals = [pooled[k][0] / pooled[k][1] for k in order]
    b.bar(range(len(order)), vals, color=colors)
    for i, k in enumerate(order):
        b.text(i, vals[i] + (.085 if k == "ablate_09" else .02), f"{pooled[k][0]}/{pooled[k][1]}",
               ha="center", va="bottom", color=TEXT, fontsize=F_ANNOT, rotation=90 if vals[i] == 0 else 0)
    b.set_ylim(0, .84); b.set_ylabel("End-of-window strong-domain pass fraction")
    b.set_xticks(range(len(order)), ["Winner"] + [f"Pair {i:02}" for i in range(12)] + [SIBLING_NAME, "Rotation"], rotation=90, ha="center")
    b.set_title("B   Single-pair ablations at 512² (pooled)")
    fig.suptitle("Matched controls constrain the mechanism in the tested rule family", y=.975)
    stem = "COMPANION_V135_FIGURE_04_CONTROLS_AND_ABLATIONS"
    outputs, size = save_set(fig, out, stem, "Controls and ablations")
    return compact_record(stem, [cp, ap], repo, outputs, size)


def render_story(repo: Path, out: Path) -> dict:
    apply_rc()
    # the nine step texts of the v0.2.1 figure, unchanged except British spelling (as in v1.29)
    steps = [("Discovery", "Search-selected rule; the large initial demonstration used a localised random patch."),
             ("Confirmation", "Frozen seeds reproduced irregular, nonmonotone large-domain growth."),
             ("Composition", "Dense regions are mixed excitations; sparse regions are near-empty."),
             ("Controls", "Pure rotation and the active Q+M sibling did not reproduce the same regime."),
             ("Homogeneous starts", "Separate whole-world random ensembles established density dependence."),
             ("First passage", "A geometrical endpoint was followed through longer observation windows."),
             ("Prepared shapes", "Large shapes persisted; relative-growth counts are not absolute formation tests."),
             ("Intervention audit", "Shuffling destroyed measured structure, yet the shuffled states crossed the geometrical endpoint."),
             ("Intervention result", "Shuffling invalidates the first-passage assay for a causal memory effect. The observed reversible domains remain.")]
    fig, ax = plt.subplots(figsize=(FIG_W,7.4), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(0, 3.12)
    for i, (title, body) in enumerate(steps):
        col = i % 3; row = 2 - i // 3; x0 = col + .04; y0 = row + .12
        ax.add_patch(Rectangle((x0, y0), .92, .80, facecolor=PANEL, edgecolor=COLORS[i % 7], lw=1.8))
        ax.text(x0 + .05, y0 + .70, f"{i + 1}. {title}", color=TEXT, fontsize=F_PANEL, weight="bold", va="top")
        ax.text(x0 + .05, y0 + .50, textwrap.fill(body, 32), color=MUTED, fontsize=F_ANNOT, va="top", linespacing=1.3)
    ax.text(1.5, 3.07, "The evidence, from selection to the shuffling test", ha="center", va="top", color=TEXT, fontsize=F_TITLE, weight="bold")
    stem = "COMPANION_V135_FIGURE_08_CORRECTED_RESEARCH_STORY"
    outputs, size = save_set(fig, out, stem, "Corrected research chronology")
    return compact_record(stem, [], repo, outputs, size)


# ---------------------------------------------------------------- Figure 6 (schematic ledger), no run needed
def render_ledger(out: Path) -> dict:
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.7), facecolor=BG)
    fig.subplots_adjust(left=.02, right=.98, top=.82, bottom=.03, wspace=.06)
    fig.suptitle("Schematic: exact charge balance does not freeze occupancy", y=.985)
    left, right = axes
    for axis in axes:
        axis.set_facecolor(PANEL); axis.set_xlim(0, 10); axis.set_ylim(1.3, 7); axis.set_aspect("equal"); axis.axis("off")
    left.set_title("Frozen winner: exact Q, variable M", color="#8a7a00", loc="center")
    left.text(5, 6.3, r"global $Q = N_{+} - N_{-} = 0$ exactly", ha="center", va="center", color=TEXT, fontsize=F_PANEL)
    left.text(5, 5.45, r"$M = N_{+} + N_{-} + N_{\mathrm{bound}}$ may change", ha="center", va="center", color=TEXT, fontsize=F_LABEL)
    left.add_patch(FancyArrowPatch((5, 5.0), (5, 4.3), arrowstyle="-|>", mutation_scale=20, color=MUTED, linewidth=1.6))
    for x, color, label in ((3.9, STATE_HEX[1], "+"), (6.1, STATE_HEX[2], "−")):
        left.add_patch(Rectangle((x - .65, 2.85), 1.3, 1.3, facecolor=color, edgecolor=TEXT, linewidth=1.6))
        left.text(x, 3.5, label, ha="center", va="center", color="#101010" if label == "+" else "white", fontsize=F_TITLE, weight="bold")
    left.text(5, 2.35, "Balanced pair creation changes M by +2\nwithout changing Q", ha="center", va="top", color=TEXT, fontsize=F_LABEL, linespacing=1.3)
    right.set_title(SIBLING_NAME, color=BLUE, loc="center")
    right.text(5, 6.3, r"global $Q = 0$ and $M = M_{\mathrm{initial}} > 0$", ha="center", va="center", color=TEXT, fontsize=F_PANEL)
    right.text(5, 5.45, "The same number of occupied cells is retained", ha="center", va="center", color=TEXT, fontsize=F_LABEL)
    for i, (color, label) in enumerate(((STATE_HEX[1], "+"), (STATE_HEX[2], "−"), (STATE_HEX[3], "b"))):
        x = 3.2 + i * 1.8
        right.add_patch(Rectangle((x - .6, 2.9), 1.2, 1.2, facecolor=color, edgecolor=TEXT, linewidth=1.6))
        right.text(x, 3.5, label, ha="center", va="center", color="white" if i else "#101010", fontsize=F_PANEL, weight="bold")
    right.text(5, 2.35, "Reactions still occur, but total M is fixed", ha="center", va="top", color=TEXT, fontsize=F_LABEL)
    model_source = Path(hf.__file__).resolve()
    panels = [hf.make_panel_record(source_task="model_schematic", seed=None, density=None, branch_point=None, intervention="frozen_winner_ledger",
                                   phase="not_applicable", state_hash="model_schematic_no_state", crop=(0, 0, 0, 0), source_path=model_source,
                                   extra={"panel": "variable_M_winner", "evidence_type": "MODEL SCHEMATIC"}),
              hf.make_panel_record(source_task="model_schematic", seed=None, density=None, branch_point=None, intervention="exact_Q_M_sibling_control",
                                   phase="not_applicable", state_hash="model_schematic_no_state", crop=(0, 0, 0, 0), source_path=model_source,
                                   extra={"panel": "exact_Q_M_control", "evidence_type": "MODEL SCHEMATIC"})]
    stem = "FIGURE_07_MECHANISM_LEDGER_V135"
    outputs, size = save_set(fig, out, stem, "Mechanism ledger (model schematic)")
    return state_record(stem, panels, "Model schematic. The winner keeps signed charge Q exactly zero but permits occupancy M to change; "
                        "the exact-Q+M sibling instead fixes a positive, nonzero occupancy.", outputs, size, major_morphology=False, out=out)


# ---------------------------------------------------------------- supplement Figure 4: multiscale memory (experiment tables only)
def render_multiscale(experiment: Path, out: Path) -> dict:
    apply_rc()
    audit_path = experiment / "INTERVENTION_AUDIT.csv"; residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"
    audit = hf._typed_rows(hf.read_verified_csv(audit_path)); residual = hf._typed_rows(hf.read_verified_csv(residual_path))
    audit = [row for row in audit if row["branch_point"] == "B" and int(row["realization"]) == 1 and row["intervention"] in {"global_shuffle", "tile_32", "tile_128"}]
    block_source = hf._mean_curves(audit, "source_block_entropy"); block_altered = hf._mean_curves(audit, "altered_block_entropy")
    mi_source = hf._mean_curves(audit, "source_mutual_information"); mi_altered = hf._mean_curves(audit, "altered_mutual_information")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W,8.0), facecolor=BG)
    fig.subplots_adjust(left=.075, right=.985, top=.905, bottom=.075, hspace=.5, wspace=.28)
    fig.suptitle("How much spatial information survives each intervention?", y=.985)
    chart(axes[0, 0], "Block entropy after intervention"); chart(axes[0, 1], "Mutual information by offset")
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        color = INTERVENTION[intervention]; name = intervention.replace("_", " ")
        scales = sorted(block_altered.get(intervention, {}))
        axes[0, 0].plot(scales, [block_altered[intervention][s] for s in scales], marker="o", color=color, label=name, lw=2)
        source_scales = sorted(block_source.get(intervention, {}))
        axes[0, 0].plot(source_scales, [block_source[intervention][s] for s in source_scales], color=color, linestyle="--", alpha=.5, lw=1.6)
        offsets = sorted(mi_altered.get(intervention, {}))
        axes[0, 1].plot(offsets, [mi_altered[intervention][o] for o in offsets], marker="o", color=color, label=name, lw=2)
        source_offsets = sorted(mi_source.get(intervention, {}))
        axes[0, 1].plot(source_offsets, [mi_source[intervention][o] for o in source_offsets], color=color, linestyle="--", alpha=.5, lw=1.6)
    axes[0, 0].set(xscale="log", xlabel="block side length (cells)", ylabel="normalised block entropy")
    axes[0, 1].set(xscale="log", xlabel="offset (cells)", ylabel="mutual information (nats)")
    key = Line2D([], [], color=MUTED, linestyle="--", label="source before intervention")
    for axis in (axes[0, 0], axes[0, 1]):
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles=handles + [key], labels=labels + ["source before intervention"])
    chart(axes[1, 0], "Immediate retained low-k power")
    positions, labels = [], []
    for index, intervention in enumerate(("global_shuffle", "tile_32", "tile_128")):
        values = [float(row["altered_low_k_power"]) / max(float(row["source_low_k_power"]), 1e-30) for row in audit if row["intervention"] == intervention]
        positions.append(index); labels.append(intervention.replace("_", " "))
        axes[1, 0].scatter(np.full(len(values), index), values, color=INTERVENTION[intervention], alpha=.75, s=36)
        if values:
            axes[1, 0].plot([index - .18, index + .18], [np.median(values)] * 2, color=TEXT, linewidth=2)
    axes[1, 0].axhline(1, color=MUTED, linestyle="--")
    axes[1, 0].set_xticks(positions, labels); axes[1, 0].set_ylabel("altered / source low-k power")
    chart(axes[1, 1], "Retained MI ratio vs residual time")
    residual_lookup = {row["task_id"]: row for row in residual}
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        points = []
        for row in audit:
            if row["intervention"] != intervention or row["task_id"] not in residual_lookup:
                continue
            source_mi = row["source_mutual_information"]; altered_mi = row["altered_mutual_information"]
            offsets = sorted(int(value) for value in source_mi)
            if offsets != [2 ** k for k in range(9)]:
                raise ValueError("retained MI summary requires offsets 1, 2, 4, ..., 256 cells")
            # Sum before dividing: weak source MI must not give a lag an equal ratio weight.
            source_total = sum(
                hf._curve_scalar(source_mi[str(o)] if str(o) in source_mi else source_mi[o])
                for o in offsets)
            altered_total = sum(
                hf._curve_scalar(altered_mi[str(o)] if str(o) in altered_mi else altered_mi[o])
                for o in offsets)
            if source_total <= 0:
                raise ValueError("retained MI summary requires positive summed source MI")
            retained = altered_total / source_total
            time, observed = hf._event_time(residual_lookup[row["task_id"]], altered=True)
            points.append((retained, time, observed))
        if points:
            axes[1, 1].scatter([p[0] for p in points], [p[1] for p in points], marker="o" if all(p[2] for p in points) else "s",
                               color=INTERVENTION[intervention], alpha=.8, s=36, label=intervention.replace("_", " "))
    axes[1, 1].set(xlabel="Summed MI: altered / source\n" + r"x/y mean; offsets $2^k$ cells, $k=0,\ldots,8$",
                   ylabel=r"$R_{\mathrm{hit}}$ or censoring time (phases)")
    axes[1, 1].margins(x=.1)
    axes[1, 1].yaxis.set_major_formatter(KFMT); axes[1, 1].legend()
    panels = [hf._aggregate_panel(audit_path, panel="block_entropy_by_scale", branch_point="B"),
              hf._aggregate_panel(audit_path, panel="mutual_information_by_offset", branch_point="B"),
              hf._aggregate_panel(audit_path, panel="retained_low_k", branch_point="B"),
              hf._aggregate_panel(residual_path, panel="retained_information_vs_residual", branch_point="B", extra={"joined_source": str(audit_path.resolve())})]
    stem = "FIGURE_05_MULTISCALE_MEMORY_V135"
    outputs, size = save_set(fig, out, stem, "Multiscale memory diagnostics")
    return state_record(stem, panels, "Immediate retained multiscale organisation and its relation to later residual first-passage time. "
                        "Solid curves are altered-state measurements; dashed curves are their matched pre-intervention sources.",
                        outputs, size, major_morphology=False, out=out)


# ---------------------------------------------------------------- state-based figures (need the retained run directory)
def render_micro_macro(run: Path, experiment: Path, out: Path) -> dict:
    apply_rc()
    representative = hf._representatives(experiment)[0]
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    source = hf._source_result(run, source_id)
    target_phase = int(source["time_to_event"]["T_hit"])
    records = hf.index_task_states(run / "sources" / source_id)
    exact = [r for r in records if int(r["phase"]) == target_phase]
    record = min(exact, key=lambda row: row["path"]) if exact else max([r for r in records if int(r["phase"]) <= target_phase], key=lambda row: int(row["phase"]))
    state, _, _, _ = hf._record_state(record)
    crop256 = hf.aligned_densest_crop(state, 256); crop64 = hf.aligned_densest_crop(state, 64, bounds=crop256)
    local = hf.crop_state(state, crop64); histogram = hf.state_histogram(local)
    probabilities = histogram / histogram.sum()
    entropy = float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])) / np.log(4.0))
    occupied = (state != 0).astype(np.float32); block = 32
    coarse = occupied.reshape(state.shape[0] // block, block, state.shape[1] // block, block).mean(axis=(1, 3))
    phase = int(record["phase"])
    fig = plt.figure(figsize=(FIG_W,8.9), facecolor=BG)
    g = fig.add_gridspec(2, 6, left=.075, right=.975, top=.905, bottom=.105, hspace=.28, wspace=.35, height_ratios=[1.12, 1])
    a0 = fig.add_subplot(g[0, 0:2]); a1 = fig.add_subplot(g[0, 2:4]); a2 = fig.add_subplot(g[0, 4:6])
    h = fig.add_subplot(g[1, 0:3]); c = fig.add_subplot(g[1, 3:6])
    show_state(a0, state, header_lines("Full 1024×1024 world", phase, state))
    show_state(a1, state, header_lines(f"256×256 crop · ({crop256[0]},{crop256[1]})", phase, state), crop=crop256)
    show_state(a2, state, header_lines(f"64×64 crop · ({crop64[0]},{crop64[1]})", phase, state), crop=crop64)
    for axis, crop, origin in [(a0, crop256, (0, 0)), (a1, crop64, crop256[:2])]:
        x, y, width, height = crop
        axis.add_patch(Rectangle((x - origin[0] - .5, y - origin[1] - .5), width, height,
                                 facecolor="none", edgecolor="white", linewidth=1.1))
    chart(h, "Four-state histogram, 64×64 crop")
    h.bar(STATE_NAMES, histogram, color=STATE_HEX, edgecolor=MUTED, linewidth=.6)
    h.set_ylabel("cells in the 64×64 crop")
    h.text(.975, .95, f"normalised entropy {entropy:.3f}", transform=h.transAxes, ha="right", va="top", fontsize=F_ANNOT, color=TEXT)
    c.set_title("Coarse occupancy field"); c.set_anchor("W")
    image = c.imshow(coarse, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    cax = c.inset_axes([1.05, 0.0, 0.05, 1.0])     # beside the image, inside the cell (fig.colorbar's own placement spilled past the canvas)
    bar = fig.colorbar(image, cax=cax); bar.set_label("occupied fraction per 32×32 block", fontsize=F_LABEL); bar.ax.tick_params(labelsize=F_TICK)
    c.set_xticks([]); c.set_yticks([])
    for spine in c.spines.values():
        spine.set_color(MUTED)
    state_legend(fig, y=.012)
    fig.suptitle("Microscopic disorder can coexist with macroscopic order", y=.985)
    base = dict(source_task=source_id, seed=seed, density=density, branch_point="source_event", intervention="untouched", requested_elapsed=None)
    full = (0, 0, state.shape[1], state.shape[0])
    panels = [
        hf._state_panel_from_record(record, crop=full, extra={"panel": "full_world"}, **base),
        hf._state_panel_from_record(record, crop=crop256, extra={"panel": "crop_256", "crop_selection": "frozen aligned maximum occupancy"}, **base),
        hf._state_panel_from_record(record, crop=crop64, extra={"panel": "crop_64", "crop_selection": "frozen nested aligned maximum occupancy"}, **base),
        hf._state_panel_from_record(record, crop=crop64, extra={"panel": "local_histogram", "histogram": histogram.tolist(), "normalized_entropy": entropy}, **base),
        hf._state_panel_from_record(record, crop=full, extra={"panel": "coarse_density", "coarse_block_size": 32}, **base),
    ]
    stem = "FIGURE_06_MICRO_DISORDER_MACRO_ORDER_V135"
    outputs, size = save_set(fig, out, stem, "Microscopic disorder, macroscopic order")
    rec = state_record(stem, panels, "A frozen event-centred representative shown at full-world, 256-cell and 64-cell scales, "
                       "with the local four-state histogram and the coarse occupancy field.", outputs, size, major_morphology=True, out=out)
    rec["entropy"] = entropy; rec["histogram"] = histogram.tolist(); rec["phase"] = phase
    return rec


def render_branch_grids(run: Path, experiment: Path, out: Path) -> list[dict]:
    integrity = hf._read_json(experiment / "INTEGRITY_AUDIT.json")
    if not integrity["passed"]:
        raise RuntimeError("publication stopped because the final integrity audit did not pass")
    realized = hf._realized(run); representatives = hf._representatives(experiment)
    requested = [0, 32768, 65536, 131072, 262144]
    columns = ["Source", "Immediately after", "+32,768", "+65,536", "+131,072", "Common +16,384"]
    interventions = ("untouched", "global_shuffle", "tile_32", "tile_128")
    recs = []
    for letter, representative in zip("AB", representatives, strict=True):
        source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
        records_by: dict[str, list[dict]] = {}; maxima = []
        for intervention in interventions:
            if intervention == "untouched":
                state_dir = run / "media_states" / "untouched" / "branch_B" / source_id
            else:
                state_dir = run / "branches" / hf._branch_entry(realized, source_id, "B", intervention)["task_id"]
            records = hf.index_task_states(state_dir)
            if not records:
                raise RuntimeError(f"no exact publication states found under {state_dir}")
            records_by[intervention] = records
            maxima.append(max(int(r["elapsed"]) for r in records if r["elapsed"] is not None))
        common_final = min(262144, *maxima)
        source = hf._source_result(run, source_id); branch = source["branch_points"]["B"]
        source_state, source_phase, _ = hf.load_exact_npz(branch["path"])
        source_record = {"path": str(Path(branch["path"]).resolve()), "phase": source_phase, "elapsed": 0,
                         "state_sha256": hf.state_sha256(source_state), "shape": list(source_state.shape)}
        q_source, m_source = q_m(source_state)
        apply_rc()
        fig, axes = plt.subplots(4, 6, figsize=(FIG_W, 12.2), facecolor=BG)
        fig.subplots_adjust(left=.045, right=.99, top=.865, bottom=.185, hspace=.6, wspace=.08)
        fig.suptitle(f"Same ingredients, different spatial history · initial occupancy {density:.2f}", y=.988)
        for ci, title in enumerate(columns):
            axes[0, ci].annotate(title, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 56), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=F_ANNOT, weight="bold", color=TEXT)
        panels = []
        for ri, intervention in enumerate(interventions):
            name = intervention.replace("_", " ")
            show_state(axes[ri, 0], source_state, ["source", f"phase {source_phase:,}", f"M = {m_source:,}" if q_source == 0 else f"Q = {q_source:,} · M = {m_source:,}"], header=F_GRID)
            panels.append(hf._state_panel_from_record(source_record, source_task=source_id, seed=seed, density=density, branch_point="B",
                                                      intervention=f"source_before_{intervention}", requested_elapsed=0,
                                                      extra={"panel": f"density_{density:.2f}_{intervention}_source"}))
            records = records_by[intervention]
            for ci, target in enumerate(requested[:4] + [common_final], start=1):
                state, record = hf._exact_at(records, target)
                actual = int(record["elapsed"]); q, m = q_m(state)
                show_state(axes[ri, ci], state, [name, f"+{actual:,}", f"M = {m:,}" if q == 0 else f"Q = {q:,} · M = {m:,}"], header=F_GRID)
                panels.append(hf._state_panel_from_record(record, source_task=source_id, seed=seed, density=density, branch_point="B",
                                                          intervention=intervention, requested_elapsed=target,
                                                          extra={"panel": f"density_{density:.2f}_{intervention}_column_{ci}", "actual_elapsed": actual}))
            axes[ri, 0].set_ylabel(name, color=INTERVENTION[intervention], fontsize=F_ANNOT, weight="bold")
        # Fixed-coordinate details of existing stored states, with no colour averaging.
        detail_crop = (512, 512, 32, 32)
        global_zero, _ = hf._exact_at(records_by["global_shuffle"], 0)
        for parent, detail, label, x in [(axes[0, 0], source_state, "Source", .20),
                                         (axes[1, 1], global_zero, "Global shuffle +0", .66)]:
            parent.add_patch(Rectangle((511.5, 511.5), 32, 32, facecolor="none", edgecolor="white", linewidth=1))
            inset = fig.add_axes([x, .042, .092, .092])
            show_state(inset, detail, [label, "32×32 at (x,y)=(512,512)"], crop=detail_crop, header=F_ANNOT)
        state_legend(fig, y=.008)
        stem = f"FIGURE_02{letter}_HISTORY_ERASURE_BRANCHES_V135"
        outputs, size = save_set(fig, out, stem, f"History-erasure branches, initial occupancy {density:.2f}")
        recs.append(state_record(stem, panels, f"Exact stored states for the preregistered median-T_hit representative at initial occupancy {density:.2f}. "
                                 "The interventions preserve the complete state inventory and change only arrangement; each panel is labelled with "
                                 "the elapsed offset of the stored state it shows.", outputs, size, major_morphology=True, out=out))
    return recs


def render_fixed_occupancy(run: Path, experiment: Path, out: Path) -> dict:
    residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"
    rows = hf._typed_rows(hf.read_verified_csv(residual_path))
    primary = [row for row in rows if row["branch_point"] == "A" and row["intervention"] == "global_shuffle" and int(row["realization"]) == 1]
    if not primary:
        raise RuntimeError("Branch-A primary rows are unavailable")
    realized = hf._realized(run)
    representative = hf._representative_for_point(run, experiment, realized, "A")
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    source = hf._source_result(run, source_id); branch = source["branch_points"]["A"]
    source_state, source_phase, _ = hf.load_exact_npz(branch["path"])
    source_record = {"path": str(Path(branch["path"]).resolve()), "phase": source_phase, "elapsed": 0, "state_sha256": hf.state_sha256(source_state), "shape": list(source_state.shape)}
    untouched_records = hf.index_task_states(run / "media_states" / "untouched" / "branch_A" / source_id)
    global_records = hf.index_task_states(run / "branches" / hf._branch_entry(realized, source_id, "A", "global_shuffle")["task_id"])
    untouched_state, untouched_record = hf._exact_at(untouched_records, 32768)
    shuffled_state, shuffled_record = hf._exact_at(global_records, 32768)
    apply_rc()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 8.2), facecolor=BG)
    fig.subplots_adjust(left=.065, right=.985, top=.905, bottom=.075, hspace=.5, wspace=.4)
    fig.suptitle("At M/N ≈ 0.35, does erasing spatial history reset the clock?", y=.985)
    chart(axes[0, 0], "Residual survival")
    hf._step_survival(axes[0, 0], primary, altered=False, label="untouched", color=INTERVENTION["untouched"])
    hf._step_survival(axes[0, 0], primary, altered=True, label="global shuffle", color=INTERVENTION["global_shuffle"])
    axes[0, 0].set(xlabel="elapsed phases", ylabel="event-free fraction", ylim=(0, 1.03), xlim=(0, 262144)); axes[0, 0].legend()
    axes[0, 0].xaxis.set_major_formatter(KFMT)
    chart(axes[0, 1], "Paired residual times")
    for row in primary:
        untouched, _ = hf._event_time(row, altered=False); altered, _ = hf._event_time(row, altered=True)
        axes[0, 1].plot([0, 1], [untouched, altered], color="#5a6b7a55", linewidth=1)
        axes[0, 1].scatter([0, 1], [untouched, altered], color=[INTERVENTION["untouched"], INTERVENTION["global_shuffle"]], s=30, zorder=3)
    axes[0, 1].set_xticks([0, 1], ["untouched", "global shuffle"]); axes[0, 1].set_ylabel(r"$R_{\mathrm{hit}}$ or censoring time (phases)")
    axes[0, 1].yaxis.set_major_formatter(KFMT)
    chart(axes[0, 2], "Crossed by fixed phase")
    fixed = [32768, 65536, 131072, 262144]
    u_frac = [sum(hf._event_time(r, altered=False)[1] and hf._event_time(r, altered=False)[0] <= t for r in primary) / len(primary) for t in fixed]
    g_frac = [sum(hf._event_time(r, altered=True)[1] and hf._event_time(r, altered=True)[0] <= t for r in primary) / len(primary) for t in fixed]
    x = np.arange(len(fixed))
    axes[0, 2].bar(x - .18, u_frac, .36, color=INTERVENTION["untouched"], label="untouched")
    axes[0, 2].bar(x + .18, g_frac, .36, color=INTERVENTION["global_shuffle"], label="global shuffle")
    axes[0, 2].set_xticks(x, [f"{t // 1024}k" for t in fixed]); axes[0, 2].set(ylim=(0, 1.42), xlabel="elapsed phases", ylabel="fraction crossed")
    axes[0, 2].set_yticks([0, .2, .4, .6, .8, 1.0]); axes[0, 2].legend(loc="upper right")   # headroom above the bars, inside the panel
    e_u = int(untouched_record["elapsed"]); e_g = int(shuffled_record["elapsed"])
    show_state(axes[1, 0], source_state, ["State-matched branch point", "+0", phase_line(source_phase, source_state)])
    show_state(axes[1, 1], untouched_state, ["untouched", f"+{e_u:,}" + ("" if e_u == 32768 else " (req. 32,768)"), phase_line(int(untouched_record["phase"]), untouched_state)])
    show_state(axes[1, 2], shuffled_state, ["global shuffle", f"+{e_g:,}" + ("" if e_g == 32768 else " (req. 32,768)"), phase_line(int(shuffled_record["phase"]), shuffled_state)])
    state_legend(fig, y=.026, blend_note=True)
    panels = [hf._aggregate_panel(residual_path, panel="branch_A_survival", branch_point="A"),
              hf._aggregate_panel(residual_path, panel="branch_A_paired_lines", branch_point="A"),
              hf._aggregate_panel(residual_path, panel="branch_A_fixed_fractions", branch_point="A"),
              hf._state_panel_from_record(source_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="source_before", requested_elapsed=0, extra={"panel": "representative_source"}),
              hf._state_panel_from_record(untouched_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="untouched", requested_elapsed=32768, extra={"panel": "representative_untouched", "actual_elapsed": e_u}),
              hf._state_panel_from_record(shuffled_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="global_shuffle", requested_elapsed=32768, extra={"panel": "representative_global", "actual_elapsed": e_g})]
    stem = "FIGURE_03_FIXED_OCCUPANCY_HISTORY_ERASURE_V135"
    outputs, size = save_set(fig, out, stem, "Fixed-occupancy history-erasure diagnostics")
    return state_record(stem, panels, "Branch-A residual first-passage diagnostics at the first sampled occupancy M/N ≥ 0.35, with the state-matched "
                        "branch point and the two branches at the stored offsets their headers give.", outputs, size, major_morphology=True, out=out)


def render_event_proximal(run: Path, experiment: Path, out: Path) -> dict:
    residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"; result_path = experiment / "HISTORY_ERASURE_RESULTS.csv"
    rows = hf._typed_rows(hf.read_verified_csv(residual_path)); results = hf._typed_rows(hf.read_verified_csv(result_path))
    b_rows = [row for row in rows if row["branch_point"] == "B" and int(row["realization"]) == 1 and row["intervention"] in {"global_shuffle", "tile_32", "tile_128"}]
    if not b_rows:
        raise RuntimeError("Branch-B rows are unavailable")
    representative = hf._representatives(experiment)[0]
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    realized = hf._realized(run)
    exact_states = []
    state, record = hf._exact_at(hf.index_task_states(run / "media_states" / "untouched" / "branch_B" / source_id), 32768)
    exact_states.append(("untouched", state, record))
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        entry = hf._branch_entry(realized, source_id, "B", intervention)
        state, record = hf._exact_at(hf.index_task_states(run / "branches" / entry["task_id"]), 32768)
        exact_states.append((intervention, state, record))
    apply_rc()
    fig = plt.figure(figsize=(FIG_W,9.2), facecolor=BG)
    top = fig.add_gridspec(1, 2, left=.065, right=.985, top=.905, bottom=.56, wspace=.36, width_ratios=[1.15, 1])
    bottom = fig.add_gridspec(1, 4, left=.03, right=.985, top=.43, bottom=.07, wspace=.12)
    survival_axis = fig.add_subplot(top[0, 0]); delay_axis = fig.add_subplot(top[0, 1])
    fig.suptitle("Erasing history 32,768 phases before the untouched event", y=.985)
    chart(survival_axis, "Event-proximal survival")
    event_colors = {**INTERVENTION, "tile_32": GREEN}
    survival_axis.axvline(32768, color=INTERVENTION["untouched"], linestyle="--", label="untouched hit by construction")
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        selected = [row for row in b_rows if row["intervention"] == intervention]
        hf._step_survival(survival_axis, selected, altered=True, label=intervention.replace("_", " "), color=event_colors[intervention])
    survival_axis.set(xlim=(0, 262144), ylim=(0, 1.03), xlabel="elapsed phases", ylabel="event-free fraction"); survival_axis.legend()
    survival_axis.xaxis.set_major_formatter(KFMT)
    chart(delay_axis, "Paired RMST delay with 95% interval\nRMST through 262,144 phases")
    delay_axis.axvspan(-8192, 8192, color=WINNER, alpha=.16, label="practical-equivalence band\n±8,192 phases")
    selected_results = [row for row in results if row["comparison"] in {"Branch B primary global", "Branch B tile 32", "Branch B tile 128"}]
    colour_of = {"primary global": SKY, "tile 32": GREEN, "tile 128": ORANGE}
    labels = []
    for index, row in enumerate(selected_results):
        point, low, high = float(row["rmst_delay"]), float(row["ci95_low"]), float(row["ci95_high"])
        short = str(row["comparison"]).replace("Branch B ", "")
        delay_axis.errorbar(point, index, xerr=[[point - low], [high - point]], fmt="o", color=colour_of[short], capsize=5, linewidth=2)
        labels.append("global\nshuffle" if short == "primary global" else short)
    delay_axis.set_yticks(range(len(selected_results)), labels); delay_axis.set_ylim(-.6, len(selected_results) - .4)
    delay_axis.axvline(0, color=MUTED, linewidth=1)
    delay_axis.set_xlabel("paired RMST delay (phases; positive = later)"); delay_axis.legend(loc="lower right")
    delay_axis.xaxis.set_major_formatter(KFMT)
    panels = [hf._aggregate_panel(residual_path, panel="branch_B_survival", branch_point="B"),
              hf._aggregate_panel(result_path, panel="branch_B_rmst_delay", branch_point="B")]
    for column, (intervention, state, record) in enumerate(exact_states):
        axis = fig.add_subplot(bottom[0, column]); actual = int(record["elapsed"])
        show_state(axis, state, [intervention.replace("_", " "), f"+{actual:,}" + ("" if actual == 32768 else " (req. 32,768)"),
                                 phase_line(int(record["phase"]), state)])
        panels.append(hf._state_panel_from_record(record, source_task=source_id, seed=seed, density=density, branch_point="B", intervention=intervention,
                                                  requested_elapsed=32768, extra={"panel": f"representative_{intervention}", "actual_elapsed": actual}))
    state_legend(fig, y=.026, blend_note=True)
    stem = "FIGURE_04_EVENT_PROXIMAL_HISTORY_ERASURE_V135"
    outputs, size = save_set(fig, out, stem, "Event-proximal history-erasure diagnostics")
    return state_record(stem, panels, "Branch-B survival and paired delay estimates use an untouched residual fixed at 32,768 phases by preregistration; "
                        "the state panels give the stored offset each one actually shows.", outputs, size, major_morphology=True, out=out)


# ---------------------------------------------------------------- command line
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True, help="repository root (the release folder); supplies the frozen tables and the rule file")
    p.add_argument("--out", type=Path, required=True, help="output folder for the rendered figures, e.g. paper/figures")
    p.add_argument("--run", type=Path, default=None,
                   help="OPTIONAL: the author-retained v0.3.2 run directory (archived state arrays). Without it the six figures that need "
                        "no state arrays are rendered (rule diagram, density, controls, timeline, ledger, multiscale).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="where to write the render manifest (default: <out>/../V135_PRINT_RENDER_MANIFEST.json with --run; without --run, "
                        "<out>/V135_PRINT_RENDER_MANIFEST_compact.json, so a compact regeneration never replaces the shipped eleven-figure manifest)")
    a = p.parse_args()
    repo = a.repo.resolve(); out = a.out.resolve()
    experiment = (repo / "experiments" / "reversible_null_phase_v032_history_erasure_20260829").resolve()
    print(f"rendering to {out}")
    recs = [render_rule_figure(repo, out), render_density(repo, out), render_controls(repo, out), render_story(repo, out),
            render_ledger(out), render_multiscale(experiment, out)]
    if a.run is not None:
        run = a.run.resolve()
        recs.append(render_micro_macro(run, experiment, out))
        recs.extend(render_branch_grids(run, experiment, out))
        recs.append(render_fixed_occupancy(run, experiment, out))
        recs.append(render_event_proximal(run, experiment, out))
        scope = "all eleven figure files (six table- or text-driven, five from the retained state arrays)"
    else:
        scope = "the six figures that need no state arrays (no --run directory given; figures 06, 02A, 02B, 03 and 04 need the retained arrays)"
    manifest = {
        "version": VERSION, "scope": scope,
        "purpose": "one print typography for every code-generated figure: canvas 12.44 in (twice the 15.8 cm text width), type at twice its printed "
                   "size, included at scale 0.5; light palette; drawn commentary removed; state-panel headers give each panel's stored phase and offset",
        "printed_type_sizes_pt": {"figure_title": F_TITLE / 2, "panel_title": F_PANEL / 2, "axis_label": F_LABEL / 2, "tick_and_annotation": F_TICK / 2,
                                  "state_panel_header": F_HEADER / 2, "branch_grid_header": F_GRID / 2},
        "derived_from": ["V021_RENDER_MANIFEST.json and V129_LIGHT_RENDER_MANIFEST.json (figures 03, 04, 08)",
                         "scripts/render_companion_v023_figures.py (rule diagram, raster)",
                         "history_figures.figure_02_history_branches, figure_03_fixed_occupancy, figure_04_event_proximal, figure_05_multiscale_memory, "
                         "figure_06_micro_macro, figure_07_mechanism_ledger (v0.3.2 campaign)"],
        "rule_sha256": RULE, "new_simulations": False, "rendered_utc": TODAY.isoformat(), "figures": recs}
    mp = a.manifest.resolve() if a.manifest else (out.parent / "V135_PRINT_RENDER_MANIFEST.json" if a.run is not None else out / "V135_PRINT_RENDER_MANIFEST_compact.json")
    mp.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(mp, "|", scope)
