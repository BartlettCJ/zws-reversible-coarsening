from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np
from PIL import Image
import torch

from .atomic import atomic_write_text, sha256_file
from .history_erasure import state_histogram, state_sha256
from .history_media import (
    choose_exact_record,
    index_task_states,
    load_exact_npz,
    verify_sidecar,
)
from .history_publication import (
    STATE_HEX,
    STATE_NAMES,
    aligned_densest_crop,
    crop_state,
    make_panel_record,
    parse_json_cell,
    read_verified_csv,
    validate_panel_record,
)


CMAP = ListedColormap(STATE_HEX)
INTERVENTION_COLORS = {
    "untouched": "#f0e442",
    "global_shuffle": "#56b4e9",
    "tile_32": "#cc79a7",
    "tile_128": "#e69f00",
    "even_translation_null": "#009e73",
}

# Publication figures use a dark canvas throughout.  Matplotlib's default
# black text otherwise leaks into state-panel titles, axis labels, and legend
# entries that do not set an explicit colour, making them effectively
# invisible in PDF/PNG exports.
plt.rcParams.update(
    {
        "text.color": "#d7e9f6",
        "axes.labelcolor": "#c8d7e3",
        "xtick.color": "#b7c7d3",
        "ytick.color": "#b7c7d3",
        "legend.labelcolor": "#d7e9f6",
    }
)


def _sidecar(path: Path) -> None:
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def _read_json(path: Path, *, require_sidecar: bool = True) -> dict[str, Any]:
    if require_sidecar:
        verify_sidecar(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _typed(value: Any) -> Any:
    parsed = parse_json_cell(value)
    if parsed is not value:
        return parsed
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return float(stripped) if any(char in stripped for char in ".eE") else int(stripped)
    except ValueError:
        return value


def _typed_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _typed(value) for key, value in row.items()} for row in rows]


def _aggregate_panel(
    source_path: Path,
    *,
    panel: str,
    branch_point: str | None = None,
    intervention: str = "aggregate",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_panel_record(
        source_task="aggregate",
        seed="multiple",
        density="pooled or stratified",
        branch_point=branch_point,
        intervention=intervention,
        phase="aggregate",
        state_hash=f"aggregate:{sha256_file(source_path)}",
        crop=(0, 0, 0, 0),
        source_path=source_path,
        extra={"panel": panel, **(extra or {})},
    )


def _state_legend(fig: plt.Figure, *, y: float = 0.01) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="#ffffff55", label=name)
        for name, color in zip(STATE_NAMES, STATE_HEX, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        frameon=False,
        labelcolor="#d7e9f6",
    )


def _state_title(state: np.ndarray, title: str, phase: int | str) -> str:
    histogram = state_histogram(state)
    q_value = int(histogram[1] - histogram[2])
    m_value = int(histogram[1:].sum())
    return f"{title}\nphase {phase:,} · Q={q_value:,} · M={m_value:,}"


def _show_state(
    axis: plt.Axes,
    state: np.ndarray,
    *,
    title: str,
    phase: int | str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    shown = crop_state(state, crop) if crop is not None else state
    axis.imshow(shown, cmap=CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest", rasterized=True)
    axis.set_title(
        _state_title(state, title, phase),
        fontsize=10,
        loc="left",
        color="#c8d7e3",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#526879")


def _save_triplet(
    fig: plt.Figure,
    output_dir: Path,
    figure_id: str,
    *,
    panels: list[dict[str, Any]],
    caption: str,
    major_morphology: bool,
) -> dict[str, Any]:
    for panel in panels:
        validate_panel_record(panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for suffix, options in (
        (".svg", {"format": "svg"}),
        (".pdf", {"format": "pdf"}),
        (".png", {"format": "png", "dpi": 300}),
    ):
        target = output_dir / f"{figure_id}{suffix}"
        temporary = target.with_name(target.stem + ".tmp" + target.suffix)
        fig.savefig(temporary, bbox_inches="tight", facecolor=fig.get_facecolor(), **options)
        os.replace(temporary, target)
        _sidecar(target)
        record: dict[str, Any] = {
            "path": str(target.resolve()),
            "sha256": sha256_file(target),
            "format": suffix[1:],
        }
        if suffix == ".png":
            with Image.open(target) as image:
                record["pixels"] = [image.width, image.height]
                if major_morphology and (image.width < 3840 or image.height < 2160):
                    raise RuntimeError(f"major morphology figure is below 3840x2160: {target}")
        outputs.append(record)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "caption": caption,
        "panels": panels,
        "outputs": outputs,
        "major_morphology": major_morphology,
    }


def _load_discovery_state(repo: Path, integrity: dict[str, Any], tick: int) -> tuple[np.ndarray, Path]:
    record = next(row for row in integrity["checkpoints"] if int(row["tick"]) == int(tick))
    path = repo / record["path"]
    verify_sidecar(path)
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"discovery checkpoint hash differs from frozen integrity record: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["state"]
    array = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = np.ascontiguousarray(array, dtype=np.uint8)
    # The frozen discovery macro-arrow predates the v0.3 extension digest
    # convention.  Its ``state_sha256_uint8`` field is the SHA-256 of the raw,
    # contiguous uint8 lattice bytes (without the shape/dtype header used by
    # ``history_erasure.state_sha256``).  Verify against that recorded
    # convention rather than comparing two intentionally different schemas.
    raw_state_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if raw_state_sha256 != record["state_sha256_uint8"]:
        raise RuntimeError(f"discovery state bytes differ from frozen integrity record: {path}")
    return array, path


def figure_01_hero(repo_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    artifact = repo / "experiments/domain_mechanism_20260825/artifacts/discovery_macro_arrow"
    macro = _read_json(artifact / "REVERSIBLE_MACRO_ARROW.json", require_sidecar=False)
    integrity = _read_json(
        artifact / "DISCOVERY_CHECKPOINT_INPUT_INTEGRITY.json", require_sidecar=False
    )
    if not macro["exact_reverse_verified_at_all_retained_boundaries"]:
        raise RuntimeError("hero source lacks exact reverse verification")
    initial_tick = int(macro["capture_ticks"]["initial"])
    peak_tick = int(macro["capture_ticks"]["peak_organisation"])
    initial, initial_path = _load_discovery_state(repo, integrity, initial_tick)
    organised, organised_path = _load_discovery_state(repo, integrity, peak_tick)
    recovered = initial
    dense_crop = aligned_densest_crop(organised, 512)

    fig, axes = plt.subplots(1, 4, figsize=(16, 9), facecolor="#071018")
    fig.suptitle(
        "A world that always adds up to zero can still organise itself - and its history runs backward exactly.",
        fontsize=20,
        color="#f6f8fa",
        y=0.96,
    )
    for axis in axes:
        axis.set_facecolor("#05070a")
    _show_state(axes[0], initial, title="Random balanced world", phase=initial_tick)
    _show_state(axes[1], organised, title="Macroscopic irregular organisation", phase=peak_tick)
    _show_state(
        axes[2],
        organised,
        title=f"Dense-domain inset · x={dense_crop[0]}, y={dense_crop[1]}",
        phase=peak_tick,
        crop=dense_crop,
    )
    _show_state(axes[3], recovered, title="Exact reverse recovery · mismatch=0", phase=0)
    _state_legend(fig, y=0.045)
    fig.text(
        0.5,
        0.015,
        "Q stays exactly zero; occupancy M is allowed to change. Reversal returns every one of 16,777,216 cells exactly.",
        ha="center",
        color="#c8d7e3",
        fontsize=11,
    )
    seed = 260824
    density = "localized composition 0.32; global initial occupancy 0.0647156"
    full_crop = (0, 0, initial.shape[1], initial.shape[0])
    panels = [
        make_panel_record(
            source_task="observe_4096_discovery",
            seed=seed,
            density=density,
            branch_point=None,
            intervention="untouched_forward",
            phase=initial_tick,
            state_hash=state_sha256(initial),
            crop=full_crop,
            source_path=initial_path,
            extra={"panel": "initial"},
        ),
        make_panel_record(
            source_task="observe_4096_discovery",
            seed=seed,
            density=density,
            branch_point=None,
            intervention="untouched_forward",
            phase=peak_tick,
            state_hash=state_sha256(organised),
            crop=full_crop,
            source_path=organised_path,
            extra={"panel": "organised"},
        ),
        make_panel_record(
            source_task="observe_4096_discovery",
            seed=seed,
            density=density,
            branch_point=None,
            intervention="untouched_forward",
            phase=peak_tick,
            state_hash=state_sha256(organised),
            crop=dense_crop,
            source_path=organised_path,
            extra={"panel": "dense_inset", "crop_selection": "frozen aligned maximum occupancy"},
        ),
        make_panel_record(
            source_task="observe_4096_discovery",
            seed=seed,
            density=density,
            branch_point=None,
            intervention="exact_reverse",
            phase=0,
            state_hash=state_sha256(recovered),
            crop=full_crop,
            source_path=initial_path,
            extra={
                "panel": "reverse_recovered",
                "mismatch_count": 0,
                "reverse_audit_path": str(
                    (repo / integrity["reversibility_audit_path"]).resolve()
                ),
            },
        ),
    ]
    return _save_triplet(
        fig,
        Path(output_dir),
        "FIGURE_01_ZERO_BALANCE_REVERSIBLE_HERO",
        panels=panels,
        caption=(
            "A localized random world with equal positive and negative charge develops irregular "
            "macroscopic organisation while exact Q remains zero and M changes; exact inverse "
            "updates recover the initial state with zero mismatches."
        ),
        major_morphology=True,
    )


def _representatives(experiment: Path) -> list[dict[str, Any]]:
    rows = _typed_rows(read_verified_csv(experiment / "REPRESENTATIVE_STATE_INDEX.csv"))
    if len(rows) != 2 or {float(row["density"]) for row in rows} != {0.20, 0.24}:
        raise RuntimeError("representative index does not contain exactly one frozen source per density")
    return sorted(rows, key=lambda row: float(row["density"]))


def _realized(run: Path) -> dict[str, Any]:
    return _read_json(run / "REALIZED_BRANCH_PLAN.json")


def _source_result(run: Path, source_task_id: str) -> dict[str, Any]:
    return _read_json(run / "sources" / source_task_id / "source_result.json")


def _branch_entry(
    realized: dict[str, Any],
    source_task_id: str,
    branch_point: str,
    intervention: str,
) -> dict[str, Any]:
    candidates = [
        row
        for row in realized["prepared_entries"]
        if row["source_task_id"] == source_task_id
        and row["branch_point"] == branch_point
        and row["intervention"] == intervention
        and int(row["realization"]) == 1
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one primary {branch_point}/{intervention} branch for {source_task_id}"
        )
    return candidates[0]


def _record_state(record: dict[str, Any]) -> tuple[np.ndarray, int, int | None, Path]:
    path = Path(record["path"])
    state, phase, elapsed = load_exact_npz(path)
    if state_sha256(state) != record["state_sha256"]:
        raise RuntimeError(f"indexed exact-state hash mismatch: {path}")
    return state, phase, elapsed, path


def _exact_at(records: list[dict[str, Any]], elapsed: int) -> tuple[np.ndarray, dict[str, Any]]:
    record = choose_exact_record(records, elapsed=elapsed, prefer_at_or_before=True)
    state, _, _, _ = _record_state(record)
    return state, record


def _state_panel_from_record(
    record: dict[str, Any],
    *,
    source_task: str,
    seed: int,
    density: float,
    branch_point: str,
    intervention: str,
    requested_elapsed: int | None,
    crop: tuple[int, int, int, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width, height = int(record["shape"][1]), int(record["shape"][0])
    return make_panel_record(
        source_task=source_task,
        seed=seed,
        density=density,
        branch_point=branch_point,
        intervention=intervention,
        phase=int(record["phase"]),
        elapsed=None if record["elapsed"] is None else int(record["elapsed"]),
        requested_elapsed=requested_elapsed,
        state_hash=record["state_sha256"],
        crop=crop or (0, 0, width, height),
        source_path=record["path"],
        extra=extra,
    )


def figure_02_history_branches(
    run_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run = Path(run_path).resolve()
    experiment = Path(experiment_path).resolve()
    integrity = _read_json(experiment / "INTEGRITY_AUDIT.json")
    if not integrity["passed"]:
        raise RuntimeError("publication stopped because the final integrity audit did not pass")
    realized = _realized(run)
    representatives = _representatives(experiment)
    requested = [0, 32768, 65536, 131072, 262144]
    columns = ["Source immediately before", "Immediately after", "+32,768", "+65,536", "+131,072", "Final/common"]
    row_specs: list[tuple[dict[str, Any], str]] = [
        (representative, intervention)
        for representative in representatives
        for intervention in ("untouched", "global_shuffle", "tile_32", "tile_128")
    ]
    records_by_branch: dict[tuple[str, str], list[dict[str, Any]]] = {}
    common_final: dict[str, int] = {}
    for representative in representatives:
        source_id = str(representative["source_task_id"])
        maxima: list[int] = []
        for intervention in ("untouched", "global_shuffle", "tile_32", "tile_128"):
            if intervention == "untouched":
                state_dir = run / "media_states" / "untouched" / "branch_B" / source_id
            else:
                entry = _branch_entry(realized, source_id, "B", intervention)
                state_dir = run / "branches" / entry["task_id"]
            records = index_task_states(state_dir)
            if not records:
                raise RuntimeError(f"no exact publication states found under {state_dir}")
            records_by_branch[(source_id, intervention)] = records
            maxima.append(
                max(int(record["elapsed"]) for record in records if record["elapsed"] is not None)
            )
        common_final[source_id] = min(262144, *maxima)
    fig, axes = plt.subplots(len(row_specs), 6, figsize=(20, 22), facecolor="#071018")
    fig.suptitle("Same ingredients, different spatial history", fontsize=24, color="#f6f8fa", y=0.995)
    panels: list[dict[str, Any]] = []
    for column, title in enumerate(columns):
        axes[0, column].text(0.5, 1.28, title, transform=axes[0, column].transAxes, ha="center", color="#c8d7e3", fontsize=11, weight="bold")

    for row_index, (representative, intervention) in enumerate(row_specs):
        source_id = str(representative["source_task_id"])
        density = float(representative["density"])
        seed = int(representative["seed"])
        source = _source_result(run, source_id)
        branch = source["branch_points"]["B"]
        source_state, source_phase, _ = load_exact_npz(branch["path"])
        source_record = {
            "path": str(Path(branch["path"]).resolve()),
            "phase": source_phase,
            "elapsed": 0,
            "state_sha256": state_sha256(source_state),
            "shape": list(source_state.shape),
        }
        _show_state(axes[row_index, 0], source_state, title="source", phase=source_phase)
        panels.append(
            _state_panel_from_record(
                source_record,
                source_task=source_id,
                seed=seed,
                density=density,
                branch_point="B",
                intervention=f"source_before_{intervention}",
                requested_elapsed=0,
                extra={"panel": f"density_{density:.2f}_{intervention}_source"},
            )
        )
        records = records_by_branch[(source_id, intervention)]
        selected_targets = requested[:4] + [common_final[source_id]]
        for column, target in enumerate(selected_targets, start=1):
            state, record = _exact_at(records, target)
            actual = int(record["elapsed"])
            label = intervention.replace("_", " ")
            if actual != target:
                label += f" · requested {target:,}"
            _show_state(axes[row_index, column], state, title=label, phase=int(record["phase"]))
            panels.append(
                _state_panel_from_record(
                    record,
                    source_task=source_id,
                    seed=seed,
                    density=density,
                    branch_point="B",
                    intervention=intervention,
                    requested_elapsed=target,
                    extra={"panel": f"density_{density:.2f}_{intervention}_column_{column}"},
                )
            )
        axes[row_index, 0].set_ylabel(
            f"ρ₀={density:.2f}\n{intervention.replace('_', ' ')}",
            color=INTERVENTION_COLORS[intervention],
            fontsize=10,
            weight="bold",
        )
    _state_legend(fig, y=0.004)
    fig.text(
        0.5,
        0.018,
        "At elapsed zero every row has identical empty/positive/negative/bound-neutral counts, Q, M, and single-site entropy; only arrangement differs.",
        ha="center",
        color="#c8d7e3",
        fontsize=11,
    )
    return _save_triplet(
        fig,
        Path(output_dir),
        "FIGURE_02_HISTORY_ERASURE_BRANCHES",
        panels=panels,
        caption=(
            "Exact synchronized states for the preregistered median-T_hit representative at each "
            "density. The interventions preserve the complete state inventory and change only arrangement."
        ),
        major_morphology=True,
    )


def _event_time(row: dict[str, Any], *, altered: bool) -> tuple[int, bool]:
    prefix = "" if altered else "untouched_"
    observed = bool(row[f"{prefix}hit_observed"])
    hit = row[f"{prefix}R_hit"]
    censor = row[f"{prefix}censor_elapsed"]
    return (int(hit) if observed and hit is not None else int(censor), observed)


def _step_survival(axis: plt.Axes, rows: list[dict[str, Any]], *, altered: bool, label: str, color: str) -> None:
    from .history_stats import kaplan_meier

    values = [_event_time(row, altered=altered) for row in rows]
    curve = kaplan_meier([value[0] for value in values], [value[1] for value in values], horizon=262144)
    xs = [int(row["time"]) for row in curve]
    ys = [float(row["survival"]) for row in curve]
    axis.step(xs, ys, where="post", label=label, color=color, linewidth=2.5)


def _chart_style(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor("#0b1721")
    axis.set_title(title, color="#f6f8fa", loc="left", fontsize=13)
    axis.tick_params(colors="#b7c7d3")
    for spine in axis.spines.values():
        spine.set_color("#40586a")
    axis.grid(color="#52687933", linewidth=0.7)


def _representative_for_point(
    run: Path, experiment: Path, realized: dict[str, Any], point: str
) -> dict[str, Any]:
    frozen = _representatives(experiment)
    available = {
        row["source_task_id"]
        for row in realized["prepared_entries"]
        if row["branch_point"] == point and row["intervention"] == "global_shuffle" and int(row["realization"]) == 1
    }
    for row in frozen:
        if row["source_task_id"] in available:
            return row
    plan = _read_json(experiment / "FROZEN_PLAN.json", require_sidecar=True)
    candidates = [task for task in plan["source_tasks"] if task["task_id"] in available]
    if not candidates:
        raise RuntimeError(f"no representative with an eligible Branch {point}")
    chosen = min(candidates, key=lambda task: (float(task["density"]), int(task["seed"]), task["task_id"]))
    source = _source_result(run, chosen["task_id"])
    return {
        "source_task_id": chosen["task_id"],
        "density": chosen["density"],
        "seed": chosen["seed"],
        "T_hit": source["time_to_event"]["T_hit"],
        "selection_rule": "frozen representative if eligible; otherwise lowest density, seed, task id",
    }


def figure_03_fixed_occupancy(
    run_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run, experiment = Path(run_path).resolve(), Path(experiment_path).resolve()
    residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"
    rows = _typed_rows(read_verified_csv(residual_path))
    primary = [row for row in rows if row["branch_point"] == "A" and row["intervention"] == "global_shuffle" and int(row["realization"]) == 1]
    if not primary:
        raise RuntimeError("Branch-A primary rows are unavailable")
    realized = _realized(run)
    representative = _representative_for_point(run, experiment, realized, "A")
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    source = _source_result(run, source_id)
    branch = source["branch_points"]["A"]
    source_state, source_phase, _ = load_exact_npz(branch["path"])
    source_record = {"path": str(Path(branch["path"]).resolve()), "phase": source_phase, "elapsed": 0, "state_sha256": state_sha256(source_state), "shape": list(source_state.shape)}
    untouched_records = index_task_states(run / "media_states" / "untouched" / "branch_A" / source_id)
    global_entry = _branch_entry(realized, source_id, "A", "global_shuffle")
    global_records = index_task_states(run / "branches" / global_entry["task_id"])
    untouched_state, untouched_record = _exact_at(untouched_records, 32768)
    shuffled_state, shuffled_record = _exact_at(global_records, 32768)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#071018")
    fig.suptitle("At M/N ≈ 0.35, does erasing spatial history reset the clock?", fontsize=21, color="#f6f8fa")
    _chart_style(axes[0, 0], "Residual first-passage survival")
    _step_survival(axes[0, 0], primary, altered=False, label="untouched", color=INTERVENTION_COLORS["untouched"])
    _step_survival(axes[0, 0], primary, altered=True, label="global shuffle", color=INTERVENTION_COLORS["global_shuffle"])
    axes[0, 0].set(xlabel="residual phases", ylabel="event-free fraction", ylim=(0, 1.03), xlim=(0, 262144))
    axes[0, 0].legend(frameon=False)

    _chart_style(axes[0, 1], "Paired residual time by progenitor")
    for index, row in enumerate(primary):
        untouched, _ = _event_time(row, altered=False)
        altered, _ = _event_time(row, altered=True)
        axes[0, 1].plot([0, 1], [untouched, altered], color="#7d90a055", linewidth=1)
        axes[0, 1].scatter([0, 1], [untouched, altered], color=[INTERVENTION_COLORS["untouched"], INTERVENTION_COLORS["global_shuffle"]], s=22)
    axes[0, 1].set_xticks([0, 1], ["untouched", "global shuffle"])
    axes[0, 1].set_ylabel("R_hit or censor time")

    _chart_style(axes[0, 2], "Transformed by fixed elapsed phase")
    fixed = [32768, 65536, 131072, 262144]
    u_frac = [sum(_event_time(row, altered=False)[1] and _event_time(row, altered=False)[0] <= t for row in primary) / len(primary) for t in fixed]
    g_frac = [sum(_event_time(row, altered=True)[1] and _event_time(row, altered=True)[0] <= t for row in primary) / len(primary) for t in fixed]
    x = np.arange(len(fixed))
    axes[0, 2].bar(x - 0.18, u_frac, 0.36, color=INTERVENTION_COLORS["untouched"], label="untouched")
    axes[0, 2].bar(x + 0.18, g_frac, 0.36, color=INTERVENTION_COLORS["global_shuffle"], label="global")
    axes[0, 2].set_xticks(x, [f"{t//1024}k" for t in fixed])
    axes[0, 2].set(ylim=(0, 1), xlabel="elapsed phases", ylabel="fraction transformed")
    axes[0, 2].legend(frameon=False)

    _show_state(axes[1, 0], source_state, title="State-matched branch point", phase=source_phase)
    _show_state(axes[1, 1], untouched_state, title="Untouched +32,768", phase=int(untouched_record["phase"]))
    _show_state(axes[1, 2], shuffled_state, title="Global shuffle +32,768", phase=int(shuffled_record["phase"]))
    _state_legend(fig, y=0.005)
    panels = [
        _aggregate_panel(residual_path, panel="branch_A_survival", branch_point="A"),
        _aggregate_panel(residual_path, panel="branch_A_paired_lines", branch_point="A"),
        _aggregate_panel(residual_path, panel="branch_A_fixed_fractions", branch_point="A"),
        _state_panel_from_record(source_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="source_before", requested_elapsed=0, extra={"panel": "representative_source"}),
        _state_panel_from_record(untouched_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="untouched", requested_elapsed=32768, extra={"panel": "representative_untouched"}),
        _state_panel_from_record(shuffled_record, source_task=source_id, seed=seed, density=density, branch_point="A", intervention="global_shuffle", requested_elapsed=32768, extra={"panel": "representative_global"}),
    ]
    return _save_triplet(fig, Path(output_dir), "FIGURE_03_FIXED_OCCUPANCY_HISTORY_ERASURE", panels=panels, caption="Branch-A residual first-passage evidence at the first sampled occupancy M/N≥0.35, with a synchronized exact-state representative.", major_morphology=True)


def figure_04_event_proximal(
    run_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run, experiment = Path(run_path).resolve(), Path(experiment_path).resolve()
    residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"
    result_path = experiment / "HISTORY_ERASURE_RESULTS.csv"
    rows = _typed_rows(read_verified_csv(residual_path))
    results = _typed_rows(read_verified_csv(result_path))
    b_rows = [row for row in rows if row["branch_point"] == "B" and int(row["realization"]) == 1 and row["intervention"] in {"global_shuffle", "tile_32", "tile_128"}]
    if not b_rows:
        raise RuntimeError("Branch-B rows are unavailable")
    representative = _representatives(experiment)[0]
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    realized = _realized(run)
    exact_states: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    untouched_records = index_task_states(run / "media_states" / "untouched" / "branch_B" / source_id)
    state, record = _exact_at(untouched_records, 32768)
    exact_states.append(("untouched", state, record))
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        entry = _branch_entry(realized, source_id, "B", intervention)
        state, record = _exact_at(index_task_states(run / "branches" / entry["task_id"]), 32768)
        exact_states.append((intervention, state, record))

    fig = plt.figure(figsize=(16, 11), facecolor="#071018")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])
    survival_axis = fig.add_subplot(grid[0, :2])
    delay_axis = fig.add_subplot(grid[0, 2:])
    fig.suptitle("Erasing history 32,768 phases before the untouched event", fontsize=21, color="#f6f8fa")
    _chart_style(survival_axis, "Event-proximal survival")
    survival_axis.axvline(32768, color=INTERVENTION_COLORS["untouched"], linestyle="--", label="untouched hit by construction")
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        selected = [row for row in b_rows if row["intervention"] == intervention]
        _step_survival(survival_axis, selected, altered=True, label=intervention.replace("_", " "), color=INTERVENTION_COLORS[intervention])
    survival_axis.set(xlim=(0, 262144), ylim=(0, 1.03), xlabel="elapsed phases", ylabel="event-free fraction")
    survival_axis.legend(frameon=False, fontsize=9)

    _chart_style(delay_axis, "Paired RMST delay with 95% interval")
    delay_axis.axvspan(-8192, 8192, color="#f0e44222", label="practical-equivalence band")
    selected_results = [row for row in results if row["comparison"] in {"Branch B primary global", "Branch B tile 32", "Branch B tile 128"}]
    for index, row in enumerate(selected_results):
        point, low, high = float(row["rmst_delay"]), float(row["ci95_low"]), float(row["ci95_high"])
        delay_axis.errorbar(point, index, xerr=[[point - low], [high - point]], fmt="o", color=list(INTERVENTION_COLORS.values())[index + 1], capsize=5, linewidth=2)
    delay_axis.set_yticks(range(len(selected_results)), [str(row["comparison"]).replace("Branch B ", "") for row in selected_results])
    delay_axis.axvline(0, color="#9fb0bf", linewidth=1)
    delay_axis.set_xlabel("paired RMST delay (phases; positive = later)")
    delay_axis.legend(frameon=False)

    panels = [
        _aggregate_panel(residual_path, panel="branch_B_survival", branch_point="B"),
        _aggregate_panel(result_path, panel="branch_B_rmst_delay", branch_point="B"),
    ]
    for column, (intervention, state, record) in enumerate(exact_states):
        axis = fig.add_subplot(grid[1, column])
        _show_state(axis, state, title=intervention.replace("_", " ") + " at +32,768", phase=int(record["phase"]))
        panels.append(_state_panel_from_record(record, source_task=source_id, seed=seed, density=density, branch_point="B", intervention=intervention, requested_elapsed=32768, extra={"panel": f"representative_{intervention}"}))
    _state_legend(fig, y=0.005)
    return _save_triplet(fig, Path(output_dir), "FIGURE_04_EVENT_PROXIMAL_HISTORY_ERASURE", panels=panels, caption="Branch-B survival and paired delay estimates use an untouched residual fixed at 32,768 phases by preregistration; exact representative states are synchronized at that elapsed phase.", major_morphology=True)


def _curve_scalar(value: Any) -> float:
    if isinstance(value, dict):
        if "mean" not in value:
            raise ValueError("structured curve value is missing its mean component")
        value = value["mean"]
    return float(value)


def _mean_curves(records: list[dict[str, Any]], field: str) -> dict[str, dict[int, float]]:
    by_intervention: dict[str, dict[int, list[float]]] = {}
    for row in records:
        mapping = row[field]
        if not isinstance(mapping, dict):
            continue
        key = str(row["intervention"])
        by_intervention.setdefault(key, {})
        for scale, value in mapping.items():
            by_intervention[key].setdefault(int(scale), []).append(_curve_scalar(value))
    return {
        intervention: {scale: float(np.mean(values)) for scale, values in curves.items()}
        for intervention, curves in by_intervention.items()
    }


def figure_05_multiscale_memory(
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    experiment = Path(experiment_path).resolve()
    audit_path = experiment / "INTERVENTION_AUDIT.csv"
    residual_path = experiment / "RESIDUAL_FIRST_PASSAGE.csv"
    audit = _typed_rows(read_verified_csv(audit_path))
    residual = _typed_rows(read_verified_csv(residual_path))
    audit = [row for row in audit if row["branch_point"] == "B" and int(row["realization"]) == 1 and row["intervention"] in {"global_shuffle", "tile_32", "tile_128"}]
    block_source = _mean_curves(audit, "source_block_entropy")
    block_altered = _mean_curves(audit, "altered_block_entropy")
    mi_source = _mean_curves(audit, "source_mutual_information")
    mi_altered = _mean_curves(audit, "altered_mutual_information")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="#071018")
    fig.suptitle("How much spatial information survives each intervention?", fontsize=21, color="#f6f8fa")
    _chart_style(axes[0, 0], "Block entropy immediately after intervention")
    _chart_style(axes[0, 1], "Categorical mutual information by offset")
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        color = INTERVENTION_COLORS[intervention]
        scales = sorted(block_altered.get(intervention, {}))
        axes[0, 0].plot(scales, [block_altered[intervention][scale] for scale in scales], marker="o", color=color, label=intervention.replace("_", " "))
        source_scales = sorted(block_source.get(intervention, {}))
        axes[0, 0].plot(source_scales, [block_source[intervention][scale] for scale in source_scales], color=color, linestyle="--", alpha=0.45)
        offsets = sorted(mi_altered.get(intervention, {}))
        axes[0, 1].plot(offsets, [mi_altered[intervention][offset] for offset in offsets], marker="o", color=color, label=intervention.replace("_", " "))
        source_offsets = sorted(mi_source.get(intervention, {}))
        axes[0, 1].plot(source_offsets, [mi_source[intervention][offset] for offset in source_offsets], color=color, linestyle="--", alpha=0.45)
    axes[0, 0].set(xscale="log", xlabel="block size", ylabel="normalized block entropy")
    axes[0, 1].set(xscale="log", yscale="symlog", xlabel="offset", ylabel="mutual information (nats)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[0, 0].add_artist(Line2D([], [], color="#9fb0bf", linestyle="--", label="source before"))

    _chart_style(axes[1, 0], "Immediate retained low-k precursor")
    positions, labels = [], []
    for index, intervention in enumerate(("global_shuffle", "tile_32", "tile_128")):
        values = [float(row["altered_low_k_power"]) / max(float(row["source_low_k_power"]), 1e-30) for row in audit if row["intervention"] == intervention]
        positions.append(index)
        labels.append(intervention.replace("_", " "))
        axes[1, 0].scatter(np.full(len(values), index), values, color=INTERVENTION_COLORS[intervention], alpha=0.7)
        if values:
            axes[1, 0].plot([index - 0.18, index + 0.18], [np.median(values)] * 2, color="white", linewidth=2)
    axes[1, 0].axhline(1, color="#9fb0bf", linestyle="--")
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylabel("altered / source low-k power")

    _chart_style(axes[1, 1], "Retained organisation versus residual time")
    residual_lookup = {row["task_id"]: row for row in residual}
    for intervention in ("global_shuffle", "tile_32", "tile_128"):
        points = []
        for row in audit:
            if row["intervention"] != intervention or row["task_id"] not in residual_lookup:
                continue
            source_mi = row["source_mutual_information"]
            altered_mi = row["altered_mutual_information"]
            offsets = sorted(int(value) for value in source_mi)
            retained = float(
                np.mean(
                    [
                        _curve_scalar(
                            altered_mi[str(offset)]
                            if str(offset) in altered_mi
                            else altered_mi[offset]
                        )
                        / max(
                            _curve_scalar(
                                source_mi[str(offset)]
                                if str(offset) in source_mi
                                else source_mi[offset]
                            ),
                            1e-30,
                        )
                        for offset in offsets
                    ]
                )
            )
            time, observed = _event_time(residual_lookup[row["task_id"]], altered=True)
            points.append((retained, time, observed))
        if points:
            axes[1, 1].scatter([p[0] for p in points], [p[1] for p in points], marker="o" if all(p[2] for p in points) else "s", color=INTERVENTION_COLORS[intervention], alpha=0.75, label=intervention.replace("_", " "))
    axes[1, 1].set(xlabel="mean retained MI ratio", ylabel="R_hit or censor time")
    axes[1, 1].legend(frameon=False)
    fig.text(0.5, 0.01, "Solid curves are altered-state measurements; dashed curves are their matched pre-intervention sources. Lines connect measured scale points and are not fitted laws.", ha="center", color="#c8d7e3", fontsize=10)
    panels = [
        _aggregate_panel(audit_path, panel="block_entropy_by_scale", branch_point="B"),
        _aggregate_panel(audit_path, panel="mutual_information_by_offset", branch_point="B"),
        _aggregate_panel(audit_path, panel="retained_low_k", branch_point="B"),
        _aggregate_panel(residual_path, panel="retained_information_vs_residual", branch_point="B", extra={"joined_source": str(audit_path.resolve())}),
    ]
    return _save_triplet(fig, Path(output_dir), "FIGURE_05_MULTISCALE_MEMORY", panels=panels, caption="Immediate retained multiscale organisation and its relation to later residual first-passage time. Solid curves are measurements after intervention; dashed curves are matched source measurements.", major_morphology=False)


def figure_06_micro_macro(
    run_path: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    run, experiment = Path(run_path).resolve(), Path(experiment_path).resolve()
    representative = _representatives(experiment)[0]
    source_id, density, seed = str(representative["source_task_id"]), float(representative["density"]), int(representative["seed"])
    source = _source_result(run, source_id)
    target_phase = int(source["time_to_event"]["T_hit"])
    records = index_task_states(run / "sources" / source_id)
    exact = [record for record in records if int(record["phase"]) == target_phase]
    if exact:
        record = min(exact, key=lambda row: row["path"])
    else:
        before = [record for record in records if int(record["phase"]) <= target_phase]
        if not before:
            raise RuntimeError("representative source lacks an exact state at or before T_hit")
        record = max(before, key=lambda row: int(row["phase"]))
    state, _, _, _ = _record_state(record)
    crop256 = aligned_densest_crop(state, 256)
    crop64 = aligned_densest_crop(state, 64, bounds=crop256)
    local = crop_state(state, crop64)
    histogram = state_histogram(local)
    probabilities = histogram / histogram.sum()
    entropy = float(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])) / np.log(4.0))
    occupied = (state != 0).astype(np.float32)
    block = 32
    coarse = occupied.reshape(state.shape[0] // block, block, state.shape[1] // block, block).mean(axis=(1, 3))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#071018")
    fig.suptitle("Microscopic disorder can coexist with macroscopic order", fontsize=21, color="#f6f8fa")
    _show_state(axes[0, 0], state, title="Full 1024×1024 world", phase=int(record["phase"]))
    _show_state(axes[0, 1], state, title=f"256×256 crop · ({crop256[0]},{crop256[1]})", phase=int(record["phase"]), crop=crop256)
    _show_state(axes[0, 2], state, title=f"64×64 crop · ({crop64[0]},{crop64[1]})", phase=int(record["phase"]), crop=crop64)
    _chart_style(axes[1, 0], "Local four-state histogram")
    axes[1, 0].bar(STATE_NAMES, histogram, color=STATE_HEX)
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_ylabel("cells in 64×64 crop")
    _chart_style(axes[1, 1], "Coarse occupancy field · 32×32 blocks")
    image = axes[1, 1].imshow(coarse, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="occupied fraction")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 2].set_facecolor("#0b1721")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.82, f"Normalized local entropy\n{entropy:.3f} / 1.000", color="#f0e442", fontsize=22, weight="bold")
    axes[1, 2].text(0.05, 0.55, "Large-scale domains are visible\nwithout a simple microscopic\ncheckerboard, stripe, or crystal.", color="#d7e9f6", fontsize=16)
    axes[1, 2].text(0.05, 0.25, "This panel describes morphology;\nit does not claim equilibrium\nor a universal phase transition.", color="#9fb0bf", fontsize=12)
    _state_legend(fig, y=0.005)
    base_kwargs = dict(source_task=source_id, seed=seed, density=density, branch_point="source_event", intervention="untouched", requested_elapsed=None)
    panels = [
        _state_panel_from_record(record, crop=(0, 0, state.shape[1], state.shape[0]), extra={"panel": "full_world"}, **base_kwargs),
        _state_panel_from_record(record, crop=crop256, extra={"panel": "crop_256", "crop_selection": "frozen aligned maximum occupancy"}, **base_kwargs),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "crop_64", "crop_selection": "frozen nested aligned maximum occupancy"}, **base_kwargs),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "local_histogram", "histogram": histogram.tolist()}, **base_kwargs),
        _state_panel_from_record(record, crop=(0, 0, state.shape[1], state.shape[0]), extra={"panel": "coarse_density", "coarse_block_size": 32}, **base_kwargs),
        _state_panel_from_record(record, crop=crop64, extra={"panel": "normalized_local_entropy", "normalized_entropy": entropy}, **base_kwargs),
    ]
    return _save_triplet(fig, Path(output_dir), "FIGURE_06_MICRO_DISORDER_MACRO_ORDER", panels=panels, caption="A frozen event-centred representative shown at full-world, 256-cell, and 64-cell scales. The local state mixture remains disordered while the coarse occupancy field is strongly organised.", major_morphology=True)


def figure_07_mechanism_ledger(output_dir: str | Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="#071018")
    fig.suptitle("The mechanism ledger: exact charge balance does not freeze occupancy", fontsize=22, color="#f6f8fa")
    for axis in axes:
        axis.set_facecolor("#0b1721")
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 10)
        axis.axis("off")
    left, right = axes
    left.set_title("Frozen winner: exact Q, variable M", color="#f0e442", fontsize=16)
    left.text(5, 8.7, r"$Q=N_{+}-N_{-}=0$ exactly", ha="center", color="white", fontsize=22)
    left.text(5, 7.7, r"$M=N_{+}+N_{-}+N_{bound}$ may change", ha="center", color="white", fontsize=19)
    for x, color, label in ((3.8, STATE_HEX[1], "+"), (6.2, STATE_HEX[2], "-")):
        left.add_patch(Rectangle((x - 0.65, 4.8), 1.3, 1.3, facecolor=color, edgecolor="white", linewidth=2))
        left.text(x, 5.45, label, ha="center", va="center", color="#05070a" if label == "+" else "white", fontsize=24, weight="bold")
    left.add_patch(FancyArrowPatch((5, 7.1), (5, 6.2), arrowstyle="-|>", mutation_scale=18, color="#9fb0bf"))
    left.text(5, 4.1, "Balanced pair creation changes M by +2\nwithout changing Q", ha="center", color="#c8d7e3", fontsize=15)
    left.text(5, 2.1, "Trajectory evidence measures when and where\nthis changing inventory becomes organised.", ha="center", color="#9fb0bf", fontsize=13)
    right.set_title("Exact-Q+M sibling control", color="#56b4e9", fontsize=16)
    right.text(5, 8.7, r"$Q=0$ and $M=M_{initial}>0$", ha="center", color="white", fontsize=22)
    right.text(5, 7.7, "The same number of occupied cells is retained", ha="center", color="white", fontsize=16)
    for i, (color, label) in enumerate(((STATE_HEX[1], "+"), (STATE_HEX[2], "-"), (STATE_HEX[3], "b"))):
        x = 3.1 + i * 1.9
        right.add_patch(Rectangle((x - 0.6, 4.8), 1.2, 1.2, facecolor=color, edgecolor="white", linewidth=2))
        right.text(x, 5.4, label, ha="center", va="center", color="white" if i else "#05070a", fontsize=22, weight="bold")
    right.text(5, 4.0, "Rearrangements may occur, but total M is fixed", ha="center", color="#c8d7e3", fontsize=15)
    right.text(5, 2.1, "This control does not reproduce the winner’s\nstrong-domain regime under the tested protocol.", ha="center", color="#9fb0bf", fontsize=13)
    fig.text(0.5, 0.03, "MODEL SCHEMATIC - not a trajectory snapshot", ha="center", color="#cc79a7", fontsize=13, weight="bold")
    model_source = Path(__file__).resolve()
    panels = [
        make_panel_record(
            source_task="model_schematic",
            seed=None,
            density=None,
            branch_point=None,
            intervention="frozen_winner_ledger",
            phase="not_applicable",
            state_hash="model_schematic_no_state",
            crop=(0, 0, 0, 0),
            source_path=model_source,
            extra={"panel": "variable_M_winner", "evidence_type": "MODEL SCHEMATIC"},
        ),
        make_panel_record(
            source_task="model_schematic",
            seed=None,
            density=None,
            branch_point=None,
            intervention="exact_Q_M_sibling_control",
            phase="not_applicable",
            state_hash="model_schematic_no_state",
            crop=(0, 0, 0, 0),
            source_path=model_source,
            extra={"panel": "exact_Q_M_control", "evidence_type": "MODEL SCHEMATIC"},
        ),
    ]
    return _save_triplet(
        fig,
        Path(output_dir),
        "FIGURE_07_MECHANISM_LEDGER",
        panels=panels,
        caption=(
            "Model schematic. The winner keeps signed charge Q exactly zero but permits occupancy M "
            "to change; the exact-Q+M sibling instead fixes a positive, nonzero occupancy."
        ),
        major_morphology=False,
    )


def figure_08_research_story(
    repo_root: str | Path,
    experiment_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    experiment = Path(experiment_path).resolve()
    decision_path = experiment / "SCIENTIFIC_DECISION_V032.json"
    decision = _read_json(decision_path)
    mechanism = decision["combined_mechanism"]
    stages = [
        ("Discovery", "Exact Q=0 reversible rule\nwith delayed domain formation", "#f0e442"),
        ("Independent confirmation", "Fresh seeds + exhaustive local audit\n+ Eshkol whole-lattice parity", "#56b4e9"),
        ("Domain composition", "Irregular high-entropy domains;\nno simple crystal signature", "#009e73"),
        ("Mechanism controls", "Pure rotation and exact-Q+M\ndo not reproduce the regime", "#cc79a7"),
        ("Density & waiting time", "Threshold becomes an observation-time\nboundary; waiting time is density-sensitive", "#e69f00"),
        ("Morphology", "Distributed simultaneous coarsening,\nnot one isolated expanding nucleus", "#d55e00"),
        ("History erasure", mechanism, "#f0e442"),
    ]
    fig, axis = plt.subplots(figsize=(16, 9), facecolor="#071018")
    axis.set_facecolor("#071018")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")
    fig.suptitle("From reversible discovery to a state-matched causal mechanism test", fontsize=22, color="#f6f8fa")
    xs = np.linspace(1.3, 14.7, len(stages))
    for index, ((title, text, color), x) in enumerate(zip(stages, xs, strict=True)):
        y = 5.1 + (0.75 if index % 2 == 0 else -0.75)
        axis.add_patch(Rectangle((x - 0.92, y - 1.05), 1.84, 2.1, facecolor="#102331", edgecolor=color, linewidth=2.5))
        axis.text(x, y + 0.58, title, ha="center", va="center", color=color, fontsize=10.5, weight="bold", wrap=True)
        axis.text(x, y - 0.25, text, ha="center", va="center", color="#d7e9f6", fontsize=8.4, wrap=True)
        if index < len(stages) - 1:
            axis.add_patch(FancyArrowPatch((x + 0.95, y), (xs[index + 1] - 0.95, 5.1 + (0.75 if (index + 1) % 2 == 0 else -0.75)), arrowstyle="-|>", mutation_scale=15, color="#7890a2", linewidth=1.5))
    corrections = [
        "recurrence → mainly rotation scaffold",
        "sustained coarsening → transient / kinetic organisation",
        "fixed threshold → observation-time boundary",
        "local nucleus → distributed coarsening",
    ]
    axis.text(0.8, 2.25, "Interpretation corrections", color="#cc79a7", fontsize=15, weight="bold")
    for index, text in enumerate(corrections):
        axis.text(1.0 + (index % 2) * 7.5, 1.62 - (index // 2) * 0.62, "• " + text, color="#c8d7e3", fontsize=12)
    axis.text(8, 0.35, "Final finite-campaign conclusion: " + mechanism, ha="center", color="#f0e442", fontsize=15, weight="bold")
    sources = [
        repo / "CODEX_COMPLETION_V030.md",
        repo / "CODEX_COMPLETION_V031.md",
        experiment / "HISTORY_ERASURE_REPORT.md",
        decision_path,
    ]
    panels = [
        _aggregate_panel(
            source,
            panel=f"research_story_source_{index + 1}",
            extra={"source_file": str(source.resolve()), "mechanism": mechanism},
        )
        for index, source in enumerate(sources)
        if source.is_file()
    ]
    return _save_triplet(
        fig,
        Path(output_dir),
        "FIGURE_08_COMPLETE_RESEARCH_STORY",
        panels=panels,
        caption=(
            "The programme moved from an exact reversible discovery through independent audits, "
            "controls, kinetics, and morphology to the final preregistered state-matched history-erasure test."
        ),
        major_morphology=False,
    )


def figure_output_paths(output_dir: str | Path) -> list[Path]:
    directory = Path(output_dir)
    return [
        directory / f"FIGURE_{index:02d}_{name}{suffix}"
        for index, name in (
            (1, "ZERO_BALANCE_REVERSIBLE_HERO"),
            (2, "HISTORY_ERASURE_BRANCHES"),
            (3, "FIXED_OCCUPANCY_HISTORY_ERASURE"),
            (4, "EVENT_PROXIMAL_HISTORY_ERASURE"),
            (5, "MULTISCALE_MEMORY"),
            (6, "MICRO_DISORDER_MACRO_ORDER"),
            (7, "MECHANISM_LEDGER"),
            (8, "COMPLETE_RESEARCH_STORY"),
        )
        for suffix in (".svg", ".pdf", ".png")
    ]
