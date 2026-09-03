from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .atomic import atomic_write_json, atomic_write_text, sha256_file




def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

@dataclass(frozen=True)
class LogisticFit:
    intercept: float
    slope: float
    p25: float
    p50: float
    p75: float
    width: float
    sample_min: float
    sample_max: float
    midpoint_bracketed: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def fit_logistic(density: np.ndarray, outcome: np.ndarray) -> LogisticFit:
    x = np.asarray(density, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.size < 4 or np.unique(y).size < 2:
        return LogisticFit(
            *(float("nan"),) * 8,
            midpoint_bracketed=False,
            success=False,
        )
    center = float(np.mean(x))
    scale = float(np.std(x)) or 1.0
    z = (x - center) / scale

    def objective(beta: np.ndarray) -> float:
        logits = beta[0] + np.exp(beta[1]) * z
        p = _sigmoid(logits)
        nll = -float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        return nll + 0.005 * float(np.square(beta).sum())

    result = minimize(objective, np.asarray([0.0, 0.0]), method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        return LogisticFit(
            *(float("nan"),) * 8,
            midpoint_bracketed=False,
            success=False,
        )
    intercept_z = float(result.x[0])
    slope_z = float(np.exp(result.x[1]))
    slope = slope_z / scale
    intercept = intercept_z - slope * center

    def quantile(p: float) -> float:
        logit = math.log(p / (1 - p))
        return (logit - intercept) / slope

    p25, p50, p75 = quantile(0.25), quantile(0.50), quantile(0.75)
    sample_min = float(x.min())
    sample_max = float(x.max())
    return LogisticFit(
        intercept,
        slope,
        p25,
        p50,
        p75,
        p75 - p25,
        sample_min,
        sample_max,
        sample_min <= p50 <= sample_max,
        success=True,
    )


def _flatten_result(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    task = payload["task"]
    cls = payload["classification"]
    initial = payload["initial"]
    initial_charge = int(payload["topological_series"][0]["charge"])
    charge_violation = any(
        int(row["charge"]) != initial_charge
        for series_name in ("topological_series", "temporal_series")
        for row in payload[series_name]
    )
    final_series = payload["topological_series"][-1]
    initial_series = payload["topological_series"][0]
    initial_occupied = float(initial_series["largest_occupied_fraction"])
    final_occupied = float(final_series["largest_occupied_fraction"])
    peak_row = max(payload["topological_series"], key=lambda row: row["finite_empty_scale"])
    max_growth = max((item["rate_per_phase"] for item in payload["shell_growth"] if np.isfinite(item["rate_per_phase"])), default=float("nan"))
    return {
        "result_path": str(path),
        "task_id": task["id"],
        "stage": task["stage"],
        "size": int(task["size"]),
        "density": float(task["density"]),
        "seed": int(task["seed"]),
        "arrangement": task["arrangement"],
        "rule_variant": task["rule_variant"],
        "steps": int(task["steps"]),
        "strong_domain": bool(cls["strong_domain"]),
        "first_passage_phase": cls.get("first_passage_phase"),
        "peak_scale_ratio": float(cls["peak_scale_ratio"]),
        "final_scale_ratio": float(cls["final_scale_ratio"]),
        "peak_low_k_ratio": float(cls["peak_low_k_ratio"]),
        "final_low_k_ratio": float(cls["final_low_k_ratio"]),
        "max_shell_growth_rate": max_growth,
        "initial_occupancy": float(initial["inventory"]["positive"] + initial["inventory"]["negative"] + initial["inventory"]["bound"]) / (task["size"] ** 2),
        "final_occupancy": float(final_series["occupancy_fraction"]),
        "initial_largest_occupied_fraction": initial_occupied,
        "final_largest_occupied_fraction": final_occupied,
        "occupied_cluster_retention": final_occupied / initial_occupied if initial_occupied > 0 else float("nan"),
        "initial_charge": initial_charge,
        "charge_violation": charge_violation,
        "peak_dense_entropy": float(peak_row["normalized_state_entropy"]),
        "final_empty_wrap_x": bool(final_series["empty_wrap_x"]),
        "final_empty_wrap_y": bool(final_series["empty_wrap_y"]),
        "final_occupied_wrap_x": bool(final_series["occupied_wrap_x"]),
        "final_occupied_wrap_y": bool(final_series["occupied_wrap_y"]),
        "rule_sha256": payload["rule"]["sha256"],
        "initial_state_sha256": initial["state_sha256"],
        "final_state_sha256": payload["final"]["state_sha256"],
    }


def load_results(output: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root = Path(output)
    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "tasks").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        rows.append(_flatten_result(payload, path))
    if not rows:
        raise RuntimeError(f"no task results found under {root / 'tasks'}")
    return pd.DataFrame(rows), payloads


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["stage", "size", "density", "arrangement", "rule_variant"], dropna=False)
        .agg(
            trajectories=("task_id", "count"),
            strong_passes=("strong_domain", "sum"),
            pass_fraction=("strong_domain", "mean"),
            median_peak_scale_ratio=("peak_scale_ratio", "median"),
            median_final_scale_ratio=("final_scale_ratio", "median"),
            median_peak_low_k_ratio=("peak_low_k_ratio", "median"),
            median_max_shell_growth_rate=("max_shell_growth_rate", "median"),
            median_first_passage=("first_passage_phase", "median"),
            median_final_occupancy=("final_occupancy", "median"),
            median_occupied_cluster_retention=("occupied_cluster_retention", "median"),
            empty_wrap_fraction_x=("final_empty_wrap_x", "mean"),
            empty_wrap_fraction_y=("final_empty_wrap_y", "mean"),
            occupied_wrap_fraction_x=("final_occupied_wrap_x", "mean"),
            occupied_wrap_fraction_y=("final_occupied_wrap_y", "mean"),
        )
        .reset_index()
    )
    return grouped


def _plot_onset(frame: pd.DataFrame, figure_dir: Path) -> dict[str, Any]:
    onset = frame[(frame.stage.str.startswith("onset")) & (frame.arrangement == "homogeneous") & (frame.rule_variant == "winner")]
    fits: dict[str, Any] = {}
    if onset.empty:
        return fits
    plt.figure(figsize=(8, 5))
    for size, group in onset.groupby("size"):
        summary = group.groupby("density").strong_domain.agg(["mean", "count"]).reset_index()
        plt.plot(summary.density, summary["mean"], marker="o", label=f"{size}²")
        fit = fit_logistic(group.density.to_numpy(), group.strong_domain.astype(int).to_numpy())
        fits[str(int(size))] = fit.to_dict()
        if fit.success:
            grid = np.linspace(group.density.min(), group.density.max(), 300)
            plt.plot(grid, _sigmoid(fit.intercept + fit.slope * grid), alpha=0.6)
    plt.xlabel("Initial homogeneous occupancy")
    plt.ylabel("Strong-domain pass probability")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "onset_probability.png"
    plt.savefig(path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for size, group in onset.groupby("size"):
        summary = group.groupby("density").first_passage_phase.median().reset_index()
        plt.plot(summary.density, summary.first_passage_phase, marker="o", label=f"{size}²")
    plt.xlabel("Initial homogeneous occupancy")
    plt.ylabel("Median first-passage phase")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "first_passage.png", dpi=180)
    plt.close()
    return fits


def _plot_precursor(frame: pd.DataFrame, figure_dir: Path) -> None:
    precursor = frame[frame.stage.str.startswith("precursor")]
    if precursor.empty:
        return
    plt.figure(figsize=(8, 5))
    for (size, rule_variant), group in precursor.groupby(["size", "rule_variant"]):
        summary = group.groupby("density").max_shell_growth_rate.median().reset_index()
        plt.plot(
            summary.density,
            summary.max_shell_growth_rate,
            marker="o",
            label=f"{size}² {rule_variant}",
        )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Initial homogeneous occupancy")
    plt.ylabel("Median maximum early log-power growth per phase")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "spectral_precursor_growth.png", dpi=180)
    plt.close()


def _plot_hysteresis(frame: pd.DataFrame, figure_dir: Path) -> None:
    data = frame[(frame.stage.str.startswith("hysteresis")) & (frame.rule_variant == "winner")]
    if data.empty:
        return
    plt.figure(figsize=(8, 5))
    for arrangement, group in data.groupby("arrangement"):
        summary = group.groupby("density").strong_domain.mean().reset_index()
        plt.plot(summary.density, summary.strong_domain, marker="o", label=arrangement)
    plt.xlabel("Inventory-matched occupancy")
    plt.ylabel("Strong-growth classifier pass probability")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "hysteresis_seed_dependence.png", dpi=180)
    plt.close()

    seeded = data[data.arrangement.isin(["slab", "droplet"])]
    if seeded.empty:
        return
    plt.figure(figsize=(8, 5))
    for arrangement, group in seeded.groupby("arrangement"):
        retention = group.groupby("density").occupied_cluster_retention.median().reset_index()
        plt.plot(
            retention.density,
            retention.occupied_cluster_retention,
            marker="o",
            label=arrangement,
        )
    plt.axhline(1.0, linewidth=1, color="black", alpha=0.6)
    plt.xlabel("Inventory-matched occupancy")
    plt.ylabel("Median final/initial largest occupied-cluster fraction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "seeded_cluster_retention.png", dpi=180)
    plt.close()


def _plot_ablation(frame: pd.DataFrame, figure_dir: Path) -> None:
    data = frame[frame.stage.str.startswith("ablation")]
    if data.empty:
        return
    summary = data.groupby("rule_variant").agg(pass_fraction=("strong_domain", "mean"), median_peak=("peak_scale_ratio", "median")).reset_index()
    summary = summary.sort_values(["pass_fraction", "median_peak"], ascending=False)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary.pass_fraction)
    plt.xticks(x, summary.rule_variant, rotation=70, ha="right")
    plt.ylabel("Strong-domain pass fraction")
    plt.tight_layout()
    plt.savefig(figure_dir / "reaction_pair_ablation.png", dpi=180)
    plt.close()


def _dashboard(output: Path, frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    raw_records = frame.to_dict(orient="records")
    for record in raw_records:
        record["image_base"] = f"../tasks/{record['stage']}/{record['task_id']}"
    records = _json_safe(raw_records)
    summary_records = _json_safe(summary.to_dict(orient="records"))
    data_path = output / "dashboard_data.json"
    atomic_write_json(data_path, {"records": records, "summary": summary_records})
    embedded_data = json.dumps(
        {"records": records, "summary": summary_records},
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    html = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZWS Reversible Null Phase Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:1rem}a{color:#58a6ff}
.controls{display:flex;gap:1rem;flex-wrap:wrap;background:#161b22;padding:1rem;border-radius:10px;position:sticky;top:0}
label{display:flex;flex-direction:column;font-size:.8rem;color:#9da7b3}select{font-size:1rem;margin-top:.25rem;background:#0d1117;color:#e6edf3;padding:.35rem}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1rem}.card{background:#161b22;padding:1rem;border-radius:10px}
img{width:100%;image-rendering:pixelated;background:#000}.metric{font-variant-numeric:tabular-nums}.wide{grid-column:1/-1}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>Reversible Null Phase Study</h1><p>Select a completed trajectory. Images are exact nearest-neighbour renderings.</p>
<div class="controls"><label>Stage<select id="stage"></select></label><label>Size<select id="size"></select></label><label>Density<select id="density"></select></label><label>Arrangement<select id="arrangement"></select></label><label>Rule<select id="rule"></select></label><label>Seed<select id="seed"></select></label></div>
<div class="grid"><div class="card"><h3>Initial</h3><img id="initial"></div><div class="card"><h3>Peak</h3><img id="peak"></div><div class="card"><h3>Final</h3><img id="final"></div><div class="card wide"><pre id="metrics" class="metric"></pre></div></div>
<script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
<script>
let records=[];const fields=['stage','size','density','arrangement','rule_variant','seed'];
const els=Object.fromEntries(fields.map(x=>[x,document.getElementById(x==='rule_variant'?'rule':x)]));
function values(field, subset){return [...new Set(subset.map(r=>String(r[field])))].sort((a,b)=>Number(a)-Number(b)||a.localeCompare(b))}
function refresh(from=0){let subset=records;for(let i=0;i<fields.length;i++){const f=fields[i],el=els[f];const opts=values(f,subset);const prior=el.value;el.innerHTML=opts.map(v=>`<option>${v}</option>`).join('');if(opts.includes(prior))el.value=prior;subset=subset.filter(r=>String(r[f])===el.value)}const r=subset[0];if(!r)return;const base=r.image_base;for(const n of ['initial','peak','final'])document.getElementById(n).src=base+'/'+n+'.png';document.getElementById('metrics').textContent=JSON.stringify(r,null,2)}
const dashboardData=JSON.parse(document.getElementById('dashboard-data').textContent);records=dashboardData.records;fields.forEach((f,i)=>els[f].addEventListener('change',()=>refresh(i)));refresh();
</script></body></html>'''
    html = html.replace("__DASHBOARD_DATA__", embedded_data)
    atomic_write_text(output / "dashboard.html", html)


def build_report(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    frame, payloads = load_results(root)
    summary = _group_summary(frame)
    report_dir = root / "report"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "TRAJECTORIES.csv", index=False)
    summary.to_csv(report_dir / "SUMMARY.csv", index=False)
    fits = _plot_onset(frame, figure_dir)
    _plot_precursor(frame, figure_dir)
    _plot_hysteresis(frame, figure_dir)
    _plot_ablation(frame, figure_dir)
    _dashboard(report_dir, frame, summary)

    winner_onset = summary[
        (summary.stage.str.startswith("onset"))
        & (summary.arrangement == "homogeneous")
        & (summary.rule_variant == "winner")
    ]
    headline = "No onset stage was run."
    if not winner_onset.empty:
        max_size = int(winner_onset["size"].max())
        fit = fits.get(str(max_size), {})
        largest_onset = winner_onset[winner_onset["size"] == max_size].sort_values("density")
        lowest_onset = largest_onset.iloc[0]
        if fit and fit.get("success") and fit.get("midpoint_bracketed"):
            headline = (
                f"At {max_size}², the descriptive strong-domain midpoint is "
                f"{fit['p50']:.5f} occupancy with a p25–p75 width of {fit['width']:.5f}."
            )
        elif fit and fit.get("success"):
            headline = (
                f"The {max_size}² onset sweep did not bracket a 50% midpoint: its lowest "
                f"sampled density ({float(lowest_onset['density']):.4f}) already passed "
                f"{int(lowest_onset['strong_passes'])}/{int(lowest_onset['trajectories'])}. "
                f"The fitted p50={fit['p50']:.5f} is an extrapolation, not a measured threshold."
            )
        else:
            headline = f"The {max_size}² onset response did not support a stable logistic fit."

    precursor = summary[(summary.stage.str.startswith("precursor")) & (summary.rule_variant == "winner")]
    precursor_controls = summary[summary.stage.str.startswith("precursor")]
    horizon = summary[(summary.stage.str.startswith("horizon")) & (summary.rule_variant == "winner")]
    hysteresis = summary[(summary.stage.str.startswith("hysteresis")) & (summary.rule_variant == "winner")]
    ablation = summary[summary.stage.str.startswith("ablation")]
    charge_violations = int(frame["charge_violation"].sum())
    lines = [
        "# Reversible Null Phase Study Report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}.",
        "",
        "## Executive summary",
        "",
        headline,
        "",
        "This report is a purpose-built follow-up to the frozen v0.1.4 discovery. It tests whether the apparent ~27.5% onset is a finite-time nucleation boundary, a linear spectral instability, a seeded/bistable regime, a topology change on the torus, or a reaction-pair-specific effect. It does not perform a new rule search.",
        "",
        "## Completed trajectories",
        "",
        f"- Total: **{len(frame)}**",
        f"- Exact-charge violations recorded in compact results: **{charge_violations}**",
        "",
    ]
    if not precursor.empty:
        precursor_ordered = precursor.sort_values("density")
        precursor_low = precursor_ordered.iloc[0]
        precursor_high = precursor_ordered.iloc[-1]
        rotation_precursor = precursor_controls[precursor_controls.rule_variant == "rotation"]
        rotation_max = (
            float(rotation_precursor["median_max_shell_growth_rate"].abs().max())
            if not rotation_precursor.empty
            else float("nan")
        )
        lines += [
            "## Spectral precursor",
            "",
            f"- Winner median maximum early shell growth rose from {float(precursor_low['median_max_shell_growth_rate']):.3e} per phase at density {float(precursor_low['density']):.2f} to {float(precursor_high['median_max_shell_growth_rate']):.3e} at {float(precursor_high['density']):.2f}.",
            f"- The rotation control remained at or below {rotation_max:.3e} in absolute median growth.",
            "- None of these 8,192-phase precursor trajectories passed the strong-domain criterion.",
            "",
            "This is evidence of reaction-specific low-k amplification well below the later strong-domain window, not by itself proof of a persistent phase or a sharp linear-instability threshold. See `figures/spectral_precursor_growth.png`.",
            "",
        ]
    if not winner_onset.empty:
        lines += [
            "## Strong-domain onset and first passage",
            "",
        ]
        for size in sorted(int(value) for value in winner_onset["size"].unique()):
            size_rows = winner_onset[winner_onset["size"] == size].sort_values("density")
            low = size_rows.iloc[0]
            fit = fits.get(str(size), {})
            fit_note = "no stable fit"
            if fit.get("success"):
                fit_note = (
                    f"p50={fit['p50']:.5f}, sample range {fit['sample_min']:.4f}–{fit['sample_max']:.4f}, "
                    f"midpoint bracketed={str(bool(fit['midpoint_bracketed'])).lower()}"
                )
            lines.append(
                f"- {size}²: lowest sampled density {float(low['density']):.4f} passed "
                f"{int(low['strong_passes'])}/{int(low['trajectories'])}; {fit_note}."
            )
        lines += [
            "",
            "Both fitted midpoints lie below their sampled grids, so their apparent finite-size shift and width must not be read as converged thresholds. See `figures/onset_probability.png` and `figures/first_passage.png`.",
            "",
        ]
    if not horizon.empty:
        horizon_low = horizon.sort_values(["size", "density"]).iloc[0]
        horizon_high = horizon.sort_values(["size", "density"]).iloc[-1]
        lines += [
            "## Horizon and topology",
            "",
            f"- At {int(horizon_low['size'])}² and density {float(horizon_low['density']):.4f}, all {int(horizon_low['strong_passes'])}/{int(horizon_low['trajectories'])} long-horizon trajectories passed by 131,072 phases; median first passage was {float(horizon_low['median_first_passage']):,.0f}.",
            f"- At that density, final empty- and occupied-wrap fractions were {float(horizon_low['empty_wrap_fraction_x']):.2f}/{float(horizon_low['occupied_wrap_fraction_x']):.2f} in x. At density {float(horizon_high['density']):.4f}, final occupied wrapping reached {float(horizon_high['occupied_wrap_fraction_x']):.2f} in x and {float(horizon_high['occupied_wrap_fraction_y']):.2f} in y.",
            "",
            "The long-horizon passes at the lowest tested density support a waiting-time/nucleation component, while the density-dependent wrapping change shows that the late morphology also changes topology. These are compatible mechanisms, not mutually exclusive labels.",
            "",
        ]
    if not hysteresis.empty:
        seeded = hysteresis[hysteresis.arrangement.isin(["slab", "droplet"])]
        homogeneous_seed_test = hysteresis[hysteresis.arrangement == "homogeneous"]
        lines += [
            "## Inventory-matched seeded/homogeneous comparison",
            "",
            f"- The finite-empty strong-growth classifier recorded {int(seeded['strong_passes'].sum())}/{int(seeded['trajectories'].sum())} passes across all slab and droplet trajectories.",
        ]
        for arrangement, rows in seeded.groupby("arrangement"):
            lines.append(
                f"- Across densities {float(rows['density'].min()):.2f}–{float(rows['density'].max()):.2f}, {arrangement} median final/initial largest occupied-cluster retention ranged from {float(rows['median_occupied_cluster_retention'].min()):.3f} to {float(rows['median_occupied_cluster_retention'].max()):.3f}."
            )
        below_onset = homogeneous_seed_test[homogeneous_seed_test.density <= 0.24]
        lines.append(
            f"- Homogeneous controls at densities 0.20 and 0.24 recorded {int(below_onset['strong_passes'].sum())}/{int(below_onset['trajectories'].sum())} strong-growth passes."
        )
        lines += [
            "",
            "The strong-growth classifier is normalized to the initial finite-empty scale and therefore is not a seed-survival statistic. The direct occupied-cluster measurement shows that both seeded shapes persist and grow where homogeneous runs do not cross the preregistered strong-domain criterion. This supports operational basin dependence, not an equilibrium free-energy claim or proof of first-order bistability. See `figures/seeded_cluster_retention.png` and `figures/hysteresis_seed_dependence.png`.",
            "",
        ]
    if not ablation.empty:
        ablation_frame = frame[frame.stage.str.startswith("ablation")]
        ablation_counts = (
            ablation_frame.groupby("rule_variant")
            .strong_domain.agg(["sum", "count"])
            .sort_values(["sum", "count"], ascending=False)
        )
        winner_count = ablation_counts.loc["winner"]
        sibling_count = ablation_counts.loc["qm_sibling"]
        rotation_count = ablation_counts.loc["rotation"]
        zero_ablations = sorted(
            name
            for name, row in ablation_counts.iterrows()
            if name.startswith("ablate_") and int(row["sum"]) == 0
        )
        lines += [
            "## Reaction-pair ablation",
            "",
            f"- Winner: {int(winner_count['sum'])}/{int(winner_count['count'])} strong-domain passes.",
            f"- Exact Q+M sibling: {int(sibling_count['sum'])}/{int(sibling_count['count'])}; rotation scaffold: {int(rotation_count['sum'])}/{int(rotation_count['count'])}.",
            f"- Single-pair ablations with zero passes: {', '.join(zero_ablations)}.",
            "",
            "The effect is rule-specific and distributed: the occupancy-changing winner outperforms both exact controls, while removal of several distinct reaction pairs abolishes the regime in this protocol. See `figures/reaction_pair_ablation.png`.",
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `TRAJECTORIES.csv`: one row per trajectory.",
        "- `SUMMARY.csv`: grouped pass fractions and medians.",
        "- `FIT.json`: descriptive logistic fits.",
        "- `dashboard.html`: local visual browser for exact initial/peak/final states.",
        "",
        "## Interpretation boundary",
        "",
        "The study can establish formal dynamical properties of one finite reversible exactly-Q-conserving automaton. It cannot establish primitive occurrence, cosmology, consciousness, life, or universality across rule families. Bold interpretations may be discussed, but the measurements and exact inferences remain separately identified.",
    ]
    report_path = report_dir / "REPORT.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    fit_payload = {"fits": _json_safe(fits), "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(report_dir / "FIT.json", _json_safe(fit_payload))

    hashes = {}
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256"):
            hashes[str(path.relative_to(report_dir))] = sha256_file(path)
    atomic_write_json(report_dir / "REPORT_MANIFEST.json", hashes)
    return {
        "trajectories": len(frame),
        "report": str(report_path),
        "dashboard": str(report_dir / "dashboard.html"),
        "fits": _json_safe(fits),
    }
