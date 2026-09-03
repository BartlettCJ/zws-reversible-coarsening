from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json, atomic_write_text


def _yes(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(result: dict[str, Any]) -> str:
    rule = result["rule"]
    local = result["local_rule_audit"]
    trace = result["trace_audit"]
    cycle = result["cycle"]
    controls = result["controls"]
    redescription = result["redescription"]
    coarse = result["coarse_graining"]
    windows = result["finite_window_measure"]
    interpretation = result["interpretation"]
    events = result.get("events", {"candidates": [], "note": "No event scan recorded."})

    lines = [
        "# ZWSLab — Experiment Report",
        "",
        f"**Rule:** `{rule['name']}`  ",
        f"**Rule file:** `{result['paths']['rule']}`  ",
        f"**Analysis device:** `{result['environment']['device']}`  ",
        "",
        "## Executive result",
        "",
        interpretation["executive_verdict"],
        "",
        "| Test | Result | Meaning |",
        "|---|---:|---|",
        f"| Exhaustive 256-block invariant | {_yes(local['valid'])} | Every local map is bijective and preserves exact integer charge. |",
        f"| Long-trace exact charge | {_yes(trace['charge_preserved'])} | No global correction or floating tolerance was used. |",
        f"| Exact reverse reconstruction | {_yes(trace['state_equal_after_reverse'])} | The finite dynamics has a unique predecessor under the inverse rule. |",
        f"| Harmless-redescription conjugacy | {_yes(redescription['passed'])} | State relabellings and block-period translations preserve the full trace exactly. |",
        f"| Additive coarse-grain composition | {_yes(coarse['passed_additive_composition'])} | The selected charge map composes exactly across tested levels. |",
        f"| Rootless finite cycle witnessed | {_yes(cycle['found'])} | A found cycle supplies a finite no-privileged-first-state witness, not a beginningless ontology. |",
        "",
        "## Local rule audit",
        "",
        f"- Reaction fraction: **{_fmt(local['reaction_fraction'])}** of local inputs change their state multiset.",
        f"- Local permutation cycles: **{local['permutation_cycle_count']}**, longest **{local['permutation_cycle_max']}**.",
        f"- Maximum exact charge error: **{local['maximum_charge_error']}**.",
        f"- Charge sectors crossed: **{local['charge_sector_crossings']}**.",
        f"- Sign-reversal physical asymmetry: **{_fmt(local['sign_asymmetry_fraction'])}** of block inputs.",
        f"- Neutral-label physical asymmetry: **{_fmt(local['neutral_label_asymmetry_fraction'])}** of block inputs.",
        (
            f"- Search disclosure: **{rule.get('metadata', {}).get('unique_rules_evaluated', 'n/a')}** "
            f"unique rules evaluated from a local family of approximately "
            f"**10^{_fmt(rule.get('metadata', {}).get('theoretical_rule_space_log10', float('nan')), 2)}** rules."
        ),
        "",
        "## Selected rule versus controls",
        "",
        "The optimization score below is a declared search heuristic, not a proposed ontological process measure.",
        "",
        "| Rule | Valid Q invariant | Quality mean±SD | Activity | Reaction | Spatial MI | Temporal MI | Perturbation distance | Patch repetition proxy | Selected beats row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(
            "| {name} | {valid} | {quality} | {activity} | {reaction} | {spatial_mi} | "
            "{temporal_mi} | {perturbation_distance} | {patch_repetition} | {win_rate} |".format(
                name=row["name"],
                valid=_yes(row["charge_preserved"]),
                quality=f"{_fmt(row['quality'])}±{_fmt(row['quality_std'])}",
                activity=_fmt(row["activity"]),
                reaction=_fmt(row["reaction"]),
                spatial_mi=_fmt(row["spatial_mi"]),
                temporal_mi=_fmt(row["temporal_mi"]),
                perturbation_distance=_fmt(row["perturbation_distance"]),
                patch_repetition=_fmt(row["patch_repetition"]),
                win_rate=(
                    "n/a"
                    if row["selected_quality_win_rate"] is None
                    else _fmt(row["selected_quality_win_rate"])
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Redescription audit",
            "",
            "| Transformation | Exact trace conjugacy | Max structural-measure Δ | Compression-ratio Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for case in redescription["cases"]:
        max_error = max(
            case["max_state_entropy_error"],
            case["max_block_entropy_error"],
            case["max_spatial_mi_error"],
            case["active_density_error"],
            case["conditional_entropy_error"],
        )
        lines.append(
            f"| {case['name']} | {_yes(case['exact_trace_equivariance'])} | "
            f"{_fmt(max_error, 8)} | {_fmt(case['compression_ratio_error'], 8)} |"
        )

    lines.extend(
        [
            "",
            "Structural candidate measures should be invariant under these exact conjugacies. "
            "Compression is retained as a deliberately coding-sensitive foil; a nonzero delta is "
            "evidence against treating its raw value as an objective process weight.",
            "",
            "## Cross-level coarse-graining",
            "",
            "| Block | Exact total Q | Exact composition | Conditional entropy | Majority next-symbol accuracy | Symbols observed |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level in coarse["levels"]:
        composition = "n/a" if level["direct_composition_exact"] is None else _yes(level["direct_composition_exact"])
        lines.append(
            f"| {level['block']} | {_yes(level['total_charge_exact'])} | {composition} | "
            f"{_fmt(level['conditional_entropy'])} | {_fmt(level['majority_prediction_accuracy'])} | "
            f"{level['macro_symbols_observed']} |"
        )
    lines.extend(["", coarse["interpretation"], "", "## Finite-window measure stress", "", windows["verdict"], ""])
    lines.extend(
        [
            "| Exhaustion | Candidate | Last value | Tail relative range | Stabilized under 5% rule |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in windows["convergence"]:
        lines.append(
            f"| {row['sequence']} | {row['candidate']} | {_fmt(row['last_value'])} | "
            f"{_fmt(row['tail_relative_range'])} | {_yes(row['stabilized'])} |"
        )

    lines.extend(
        [
            "",
            "## Exact cycle search",
            "",
            f"- Found: **{cycle['found']}**",
            f"- Transient: **{cycle['transient']}**",
            f"- Period: **{cycle['period']}**",
            f"- Ticks examined: **{cycle['ticks_examined']}**",
            f"- Note: {cycle['note']}",
            "",
            "## Automated visual-review candidates",
            "",
            events.get("note", "These are bookmarks for inspection, not discoveries."),
            "",
            "| Event | Tick | Review score | Reasons |",
            "|---:|---:|---:|---|",
        ]
    )
    for event in events.get("candidates", []):
        lines.append(
            f"| {event['event_id']} | {event['tick']} | {_fmt(event['score'])} | "
            f"{'; '.join(event['reasons'])} |"
        )
    lines.extend(["", "## What this bears on", ""])
    for item in interpretation["supported_if_positive"]:
        lines.append(f"- {item}")
    lines.extend(["", "## What this does **not** test", ""])
    for item in interpretation["not_tested"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Threats and stopping rule",
            "",
        ]
    )
    for item in interpretation["threats"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Animation: `{result['paths']['animation']}`",
            f"- Final frame: `{result['paths']['final_frame']}`",
            f"- Control chart: `{result['paths']['control_chart']}`",
            f"- Smooth zoom/pan explorer: `{result['paths'].get('explorer_html')}`",
            f"- Auto-event contact sheet: `{result['paths'].get('event_contact_sheet')}`",
            f"- Exact trajectory binary: `{result['paths'].get('trajectory_binary')}`",
            f"- Eshkol exact oracle: `{result['paths']['eshkol_oracle']}`",
            f"- Machine-readable result: `{result['paths']['json']}`",
            "",
            "---",
            "",
            "This report distinguishes mathematical compatibility, computational behaviour, and metaphysical interpretation. A cellular automaton run is an occurring physical computation on the test machine, but its output does not decide Execution–Trace Non-Equivalence or establish that the simulated rule is the base of reality.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str, title: str = "ZWSLab Report") -> str:
    """Small dependency-free report renderer; the Markdown remains canonical."""
    blocks: list[str] = []
    in_list = False
    in_table = False
    table_rows: list[list[str]] = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            blocks.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        rows = [row for row in table_rows if not all(set(cell) <= {"-", ":"} for cell in row)]
        if rows:
            blocks.append("<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[1:]:
                blocks.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
            blocks.append("</tbody></table>")
        in_table = False
        table_rows = []

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            flush_list()
            in_table = True
            table_rows.append([cell.strip().replace("**", "").replace("`", "") for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("- "):
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{escape(line[2:])}</li>")
        else:
            flush_list()
            if line.startswith("# "):
                blocks.append(f"<h1>{escape(line[2:])}</h1>")
            elif line.startswith("## "):
                blocks.append(f"<h2>{escape(line[3:])}</h2>")
            elif line == "---":
                blocks.append("<hr>")
            elif line:
                blocks.append(f"<p>{escape(line)}</p>")
    flush_table()
    flush_list()
    body = "\n".join(blocks)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
body{{max-width:1100px;margin:2rem auto;padding:0 1.2rem;font:16px/1.55 system-ui,sans-serif;color:#17202a}}
h1,h2{{color:#243b53}}table{{border-collapse:collapse;width:100%;display:block;overflow:auto;margin:1rem 0}}
th,td{{border:1px solid #cbd5e1;padding:.45rem .6rem;text-align:left}}th{{background:#eef2f7}}code{{background:#f1f5f9;padding:.1rem .25rem}}li{{margin:.3rem 0}}
</style></head><body>{body}</body></html>'''


def write_reports(result: dict[str, Any], directory: str | Path) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(result)
    md_path = root / "REPORT.md"
    html_path = root / "REPORT.html"
    atomic_write_text(md_path, markdown)
    atomic_write_text(html_path, markdown_to_html(markdown))
    return md_path, html_path


def write_json(result: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, result)
    return destination
