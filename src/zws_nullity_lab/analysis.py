from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from typing import Any

import torch

from .atomic import atomic_torch_save, atomic_write_json
from .coarsegrain import audit_coarse_graining
from .config import save_config
from .control import RunControl, WorkloadHibernate, WorkloadStop
from .cycle import find_cycle
from .eshkol_export import export_eshkol_oracle
from .events import save_event_captures, scan_interesting_events
from .explorer import export_explorer_bundle
from .initial import SeedSpec, balanced_random, paired_impulse, perturb_charge_preserving
from .margolus import audit_trace, run_trajectory
from .measures import stress_finite_windows
from .metrics import evaluate_population
from .redescription import audit_state_relabellings
from .report import write_json, write_reports
from .resources import ResourceGovernor
from .rules import ReversibleRule, stack_rules
from .search import config_hash, environment_manifest, resolve_device
from .visualize import plot_control_metrics, save_state_png, save_trajectory_gif


ANALYSIS_PROGRESS_SCHEMA = "zws-nullity-analysis-progress-v1"


def _rule_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(rule.table.numpy().tobytes()).hexdigest()


def local_rule_audit(rule: ReversibleRule) -> dict[str, Any]:
    charges = rule.alphabet.block_charge_cpu
    delta = charges[rule.table] - charges
    seen = torch.zeros(256, dtype=torch.bool)
    cycle_lengths: list[int] = []
    for start in range(256):
        if bool(seen[start]):
            continue
        current = start
        length = 0
        while not bool(seen[current]):
            seen[current] = True
            current = int(rule.table[current])
            length += 1
        cycle_lengths.append(length)
    return {
        "schema": "zws-local-rule-audit-v1",
        "valid": rule.is_valid(),
        "is_permutation": rule.is_permutation(),
        "preserves_charge": rule.preserves_charge(),
        "maximum_charge_error": int(delta.abs().max()),
        "charge_sector_crossings": int((delta != 0).sum()),
        "reaction_fraction": rule.reaction_fraction(),
        "sign_asymmetry_fraction": rule.sign_asymmetry(),
        "neutral_label_asymmetry_fraction": rule.neutral_label_asymmetry(),
        "reactive_inputs": int(rule.reaction_mask().sum()),
        "permutation_cycle_count": len(cycle_lengths),
        "permutation_cycle_min": min(cycle_lengths),
        "permutation_cycle_max": max(cycle_lengths),
        "permutation_cycle_mean": sum(cycle_lengths) / len(cycle_lengths),
        "permutation_cycle_lengths": sorted(cycle_lengths, reverse=True),
    }


def cross_device_audit(initial: torch.Tensor, rule: ReversibleRule, steps: int = 64) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "performed": False,
            "passed": None,
            "reason": "CUDA unavailable in the current environment; CPU reference still ran.",
        }
    cpu_trace = run_trajectory(initial.cpu(), rule, steps).cpu()
    gpu_trace = run_trajectory(initial.cuda(), rule, steps).cpu()
    mismatch = torch.nonzero(cpu_trace != gpu_trace, as_tuple=False)
    return {
        "performed": True,
        "passed": mismatch.numel() == 0,
        "mismatching_cells": int(mismatch.shape[0]),
        "first_mismatch": mismatch[0].tolist() if mismatch.numel() else None,
        "note": "All state operations are integer lookup/reshape/roll operations.",
    }


def _control_rows(
    selected: ReversibleRule,
    initial: torch.Tensor,
    perturbed: torch.Tensor,
    device: torch.device,
    steps: int,
    burn_in: int,
    sample_every: int,
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    controls = [
        selected,
        ReversibleRule.identity(),
        ReversibleRule.spatial_rotation(1),
        ReversibleRule.random(generator, name="random-conserving-control"),
        ReversibleRule.random_unconstrained(generator, name="invariant-breaking-control"),
    ]
    if initial.ndim != 3 or initial.shape != perturbed.shape:
        raise ValueError("holdout initial/perturbed states must be [R,H,W]")
    replicates, height, width = initial.shape
    batch_initial = (
        initial.unsqueeze(0)
        .expand(len(controls), replicates, height, width)
        .reshape(len(controls) * replicates, height, width)
        .to(device)
        .contiguous()
    )
    batch_perturbed = (
        perturbed.unsqueeze(0)
        .expand(len(controls), replicates, height, width)
        .reshape(len(controls) * replicates, height, width)
        .to(device)
        .contiguous()
    )
    tables = stack_rules(controls, device=device).repeat_interleave(replicates, dim=0)
    metrics = evaluate_population(
        batch_initial,
        batch_perturbed,
        tables,
        steps=steps,
        burn_in=burn_in,
        sample_every=sample_every,
    )
    metric_names = (
        "quality",
        "activity",
        "reaction",
        "state_entropy",
        "block_entropy",
        "spatial_mi",
        "temporal_mi",
        "recurrence",
        "perturbation_distance",
        "patch_repetition",
    )
    rows: list[dict[str, Any]] = []
    per_rule_quality: list[torch.Tensor] = []
    for index, control in enumerate(controls):
        selection = slice(index * replicates, (index + 1) * replicates)
        row: dict[str, Any] = {
            "name": control.name,
            "rule_valid": control.is_valid(),
            "charge_preserved": bool(metrics.charge_preserved[selection].all().cpu()),
            "holdout_replicates": replicates,
            "per_replicate": {},
        }
        for metric_name in metric_names:
            values = getattr(metrics, metric_name)[selection].detach().cpu().float()
            row[metric_name] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(unbiased=False))
            row["per_replicate"][metric_name] = [float(value) for value in values.tolist()]
        per_rule_quality.append(getattr(metrics, "quality")[selection].detach().cpu().float())
        rows.append(row)
    selected_quality = per_rule_quality[0]
    rows[0]["quality_difference_from_selected"] = 0.0
    rows[0]["selected_quality_win_rate"] = None
    for index in range(1, len(rows)):
        difference = selected_quality - per_rule_quality[index]
        rows[index]["quality_difference_from_selected"] = float(difference.mean())
        rows[index]["selected_quality_win_rate"] = float((difference > 0).float().mean())
    return rows


def _interpretation(
    local: dict[str, Any],
    trace: dict[str, Any],
    controls: list[dict[str, Any]],
    cycle: dict[str, Any],
    windows: dict[str, Any],
) -> dict[str, Any]:
    selected = controls[0]
    nonreactive = controls[1:3]
    nontrivial = selected["reaction"] > max(row["reaction"] for row in nonreactive) + 1e-6
    exact = bool(local["valid"] and trace["charge_preserved"] and trace["state_equal_after_reverse"])
    if exact and nontrivial:
        executive = (
            "The selected local rule is an exact reversible signed-nullity dynamics and exhibits "
            "state-multiset-changing interactions absent from the static and purely spatial controls. "
            "This is a positive formal witness for the Signed Life research conjecture in this rule "
            "family. It is not evidence that the universe uses the rule or that abstract mathematics "
            "occurs by itself."
        )
    elif exact:
        executive = (
            "The exact nullity and reversibility obligations passed, but the selected behaviour did "
            "not exceed the declared nonreactive controls on the interaction criterion. This is a "
            "valid dynamics but not yet a useful Signed Life witness."
        )
    else:
        executive = (
            "A load-bearing exactness obligation failed. Do not interpret higher-level behaviour "
            "until the implementation or rule is repaired."
        )
    return {
        "executive_verdict": executive,
        "supported_if_positive": [
            "Rich, changing finite dynamics can coexist with an exact zero global signed total when the invariant is enforced locally rather than repaired globally.",
            "Local reversible interactions can include pair conversion, splitting-like redistribution, merging-like binding, and annihilation-like conversion into neutral excitations while retaining Q=0.",
            "A finite toroidal reversible orbit has exact predecessors; when a cycle is witnessed it has no dynamically privileged first member.",
            "A proposed finite-window process statistic can be rejected when it depends materially on window order, shape, scale, or harmless redescription.",
            "The additive charge component of a cross-level map can be made compositionally transitive; causal-organisational closure remains an empirical separate obligation.",
        ],
        "not_tested": [
            "Execution–Trace Non-Equivalence (ETNE), primitive occurrence, or why this computation is occurring rather than merely specified.",
            "Whether any natural invariant of our physical universe is exactly zero.",
            "Strict potentialism as an ontology or recovery of continuum/quantum/relativistic physics.",
            "Conscious actuality, conscious character, the Ground of Experience Conjecture, or AI consciousness.",
            "Uniqueness of a process or observer measure.",
        ],
        "threats": [
            "The rule family is restricted to reversible 2x2 Margolus permutations over four states.",
            "Evolution optimizes declared proxies and can exploit them; all component metrics and controls must be inspected.",
            "Patch repetition is not causal reproduction, and mutual information is not autonomous agency.",
            "Automated event captures are review bookmarks, not evidence of organisms or replication.",
            "Finite cycles are guaranteed in principle by a finite bijection; only the period and internal organisation are discoveries.",
            "A negative search result constrains only the searched family, seeds, lattice sizes, metrics, and compute budget.",
            "A positive computational witness establishes formal possibility, not physical instantiation or metaphysical truth.",
            f"Finite-window verdict recorded by this run: {windows['verdict']}",
            f"Cycle witness found under cap: {cycle['found']}.",
        ],
    }


def _save_progress(path: Path, rule: ReversibleRule, config: dict[str, Any], results: dict[str, Any]) -> None:
    atomic_torch_save(
        path,
        {
            "schema": ANALYSIS_PROGRESS_SCHEMA,
            "rule_sha256": _rule_digest(rule),
            "config_sha256": config_hash(config),
            "results": results,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_progress(path: Path, rule: ReversibleRule, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != ANALYSIS_PROGRESS_SCHEMA:
        raise ValueError("unsupported analysis progress schema")
    if value.get("rule_sha256") != _rule_digest(rule):
        raise ValueError("analysis progress belongs to a different rule")
    if value.get("config_sha256") != config_hash(config):
        raise ValueError("analysis progress belongs to a different configuration")
    return dict(value.get("results", {}))


def _analysis_gate(
    root: Path,
    stage: str,
    control: RunControl,
    governor: ResourceGovernor,
    device: torch.device,
    resources: dict[str, Any],
) -> None:
    while True:
        flags = control.read()
        if flags.stop:
            raise WorkloadStop(f"STOP requested before analysis stage {stage}")
        if flags.hibernate:
            raise WorkloadHibernate(f"HIBERNATE requested before analysis stage {stage}")
        if flags.pause:
            if device.type == "cuda" and bool(resources.get("release_cuda_cache", True)):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            atomic_write_json(
                root / "analysis_status.json",
                {
                    "status": "paused",
                    "stage": stage,
                    "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(max(0.5, float(resources.get("poll_seconds", 5.0))))
            continue
        if not governor.enabled:
            break
        if device.type == "cuda" and bool(resources.get("release_cuda_cache", True)):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        decision = governor.observe()
        if decision.runnable:
            break
        atomic_write_json(
            root / "analysis_status.json",
            {
                "status": "yielding",
                "stage": stage,
                "reason": list(decision.reasons),
                "resource": decision.to_dict(),
                "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if governor.should_hibernate():
            raise WorkloadHibernate(f"resource pressure before analysis stage {stage}")
        time.sleep(governor.poll_seconds)
    atomic_write_json(
        root / "analysis_status.json",
        {
            "status": "running",
            "stage": stage,
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def analyze_rule(
    rule: ReversibleRule,
    config: dict[str, Any],
    output_directory: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if not rule.is_valid():
        raise ValueError("selected analysis rule must be bijective and exactly charge-preserving")
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = root / "analysis.json"
    if resume and completed.is_file():
        return __import__("json").loads(completed.read_text(encoding="utf-8"))
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    rule_path = rule.save(root / "selected_rule.json")
    save_config(config, root / "analysis_config.yaml")
    device = resolve_device(str(config["run"].get("device", "auto")))
    evaluation = config["evaluation"]
    analysis = config["analysis"]
    resources = config["resources"]
    visualization = config["visualization"]
    seed = int(config["run"]["seed"])
    spec = SeedSpec(
        positive_fraction=float(evaluation["positive_fraction"]),
        bound_fraction=float(evaluation["bound_fraction"]),
        localized=bool(evaluation.get("localized", False)),
    )
    holdout_replicates = int(analysis.get("holdout_replicates", 8))
    initial_cpu = balanced_random(
        holdout_replicates,
        int(evaluation["height"]),
        int(evaluation["width"]),
        spec=spec,
        seed=seed + 9001,
        device="cpu",
    )
    perturbed_cpu = perturb_charge_preserving(initial_cpu, seed=seed + 9002)
    progress_path = root / "analysis_progress.pt"
    progress = _load_progress(progress_path, rule, config) if resume else {}
    control = RunControl(root.parent if root.name == "analysis" else root)
    governor = ResourceGovernor(resources, run_directory=root)

    def stage(name: str) -> None:
        _save_progress(progress_path, rule, config, progress)
        _analysis_gate(root, name, control, governor, device, resources)

    stage("local_rule_audit")
    if "local_rule_audit" not in progress:
        progress["local_rule_audit"] = local_rule_audit(rule)

    stage("trace_and_device_audits")
    if "trace_audit" not in progress:
        progress["trace_audit"] = asdict(
            audit_trace(initial_cpu, rule, min(int(analysis["steps"]), 512))
        )
    if "cross_device_audit" not in progress:
        progress["cross_device_audit"] = cross_device_audit(
            initial_cpu[:1], rule, steps=min(64, int(analysis["steps"]))
        )

    stage("holdout_controls")
    if "controls" not in progress:
        progress["controls"] = _control_rows(
            rule,
            initial_cpu,
            perturbed_cpu,
            device,
            steps=min(int(evaluation["steps"]), 192),
            burn_in=min(
                int(evaluation["burn_in"]),
                max(1, min(int(evaluation["steps"]), 192) // 3),
            ),
            sample_every=int(evaluation["sample_every"]),
            seed=seed + 9003,
        )

    trajectory_path = assets / "trajectory.pt"
    stage("trajectory")
    if trajectory_path.is_file():
        trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    else:
        trajectory = run_trajectory(
            initial_cpu[:1].to(device), rule, int(analysis["steps"])
        ).cpu()
        atomic_torch_save(trajectory_path, trajectory.to(torch.uint8))

    stage("redescription")
    if "redescription" not in progress:
        progress["redescription"] = audit_state_relabellings(
            initial_cpu[:1], rule, steps=min(96, int(analysis["steps"]))
        ).to_dict()

    stage("coarse_graining")
    if "coarse_graining" not in progress:
        progress["coarse_graining"] = audit_coarse_graining(
            trajectory, tuple(int(x) for x in analysis.get("coarse_blocks", [2, 4]))
        ).to_dict()

    stage("finite_window_measure")
    if "finite_window_measure" not in progress:
        progress["finite_window_measure"] = stress_finite_windows(
            trajectory, points=int(analysis.get("window_points", 8))
        ).to_dict()

    stage("cycle")
    if "cycle" not in progress:
        cycle_seed = paired_impulse(
            int(analysis["cycle_height"]), int(analysis["cycle_width"]), separation=1
        )
        progress["cycle"] = find_cycle(
            cycle_seed,
            rule,
            max_ticks=int(analysis["cycle_max_ticks"]),
        ).to_dict()

    stage("event_scan_and_visualization")
    if "events" not in progress:
        events, series = scan_interesting_events(
            trajectory,
            rule,
            max_events=int(visualization.get("auto_capture_events", 20)),
            min_separation=int(visualization.get("event_min_separation", 12)),
            stride=int(visualization.get("event_scan_stride", 1)),
            padding=int(visualization.get("capture_padding", 10)),
        )
        event_paths = save_event_captures(
            trajectory,
            events,
            assets / "events",
            contact_sheet_columns=int(visualization.get("contact_sheet_columns", 4)),
        )
        progress["events"] = {
            "note": "Automated multi-metric review bookmarks; not organism or replication claims.",
            "candidates": [event.to_dict() for event in events],
            "paths": event_paths,
            "metric_series": series,
        }
    else:
        from .events import EventCandidate

        events = [
            EventCandidate(
                **{
                    **row,
                    "reasons": tuple(row["reasons"]),
                    "crop": tuple(row["crop"]),
                }
            )
            for row in progress["events"]["candidates"]
        ]
        series = progress["events"]["metric_series"]

    final_frame = save_state_png(trajectory[-1, 0], assets / "final_state.png")
    animation = save_trajectory_gif(
        trajectory,
        assets / "trajectory.gif",
        stride=int(analysis.get("gif_stride", 2)),
        max_frames=int(analysis.get("gif_frames", 160)),
    )
    metric_names = [row["name"] for row in progress["controls"]]
    metric_values = {
        key: [float(row[key]) for row in progress["controls"]]
        for key in (
            "activity",
            "reaction",
            "spatial_mi",
            "temporal_mi",
            "perturbation_distance",
        )
    }
    control_chart = plot_control_metrics(metric_names, metric_values, assets / "controls.png")
    explorer_paths = export_explorer_bundle(
        trajectory,
        root / "explorer",
        rule_name=rule.name,
        rule_sha256=_rule_digest(rule),
        events=events,
        metric_series=series,
        fps=20,
    )
    oracle_path = export_eshkol_oracle(rule, root / "eshkol_exact_oracle.esk")
    json_path = root / "analysis.json"

    result: dict[str, Any] = {
        "schema": "zws-nullity-analysis-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": rule.to_dict(),
        "environment": environment_manifest(device, config),
        "local_rule_audit": progress["local_rule_audit"],
        "trace_audit": progress["trace_audit"],
        "cross_device_audit": progress["cross_device_audit"],
        "holdout": {
            "replicates": holdout_replicates,
            "seed_base": seed + 9001,
            "search_seed_base": seed,
            "separated_from_search": True,
        },
        "controls": progress["controls"],
        "redescription": progress["redescription"],
        "coarse_graining": progress["coarse_graining"],
        "finite_window_measure": progress["finite_window_measure"],
        "cycle": progress["cycle"],
        "events": progress["events"],
        "interpretation": {},
        "paths": {
            "rule": str(rule_path),
            "animation": str(animation),
            "trajectory_tensor": str(trajectory_path),
            "final_frame": str(final_frame),
            "control_chart": str(control_chart),
            "event_contact_sheet": progress["events"]["paths"].get("contact_sheet"),
            "events_json": progress["events"]["paths"].get("events_json"),
            "explorer_html": explorer_paths["html"],
            "explorer_directory": explorer_paths["directory"],
            "trajectory_binary": explorer_paths["binary"],
            "eshkol_oracle": str(oracle_path),
            "json": str(json_path),
        },
    }
    result["interpretation"] = _interpretation(
        result["local_rule_audit"],
        result["trace_audit"],
        result["controls"],
        result["cycle"],
        result["finite_window_measure"],
    )
    write_json(result, json_path)
    md_path, html_path = write_reports(result, root)
    result["paths"]["report_markdown"] = str(md_path)
    result["paths"]["report_html"] = str(html_path)
    write_json(result, json_path)
    _save_progress(progress_path, rule, config, progress)
    atomic_write_json(
        root / "analysis_status.json",
        {
            "status": "complete",
            "stage": "complete",
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result
