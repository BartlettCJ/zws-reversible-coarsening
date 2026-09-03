from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .atomic import atomic_write_text


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "zws-nullity",
        "output_root": "results",
        "seed": 260824,
        "device": "auto",
        "deterministic": True,
    },
    "search": {
        "population": 96,
        "generations": 60,
        "elites": 12,
        "random_injection": 8,
        "mutation_swaps": [1, 2, 4, 8, 16],
        "novelty_weight": 0.08,
        "novelty_k": 8,
        "replicates": 2,
        "save_top": 8,
        "checkpoint_every": 10,
    },
    "evaluation": {
        "height": 96,
        "width": 96,
        "steps": 160,
        "burn_in": 40,
        "sample_every": 4,
        "positive_fraction": 0.12,
        "bound_fraction": 0.08,
        "localized": False,
    },
    "analysis": {
        "steps": 512,
        "holdout_replicates": 8,
        "cycle_height": 4,
        "cycle_width": 4,
        "cycle_max_ticks": 200000,
        "window_points": 8,
        "coarse_blocks": [2, 4],
        "gif_frames": 160,
        "gif_stride": 2,
    },
    "resilience": {
        "candidate_batch": 16,
        "checkpoint_every_chunks": 1,
        "checkpoint_keep": 4,
        "archive_improvements": True,
        "archive_novelty_champions": True,
        "archive_metric_champions": True,
        "max_archived_rules": 64,
        "manual_pause_mode": "sleep",
    },
    "resources": {
        "enabled": False,
        "mode": "sleep",
        "gpu_index": 0,
        "poll_seconds": 5.0,
        "sample_interval_seconds": 0.75,
        "busy_confirm_samples": 2,
        "idle_confirm_samples": 3,
        "hibernate_after_seconds": 900.0,
        "cpu_busy_percent": 78.0,
        "cpu_resume_percent": 55.0,
        "min_available_ram_gib": 8.0,
        "yield_for_external_gpu_processes": True,
        "external_gpu_memory_mib": 1536.0,
        "gpu_driver_allowance_mib": 768.0,
        "yield_for_gpu_utilization": True,
        "gpu_busy_utilization_percent": 40.0,
        "gpu_resume_utilization_percent": 20.0,
        "gpu_busy_temperature_c": 83.0,
        "gpu_resume_temperature_c": 76.0,
        "yield_when_user_active": False,
        "required_user_idle_seconds": 90.0,
        "release_cuda_cache": True,
        "below_normal_priority": True,
        "leave_logical_cores": 4,
    },
    "visualization": {
        "save_explorer": True,
        "auto_capture_events": 20,
        "event_min_separation": 12,
        "event_scan_stride": 1,
        "capture_padding": 10,
        "contact_sheet_columns": 4,
        "trajectory_binary": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        source = Path(path)
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a mapping")
        config = _deep_merge(config, data)
        config["_source"] = str(source.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    search = config["search"]
    evaluation = config["evaluation"]
    analysis = config["analysis"]
    resilience = config["resilience"]
    resources = config["resources"]
    visualization = config["visualization"]
    population = int(search["population"])
    elites = int(search["elites"])
    injection = int(search["random_injection"])
    if population < 4:
        raise ValueError("search.population must be at least 4")
    if not 1 <= elites < population:
        raise ValueError("search.elites must be in [1, population)")
    if injection < 0 or elites + injection > population:
        raise ValueError("invalid random_injection")
    if int(search["replicates"]) < 1:
        raise ValueError("search.replicates must be >= 1")
    if int(evaluation["height"]) % 2 or int(evaluation["width"]) % 2:
        raise ValueError("evaluation dimensions must be even")
    if int(evaluation["burn_in"]) >= int(evaluation["steps"]):
        raise ValueError("evaluation.burn_in must be less than evaluation.steps")
    if int(analysis["cycle_height"]) % 2 or int(analysis["cycle_width"]) % 2:
        raise ValueError("cycle dimensions must be even")
    candidate_batch = int(resilience["candidate_batch"])
    if candidate_batch < 1:
        raise ValueError("resilience.candidate_batch must be >= 1")
    if int(resilience["checkpoint_every_chunks"]) < 1:
        raise ValueError("resilience.checkpoint_every_chunks must be >= 1")
    if int(resilience["checkpoint_keep"]) < 2:
        raise ValueError("resilience.checkpoint_keep must be >= 2")
    if str(resilience.get("manual_pause_mode", "sleep")).lower() not in {"sleep", "exit"}:
        raise ValueError("resilience.manual_pause_mode must be sleep or exit")
    if str(resources.get("mode", "sleep")).lower() not in {"sleep", "hibernate"}:
        raise ValueError("resources.mode must be sleep or hibernate")
    if float(resources["cpu_resume_percent"]) > float(resources["cpu_busy_percent"]):
        raise ValueError("resources.cpu_resume_percent must not exceed cpu_busy_percent")
    if float(resources["gpu_resume_utilization_percent"]) > float(
        resources["gpu_busy_utilization_percent"]
    ):
        raise ValueError(
            "resources.gpu_resume_utilization_percent must not exceed busy threshold"
        )
    if int(visualization["auto_capture_events"]) < 0:
        raise ValueError("visualization.auto_capture_events must be >= 0")
    if int(visualization["event_min_separation"]) < 1:
        raise ValueError("visualization.event_min_separation must be >= 1")


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    return atomic_write_text(destination, yaml.safe_dump(clean, sort_keys=False))
