from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import torch

from .atomic import atomic_torch_save, atomic_write_json
from .checkpoint import checkpoint_summary, load_checkpoint, save_checkpoint
from .config import save_config
from .control import (
    RunControl,
    SignalLatch,
    WorkloadHibernate,
    WorkloadPaused,
    WorkloadStop,
)
from .initial import SeedSpec, balanced_random, perturb_charge_preserving
from .metrics import PopulationMetrics, evaluate_population, novelty_scores
from .resources import ResourceGovernor, set_background_priority
from .rules import ReversibleRule, stack_rules
from .visualize import plot_search_history, save_state_png


_METRIC_NAMES = (
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


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def config_hash(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def charge_preserving_rule_space_log10() -> float:
    """log10 of the complete local rule family size: product_q |sector_q|!."""
    from .alphabet import DEFAULT_ALPHABET

    return sum(
        math.lgamma(int(indices.numel()) + 1) / math.log(10.0)
        for indices in DEFAULT_ALPHABET.charge_sectors.values()
    )


def _rule_digest(rule: ReversibleRule) -> str:
    return hashlib.sha256(rule.table.numpy().tobytes()).hexdigest()


def environment_manifest(device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    return {
        "schema": "zws-nullity-run-manifest-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": gpu,
        "config_sha256": config_hash(config),
        "seed": int(config["run"]["seed"]),
        "deterministic_requested": bool(config["run"].get("deterministic", True)),
    }


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    best_total: float
    best_quality: float
    mean_quality: float
    best_novelty: float
    best_rule: str
    activity: float
    reaction: float
    state_entropy: float
    block_entropy: float
    spatial_mi: float
    temporal_mi: float
    recurrence: float
    perturbation_distance: float
    patch_repetition: float
    simple_baseline_distance: float
    sign_asymmetry: float


@dataclass(frozen=True)
class SearchResult:
    run_directory: str
    device: str
    best_rule_path: str
    top_rule_paths: list[str]
    generations: list[GenerationRecord]
    unique_rules_evaluated: int
    rule_replicate_evaluations: int
    theoretical_rule_space_log10: float
    searched_fraction_log10_upper_bound: float
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_directory": self.run_directory,
            "device": self.device,
            "best_rule_path": self.best_rule_path,
            "top_rule_paths": self.top_rule_paths,
            "generations": [asdict(item) for item in self.generations],
            "unique_rules_evaluated": self.unique_rules_evaluated,
            "rule_replicate_evaluations": self.rule_replicate_evaluations,
            "theoretical_rule_space_log10": self.theoretical_rule_space_log10,
            "searched_fraction_log10_upper_bound": self.searched_fraction_log10_upper_bound,
            "resumed": self.resumed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, resumed: bool = False) -> "SearchResult":
        return cls(
            run_directory=str(value["run_directory"]),
            device=str(value["device"]),
            best_rule_path=str(value["best_rule_path"]),
            top_rule_paths=[str(item) for item in value["top_rule_paths"]],
            generations=[GenerationRecord(**item) for item in value["generations"]],
            unique_rules_evaluated=int(value["unique_rules_evaluated"]),
            rule_replicate_evaluations=int(value["rule_replicate_evaluations"]),
            theoretical_rule_space_log10=float(value["theoretical_rule_space_log10"]),
            searched_fraction_log10_upper_bound=float(
                value["searched_fraction_log10_upper_bound"]
            ),
            resumed=resumed,
        )


def _aggregate(metrics: PopulationMetrics, population: int, replicates: int) -> dict[str, torch.Tensor]:
    result = {
        name: getattr(metrics, name).reshape(population, replicates).mean(dim=1)
        for name in _METRIC_NAMES
    }
    result["descriptor"] = metrics.descriptor.reshape(population, replicates, -1).mean(dim=1)
    result["charge_preserved"] = metrics.charge_preserved.reshape(population, replicates).all(dim=1)
    return result


def _write_history(path: Path, records: list[GenerationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _make_initial_batch(
    population: int,
    replicates: int,
    height: int,
    width: int,
    spec: SeedSpec,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = balanced_random(replicates, height, width, spec=spec, seed=seed, device="cpu")
    perturbed = perturb_charge_preserving(base, seed=seed + 7919)
    base = (
        base.unsqueeze(0)
        .expand(population, replicates, height, width)
        .reshape(population * replicates, height, width)
        .contiguous()
    )
    perturbed = (
        perturbed.unsqueeze(0)
        .expand(population, replicates, height, width)
        .reshape(population * replicates, height, width)
        .contiguous()
    )
    return base.to(device), perturbed.to(device)


def _next_population(
    rules: list[ReversibleRule],
    ranking: torch.Tensor,
    config: dict[str, Any],
    generator: torch.Generator,
    generation: int,
) -> list[ReversibleRule]:
    search = config["search"]
    population = int(search["population"])
    elite_count = int(search["elites"])
    injection = int(search["random_injection"])
    mutations = [int(value) for value in search["mutation_swaps"]]
    elite_indices = ranking[:elite_count].detach().cpu().tolist()
    elites = [rules[int(index)] for index in elite_indices]
    next_rules: list[ReversibleRule] = [
        ReversibleRule(
            elite.table.clone(),
            name=f"g{generation + 1:04d}_elite_{rank:03d}",
            metadata={**elite.metadata, "elite_from": elite.name},
        )
        for rank, elite in enumerate(elites)
    ]
    offspring_count = population - elite_count - injection
    weights = torch.linspace(elite_count, 1, elite_count, dtype=torch.float32)
    for child_index in range(offspring_count):
        parent_slot = int(torch.multinomial(weights, 1, generator=generator))
        mutation_index = int(torch.randint(len(mutations), (1,), generator=generator))
        swaps = mutations[mutation_index]
        next_rules.append(
            elites[parent_slot].mutated(
                swaps,
                generator=generator,
                name=f"g{generation + 1:04d}_child_{child_index:03d}",
            )
        )
    for random_index in range(injection):
        next_rules.append(
            ReversibleRule.random(
                generator,
                name=f"g{generation + 1:04d}_inject_{random_index:03d}",
            )
        )
    if len(next_rules) != population:
        raise AssertionError("population construction error")
    return next_rules


def _initial_rules(population: int, generator: torch.Generator) -> list[ReversibleRule]:
    seeds: list[ReversibleRule] = [
        ReversibleRule.identity("g0000_identity"),
        ReversibleRule.spatial_rotation(1, "g0000_rotate90"),
        ReversibleRule.spatial_rotation(2, "g0000_rotate180"),
        ReversibleRule.spatial_rotation(3, "g0000_rotate270"),
        ReversibleRule.reactive_rotation(4, name="g0000_reactive04"),
        ReversibleRule.reactive_rotation(12, name="g0000_reactive12"),
        ReversibleRule.reactive_rotation(24, name="g0000_reactive24"),
        ReversibleRule.reactive_rotation(48, name="g0000_reactive48"),
    ]
    rules = seeds[:population]
    mutation_choices = [1, 2, 4, 8, 16, 32, 64]
    while len(rules) < population:
        index = len(rules)
        if index % 4 == 0:
            rules.append(ReversibleRule.random(generator, name=f"g0000_random_{index:03d}"))
        else:
            parent = seeds[index % len(seeds)]
            swaps = mutation_choices[index % len(mutation_choices)]
            rules.append(parent.mutated(swaps, generator, name=f"g0000_seeded_{index:03d}"))
    return rules


def _new_partial(population: int, height: int, width: int) -> dict[str, torch.Tensor]:
    partial: dict[str, torch.Tensor] = {
        name: torch.full((population,), float("nan"), dtype=torch.float32)
        for name in _METRIC_NAMES
    }
    partial["descriptor"] = torch.full((population, 9), float("nan"), dtype=torch.float32)
    partial["charge_preserved"] = torch.zeros(population, dtype=torch.bool)
    partial["representative_final_state"] = torch.zeros(
        (population, height, width), dtype=torch.uint8
    )
    return partial


def _rules_to_dict(rules: list[ReversibleRule]) -> list[dict[str, Any]]:
    return [rule.to_dict() for rule in rules]


def _rules_from_dict(rows: list[dict[str, Any]]) -> list[ReversibleRule]:
    return [
        ReversibleRule(row["table"], name=row.get("name", "restored"), metadata=row.get("metadata"))
        for row in rows
    ]


def _state_payload(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": config_hash(config),
        "checkpoint_sequence": int(state["checkpoint_sequence"]),
        "stage": state["stage"],
        "generation": int(state["generation"]),
        "next_candidate": int(state["next_candidate"]),
        "rules": _rules_to_dict(state["rules"]),
        "history": [asdict(item) for item in state["history"]],
        "unique_rule_digests": sorted(state["unique_rule_digests"]),
        "generator_state": state["generator"].get_state(),
        "python_random_state": random.getstate(),
        "partial": state["partial"],
        "ranking": state.get("ranking"),
        "champions": state.get("champions", {}),
        "archive_count": int(state.get("archive_count", 0)),
        "device": str(state["device"]),
        "chunks_completed": int(state.get("chunks_completed", 0)),
    }


def _restore_state(
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.set_state(payload["generator_state"])
    random.setstate(payload["python_random_state"])
    return {
        "checkpoint_sequence": int(payload["checkpoint_sequence"]),
        "stage": str(payload["stage"]),
        "generation": int(payload["generation"]),
        "next_candidate": int(payload["next_candidate"]),
        "rules": _rules_from_dict(payload["rules"]),
        "history": [GenerationRecord(**item) for item in payload.get("history", [])],
        "unique_rule_digests": set(payload.get("unique_rule_digests", [])),
        "generator": generator,
        "partial": payload["partial"],
        "ranking": payload.get("ranking"),
        "champions": dict(payload.get("champions", {})),
        "archive_count": int(payload.get("archive_count", 0)),
        "device": device,
        "chunks_completed": int(payload.get("chunks_completed", 0)),
    }


def _save_state_checkpoint(root: Path, state: dict[str, Any], config: dict[str, Any]) -> Path:
    state["checkpoint_sequence"] = int(state["checkpoint_sequence"]) + 1
    return save_checkpoint(
        root,
        _state_payload(state, config),
        keep=int(config["resilience"]["checkpoint_keep"]),
    )


def _status(
    root: Path,
    state: dict[str, Any],
    *,
    status: str,
    message: str | None = None,
    resource: dict[str, Any] | None = None,
) -> None:
    population = len(state.get("rules", []))
    atomic_write_json(
        root / "status.json",
        {
            "schema": "zws-nullity-live-status-v2",
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "status": status,
            "message": message,
            "stage": state.get("stage"),
            "generation": int(state.get("generation", 0)),
            "next_candidate": int(state.get("next_candidate", 0)),
            "population": population,
            "generation_fraction": (
                float(state.get("next_candidate", 0)) / max(1, population)
            ),
            "history_records": len(state.get("history", [])),
            "checkpoint_sequence": int(state.get("checkpoint_sequence", 0)),
            "chunks_completed": int(state.get("chunks_completed", 0)),
            "resource": resource,
            "checkpoint": checkpoint_summary(root),
        },
    )


def _release_cuda(device: torch.device, enabled: bool = True) -> None:
    if device.type == "cuda" and enabled:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _wait_for_permission(
    root: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    control: RunControl,
    signal_latch: SignalLatch,
    governor: ResourceGovernor,
) -> None:
    resources = config["resources"]
    resilience = config["resilience"]
    while True:
        flags = control.read()
        if flags.stop:
            _status(root, state, status="stopped", message="STOP control file observed")
            raise WorkloadStop("STOP control file observed")
        if flags.hibernate:
            _status(root, state, status="hibernating", message="HIBERNATE control file observed")
            raise WorkloadHibernate("HIBERNATE control file observed")
        if signal_latch.requested.is_set():
            _status(root, state, status="paused", message="console signal observed")
            raise WorkloadPaused("console signal observed")
        if flags.pause:
            _release_cuda(state["device"], bool(resources.get("release_cuda_cache", True)))
            if str(resilience.get("manual_pause_mode", "sleep")).lower() == "exit":
                _status(root, state, status="paused", message="PAUSE control file observed")
                raise WorkloadPaused("PAUSE control file observed")
            _status(
                root,
                state,
                status="paused",
                message="PAUSE file present; sleeping with the last checkpoint safe",
            )
            time.sleep(max(0.5, float(resources.get("poll_seconds", 5.0))))
            continue
        if not governor.enabled:
            return
        _release_cuda(state["device"], bool(resources.get("release_cuda_cache", True)))
        decision = governor.observe()
        if decision.runnable:
            return
        _status(
            root,
            state,
            status="yielding",
            message="; ".join(decision.reasons),
            resource=decision.to_dict(),
        )
        if governor.should_hibernate():
            raise WorkloadHibernate(
                f"external resource pressure persisted for {governor.busy_seconds():.0f}s"
            )
        time.sleep(governor.poll_seconds)


def _archive_rule(
    root: Path,
    state: dict[str, Any],
    category: str,
    score: float,
    rule: ReversibleRule,
    generation: int,
    representative_state: torch.Tensor,
    details: dict[str, Any],
    config: dict[str, Any],
) -> None:
    resilience = config["resilience"]
    maximum = int(resilience.get("max_archived_rules", 64))
    if int(state.get("archive_count", 0)) >= maximum:
        return
    current = state["champions"].get(category)
    if current is not None and float(current["score"]) >= float(score):
        return
    digest = _rule_digest(rule)
    directory = root / "interesting" / category
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"g{generation:04d}_{score:.6f}_{digest[:12]}"
    rule_path = directory / f"{stem}.rule.json"
    state_path = directory / f"{stem}.state.pt"
    preview_path = directory / f"{stem}.png"
    metadata_path = directory / f"{stem}.json"
    atomic_write_json(rule_path, rule.to_dict())
    atomic_torch_save(state_path, representative_state.detach().cpu().to(torch.uint8))
    save_state_png(representative_state, preview_path, scale=4)
    atomic_write_json(
        metadata_path,
        {
            "schema": "zws-nullity-interesting-rule-v1",
            "category": category,
            "score": float(score),
            "generation": generation,
            "rule_sha256": digest,
            "rule_path": str(rule_path),
            "state_path": str(state_path),
            "preview_png": str(preview_path),
            "details": details,
            "note": "Automatically archived search champion; human review is required.",
        },
    )
    state["champions"][category] = {
        "score": float(score),
        "generation": generation,
        "digest": digest,
        "rule_path": str(rule_path),
        "metadata_path": str(metadata_path),
    }
    state["archive_count"] = int(state.get("archive_count", 0)) + 1


def _generation_ranking(
    state: dict[str, Any], config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    partial = state["partial"]
    if torch.isnan(partial["quality"]).any():
        raise RuntimeError("cannot rank an incomplete generation")
    descriptor = partial["descriptor"]
    novelty = novelty_scores(descriptor, k=int(config["search"]["novelty_k"]))
    novelty_scaled = torch.tanh(novelty / novelty.median().clamp_min(1e-6))
    rules = state["rules"]
    simplicity_distance = torch.tensor(
        [rule.simple_baseline_distance() for rule in rules], dtype=torch.float32
    )
    rule_structure_reward = torch.sigmoid((simplicity_distance - 0.01) / 0.02) * torch.sigmoid(
        (0.65 - simplicity_distance) / 0.08
    )
    sign_asymmetry = torch.tensor([rule.sign_asymmetry() for rule in rules], dtype=torch.float32)
    asymmetry_reward = torch.sigmoid((sign_asymmetry - 0.015) / 0.015) * torch.sigmoid(
        (0.94 - sign_asymmetry) / 0.05
    )
    total = (
        partial["quality"]
        + float(config["search"]["novelty_weight"]) * novelty_scaled
        + 0.06 * rule_structure_reward
        + 0.04 * asymmetry_reward
    )
    ranking = torch.argsort(total, descending=True)
    return ranking, total, novelty, torch.stack([simplicity_distance, sign_asymmetry], dim=1)


def _record_generation(
    root: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    ranking: torch.Tensor,
    total: torch.Tensor,
    novelty: torch.Tensor,
    structure: torch.Tensor,
) -> GenerationRecord:
    best = int(ranking[0])
    partial = state["partial"]
    record = GenerationRecord(
        generation=int(state["generation"]),
        best_total=float(total[best]),
        best_quality=float(partial["quality"][best]),
        mean_quality=float(partial["quality"].mean()),
        best_novelty=float(novelty[best]),
        best_rule=state["rules"][best].name,
        activity=float(partial["activity"][best]),
        reaction=float(partial["reaction"][best]),
        state_entropy=float(partial["state_entropy"][best]),
        block_entropy=float(partial["block_entropy"][best]),
        spatial_mi=float(partial["spatial_mi"][best]),
        temporal_mi=float(partial["temporal_mi"][best]),
        recurrence=float(partial["recurrence"][best]),
        perturbation_distance=float(partial["perturbation_distance"][best]),
        patch_repetition=float(partial["patch_repetition"][best]),
        simple_baseline_distance=float(structure[best, 0]),
        sign_asymmetry=float(structure[best, 1]),
    )
    state["history"].append(record)
    _write_history(root / "generation_history.csv", state["history"])
    generation = int(state["generation"])
    representative = partial["representative_final_state"]
    if bool(config["resilience"].get("archive_improvements", True)):
        _archive_rule(
            root,
            state,
            "best_total",
            float(total[best]),
            state["rules"][best],
            generation,
            representative[best],
            asdict(record),
            config,
        )
    if bool(config["resilience"].get("archive_novelty_champions", True)):
        index = int(torch.argmax(novelty))
        _archive_rule(
            root,
            state,
            "novelty",
            float(novelty[index]),
            state["rules"][index],
            generation,
            representative[index],
            {"novelty": float(novelty[index]), "total": float(total[index])},
            config,
        )
    if bool(config["resilience"].get("archive_metric_champions", True)):
        for category in ("reaction", "recurrence", "perturbation_distance", "spatial_mi"):
            index = int(torch.argmax(partial[category]))
            _archive_rule(
                root,
                state,
                category,
                float(partial[category][index]),
                state["rules"][index],
                generation,
                representative[index],
                {category: float(partial[category][index]), "total": float(total[index])},
                config,
            )
    return record


def _finalize(
    root: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    resumed: bool,
) -> SearchResult:
    ranking = torch.as_tensor(state["ranking"], dtype=torch.int64)
    rules_dir = root / "rules"
    rules_dir.mkdir(exist_ok=True)
    population = int(config["search"]["population"])
    generations_count = int(config["search"]["generations"])
    replicates = int(config["search"]["replicates"])
    top_count = min(int(config["search"]["save_top"]), population)
    top_paths: list[str] = []
    for rank, index in enumerate(ranking[:top_count].tolist()):
        rule = state["rules"][int(index)]
        rule.metadata.update(
            {
                "final_rank": rank,
                "search_config_sha256": config_hash(config),
                "search_run": str(root),
                "unique_rules_evaluated": len(state["unique_rule_digests"]),
                "rule_replicate_evaluations": population * generations_count * replicates,
                "theoretical_rule_space_log10": charge_preserving_rule_space_log10(),
            }
        )
        path = rules_dir / f"rank_{rank:03d}_{rule.name}.json"
        atomic_write_json(path, rule.to_dict())
        top_paths.append(str(path))
    result = SearchResult(
        run_directory=str(root),
        device=str(state["device"]),
        best_rule_path=top_paths[0],
        top_rule_paths=top_paths,
        generations=state["history"],
        unique_rules_evaluated=len(state["unique_rule_digests"]),
        rule_replicate_evaluations=population * generations_count * replicates,
        theoretical_rule_space_log10=charge_preserving_rule_space_log10(),
        searched_fraction_log10_upper_bound=(
            math.log10(max(1, len(state["unique_rule_digests"])))
            - charge_preserving_rule_space_log10()
        ),
        resumed=resumed,
    )
    atomic_write_json(root / "search_result.json", result.to_dict())
    if state["history"]:
        plot_search_history(
            [asdict(item) for item in state["history"]], root / "search_history.png"
        )
    state["stage"] = "complete"
    _save_state_checkpoint(root, state, config)
    _status(root, state, status="complete", message="search completed normally")
    return result


def run_search(
    config: dict[str, Any],
    run_directory: str | Path | None = None,
    *,
    resume: bool = False,
    stop_after_chunks: int | None = None,
) -> SearchResult:
    run = config["run"]
    search = config["search"]
    evaluation = config["evaluation"]
    resilience = config["resilience"]
    resources = config["resources"]
    seed = int(run["seed"])
    device = resolve_device(str(run.get("device", "auto")))
    if bool(run.get("deterministic", True)):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if run_directory is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_directory = Path(str(run["output_root"])) / f"{run['name']}_{stamp}"
    root = Path(run_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed_path = root / "search_result.json"
    if resume and completed_path.is_file():
        return SearchResult.from_dict(
            json.loads(completed_path.read_text(encoding="utf-8")), resumed=True
        )
    if not resume and (root / "config.resolved.yaml").exists():
        raise FileExistsError(
            f"run directory already contains a resolved run: {root}; use --resume"
        )
    (root / "rules").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    resolved_path = root / "config.resolved.yaml"
    if resume:
        if not resolved_path.is_file():
            raise FileNotFoundError(f"resolved configuration missing for resume: {resolved_path}")
    else:
        save_config(config, resolved_path)
        atomic_write_json(root / "manifest.json", environment_manifest(device, config))

    population = int(search["population"])
    generations_count = int(search["generations"])
    replicates = int(search["replicates"])
    height = int(evaluation["height"])
    width = int(evaluation["width"])
    candidate_batch = min(population, max(1, int(resilience["candidate_batch"])))
    spec = SeedSpec(
        positive_fraction=float(evaluation["positive_fraction"]),
        bound_fraction=float(evaluation["bound_fraction"]),
        localized=bool(evaluation.get("localized", False)),
    )

    if bool(resources.get("enabled", False)) and bool(
        resources.get("below_normal_priority", True)
    ):
        atomic_write_json(
            root / "process_policy.json",
            set_background_priority(
                leave_logical_cores=int(resources.get("leave_logical_cores", 0))
            ),
        )

    resumed = bool(resume)
    if resume:
        payload, loaded_path = load_checkpoint(
            root, expected_config_sha256=config_hash(config)
        )
        state = _restore_state(payload, config=config, device=device)
        atomic_write_json(root / "resume.json", {"checkpoint": str(loaded_path), "utc": datetime.now(timezone.utc).isoformat()})
    else:
        random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rules = _initial_rules(population, generator)
        state = {
            "checkpoint_sequence": 0,
            "stage": "evaluating",
            "generation": 0,
            "next_candidate": 0,
            "rules": rules,
            "history": [],
            "unique_rule_digests": {_rule_digest(rule) for rule in rules},
            "generator": generator,
            "partial": _new_partial(population, height, width),
            "ranking": None,
            "champions": {},
            "archive_count": 0,
            "device": device,
            "chunks_completed": 0,
        }
        _save_state_checkpoint(root, state, config)

    control = RunControl(root)
    governor = ResourceGovernor(resources, run_directory=root)
    checkpoint_every_chunks = max(1, int(resilience["checkpoint_every_chunks"]))
    chunks_this_invocation = 0

    with SignalLatch() as signal_latch:
        try:
            while int(state["generation"]) < generations_count:
                if state["stage"] == "finalizing":
                    return _finalize(root, state, config, resumed=resumed)
                if state["stage"] != "evaluating":
                    raise RuntimeError(f"unsupported restored search stage: {state['stage']}")
                generation = int(state["generation"])
                if int(state["next_candidate"]) == 0:
                    state["unique_rule_digests"].update(
                        _rule_digest(rule) for rule in state["rules"]
                    )
                while int(state["next_candidate"]) < population:
                    _wait_for_permission(
                        root, state, config, control, signal_latch, governor
                    )
                    start = int(state["next_candidate"])
                    end = min(population, start + candidate_batch)
                    count = end - start
                    _status(
                        root,
                        state,
                        status="running",
                        message=f"evaluating generation {generation}, candidates {start}:{end}",
                    )
                    initial, perturbed = _make_initial_batch(
                        count,
                        replicates,
                        height,
                        width,
                        spec,
                        seed + generation * 104729,
                        device,
                    )
                    tables = stack_rules(state["rules"][start:end], device=device).repeat_interleave(
                        replicates, dim=0
                    )
                    metrics = evaluate_population(
                        initial,
                        perturbed,
                        tables,
                        steps=int(evaluation["steps"]),
                        burn_in=int(evaluation["burn_in"]),
                        sample_every=int(evaluation["sample_every"]),
                    )
                    aggregate = _aggregate(metrics, count, replicates)
                    for name in _METRIC_NAMES:
                        state["partial"][name][start:end] = aggregate[name].detach().cpu()
                    state["partial"]["descriptor"][start:end] = aggregate[
                        "descriptor"
                    ].detach().cpu()
                    state["partial"]["charge_preserved"][start:end] = aggregate[
                        "charge_preserved"
                    ].detach().cpu()
                    representative = metrics.final_state.reshape(
                        count, replicates, height, width
                    )[:, 0]
                    state["partial"]["representative_final_state"][start:end] = (
                        representative.detach().cpu().to(torch.uint8)
                    )
                    state["next_candidate"] = end
                    state["chunks_completed"] = int(state["chunks_completed"]) + 1
                    chunks_this_invocation += 1
                    del initial, perturbed, tables, metrics, aggregate, representative
                    _release_cuda(device, bool(resources.get("release_cuda_cache", True)))
                    if int(state["chunks_completed"]) % checkpoint_every_chunks == 0:
                        _save_state_checkpoint(root, state, config)
                    _status(
                        root,
                        state,
                        status="running",
                        message=f"completed generation {generation} candidates 0:{end}",
                    )
                    if signal_latch.requested.is_set():
                        _save_state_checkpoint(root, state, config)
                        raise WorkloadPaused("console signal observed after a safe chunk")
                    if stop_after_chunks is not None and chunks_this_invocation >= stop_after_chunks:
                        _save_state_checkpoint(root, state, config)
                        raise WorkloadPaused("planned test pause after safe chunk")

                ranking, total, novelty, structure = _generation_ranking(state, config)
                record = _record_generation(
                    root, state, config, ranking, total, novelty, structure
                )
                state["ranking"] = ranking.tolist()
                generation = int(state["generation"])
                checkpoint_every = int(search["checkpoint_every"])
                if generation == 0 or (generation + 1) % checkpoint_every == 0:
                    best = int(ranking[0])
                    state["rules"][best].save(
                        root / "checkpoints" / f"generation_{generation:04d}_best.json"
                    )
                    atomic_torch_save(
                        root / "checkpoints" / f"generation_{generation:04d}_state.pt",
                        {
                            "generation": generation,
                            "rule_table": state["rules"][best].table,
                            "final_state": state["partial"]["representative_final_state"][best],
                            "record": asdict(record),
                        },
                    )
                if generation + 1 >= generations_count:
                    state["stage"] = "finalizing"
                    _save_state_checkpoint(root, state, config)
                    return _finalize(root, state, config, resumed=resumed)
                state["rules"] = _next_population(
                    state["rules"], ranking, config, state["generator"], generation
                )
                state["generation"] = generation + 1
                state["next_candidate"] = 0
                state["partial"] = _new_partial(population, height, width)
                state["ranking"] = None
                state["stage"] = "evaluating"
                _save_state_checkpoint(root, state, config)
        except WorkloadPaused as exc:
            _save_state_checkpoint(root, state, config)
            _status(root, state, status="paused", message=str(exc))
            raise
        except WorkloadHibernate as exc:
            _save_state_checkpoint(root, state, config)
            _status(root, state, status="hibernated", message=str(exc))
            raise
        except WorkloadStop as exc:
            _save_state_checkpoint(root, state, config)
            _status(root, state, status="stopped", message=str(exc))
            raise

    raise RuntimeError("search loop terminated without a result")
