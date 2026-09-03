from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .config import StudyConfig


@dataclass(frozen=True)
class Task:
    stage: str
    size: int
    density: float
    seed: int
    steps: int
    sample_every: int
    topology_every: int
    checkpoint_every: int
    arrangement: str
    rule_variant: str
    early_fit_start: int
    early_fit_end: int
    shells: tuple[tuple[float, float], ...]

    @property
    def id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = self.id
        return payload


def generate_tasks(config: StudyConfig) -> list[Task]:
    tasks: list[Task] = []
    for stage_name, stage in config.stages.items():
        if not stage.enabled:
            continue
        variants = stage.rule_variants
        if stage_name.startswith("ablation") and "single_ablations" in variants:
            variants = tuple(v for v in variants if v != "single_ablations") + tuple(
                f"ablate_{i:02d}" for i in range(12)
            )
        for size in stage.sizes:
            if size % 2:
                raise ValueError(f"stage {stage_name}: size must be even")
            for density in stage.densities:
                for seed in stage.seeds:
                    for arrangement in stage.arrangements:
                        for variant in variants:
                            tasks.append(
                                Task(
                                    stage=stage_name,
                                    size=size,
                                    density=density,
                                    seed=seed,
                                    steps=stage.steps,
                                    sample_every=stage.sample_every,
                                    topology_every=stage.topology_every,
                                    checkpoint_every=stage.checkpoint_every,
                                    arrangement=arrangement,
                                    rule_variant=variant,
                                    early_fit_start=stage.early_fit_start,
                                    early_fit_end=stage.early_fit_end,
                                    shells=stage.shells,
                                )
                            )
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("task plan contains duplicate IDs")
    return tasks
