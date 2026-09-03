from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import yaml


@dataclass(frozen=True)
class StageConfig:
    enabled: bool = True
    sizes: tuple[int, ...] = (128,)
    densities: tuple[float, ...] = (0.26, 0.28, 0.30)
    seeds: tuple[int, ...] = (1, 2)
    steps: int = 512
    sample_every: int = 64
    topology_every: int = 128
    checkpoint_every: int = 512
    batch_size: int = 2
    arrangements: tuple[str, ...] = ("homogeneous",)
    rule_variants: tuple[str, ...] = ("winner",)
    early_fit_start: int = 64
    early_fit_end: int = 256
    shells: tuple[tuple[float, float], ...] = ((1, 2), (2, 4), (4, 8), (8, 16))

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "StageConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            sizes=tuple(int(x) for x in payload.get("sizes", [128])),
            densities=tuple(float(x) for x in payload.get("densities", [0.26, 0.28, 0.30])),
            seeds=tuple(int(x) for x in payload.get("seeds", [1, 2])),
            steps=int(payload.get("steps", 512)),
            sample_every=int(payload.get("sample_every", 64)),
            topology_every=int(payload.get("topology_every", 128)),
            checkpoint_every=int(payload.get("checkpoint_every", max(int(payload.get("topology_every", 128)), 512))),
            batch_size=int(payload.get("batch_size", 2)),
            arrangements=tuple(str(x) for x in payload.get("arrangements", ["homogeneous"])),
            rule_variants=tuple(str(x) for x in payload.get("rule_variants", ["winner"])),
            early_fit_start=int(payload.get("early_fit_start", 64)),
            early_fit_end=int(payload.get("early_fit_end", 256)),
            shells=tuple(tuple(float(v) for v in pair) for pair in payload.get("shells", [[1, 2], [2, 4], [4, 8], [8, 16]])),
        )


@dataclass(frozen=True)
class ResourceConfig:
    enabled: bool = True
    max_cpu_percent: float = 85.0
    min_available_ram_gib: float = 8.0
    max_gpu_temperature_c: float = 84.0
    max_external_gpu_memory_mib: float = 1536.0
    sleep_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ResourceConfig":
        return cls(**{k: payload.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class StudyConfig:
    schema: str
    name: str
    rule_path: str
    device: str = "auto"
    output_root: str = "results/reversible_null_phase"
    low_k_max: float = 8.0
    criterion: dict[str, float] = field(default_factory=dict)
    nucleation: dict[str, Any] = field(default_factory=dict)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    stages: dict[str, StageConfig] = field(default_factory=dict)
    source_path: str | None = None
    # The rule path exactly as written in the YAML file (scheme-2 hashes use it).
    rule_path_declared: str | None = None

    # Configuration-hash scheme written by this package version.
    #   1 (package <= 0.3.1): the hashed payload carried the rule file's RESOLVED
    #     ABSOLUTE path, so a hash could only be reproduced at the original checkout
    #     location.  Plans recorded by the v0.3.0 and v0.3.1 campaigns carry such
    #     values; verify them with ``payload_sha256`` on the payload embedded in the
    #     plan, or with ``legacy_sha256`` at the original location.
    #   2 (package >= 0.4.0): the payload carries the rule path as DECLARED in the
    #     YAML file (POSIX separators), so the hash depends only on file contents.
    HASH_SCHEME: ClassVar[int] = 2
    LEGACY_HASH_SCHEME: ClassVar[int] = 1

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        schema = str(payload.get("schema", ""))
        if schema != "zws-reversible-null-phase-study-v1":
            raise ValueError(f"unsupported configuration schema: {schema}")
        stages_payload = payload.get("stages", {})
        stages = {
            str(name): StageConfig.from_mapping(dict(value))
            for name, value in stages_payload.items()
        }
        rule_path = Path(str(payload["rule_path"]))
        if not rule_path.is_absolute():
            rule_path = (source.parent / rule_path).resolve()
        config = cls(
            schema=schema,
            name=str(payload.get("name", source.stem)),
            rule_path=str(rule_path),
            device=str(payload.get("device", "auto")),
            output_root=str(payload.get("output_root", "results/reversible_null_phase")),
            low_k_max=float(payload.get("low_k_max", 8.0)),
            criterion={str(k): float(v) for k, v in dict(payload.get("criterion", {})).items()},
            nucleation=dict(payload.get("nucleation", {})),
            resource=ResourceConfig.from_mapping(dict(payload.get("resource", {}))),
            stages=stages,
            source_path=str(source),
            rule_path_declared=str(payload["rule_path"]),
        )
        config.validate_nucleation()
        return config

    def validate_nucleation(self) -> None:
        if not self.nucleation:
            return
        required = {
            "observation_every",
            "persistence_followups",
            "fixed_readout_phases",
            "event_window_before",
            "event_window_after",
            "event_crop_fraction",
        }
        missing = sorted(required - self.nucleation.keys())
        if missing:
            raise ValueError(f"nucleation protocol is missing fields: {missing}")
        observation = int(self.nucleation["observation_every"])
        if observation <= 0:
            raise ValueError("nucleation observation_every must be positive")
        if int(self.nucleation["persistence_followups"]) < 0:
            raise ValueError("nucleation persistence_followups must be nonnegative")
        for setting_name in ("event_window_before", "event_window_after"):
            value = int(self.nucleation[setting_name])
            if value < 0 or value % observation:
                raise ValueError(
                    f"nucleation {setting_name} must be a nonnegative observation multiple"
                )
        fraction = float(self.nucleation["event_crop_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("nucleation event_crop_fraction must lie in (0,1]")
        for name, stage in self.stages.items():
            if (
                stage.topology_every != observation
                or observation % stage.sample_every
            ):
                raise ValueError(
                    f"nucleation stage {name} must measure topology every {observation} "
                    "phases and temporal metrics on a divisor of that interval"
                )
            if stage.steps % observation:
                raise ValueError(f"nucleation stage {name} horizon must be an observation multiple")
        for setting_name in ("primary_stage", "area_stage", "seeded_stage"):
            if (
                setting_name in self.nucleation
                and str(self.nucleation[setting_name]) not in self.stages
            ):
                raise ValueError(
                    f"nucleation {setting_name} does not name a configured stage"
                )

    def declared_rule_path(self) -> str:
        """The rule path as written in the configuration file, with POSIX separators.

        Falls back to the rule file's name for configurations built in code.  A
        relative declared path makes the scheme-2 hash independent of where the
        checkout lives; an absolute declared path is location-bound by choice.
        """
        declared = self.rule_path_declared or Path(self.rule_path).name
        return declared.replace("\\", "/")

    def canonical_payload(self, scheme: int | None = None) -> dict[str, Any]:
        scheme = self.HASH_SCHEME if scheme is None else int(scheme)
        if scheme not in (self.LEGACY_HASH_SCHEME, self.HASH_SCHEME):
            raise ValueError(f"unknown configuration hash scheme: {scheme}")
        legacy = scheme == self.LEGACY_HASH_SCHEME
        payload = {
            "schema": self.schema,
            "name": self.name,
            # Scheme 1 hashed the resolved absolute path (location-dependent);
            # scheme 2 hashes the path as declared in the file.
            "rule_path": self.rule_path if legacy else self.declared_rule_path(),
            "device": self.device,
            "output_root": self.output_root,
            "low_k_max": self.low_k_max,
            "criterion": self.criterion,
            "resource": self.resource.__dict__,
            "stages": {name: stage.__dict__ for name, stage in self.stages.items()},
        }
        if not legacy:
            payload["hash_scheme"] = scheme
        # Keep every v0.3 configuration hash byte-for-byte stable.  The additive
        # protocol block participates in v0.3.1 hashes only when it is present.
        if self.nucleation:
            payload["nucleation"] = self.nucleation
        return payload

    @staticmethod
    def payload_sha256(payload: dict[str, Any]) -> str:
        """Hash a canonical payload, including one read back from a recorded plan."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def sha256(self, scheme: int | None = None) -> str:
        return self.payload_sha256(self.canonical_payload(scheme))

    def legacy_sha256(self) -> str:
        """Scheme-1 hash (package <= 0.3.1); matches recorded values only at the
        original checkout location, or after substituting the recorded rule path."""
        return self.sha256(self.LEGACY_HASH_SCHEME)
