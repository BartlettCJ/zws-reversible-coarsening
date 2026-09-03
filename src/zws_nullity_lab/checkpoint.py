from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from .atomic import atomic_torch_save, atomic_write_json, atomic_write_text, sha256_file


CHECKPOINT_SCHEMA = "zws-nullity-search-checkpoint-v2"


def checkpoint_directory(run_directory: str | Path) -> Path:
    directory = Path(run_directory).resolve() / "checkpoints" / "resumable"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _checkpoint_name(payload: dict[str, Any]) -> str:
    sequence = int(payload.get("checkpoint_sequence", 0))
    generation = int(payload.get("generation", 0))
    next_candidate = int(payload.get("next_candidate", 0))
    stage = str(payload.get("stage", "evaluating")).replace("/", "-")
    return f"checkpoint_{sequence:08d}_g{generation:04d}_c{next_candidate:04d}_{stage}.pt"


def save_checkpoint(
    run_directory: str | Path,
    payload: dict[str, Any],
    *,
    keep: int = 4,
) -> Path:
    directory = checkpoint_directory(run_directory)
    value = dict(payload)
    value["schema"] = CHECKPOINT_SCHEMA
    value["saved_unix"] = time.time()
    path = directory / _checkpoint_name(value)
    atomic_torch_save(path, value)
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    atomic_write_json(
        directory / "LATEST.json",
        {
            "schema": "zws-nullity-latest-checkpoint-v1",
            "path": path.name,
            "sha256": digest,
            "generation": int(value.get("generation", 0)),
            "next_candidate": int(value.get("next_candidate", 0)),
            "stage": value.get("stage"),
            "checkpoint_sequence": int(value.get("checkpoint_sequence", 0)),
        },
    )
    _prune(directory, keep=max(2, int(keep)))
    return path


def _prune(directory: Path, keep: int) -> None:
    candidates = sorted(
        directory.glob("checkpoint_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    for path in candidates[keep:]:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)


def _verified(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return bool(expected) and sha256_file(path) == expected


def available_checkpoints(run_directory: str | Path) -> list[Path]:
    directory = checkpoint_directory(run_directory)
    candidates = sorted(
        directory.glob("checkpoint_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    return [path for path in candidates if _verified(path)]


def load_checkpoint(
    run_directory: str | Path,
    *,
    expected_config_sha256: str | None = None,
) -> tuple[dict[str, Any], Path]:
    candidates = available_checkpoints(run_directory)
    if not candidates:
        raise FileNotFoundError(f"no valid resumable checkpoint found under {run_directory}")
    errors: list[str] = []
    for path in candidates:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            errors.append(f"{path.name}: ValueError: unsupported checkpoint schema")
            continue
        # A configuration mismatch is not corruption and must not silently fall
        # back to an older checkpoint from the same run.  Refuse the resume
        # before any resolved configuration or scientific output is changed.
        if expected_config_sha256 is not None and payload.get("config_sha256") != expected_config_sha256:
            raise ValueError(
                "checkpoint configuration hash does not match the resolved configuration"
            )
        return payload, path
    raise RuntimeError("all verified checkpoints failed to load: " + " | ".join(errors))


def checkpoint_summary(run_directory: str | Path) -> dict[str, Any]:
    paths = available_checkpoints(run_directory)
    if not paths:
        return {"available": False, "count": 0}
    payload, selected = load_checkpoint(run_directory)
    return {
        "available": True,
        "count": len(paths),
        "selected": str(selected),
        "schema": payload.get("schema"),
        "generation": payload.get("generation"),
        "next_candidate": payload.get("next_candidate"),
        "stage": payload.get("stage"),
        "checkpoint_sequence": payload.get("checkpoint_sequence"),
        "history_records": len(payload.get("history", [])),
    }
