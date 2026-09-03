from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .atomic import sha256_file
from .history_erasure import HistoryConfig, state_histogram, state_sha256
from .history_runner import _save_exact_state
from .rule import ReversibleRule
from .runner import resolve_device
from .simulator import step_batched


RULE_SHA256 = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"


def verify_sidecar(path: str | Path) -> None:
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if not target.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing artifact or sidecar: {target}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    actual = sha256_file(target).lower()
    if actual != expected:
        raise RuntimeError(f"artifact checksum mismatch: {target}")


def load_verified_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    verify_sidecar(target)
    return json.loads(target.read_text(encoding="utf-8"))


def load_exact_npz(path: str | Path) -> tuple[np.ndarray, int, int | None]:
    target = Path(path)
    verify_sidecar(target)
    with np.load(target, allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.uint8)
        phase = int(np.asarray(payload["phase"]).reshape(-1)[0])
        raw_elapsed = int(np.asarray(payload["elapsed"]).reshape(-1)[0])
    if state.ndim != 2 or np.any(state > 3):
        raise RuntimeError(f"invalid exact state: {target}")
    elapsed = None if raw_elapsed < 0 else raw_elapsed
    return np.ascontiguousarray(state), phase, elapsed


def exact_state_metadata(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    state, phase, elapsed = load_exact_npz(target)
    histogram = state_histogram(state).tolist()
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "state_sha256": state_sha256(state),
        "phase": phase,
        "elapsed": elapsed,
        "shape": list(state.shape),
        "histogram": histogram,
        "Q": int(histogram[1] - histogram[2]),
        "M": int(sum(histogram[1:])),
    }


def _has_exact_state_schema(path: Path) -> bool:
    verify_sidecar(path)
    with np.load(path, allow_pickle=False) as payload:
        return {"state", "phase", "elapsed"}.issubset(payload.files)


def index_task_states(task_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(task_dir)
    state_dir = directory / "states"
    if not state_dir.is_dir():
        return []
    records = [
        exact_state_metadata(path)
        for path in sorted(state_dir.glob("*.npz"))
        if _has_exact_state_schema(path)
    ]
    keys = [(record["phase"], record["elapsed"], record["state_sha256"]) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate exact-state records under {state_dir}")
    return sorted(
        records,
        key=lambda record: (
            -1 if record["elapsed"] is None else int(record["elapsed"]),
            int(record["phase"]),
            record["path"],
        ),
    )


def choose_exact_record(
    records: Iterable[dict[str, Any]],
    *,
    elapsed: int,
    prefer_at_or_before: bool = True,
) -> dict[str, Any]:
    rows = [row for row in records if row.get("elapsed") is not None]
    if not rows:
        raise RuntimeError("no elapsed exact states available")
    exact = [row for row in rows if int(row["elapsed"]) == int(elapsed)]
    if exact:
        return min(exact, key=lambda row: row["path"])
    if prefer_at_or_before:
        before = [row for row in rows if int(row["elapsed"]) <= int(elapsed)]
        if before:
            return max(before, key=lambda row: (int(row["elapsed"]), row["path"]))
    return min(rows, key=lambda row: (abs(int(row["elapsed"]) - int(elapsed)), row["path"]))


def _source_metric_hash(source: dict[str, Any], phase: int) -> str:
    row = next((row for row in source["series"] if int(row["phase"]) == int(phase)), None)
    if row is None:
        raise RuntimeError(f"source lacks observation hash at phase {phase}")
    return str(row["state_sha256"])


def reconstruct_untouched_media_states(
    config_path: str | Path,
    run_path: str | Path,
    source_task_ids: Iterable[str],
    *,
    branch_point: str = "B",
    elapsed_targets: Iterable[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct exact untouched suffix states and verify every state against source hashes.

    This is a deterministic media replay from an already saved branch state. It does not
    add a scientific replicate or alter any outcome. Every retained replay phase must match
    the corresponding progenitor observation hash byte-for-byte.
    """

    if branch_point not in {"A", "B"}:
        raise ValueError("branch_point must be A or B")
    config = HistoryConfig.load(config_path)
    run = Path(run_path).resolve()
    rule = ReversibleRule.load(config.rule_path)
    rule.assert_valid()
    if rule.sha256() != RULE_SHA256 or rule.sha256() != config.rule_sha256_uint8:
        raise RuntimeError("frozen winner hash mismatch before media replay")
    device = resolve_device(config.device)
    targets = sorted(
        set(
            int(value)
            for value in (
                elapsed_targets if elapsed_targets is not None else config.fixed_elapsed_phases
            )
            if int(value) >= 0
        )
    )
    if not targets or targets[0] != 0:
        targets.insert(0, 0)

    source_ids = [str(value) for value in source_task_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_task_ids contains duplicates")
    if not source_ids:
        return {}
    contexts: list[dict[str, Any]] = []
    for source_task_id in source_ids:
        source_path = run / "sources" / source_task_id / "source_result.json"
        source = load_verified_json(source_path)
        branch = source["branch_points"].get(branch_point)
        if not branch or not branch.get("eligible"):
            raise RuntimeError(
                f"representative source lacks eligible Branch {branch_point}: {source_task_id}"
            )
        branch_phase = int(branch["phase"])
        branch_state_path = Path(branch["path"])
        state, saved_phase, _ = load_exact_npz(branch_state_path)
        if saved_phase != branch_phase or state_sha256(state) != branch["state_sha256"]:
            raise RuntimeError(f"Branch-{branch_point} source capture mismatch: {source_task_id}")
        allowed = [value for value in targets if branch_phase + value <= config.source_horizon]
        media_dir = (
            run
            / "media_states"
            / "untouched"
            / f"branch_{branch_point}"
            / source_task_id
            / "states"
        )
        media_dir.mkdir(parents=True, exist_ok=True)
        contexts.append(
            {
                "source_task_id": source_task_id,
                "source": source,
                "branch_phase": branch_phase,
                "state": state,
                "allowed": set(allowed),
                "media_dir": media_dir,
            }
        )
    shapes = {tuple(context["state"].shape) for context in contexts}
    if len(shapes) != 1:
        raise RuntimeError("media replay sources do not share one lattice shape")
    tensor = torch.stack(
        [torch.from_numpy(context["state"]) for context in contexts]
    ).to(device=device)
    table = (
        rule.table.to(device=device)
        .unsqueeze(0)
        .expand(len(contexts), -1)
        .contiguous()
    )
    output: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in source_ids}
    elapsed = 0
    union_targets = sorted(set().union(*(context["allowed"] for context in contexts)))
    for target in union_targets:
        while elapsed < target:
            evolved = step_batched(tensor, table, elapsed)
            if not isinstance(evolved, torch.Tensor):
                raise RuntimeError("unexpected simulator return during media replay")
            tensor = evolved
            elapsed += 1
        for index, context in enumerate(contexts):
            if target not in context["allowed"]:
                continue
            source_task_id = context["source_task_id"]
            source = context["source"]
            branch_phase = int(context["branch_phase"])
            media_dir = context["media_dir"]
            absolute = branch_phase + elapsed
            array = tensor[index].detach().cpu().numpy().astype(np.uint8, copy=False)
            expected = _source_metric_hash(source, absolute)
            actual = state_sha256(array)
            if actual != expected:
                raise RuntimeError(
                    f"untouched media replay mismatch for {source_task_id} at {absolute}: "
                    f"{actual} != {expected}"
                )
            path = media_dir / f"fixed_elapsed_{elapsed:06d}_phase_{absolute:06d}.npz"
            if path.exists():
                existing, phase_existing, elapsed_existing = load_exact_npz(path)
                if (
                    phase_existing != absolute
                    or elapsed_existing != elapsed
                    or state_sha256(existing) != actual
                ):
                    raise RuntimeError(f"existing media replay state is inconsistent: {path}")
            else:
                _save_exact_state(path, array, phase=absolute, elapsed=elapsed)
            output[source_task_id].append(exact_state_metadata(path))
    del tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output
