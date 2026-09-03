from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(directory: Path) -> None:
    """Best-effort directory sync after an atomic rename.

    Windows does not expose a portable directory fsync through Python. The file
    itself is still flushed and replaced atomically; POSIX additionally syncs
    the containing directory where the platform permits it.
    """
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
def _replace_with_retry(source: Path, destination: Path) -> None:
    """Atomic replace with a short Windows-friendly sharing/AV retry window."""
    attempts = 8
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.8, 0.025 * (2**attempt)))


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, destination)
        _sync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        # Windows requires a write-capable descriptor for fsync. Opening the
        # completed checkpoint read/write preserves the bytes while allowing
        # the durability flush on both Windows and POSIX.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, destination)
        _sync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
