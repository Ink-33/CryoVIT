"""Lightweight stage timers for CryoVIT feature extraction and inference I/O.

Enabled with ``CRYOVIT_PERF=1``. When disabled, ``probe_stage`` is a no-op.

Usage::

    with probe_stage("dino.forward"):
        ...

    probe_dump(path)  # append one JSON line
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCK = threading.Lock()
_STAGES: dict[str, dict[str, float]] = {}
_ENABLED: bool | None = None


def perf_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("CRYOVIT_PERF", "").strip() not in ("", "0", "false", "False")
    return _ENABLED


@contextmanager
def probe_stage(name: str) -> Iterator[None]:
    """Time a named stage into the thread-safe accumulator."""
    if not perf_enabled():
        yield
        return
    t0 = time.perf_counter()
    cuda_sync = _maybe_cuda_sync()
    try:
        yield
    finally:
        cuda_sync()
        dt = time.perf_counter() - t0
        with _LOCK:
            rec = _STAGES.setdefault(name, {"count": 0.0, "total_s": 0.0, "max_s": 0.0})
            rec["count"] += 1
            rec["total_s"] += dt
            rec["max_s"] = max(rec["max_s"], dt)


def _maybe_cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.synchronize
    except Exception:
        pass
    return lambda: None


def probe_reset() -> None:
    with _LOCK:
        _STAGES.clear()


def probe_snapshot() -> dict[str, dict[str, float]]:
    with _LOCK:
        return {k: dict(v) for k, v in _STAGES.items()}


def probe_peak_vram_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def probe_dump(path: str | Path, **extra) -> Path | None:
    """Write one JSONL row of accumulated stages. Returns the path if written."""
    if not perf_enabled():
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stages = probe_snapshot()
    row = {
        "ts": time.time(),
        "stages": stages,
        "peak_vram_bytes": probe_peak_vram_bytes(),
        **extra,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    probe_reset()
    return path
