"""Checkpoint cache for Colab-disconnect resilience.

Each notebook module ends with a save_checkpoint() call so the next module
can pick up via load_or_compute() even after a runtime restart.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

DEFAULT_ROOT = Path("/content/checkpoints")


def _resolve_root(root: Path | str | None) -> Path:
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_checkpoint(name: str, *, root: Path | str | None = None, **arrays: np.ndarray) -> Path:
    """Save named arrays to <root>/<name>.npz."""
    path = _resolve_root(root) / f"{name}.npz"
    np.savez(path, **arrays)
    return path


def load_or_compute(name: str, *, root: Path | str | None = None, fn) -> dict:
    """Load <root>/<name>.npz if it exists; else call fn(), save its result, and return it."""
    path = _resolve_root(root) / f"{name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k].copy() for k in data.files}
    result = fn()
    if not isinstance(result, dict):
        raise TypeError(f"fn() must return a dict of arrays; got {type(result).__name__}")
    save_checkpoint(name, root=root, **result)
    return result
