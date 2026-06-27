"""YAML config loader with portable ``${REPO}`` / ``${ROOT}`` path substitution.

  REPO  = repository root (this file's grandparent).
  ROOT  = scratch base for data/cache/features. Taken from the ``SHAPEDEM_ROOT``
          environment variable, else ``<REPO>/_workspace`` (git-ignored).

So the repo runs anywhere with no hard-coded paths; point ``SHAPEDEM_ROOT`` at a
large scratch disk if your home directory is small.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT = _REPO / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access and dotted .get()."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        # Wrap nested dicts so cfg.paths.data_dir works recursively
        return Config(v) if isinstance(v, dict) else v

    def dotget(self, path: str, default: Any = None):
        """Walk a dotted key like "extract.n_points" without raising on missing intermediates."""
        cur: Any = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _subst(obj, mapping):
    """Recursively replace ${REPO}/${ROOT} placeholders so YAML stays portable."""
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace(k, v)
        return obj
    if isinstance(obj, dict):
        return {k: _subst(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, mapping) for v in obj]
    # Non-string leaves (int, float, bool, None) pass through unchanged
    return obj


def repo_root() -> Path:
    return _REPO


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else _DEFAULT
    with open(p) as f:
        data = yaml.safe_load(f)
    # Resolve ROOT from env so large data can live on a different disk
    root = os.environ.get("SHAPEDEM_ROOT", str(_REPO / "_workspace"))
    data = _subst(data, {"${REPO}": str(_REPO), "${ROOT}": root})
    cfg = Config(data)
    # Pre-create output directories so downstream code never needs to check
    for key in ("data_dir", "work_dir", "features_dir", "cache_dir", "results_dir"):
        d = cfg["paths"].get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    return cfg
