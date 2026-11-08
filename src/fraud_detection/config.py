"""Configuration loading and path helpers.

The YAML file in ``configs/`` is the single source of truth. ``load_config`` adds a
``root`` key (repo root) so every path in the config can be resolved relative to it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and attach the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["root"] = str(path.resolve().parents[1])
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` against the repo root (absolute paths pass through)."""
    return Path(cfg["root"]) / cfg["paths"][key]


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of the config (excluding the machine-specific ``root``)."""
    clean = {k: v for k, v in cfg.items() if k != "root"}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
