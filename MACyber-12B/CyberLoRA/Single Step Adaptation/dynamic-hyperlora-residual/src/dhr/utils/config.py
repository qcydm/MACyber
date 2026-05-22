from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_recursive(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Cyclic config inheritance detected: {config_path}")
    seen.add(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    base_ref = raw.pop("base_config", None)
    if base_ref is None:
        return raw

    base_path = (config_path.parent / str(base_ref)).resolve()
    base_cfg = _load_recursive(base_path, seen)
    return _deep_merge(base_cfg, raw)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load config with optional recursive `base_config` inheritance."""
    return _load_recursive(Path(config_path), seen=set())

