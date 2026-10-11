"""Load config.yaml and resolve project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def path(rel: str) -> Path:
    """Resolve a config-relative path against the repository root."""
    return ROOT / rel


def data_dir() -> Path:
    """Processed data if the full pipeline has run, otherwise the bundled sample.
    Override with the ATHLETEIQ_DATA_DIR environment variable."""
    if os.getenv("ATHLETEIQ_DATA_DIR"):
        return Path(os.environ["ATHLETEIQ_DATA_DIR"])
    cfg = load_config()["data"]
    processed = path(cfg["processed_dir"])
    if (processed / "sessions.parquet").exists():
        return processed
    return path(cfg["sample_dir"])
