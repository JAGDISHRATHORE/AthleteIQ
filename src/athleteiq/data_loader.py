"""Download and load the public Endomondo (FitRec) running-workout dataset.

Source: the FitRec / Endomondo dataset (Ni, Muhlstein & McAuley, WWW 2019),
as re-packaged on Hugging Face at ``rricc22/endomondo-hr-prediction-v2``:
40,186 running workouts from 761 athletes, each resampled to 500 points of
heart rate, speed, altitude and timestamp. See docs/data_card.md.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, path

ROWS_API = "https://datasets-server.huggingface.co/rows"
SPLIT_SIZES = {"train": 28130, "validation": 6027, "test": 6029}
SERIES_COLS = ["heart_rate", "speed", "altitude", "timestamp"]


def _download_via_hub(dataset: str, dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        local = hf_hub_download(dataset, "dataset.parquet", repo_type="dataset")
    except Exception as exc:  # network policy, missing file, etc.
        print(f"hub download failed ({exc.__class__.__name__}); falling back to rows API")
        return False
    pd.read_parquet(local).to_parquet(dest)
    return True


def _fetch_page(dataset: str, split: str, offset: int, length: int = 100) -> pd.DataFrame:
    url = (f"{ROWS_API}?dataset={dataset}&config=default&split={split}"
           f"&offset={offset}&length={length}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            return pd.DataFrame([r["row"] for r in payload["rows"]])
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"could not fetch {split}@{offset}")


def _download_via_rows_api(dataset: str, dest: Path, workers: int = 8) -> None:
    jobs = [(s, o) for s, n in SPLIT_SIZES.items() for o in range(0, n, 100)]
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda j: _fetch_page(dataset, *j), jobs))
    pd.concat(frames, ignore_index=True).to_parquet(dest)


def download_raw(force: bool = False) -> Path:
    """Download the raw dataset to data/raw (about 330 MB)."""
    cfg = load_config()["data"]
    dest = path(cfg["raw_file"])
    if dest.exists() and not force:
        print(f"raw data already at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _download_via_hub(cfg["hf_dataset"], dest):
        _download_via_rows_api(cfg["hf_dataset"], dest)
    print(f"saved {dest}")
    return dest


def load_raw(file: str | Path | None = None) -> pd.DataFrame:
    """Load the raw workouts, one row per workout with array-valued series columns."""
    f = Path(file) if file else path(load_config()["data"]["raw_file"])
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - run `python scripts/download_data.py` first")
    df = pd.read_parquet(f)
    for c in SERIES_COLS:
        df[c] = df[c].apply(lambda v: np.asarray(v, dtype=float))
    return df.sort_values(["user_id", "workout_id"]).reset_index(drop=True)


def explode_series(raw: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (workout, sample) with seconds since workout start."""
    rows = []
    for r in raw.itertuples(index=False):
        ts = np.asarray(r.timestamp, dtype=float)
        rows.append(pd.DataFrame({
            "workout_id": r.workout_id,
            "user_id": r.user_id,
            "t_s": (ts - ts[0]).astype("float32"),
            "heart_rate": np.asarray(r.heart_rate, dtype="float32"),
            "speed_kmh": np.asarray(r.speed, dtype="float32"),
            "altitude_m": np.asarray(r.altitude, dtype="float32"),
        }))
    return pd.concat(rows, ignore_index=True)
