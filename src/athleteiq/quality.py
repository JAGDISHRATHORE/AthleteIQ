"""Sensor-quality checks for a single workout's heart-rate / GPS series.

Consumer wearables produce dropouts, spikes and GPS jumps. Every session gets
these measurements plus a 0-1 quality score and a label, so analyses and the
dashboard can show (or exclude) sessions whose numbers should not be trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def _longest_constant_run_s(hr: np.ndarray, dt: np.ndarray) -> float:
    """Longest stretch (seconds) over which HR did not change at all."""
    if len(hr) < 2:
        return 0.0
    same = np.isclose(np.diff(hr), 0.0)
    best = cur = 0.0
    for s, d in zip(same, dt):
        cur = cur + d if s else 0.0
        best = max(best, cur)
    return float(best)


def check_workout(hr: np.ndarray, speed: np.ndarray, ts: np.ndarray,
                  cfg: dict | None = None) -> dict:
    q = cfg or load_config()["quality"]
    hr = np.asarray(hr, float)
    speed = np.asarray(speed, float)
    dt = np.diff(np.asarray(ts, float))
    n = max(len(hr), 1)

    out_of_range = float(np.mean((hr < q["hr_min_valid"]) | (hr > q["hr_max_valid"])))
    hr_spikes = int(np.sum(np.abs(np.diff(hr)) > q["hr_spike_bpm"]))
    speed_bad = float(np.mean(speed > q["speed_max_valid_kmh"]))
    gps_jumps = int(np.sum(np.abs(np.diff(speed)) > q["speed_jump_kmh"]))
    gaps = int(np.sum(dt > q["gap_s"]))
    max_gap = float(dt.max()) if len(dt) else 0.0
    flat = _longest_constant_run_s(hr, dt)

    # Penalties are deliberately simple and documented in docs/methodology.md.
    penalty = (
        2.0 * out_of_range
        + 1.0 * speed_bad
        + 0.5 * min(hr_spikes / n * 10, 1)
        + 0.5 * min(gps_jumps / n * 10, 1)
        + 0.3 * min(gaps / 5, 1)
        + 0.3 * (flat >= q["flatline_s"])
    )
    score = float(np.clip(1.0 - penalty, 0.0, 1.0))
    label = "good" if score >= 0.85 else ("check" if score >= 0.6 else "poor")
    return {
        "q_hr_out_of_range_frac": out_of_range,
        "q_hr_spikes": hr_spikes,
        "q_speed_implausible_frac": speed_bad,
        "q_gps_jumps": gps_jumps,
        "q_gaps": gaps,
        "q_max_gap_s": max_gap,
        "q_hr_flatline_s": flat,
        "quality_score": score,
        "quality_label": label,
    }


def quality_table(raw: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()["quality"]
    rows = [
        {"workout_id": r.workout_id, **check_workout(r.heart_rate, r.speed, r.timestamp, cfg)}
        for r in raw.itertuples(index=False)
    ]
    return pd.DataFrame(rows)
