"""Athlete-specific baselines and deviations.

Every baseline uses only sessions that happened *before* the one being scored
(``closed="left"`` windows / ``shift(1)``), so a session is always compared
against what was known at the time - never against itself or the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config

BASELINE_METRICS = ["hr_mean", "avg_speed_kmh", "efficiency_m_per_beat", "trimp", "distance_km"]


def add_baselines(sessions: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    a = load_config()["athlete"]
    window = f"{a['baseline_window_days']}D"
    minp = a["baseline_min_sessions"]
    metrics = metrics or BASELINE_METRICS
    floors = a.get("z_std_floor", {})
    s = sessions.sort_values(["user_id", "start_time"]).copy()
    parts = []
    for _, g in s.groupby("user_id", sort=False):
        g = g.copy()
        gi = g.set_index("start_time")
        for m in metrics:
            roll = gi[m].rolling(window, closed="left", min_periods=minp)
            mean, std = roll.mean().to_numpy(), roll.std().to_numpy()
            g[f"{m}_base28"] = mean
            std = np.fmax(std, floors.get(m, 0.0))
            g[f"{m}_z28"] = np.where(std > 0, (g[m].to_numpy() - mean) / std, np.nan)
            g[f"{m}_last5"] = g[m].shift(1).rolling(5, min_periods=1).mean()
        g["days_since_last"] = g["start_time"].diff().dt.total_seconds() / 86400
        parts.append(g)
    out = pd.concat(parts).sort_index()
    z = a["anomaly_z"]
    zcols = [f"{m}_z28" for m in metrics]
    out["anomaly_metrics"] = out[zcols].apply(
        lambda r: ",".join(c.replace("_z28", "") for c, v in r.items() if pd.notna(v) and abs(v) > z), axis=1)
    out["is_anomaly"] = out["anomaly_metrics"] != ""
    return out
