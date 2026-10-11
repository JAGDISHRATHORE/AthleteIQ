"""Turn raw per-workout series into an analysis-ready session table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .intensity import edwards_trimp, online_hrmax, time_in_zones_min
from .quality import quality_table


def _session_summary(r) -> dict:
    ts = np.asarray(r.timestamp, float)
    hr = np.asarray(r.heart_rate, float)
    sp = np.asarray(r.speed, float)
    alt = np.asarray(r.altitude, float)
    dt = np.minimum(np.diff(ts), 60.0)              # cap gaps at 60 s
    w = np.append(dt, 0.0)                          # time each sample "owns"
    moving_s = float(dt.sum())
    dalt = np.diff(alt)
    return {
        "workout_id": r.workout_id,
        "user_id": r.user_id,
        "start_time": pd.Timestamp(ts[0], unit="s"),
        "duration_min": (ts[-1] - ts[0]) / 60.0,
        "moving_min": moving_s / 60.0,
        "distance_km": float(np.sum(sp[:-1] * dt) / 3600.0),
        "avg_speed_kmh": float(np.average(sp, weights=w)) if w.sum() > 0 else float(sp.mean()),
        "speed_std_kmh": float(sp.std()),
        "speed_max_kmh": float(sp.max()),
        "elev_gain_m": float(dalt[dalt > 0].sum()),
        "elev_loss_m": float(abs(dalt[dalt < 0].sum())),
        "hr_mean": float(np.average(hr, weights=w)) if w.sum() > 0 else float(hr.mean()),
        "hr_std": float(hr.std()),
        "hr_min": float(hr.min()),
        "hr_max": float(hr.max()),
        "hr_drift": float(hr[len(hr) // 2:].mean() - hr[: len(hr) // 2].mean()),
        "label_workout_type": r.workout_type,       # provided by the dataset packager; HR-derived
    }


def build_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    s = pd.DataFrame([_session_summary(r) for r in raw.itertuples(index=False)])
    s = s.merge(quality_table(raw), on="workout_id", how="left")
    s = s.sort_values(["user_id", "start_time"]).reset_index(drop=True)

    s["date"] = s["start_time"].dt.normalize()
    s["hour"] = s["start_time"].dt.hour
    s["dayofweek"] = s["start_time"].dt.dayofweek
    s["session_idx"] = s.groupby("user_id").cumcount()
    s["pace_min_per_km"] = np.where(s["avg_speed_kmh"] > 0, 60.0 / s["avg_speed_kmh"], np.nan)
    # Efficiency index: metres covered per heartbeat. Higher = more speed for the same cardiac cost.
    s["efficiency_m_per_beat"] = (s["avg_speed_kmh"] * 1000 / 60) / s["hr_mean"]

    a = cfg["athlete"]
    s["hrmax_est"] = online_hrmax(s, a["hrmax_default"], a["hrmax_min_sessions"])
    s["hrmax_reliable"] = s["session_idx"] + 1 >= a["hrmax_min_sessions"]

    by_id = raw.set_index("workout_id")
    zones = np.vstack([
        time_in_zones_min(by_id.at[w, "heart_rate"], by_id.at[w, "timestamp"], hm)
        for w, hm in zip(s["workout_id"], s["hrmax_est"])
    ])
    for k in range(6):
        s[f"z{k}_min"] = zones[:, k]
    s["trimp"] = [edwards_trimp(z) for z in zones]
    s["pct_hrmax_mean"] = s["hr_mean"] / s["hrmax_est"]
    return s
