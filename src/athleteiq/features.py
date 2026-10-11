"""Feature table for the prediction task.

Task: predict a session's mean heart rate (bpm) from what the athlete *did*
(the external load: duration, distance, speed, terrain, time of day) and what
was known about the athlete *before* the session (personal history and recent
training load). In other words: "given this planned workout, how hard will it
be for this athlete's heart?"

Leakage rules
- No feature uses the current session's heart rate.
- History features use only earlier sessions (shift / closed-left windows).
- The packager's RECOVERY/STEADY/INTENSIVE label is derived from heart rate,
  so it is never a feature - it is only used to slice errors afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .training_load import prior_load_features

EXTERNAL_FEATURES = [
    "duration_min", "distance_km", "avg_speed_kmh", "speed_std_kmh", "speed_max_kmh",
    "elev_gain_m", "elev_loss_m", "hour", "dayofweek",
]
HISTORY_FEATURES = [
    "hr_mean_last5", "hr_mean_base28", "efficiency_m_per_beat_last5", "avg_speed_kmh_last5",
    "trimp_last5", "prior_hr_at_speed", "prior_hr_speed_slope", "log_prior_sessions",
    "days_since_last", "prior_acute", "prior_chronic", "prior_acwr", "prior_ewma_acwr",
]
FEATURES = EXTERNAL_FEATURES + HISTORY_FEATURES
TARGET = "hr_mean"

PRETTY = {
    "duration_min": "Duration (min)", "distance_km": "Distance (km)",
    "avg_speed_kmh": "Avg speed (km/h)", "speed_std_kmh": "Speed variability",
    "speed_max_kmh": "Max speed (km/h)", "elev_gain_m": "Elevation gain (m)",
    "elev_loss_m": "Elevation loss (m)", "hour": "Hour of day", "dayofweek": "Day of week",
    "hr_mean_last5": "Avg HR, last 5 sessions", "hr_mean_base28": "Avg HR, 28-day baseline",
    "efficiency_m_per_beat_last5": "Efficiency, last 5 sessions",
    "avg_speed_kmh_last5": "Avg speed, last 5 sessions", "trimp_last5": "Load (TRIMP), last 5 sessions",
    "prior_hr_at_speed": "Personal HR expected at this speed",
    "prior_hr_speed_slope": "Personal HR-per-km/h slope",
    "log_prior_sessions": "History length (log sessions)", "days_since_last": "Days since last session",
    "prior_acute": "Acute load (7 d)", "prior_chronic": "Chronic load (28 d)",
    "prior_acwr": "Acute:chronic ratio", "prior_ewma_acwr": "EWMA acute:chronic ratio",
}


def _expanding_speed_hr_fit(g: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Per-athlete linear fit hr_mean ~ avg_speed on *previous* sessions only,
    evaluated at the current session's speed. Uses cumulative sums, O(n)."""
    x, y = g["avg_speed_kmh"].to_numpy(), g["hr_mean"].to_numpy()
    n = np.arange(len(g))                                    # number of prior sessions
    sx = np.concatenate([[0], np.cumsum(x)[:-1]])
    sy = np.concatenate([[0], np.cumsum(y)[:-1]])
    sxx = np.concatenate([[0], np.cumsum(x * x)[:-1]])
    sxy = np.concatenate([[0], np.cumsum(x * y)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        var = sxx / n - (sx / n) ** 2
        cov = sxy / n - (sx / n) * (sy / n)
        slope = np.where((n >= min_n) & (var > 0.05), cov / var, np.nan)
        slope = np.clip(slope, -5, 15)                       # physiologically sane range
        intercept = sy / n - slope * sx / n
    return pd.DataFrame({"prior_hr_speed_slope": slope,
                         "prior_hr_at_speed": intercept + slope * x}, index=g.index)


def build_feature_table(sessions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    s = sessions.sort_values(["user_id", "start_time"]).copy()
    fits = pd.concat([_expanding_speed_hr_fit(g) for _, g in s.groupby("user_id", sort=False)])
    s = s.join(fits)
    s["log_prior_sessions"] = np.log1p(s["session_idx"])
    loads = prior_load_features(s, daily).drop(columns=["user_id", "date"])
    s = s.merge(loads, on="workout_id", how="left")
    return s
