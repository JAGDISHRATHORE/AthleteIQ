"""Heart-rate zones and session intensity (Edwards TRIMP).

Zones follow the common 5-zone %HRmax scheme used by Edwards (1993):
Z1 50-60%, Z2 60-70%, Z3 70-80%, Z4 80-90%, Z5 90-100%. Time below 50% HRmax
counts as Z0 and contributes no load.

The dataset has no age or lab-measured HRmax, so HRmax is *estimated* per
athlete as the highest session max HR observed so far (an online estimate that
never looks at future sessions), with a population default until the athlete
has enough sessions. This is stated wherever zones are shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_EDGES = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # lower bounds of Z1..Z5 as %HRmax
ZONE_NAMES = ["Z0 <50%", "Z1 50-60%", "Z2 60-70%", "Z3 70-80%", "Z4 80-90%", "Z5 90-100%"]


def zone_index(hr: np.ndarray, hrmax: float) -> np.ndarray:
    """0..5 zone index for each HR sample."""
    return np.searchsorted(ZONE_EDGES, np.asarray(hr, float) / hrmax, side="right")


def time_in_zones_min(hr: np.ndarray, ts: np.ndarray, hrmax: float, gap_cap_s: float = 60) -> np.ndarray:
    """Minutes spent in zones 0..5. Each sample owns the interval to the next one
    (capped so recording gaps do not inflate time)."""
    ts = np.asarray(ts, float)
    dt = np.minimum(np.diff(ts, append=ts[-1]), gap_cap_s)
    z = zone_index(hr, hrmax)
    return np.bincount(z, weights=dt, minlength=6)[:6] / 60.0


def edwards_trimp(zone_minutes: np.ndarray) -> float:
    """Edwards TRIMP = sum(minutes in zone k * k) for k = 1..5."""
    return float(np.dot(zone_minutes[1:6], np.arange(1, 6)))


def online_hrmax(sessions: pd.DataFrame, default: float, min_sessions: int) -> pd.Series:
    """Per-session HRmax estimate using only that athlete's sessions up to and
    including the current one (the current session's HR is known once it ends).
    Falls back to ``max(default, running max)`` until ``min_sessions`` exist."""
    s = sessions.sort_values(["user_id", "start_time"])
    running = s.groupby("user_id")["hr_max"].cummax()
    n = s.groupby("user_id").cumcount() + 1
    est = running.where(n >= min_sessions, np.maximum(running, default))
    return est.reindex(sessions.index)
