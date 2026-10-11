"""Training load: daily load, acute/chronic load, ACWR, weekly monotony and strain.

Load = Edwards TRIMP summed per calendar day. Days without a recorded
heart-rate workout count as zero load. The dataset contains only workouts the
athlete chose to record *with* HR, so absolute loads are a lower bound.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .config import load_config


def daily_load(sessions: pd.DataFrame) -> pd.DataFrame:
    """Continuous daily series per athlete with acute/chronic load and ACWR.

    acute   = rolling mean of the last 7 days (including today)
    chronic = rolling mean of the last 28 days (including today)
    acwr    = acute / chronic, only once 28 days of history exist
    ewma_*  = exponentially weighted variants (Williams et al., 2017)
    """
    lc = load_config()["load"]
    a, c = lc["acute_days"], lc["chronic_days"]
    per_day = (sessions.groupby(["user_id", "date"])
               .agg(load=("trimp", "sum"), sessions=("workout_id", "count"),
                    distance_km=("distance_km", "sum"), duration_min=("duration_min", "sum"))
               .reset_index())
    out = []
    for uid, g in per_day.groupby("user_id"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx).fillna({"load": 0, "sessions": 0,
                                                     "distance_km": 0, "duration_min": 0})
        g["user_id"] = uid
        g["acute"] = g["load"].rolling(a, min_periods=1).mean()
        g["chronic"] = g["load"].rolling(c, min_periods=c).mean()
        g["acwr"] = np.where(g["chronic"] > 0, g["acute"] / g["chronic"], np.nan)
        g["ewma_acute"] = g["load"].ewm(alpha=2 / (a + 1), adjust=False).mean()
        g["ewma_chronic"] = g["load"].ewm(alpha=2 / (c + 1), adjust=False).mean()
        g["ewma_acwr"] = np.where(g["ewma_chronic"] > 0, g["ewma_acute"] / g["ewma_chronic"], np.nan)
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def weekly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly load, monotony (mean/sd of daily load) and strain (load * monotony),
    after Foster (1998), computed in SQL with DuckDB."""
    con = duckdb.connect()
    con.register("daily", daily)
    df = con.execute("""
        SELECT user_id,
               date_trunc('week', date)               AS week,
               SUM(load)                              AS weekly_load,
               SUM(sessions)::INT                     AS sessions,
               SUM(distance_km)                       AS distance_km,
               SUM(duration_min)                      AS duration_min,
               AVG(load) / NULLIF(STDDEV_POP(load), 0) AS monotony,
               SUM(load) * AVG(load) / NULLIF(STDDEV_POP(load), 0) AS strain,
               COUNT(*)                               AS days_observed
        FROM daily
        GROUP BY ALL
        ORDER BY user_id, week
    """).df()
    con.close()
    df["week"] = pd.to_datetime(df["week"])
    return df


def prior_load_features(sessions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Load known *before* each session's day: acute/chronic/ACWR as of the previous day."""
    d = daily.sort_values(["user_id", "date"]).copy()
    for col in ["acute", "chronic", "acwr", "ewma_acute", "ewma_chronic", "ewma_acwr"]:
        d[f"prior_{col}"] = d.groupby("user_id")[col].shift(1)
    cols = ["user_id", "date"] + [c for c in d.columns if c.startswith("prior_")]
    return sessions[["workout_id", "user_id", "date"]].merge(d[cols], on=["user_id", "date"], how="left")
