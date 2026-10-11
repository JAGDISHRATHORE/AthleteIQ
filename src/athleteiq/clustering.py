"""Athlete training-pattern discovery (k-means on per-athlete training profiles)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_FEATURES = {
    "sessions_per_week": "sessions / active week",
    "median_duration_min": "session length",
    "median_speed_kmh": "running speed",
    "median_pct_hrmax": "intensity (%HRmax)",
    "share_z4_z5": "time in Z4-Z5",
    "median_elev_gain_m": "hilliness",
}


def athlete_profiles(sessions: pd.DataFrame, min_sessions: int = 10) -> pd.DataFrame:
    s = sessions.copy()
    s["week"] = s["start_time"].dt.to_period("W")
    zone_cols = [f"z{k}_min" for k in range(6)]
    s["zone_total"] = s[zone_cols].sum(axis=1)
    g = s.groupby("user_id")
    prof = pd.DataFrame({
        "n_sessions": g.size(),
        "sessions_per_week": g.size() / g["week"].nunique(),
        "median_duration_min": g["duration_min"].median(),
        "median_speed_kmh": g["avg_speed_kmh"].median(),
        "median_pct_hrmax": g["pct_hrmax_mean"].median(),
        "share_z4_z5": g["z4_min"].sum().add(g["z5_min"].sum()) / g["zone_total"].sum(),
        "median_elev_gain_m": g["elev_gain_m"].median(),
    })
    return prof[prof["n_sessions"] >= min_sessions]


def _describe(z_centroid: pd.Series) -> str:
    top = z_centroid.abs().sort_values(ascending=False).index[:2]
    words = [("higher " if z_centroid[f] > 0 else "lower ") + PROFILE_FEATURES[f] for f in top]
    return ", ".join(words)


def cluster_athletes(profiles: pd.DataFrame, k_range=range(2, 7), seed: int = 42):
    X = profiles[list(PROFILE_FEATURES)].copy()
    X["median_elev_gain_m"] = np.log1p(X["median_elev_gain_m"])
    Z = StandardScaler().fit_transform(X)
    scores = {}
    for k in k_range:
        labels = KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels))
    best_k = max(scores, key=scores.get)
    km = KMeans(best_k, n_init=20, random_state=seed).fit(Z)
    out = profiles.copy()
    out["cluster"] = km.labels_
    zc = pd.DataFrame(km.cluster_centers_, columns=list(PROFILE_FEATURES))
    summary = out.groupby("cluster")[list(PROFILE_FEATURES)].median()
    summary["athletes"] = out.groupby("cluster").size()
    summary["description"] = [_describe(zc.loc[c]) for c in summary.index]
    out["cluster_description"] = out["cluster"].map(summary["description"])
    return out, summary, scores
