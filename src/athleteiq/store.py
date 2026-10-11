"""Read-only access to processed AthleteIQ tables (used by the dashboard and the assistant)."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd

from .config import data_dir, load_config, path
from .explainability import local_explanation
from .features import FEATURES


class AthleteStore:
    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory) if directory else data_dir()
        self.is_sample = self.dir.name == "sample"

    # ---------------------------------------------------------------- tables
    def _read(self, name: str) -> pd.DataFrame:
        f = self.dir / f"{name}.parquet"
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    @cached_property
    def sessions_all(self) -> pd.DataFrame:
        return self._read("sessions").sort_values(["user_id", "start_time"]).reset_index(drop=True)

    @cached_property
    def daily_all(self) -> pd.DataFrame:
        return self._read("daily_load")

    @cached_property
    def weekly_all(self) -> pd.DataFrame:
        return self._read("weekly")

    @cached_property
    def predictions(self) -> pd.DataFrame:
        return self._read("predictions")

    @cached_property
    def shap(self) -> pd.DataFrame:
        return self._read("shap_values")

    @cached_property
    def cluster_summary(self) -> pd.DataFrame:
        return self._read("cluster_summary")

    @cached_property
    def metrics(self) -> dict:
        f = path("models") / "metrics.json"
        return json.load(open(f)) if f.exists() else {}

    @cached_property
    def model_meta(self) -> dict:
        f = path("models") / "model_meta.json"
        return json.load(open(f)) if f.exists() else {}

    # ---------------------------------------------------------------- lookups
    def athletes(self) -> pd.DataFrame:
        s = self.sessions_all
        g = s.groupby("user_id")
        out = pd.DataFrame({
            "sessions": g.size(),
            "first": g["start_time"].min(),
            "last": g["start_time"].max(),
            "total_km": g["distance_km"].sum().round(1),
            "cluster": g["cluster"].first() if "cluster" in s else None,
            "cluster_description": g["cluster_description"].first() if "cluster_description" in s else None,
            "has_predictions": g["workout_id"].apply(lambda w: w.isin(self.predictions.get("workout_id", [])).any()),
        })
        return out.sort_values("sessions", ascending=False)

    def has_athlete(self, user_id: int) -> bool:
        return int(user_id) in set(self.sessions_all["user_id"].unique())

    def sessions(self, user_id: int) -> pd.DataFrame:
        return self.sessions_all[self.sessions_all.user_id == int(user_id)]

    def session(self, workout_id: int) -> pd.Series | None:
        s = self.sessions_all[self.sessions_all.workout_id == int(workout_id)]
        return None if s.empty else s.iloc[0]

    def daily(self, user_id: int) -> pd.DataFrame:
        return self.daily_all[self.daily_all.user_id == int(user_id)]

    def weekly(self, user_id: int) -> pd.DataFrame:
        return self.weekly_all[self.weekly_all.user_id == int(user_id)]

    def series(self, workout_id: int) -> pd.DataFrame:
        f = self.dir / "series.parquet"
        if f.exists():
            return pd.read_parquet(f, filters=[("workout_id", "==", int(workout_id))])
        raw = path(load_config()["data"]["raw_file"])
        if not raw.exists():
            return pd.DataFrame()
        r = pd.read_parquet(raw, filters=[("workout_id", "==", int(workout_id))])
        if r.empty:
            return r
        row = r.iloc[0]
        ts = pd.Series(row["timestamp"], dtype=float)
        return pd.DataFrame({"workout_id": int(workout_id), "t_s": ts - ts.iloc[0],
                             "heart_rate": row["heart_rate"], "speed_kmh": row["speed"],
                             "altitude_m": row["altitude"]})

    def prediction(self, workout_id: int) -> pd.Series | None:
        p = self.predictions
        if p.empty:
            return None
        p = p[p.workout_id == int(workout_id)]
        return None if p.empty else p.iloc[0]

    def predicted_sessions(self, user_id: int) -> pd.DataFrame:
        p = self.predictions
        return p[p.user_id == int(user_id)].sort_values("start_time") if not p.empty else p

    def explanation(self, workout_id: int, top: int = 8) -> pd.DataFrame | None:
        sh = self.shap[self.shap.workout_id == int(workout_id)] if not self.shap.empty else self.shap
        s = self.session(workout_id)
        if sh.empty or s is None:
            return None
        return local_explanation(sh.iloc[0][FEATURES], s, self.model_meta.get("shap_base", 0.0), top)
