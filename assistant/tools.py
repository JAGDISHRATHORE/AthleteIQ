"""Typed tools the assistant may call. Each tool reads precomputed tables and
returns structured data plus *citations*: every number the assistant is
allowed to say, with the metric it comes from and whether it is an observed
value or a model estimate.

Time reference: the dataset is historical (2007-2016), so "last N weeks"
always means the N*7 days ending on the athlete's most recent recorded workout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from athleteiq.store import AthleteStore  # noqa: E402

ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]


METRIC_NAMES = {"hr_mean": "average HR", "avg_speed_kmh": "speed", "efficiency_m_per_beat": "efficiency",
                "trimp": "load", "distance_km": "distance"}


def _r(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(round(float(x))) if nd == 0 else round(float(x), nd)


def _metric_names(s) -> str:
    return ", ".join(METRIC_NAMES.get(m, m) for m in str(s or "").split(",") if m)


def _cite(metric: str, value, unit: str = "", kind: str = "observed", source: str = "") -> dict:
    return {"metric": metric, "value": value, "unit": unit, "kind": kind, "source": source}


def _definition(metric: str, value, unit: str = "") -> dict:
    """A method constant (e.g. the 28-day window) - cited so every number in an answer is traceable."""
    return _cite(metric, value, unit, kind="definition", source="docs/methodology.md")


class ToolError(Exception):
    pass


class AthleteTools:
    def __init__(self, store: AthleteStore | None = None):
        self.store = store or AthleteStore()

    # ------------------------------------------------------------------ helpers
    def _check_athlete(self, user_id) -> int:
        if user_id is None:
            raise ToolError("Which athlete? Please give an athlete id.")
        uid = int(user_id)
        if not self.store.has_athlete(uid):
            raise ToolError(f"Athlete {uid} is not in the loaded data.")
        return uid

    # ------------------------------------------------------------------ tools
    def athlete_overview(self, user_id: int) -> dict:
        uid = self._check_athlete(user_id)
        s = self.store.sessions(uid)
        last = s["start_time"].max()
        recent = s[s.start_time > last - pd.Timedelta(days=28)]
        prev = s[(s.start_time <= last - pd.Timedelta(days=28)) & (s.start_time > last - pd.Timedelta(days=56))]
        eff_r, eff_p = recent["efficiency_m_per_beat"].mean(), prev["efficiency_m_per_beat"].mean()
        data = {
            "user_id": uid, "sessions_total": int(len(s)),
            "first_session": str(s.start_time.min().date()), "last_session": str(last.date()),
            "total_km": _r(s.distance_km.sum()),
            "last28_sessions": int(len(recent)), "last28_km": _r(recent.distance_km.sum()),
            "last28_avg_hr": _r(recent.hr_mean.mean()), "last28_avg_pace_min_km": _r(recent.pace_min_per_km.mean(), 2),
            "efficiency_last28": _r(eff_r, 3), "efficiency_prev28": _r(eff_p, 3),
            "hrmax_estimate": _r(s.hrmax_est.iloc[-1], 0),
            "training_pattern": s["cluster_description"].iloc[0] if "cluster_description" in s and pd.notna(s["cluster_description"].iloc[0]) else None,
        }
        src = "sessions table"
        cites = [
            _cite("total sessions", data["sessions_total"], "", source=src),
            _cite("total distance", data["total_km"], "km", source=src),
            _cite("sessions in last 28 days", data["last28_sessions"], "", source=src),
            _cite("distance in last 28 days", data["last28_km"], "km", source=src),
            _cite("mean session HR, last 28 days", data["last28_avg_hr"], "bpm", source=src),
            _cite("mean pace, last 28 days", data["last28_avg_pace_min_km"], "min/km", source=src),
            _cite("efficiency, last 28 days", data["efficiency_last28"], "m/beat", source=src),
            _cite("efficiency, previous 28 days", data["efficiency_prev28"], "m/beat", source=src),
            _cite("estimated HRmax", data["hrmax_estimate"], "bpm", kind="estimate",
                  source="highest session max HR observed (no lab test)"),
            _definition("comparison window", 28, "days"),
        ]
        return {"tool": "athlete_overview", "data": data, "citations": cites}

    def workload_trend(self, user_id: int, weeks: int = 4) -> dict:
        uid = self._check_athlete(user_id)
        weeks = int(max(1, min(weeks or 4, 26)))
        d = self.store.daily(uid).sort_values("date")
        end = d["date"].max()
        span = pd.Timedelta(days=7 * weeks)
        rec = d[d.date > end - span]
        prev = d[(d.date <= end - span) & (d.date > end - 2 * span)]
        load_r, load_p = float(rec.load.sum()), float(prev.load.sum())
        pct = (load_r - load_p) / load_p * 100 if load_p > 0 else None
        last_row = d.iloc[-1]
        w = self.store.weekly(uid).sort_values("week").tail(weeks)
        data = {
            "user_id": uid, "weeks": weeks, "period_end": str(end.date()),
            "load_recent": _r(load_r, 0), "load_previous": _r(load_p, 0), "pct_change": _r(pct, 0),
            "sessions_recent": int(rec.sessions.sum()), "sessions_previous": int(prev.sessions.sum()),
            "km_recent": _r(rec.distance_km.sum()), "km_previous": _r(prev.distance_km.sum()),
            "acwr_latest": _r(last_row.acwr, 2),
            "weekly": [{"week": str(r.week.date()), "load": _r(r.weekly_load, 0), "sessions": int(r.sessions)}
                       for r in w.itertuples()],
        }
        src = "daily_load table (Edwards TRIMP per day)"
        cites = [
            _cite(f"training load, last {weeks} weeks", data["load_recent"], "TRIMP", source=src),
            _cite(f"training load, previous {weeks} weeks", data["load_previous"], "TRIMP", source=src),
            _cite("change in load", data["pct_change"], "%", source=src),
            _cite(f"sessions, last {weeks} weeks", data["sessions_recent"], "", source=src),
            _cite(f"sessions, previous {weeks} weeks", data["sessions_previous"], "", source=src),
            _cite(f"distance, last {weeks} weeks", data["km_recent"], "km", source=src),
            _cite(f"distance, previous {weeks} weeks", data["km_previous"], "km", source=src),
            _cite("acute:chronic workload ratio (latest)", data["acwr_latest"], "", source=src),
        ]
        return {"tool": "workload_trend", "data": data, "citations": cites}

    def session_detail(self, user_id: int | None = None, workout_id: int | None = None) -> dict:
        if workout_id is not None:
            s = self.store.session(workout_id)
            if s is None:
                raise ToolError(f"Workout {workout_id} is not in the loaded data.")
        else:
            uid = self._check_athlete(user_id)
            s = self.store.sessions(uid).iloc[-1]
        zones = {z: _r(s[f"z{i + 1}_min"]) for i, z in enumerate(ZONES)}
        anomalies = [METRIC_NAMES.get(m, m) for m in str(s.get("anomaly_metrics") or "").split(",") if m]
        data = {
            "workout_id": int(s.workout_id), "user_id": int(s.user_id), "date": str(s.start_time.date()),
            "duration_min": _r(s.duration_min), "distance_km": _r(s.distance_km, 2),
            "pace_min_km": _r(s.pace_min_per_km, 2), "avg_hr": _r(s.hr_mean), "max_hr": _r(s.hr_max),
            "pct_hrmax": _r(s.pct_hrmax_mean * 100, 0), "trimp": _r(s.trimp, 0),
            "minutes_in_zone": zones, "hr_vs_baseline_z": _r(s.get("hr_mean_z28"), 1),
            "efficiency_vs_baseline_z": _r(s.get("efficiency_m_per_beat_z28"), 1),
            "unusual_metrics": anomalies, "data_quality": s.quality_label,
        }
        src = f"sessions table, workout {data['workout_id']}"
        cites = [
            _cite("duration", data["duration_min"], "min", source=src),
            _cite("distance", data["distance_km"], "km", source=src),
            _cite("pace", data["pace_min_km"], "min/km", source=src),
            _cite("average HR", data["avg_hr"], "bpm", source=src),
            _cite("max HR", data["max_hr"], "bpm", source=src),
            _cite("average HR as % of estimated HRmax", data["pct_hrmax"], "%", kind="estimate", source=src),
            _cite("session load", data["trimp"], "TRIMP", source=src),
            _cite("HR vs 28-day baseline", data["hr_vs_baseline_z"], "z", source=src),
            _cite("efficiency vs 28-day baseline", data["efficiency_vs_baseline_z"], "z", source=src),
            _definition("baseline window", 28, "days"),
        ] + [_cite(f"minutes in {z}", v, "min", source=src) for z, v in zones.items()]
        return {"tool": "session_detail", "data": data, "citations": cites}

    def explain_prediction(self, workout_id: int | None = None, user_id: int | None = None) -> dict:
        if workout_id is None:
            uid = self._check_athlete(user_id)
            ps = self.store.predicted_sessions(uid)
            if ps.empty:
                raise ToolError(f"Athlete {uid} has no model predictions (their sessions were used for training).")
            workout_id = int(ps.iloc[-1].workout_id)
        p = self.store.prediction(workout_id)
        if p is None:
            raise ToolError(f"Workout {workout_id} has no model prediction (it was a training session).")
        exp = self.store.explanation(workout_id, top=5)
        drivers = [] if exp is None else [
            {"feature": r.label, "value": _r(r.value, 2), "effect_bpm": _r(r.contribution_bpm, 1)}
            for r in exp.itertuples()]
        data = {
            "workout_id": int(workout_id), "user_id": int(p.user_id), "date": str(pd.Timestamp(p.start_time).date()),
            "predicted_hr": _r(p.pred_lgbm_personal), "interval_low": _r(p.pi_low), "interval_high": _r(p.pi_high),
            "actual_hr": _r(p.hr_mean), "error": _r(p.pred_lgbm_personal - p.hr_mean),
            "personal_offset": _r(p.personal_offset), "model_average": _r(self.store.model_meta.get("shap_base"), 1),
            "drivers": drivers,
        }
        src = f"predictions + SHAP tables, workout {workout_id}"
        cites = [
            _cite("predicted average HR", data["predicted_hr"], "bpm", kind="estimate", source=src),
            _cite("90% interval low", data["interval_low"], "bpm", kind="estimate", source=src),
            _cite("90% interval high", data["interval_high"], "bpm", kind="estimate", source=src),
            _cite("actual average HR", data["actual_hr"], "bpm", source=src),
            _cite("prediction error", data["error"], "bpm", kind="estimate", source=src),
            _cite("personal correction", data["personal_offset"], "bpm", kind="estimate", source=src),
            _cite("model average prediction", data["model_average"], "bpm", kind="estimate", source=src),
            _definition("prediction interval level", 90, "%"),
            _definition("recent-history window", 5, "sessions"),
            _definition("baseline window", 28, "days"),
        ] + [_cite(f"effect of {d['feature']}", d["effect_bpm"], "bpm", kind="estimate", source=src) for d in drivers] \
          + [_cite(f"value of {d['feature']}", d["value"], "", source=src) for d in drivers]
        return {"tool": "explain_prediction", "data": data, "citations": cites}

    def find_anomalies(self, user_id: int, limit: int = 5) -> dict:
        uid = self._check_athlete(user_id)
        s = self.store.sessions(uid)
        a = s[s.is_anomaly].sort_values("start_time", ascending=False)
        rows = [{"workout_id": int(r.workout_id), "date": str(r.start_time.date()),
                 "metrics": _metric_names(r.anomaly_metrics), "avg_hr": _r(r.hr_mean), "hr_z": _r(r.hr_mean_z28),
                 "distance_km": _r(r.distance_km, 2)} for r in a.head(int(limit)).itertuples()]
        scored = int(s["hr_mean_z28"].notna().sum())
        data = {"user_id": uid, "anomalous_sessions": int(len(a)), "sessions_scored": scored, "recent": rows}
        src = "sessions table (|z| > 2 vs prior 28-day personal baseline)"
        cites = [_cite("unusual sessions", data["anomalous_sessions"], "", source=src),
                 _cite("sessions with a baseline", scored, "", source=src),
                 _definition("baseline window", 28, "days"), _definition("anomaly threshold", 2, "SD")]
        for r in rows:
            cites += [_cite(f"avg HR on {r['date']}", r["avg_hr"], "bpm", source=src),
                      _cite(f"HR z-score on {r['date']}", r["hr_z"], "z", source=src),
                      _cite(f"distance on {r['date']}", r["distance_km"], "km", source=src)]
        return {"tool": "find_anomalies", "data": data, "citations": cites}

    # ------------------------------------------------------------------ registry
    def call(self, name: str, args: dict[str, Any]) -> dict:
        fn = getattr(self, name, None)
        if name not in TOOL_SPECS or fn is None:
            raise ToolError(f"unknown tool {name}")
        return fn(**args)


TOOL_SPECS: dict[str, dict] = {
    "athlete_overview": {
        "description": "Summary of an athlete: totals, last-28-day volume, HR, pace, efficiency trend, estimated HRmax, training pattern.",
        "input_schema": {"type": "object", "properties": {"user_id": {"type": "integer"}}, "required": ["user_id"]},
    },
    "workload_trend": {
        "description": "Training load (Edwards TRIMP) over the last N weeks vs the N weeks before, sessions, distance, latest ACWR, weekly breakdown.",
        "input_schema": {"type": "object", "properties": {"user_id": {"type": "integer"},
                                                          "weeks": {"type": "integer", "default": 4}},
                         "required": ["user_id"]},
    },
    "session_detail": {
        "description": "One workout's metrics (latest for the athlete if no workout_id): distance, pace, HR, zones, load, deviation from personal baseline, data quality.",
        "input_schema": {"type": "object", "properties": {"user_id": {"type": "integer"},
                                                          "workout_id": {"type": "integer"}}},
    },
    "explain_prediction": {
        "description": "Model-predicted average HR for a workout with its 90% interval, the actual value and the top SHAP drivers (latest predicted workout if no workout_id).",
        "input_schema": {"type": "object", "properties": {"workout_id": {"type": "integer"},
                                                          "user_id": {"type": "integer"}}},
    },
    "find_anomalies": {
        "description": "Recent sessions that deviated more than 2 SD from the athlete's own prior 28-day baseline.",
        "input_schema": {"type": "object", "properties": {"user_id": {"type": "integer"},
                                                          "limit": {"type": "integer", "default": 5}},
                         "required": ["user_id"]},
    },
}
