"""Leakage-safe splits, baselines, ML models, online personalisation and
conformal prediction intervals for session mean heart rate."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .features import FEATURES, TARGET

# --------------------------------------------------------------------------- splits


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Return a split label per row.

    * ``holdout``  - every session of ~15% of athletes the model never trains on
                     (tests generalisation to brand-new athletes).
    * ``calib``    - every session of another ~10% of unseen athletes, used only
                     to calibrate prediction intervals (so intervals also see
                     athletes with very short histories).
    * ``train`` / ``val`` / ``test`` - for every other athlete with enough
      sessions: the first 70% of their sessions in time, the next 15%, the most
      recent 15%. The model is always evaluated on each athlete's *future*.
    * athletes with too few sessions contribute to ``train`` only.
    """
    m = load_config()["model"]
    counts = df.groupby("user_id")["workout_id"].transform("count")
    eligible_ids = np.sort(df.loc[counts >= m["min_sessions_per_athlete"], "user_id"].unique())
    rng = np.random.default_rng(m["random_state"])
    n_hold = int(round(len(eligible_ids) * m["holdout_athlete_frac"]))
    n_cal = int(round(len(eligible_ids) * m["calib_athlete_frac"]))
    picked = rng.choice(eligible_ids, n_hold + n_cal, replace=False).tolist()
    holdout, calib = set(picked[:n_hold]), set(picked[n_hold:])

    order = df.groupby("user_id")["start_time"].rank(method="first") - 1
    frac = order / counts
    split = np.where(frac < m["train_frac"], "train",
                     np.where(frac < m["train_frac"] + m["val_frac"], "val", "test"))
    split = pd.Series(split, index=df.index)
    split[counts < m["min_sessions_per_athlete"]] = "train"
    split[df["user_id"].isin(holdout)] = "holdout"
    split[df["user_id"].isin(calib)] = "calib"
    return split


# --------------------------------------------------------------------------- metrics


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "within_5bpm": float(np.mean(np.abs(err) <= 5)),
    }


# --------------------------------------------------------------------------- baselines


def predict_baselines(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    last5 = df["hr_mean_last5"].fillna(global_mean)
    speed_fit = df["prior_hr_at_speed"].fillna(last5)
    return pd.DataFrame({
        "pred_global_mean": np.full(len(df), global_mean),
        "pred_athlete_last5": last5.to_numpy(),
        "pred_personal_speed_fit": speed_fit.to_numpy(),
    }, index=df.index)


# --------------------------------------------------------------------------- ML models


def fit_ridge(train: pd.DataFrame):
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    model.fit(train[FEATURES], train[TARGET])
    return model


def fit_lgbm(train: pd.DataFrame, val: pd.DataFrame, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.06, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        objective="l1", random_state=seed, verbose=-1,
    )
    model.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
              eval_metric="l1", callbacks=[lgb.early_stopping(150, verbose=False)])
    return model


# --------------------------------------------------------------------------- personalisation


def online_offsets(df: pd.DataFrame, residual: pd.Series, k: float) -> pd.Series:
    """Athlete-specific correction learned *online*.

    For each session, the offset is the shrunk mean of the model's errors on
    that athlete's earlier evaluated sessions: sum(prior residuals) / (n + k).
    The true HR of a session becomes known once it is recorded, so this is
    exactly what a deployed system could do. ``k`` controls shrinkage towards
    zero (the global model) when an athlete has little history.
    """
    d = pd.DataFrame({"user_id": df["user_id"], "t": df["start_time"], "r": residual})
    d = d.sort_values(["user_id", "t"])
    g = d.groupby("user_id")["r"]
    prior_sum = g.cumsum() - d["r"]
    prior_n = g.cumcount()
    return (prior_sum / (prior_n + k)).reindex(df.index)


# --------------------------------------------------------------------------- conformal


@dataclass
class ConformalInterval:
    """Split-conformal interval: pred +/- q, where q is the finite-sample
    corrected (1 - alpha) quantile of |residual| on a calibration set."""
    alpha: float
    q: float = float("nan")

    def fit(self, residuals: np.ndarray) -> ConformalInterval:
        r = np.sort(np.abs(np.asarray(residuals, float)))
        n = len(r)
        rank = min(int(np.ceil((n + 1) * (1 - self.alpha))), n)
        self.q = float(r[rank - 1])
        return self

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pred - self.q, pred + self.q

    @staticmethod
    def coverage(y, lo, hi) -> float:
        y = np.asarray(y, float)
        return float(np.mean((y >= lo) & (y <= hi)))


class MondrianConformal:
    """Group-conditional (Mondrian) split conformal: a separate half-width per
    group, so coverage holds within each group and not just on average. Groups
    here are history-length bands - new athletes get wider, honest intervals."""

    def __init__(self, alpha: float, edges: list[int]):
        self.alpha, self.edges = alpha, list(edges)
        self.q: dict[int, float] = {}

    def group(self, n_prior) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(n_prior), side="right") - 1

    def fit(self, residuals, n_prior, min_group: int = 50) -> MondrianConformal:
        g = self.group(n_prior)
        r = np.asarray(residuals, float)
        pooled = ConformalInterval(self.alpha).fit(r).q
        for k in range(len(self.edges)):
            rk = r[g == k]
            # too few calibration points in a group -> fall back to the pooled width
            self.q[k] = ConformalInterval(self.alpha).fit(rk).q if len(rk) >= min_group else pooled
        return self

    def interval(self, pred, n_prior):
        q = np.array([self.q[k] for k in self.group(n_prior)])
        pred = np.asarray(pred, float)
        return pred - q, pred + q

    def labels(self) -> dict[int, str]:
        e = self.edges + [None]
        return {k: (f"{e[k]}+" if e[k + 1] is None else f"{e[k]}-{e[k + 1] - 1}") for k in range(len(self.edges))}
