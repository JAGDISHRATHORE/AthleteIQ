"""Tests for leakage-safe splitting, features, personalisation and conformal intervals."""
import numpy as np
import pandas as pd
import pytest

from athleteiq.features import FEATURES, _expanding_speed_hr_fit
from athleteiq.model import (
    ConformalInterval,
    MondrianConformal,
    assign_splits,
    online_offsets,
    regression_metrics,
)


def _many_athletes(n_athletes=40, n_sessions=30):
    rows = []
    for u in range(n_athletes):
        for i in range(n_sessions):
            rows.append({"workout_id": u * 1000 + i, "user_id": u,
                         "start_time": pd.Timestamp("2014-01-01") + pd.Timedelta(days=3 * i + u)})
    return pd.DataFrame(rows)


def test_splits_are_chronological_per_athlete():
    df = _many_athletes()
    df["split"] = assign_splits(df)
    assert set(df.split) == {"train", "val", "test", "holdout", "calib"}
    for _, g in df[df.split.isin(["train", "val", "test"])].groupby("user_id"):
        assert g[g.split == "train"].start_time.max() < g[g.split == "val"].start_time.min()
        assert g[g.split == "val"].start_time.max() < g[g.split == "test"].start_time.min()


def test_holdout_athletes_never_in_training():
    df = _many_athletes()
    df["split"] = assign_splits(df)
    unseen = set(df[df.split.isin(["holdout", "calib"])].user_id)
    assert unseen and not unseen & set(df[df.split == "train"].user_id)
    assert not set(df[df.split == "holdout"].user_id) & set(df[df.split == "calib"].user_id)


def test_feature_list_has_no_target_leak():
    banned = {"hr_mean", "hr_max", "hr_min", "hr_std", "trimp", "pct_hrmax_mean", "label_workout_type",
              "efficiency_m_per_beat", "hr_mean_z28", "hr_drift"}
    assert not banned & set(FEATURES)
    assert not [f for f in FEATURES if f.startswith("z") and f.endswith("_min")]


def test_expanding_fit_uses_only_prior_sessions():
    speed = np.array([10, 11, 12, 10, 11, 12, 13, 9], float)
    g = pd.DataFrame({"avg_speed_kmh": speed, "hr_mean": 100 + 4 * speed})
    out = _expanding_speed_hr_fit(g, min_n=5)
    assert out.prior_hr_at_speed.iloc[:5].isna().all()
    assert out.prior_hr_speed_slope.iloc[5] == pytest.approx(4.0)
    assert out.prior_hr_at_speed.iloc[7] == pytest.approx(136.0)
    # changing the current session's HR must not change its own feature
    g2 = g.copy()
    g2.loc[7, "hr_mean"] = 999
    assert _expanding_speed_hr_fit(g2, 5).prior_hr_at_speed.iloc[7] == pytest.approx(136.0)


def test_online_offsets_use_only_earlier_residuals():
    df = pd.DataFrame({"user_id": [1, 1, 1, 2], "start_time": pd.date_range("2020", periods=4)})
    res = pd.Series([4.0, 4.0, 100.0, 7.0])
    off = online_offsets(df, res, k=1)
    assert off.tolist() == [0.0, 2.0, pytest.approx(8 / 3), 0.0]


def test_conformal_coverage_on_synthetic_data():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(0, 5, 2000), rng.normal(0, 5, 20000)
    ci = ConformalInterval(0.1).fit(cal)
    lo, hi = ci.interval(np.zeros_like(test))
    assert 0.88 <= ConformalInterval.coverage(test, lo, hi) <= 0.92


def test_mondrian_gives_wider_intervals_to_noisier_group():
    rng = np.random.default_rng(1)
    n_prior = np.r_[np.zeros(3000), np.full(3000, 30)]
    resid = np.r_[rng.normal(0, 10, 3000), rng.normal(0, 3, 3000)]
    m = MondrianConformal(0.1, [0, 5, 20]).fit(resid, n_prior)
    assert m.q[0] > 2 * m.q[2]
    lo, hi = m.interval(np.zeros(2), [0, 40])
    assert hi[0] > hi[1]


def test_regression_metrics():
    m = regression_metrics([100, 110, 120], [101, 108, 120])
    assert m["mae"] == pytest.approx(1.0) and m["within_5bpm"] == 1.0
