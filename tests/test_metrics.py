"""Unit tests for the sports-science calculations (hand-checked numbers)."""
import numpy as np
import pandas as pd
import pytest

from athleteiq.baselines import add_baselines
from athleteiq.intensity import edwards_trimp, online_hrmax, time_in_zones_min, zone_index
from athleteiq.quality import check_workout
from athleteiq.training_load import daily_load, prior_load_features


def test_zone_index_boundaries():
    # HRmax 200: 99 -> Z0, 100 -> Z1 (50%), 139 -> Z2, 180 -> Z5 (90%)
    assert zone_index(np.array([99, 100, 139, 160, 180, 200]), 200).tolist() == [0, 1, 2, 4, 5, 5]


def test_time_in_zones_and_trimp():
    # 10 samples, 60 s apart, all at 85% HRmax -> 9 minutes in Z4 (last sample owns no time)
    ts = np.arange(10) * 60.0
    hr = np.full(10, 170.0)
    z = time_in_zones_min(hr, ts, 200)
    assert z[4] == pytest.approx(9.0) and z.sum() == pytest.approx(9.0)
    assert edwards_trimp(z) == pytest.approx(36.0)  # 9 min * zone 4


def test_time_in_zones_caps_recording_gaps():
    ts = np.array([0.0, 60.0, 3660.0])  # one-hour gap
    z = time_in_zones_min(np.array([150.0, 150.0, 150.0]), ts, 200)
    assert z.sum() == pytest.approx(2.0)  # 60 s + gap capped at 60 s


def test_online_hrmax_never_looks_ahead():
    s = pd.DataFrame({"user_id": 1, "start_time": pd.date_range("2020-01-01", periods=6),
                      "hr_max": [170, 175, 172, 180, 185, 200]})
    est = online_hrmax(s, default=190, min_sessions=5)
    assert est.tolist() == [190, 190, 190, 190, 185, 200]


def test_quality_flags_bad_data():
    ts = np.arange(100) * 5.0
    good = check_workout(np.full(100, 150.0) + np.sin(np.arange(100)), np.full(100, 11.0), ts)
    assert good["quality_label"] == "good"
    hr = np.full(100, 150.0)
    hr[::3] = 250  # out of range and spiky
    bad = check_workout(hr, np.full(100, 45.0), ts)
    assert bad["quality_label"] == "poor"
    assert bad["q_hr_spikes"] > 0 and bad["q_speed_implausible_frac"] == 1.0


def _sessions(n=10):
    return pd.DataFrame({
        "workout_id": range(n), "user_id": 1,
        "start_time": pd.date_range("2020-01-01", periods=n, freq="2D"),
        "hr_mean": [150.0] * (n - 1) + [175.0], "avg_speed_kmh": 11.0, "efficiency_m_per_beat": 1.2,
        "trimp": 100.0, "distance_km": 10.0, "duration_min": 60.0,
    })


def test_baselines_exclude_current_session():
    out = add_baselines(_sessions())
    last = out.iloc[-1]
    assert last.hr_mean_base28 == pytest.approx(150.0)  # baseline built only from earlier sessions
    assert last.hr_mean_z28 > 2 and last.is_anomaly and "hr_mean" in last.anomaly_metrics
    assert pd.isna(out.iloc[0].hr_mean_base28)  # no history -> no baseline


def test_daily_load_acwr_and_prior_shift():
    s = _sessions(30).assign(date=lambda d: d.start_time.dt.normalize())
    d = daily_load(s)
    assert len(d) == 59  # continuous calendar
    assert d.load.sum() == pytest.approx(3000)
    assert d.chronic.iloc[:27].isna().all() and d.chronic.iloc[27:].notna().all()
    pl = prior_load_features(s, d)
    # load known before a session must equal the previous day's acute value, not today's
    day = s.date.iloc[20]
    prev = d[d.date == day - pd.Timedelta(days=1)].acute.iloc[0]
    assert pl[pl.date == day].prior_acute.iloc[0] == pytest.approx(prev)
