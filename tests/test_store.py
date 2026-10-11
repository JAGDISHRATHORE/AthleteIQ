"""Integration checks on the bundled sample of real processed data."""
import json

import numpy as np

from athleteiq.config import path
from athleteiq.features import FEATURES
from athleteiq.store import AthleteStore


def test_sample_tables_load_and_are_consistent():
    st = AthleteStore(path("data/sample"))
    s = st.sessions_all
    assert s.user_id.nunique() >= 8
    assert s.workout_id.is_unique
    assert set(FEATURES) <= set(s.columns)
    assert (s.distance_km > 0).all() and s.hr_mean.between(60, 220).all()
    assert set(st.predictions.workout_id) <= set(s.workout_id)
    # predictions only ever exist for sessions that were not used for training
    assert not (st.predictions.split == "train").any()


def test_prediction_intervals_contain_point_prediction():
    p = AthleteStore(path("data/sample")).predictions
    assert (p.pi_low < p.pred_lgbm_personal).all() and (p.pred_lgbm_personal < p.pi_high).all()


def test_shap_values_add_up_to_prediction():
    """SHAP additivity: base + sum(contributions) == raw LightGBM prediction."""
    st = AthleteStore(path("data/sample"))
    sh = st.shap.set_index("workout_id")
    p = st.predictions.set_index("workout_id")
    w = sh.index[:50]
    recon = st.model_meta["shap_base"] + sh.loc[w, FEATURES].sum(axis=1)
    assert np.allclose(recon, p.loc[w, "pred_lgbm"], atol=0.05)


def test_reported_metrics_show_ml_beats_baselines():
    m = json.load(open(path("models/metrics.json")))
    for split in ["test", "holdout"]:
        r = m["by_split"][split]
        assert r["lgbm"]["mae"] < r["personal_speed_fit"]["mae"] < r["athlete_last5"]["mae"] < r["global_mean"]["mae"]
