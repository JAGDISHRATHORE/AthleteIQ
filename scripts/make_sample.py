"""Create the small bundled sample in data/sample/ so the dashboard, assistant
and tests run straight after cloning, without the 330 MB download.

The sample is a subset of the *real* processed data (12 athletes); it is not
synthetic. Model metrics in the README always come from the full pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from athleteiq.config import load_config, path  # noqa: E402
from athleteiq.data_loader import explode_series, load_raw  # noqa: E402

N_TEST, N_HOLDOUT, SERIES_PER_ATHLETE = 8, 4, 40


def main():
    cfg = load_config()["data"]
    src, dst = path(cfg["processed_dir"]), path(cfg["sample_dir"])
    dst.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(src / "sessions.parquet")
    per = s.groupby("user_id").agg(n=("workout_id", "size"), split=("split", lambda x: set(x)),
                                   cluster=("cluster", "first"))
    rng = np.random.default_rng(7)
    test_pool = per[(per.split.apply(lambda x: "test" in x)) & per.n.between(60, 400)]
    # spread the picks over training-pattern clusters
    picks = []
    for _, g in test_pool.groupby("cluster"):
        per_cluster = -(-N_TEST // test_pool.cluster.nunique())
        picks += rng.choice(g.index, min(len(g), per_cluster), replace=False).tolist()
    picks = picks[:N_TEST]
    hold_pool = per[(per.split.apply(lambda x: "holdout" in x)) & per.n.between(30, 300)]
    picks += rng.choice(hold_pool.index, N_HOLDOUT, replace=False).tolist()
    picks = [int(p) for p in picks]

    for name in ["sessions", "daily_load", "weekly", "predictions", "shap_values"]:
        df = pd.read_parquet(src / f"{name}.parquet")
        if "user_id" in df:
            df = df[df.user_id.isin(picks)]
        else:  # shap_values keyed by workout id
            df = df[df.workout_id.isin(s[s.user_id.isin(picks)].workout_id)]
        df.to_parquet(dst / f"{name}.parquet", index=False)
    for name in ["athlete_profiles", "cluster_summary"]:
        pd.read_parquet(src / f"{name}.parquet").to_parquet(dst / f"{name}.parquet")

    keep = (s[s.user_id.isin(picks)].sort_values("start_time")
            .groupby("user_id").tail(SERIES_PER_ATHLETE).workout_id)
    raw = load_raw()
    series = explode_series(raw[raw.workout_id.isin(keep)])
    series.to_parquet(dst / "series.parquet", index=False)
    print(f"sample: {len(picks)} athletes, {s.user_id.isin(picks).sum()} sessions, "
          f"{len(keep)} workouts with full series -> {dst}")


if __name__ == "__main__":
    main()
