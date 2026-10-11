"""End-to-end pipeline: raw data -> processed tables -> models -> reports."""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from .baselines import add_baselines
from .clustering import athlete_profiles, cluster_athletes
from .config import load_config, path
from .data_loader import load_raw
from .explainability import global_importance, shap_values
from .features import FEATURES, TARGET, build_feature_table
from .model import (
    ConformalInterval,
    MondrianConformal,
    assign_splits,
    fit_lgbm,
    fit_ridge,
    online_offsets,
    predict_baselines,
    regression_metrics,
)
from .preprocessing import build_sessions
from .training_load import daily_load, weekly_summary


def _out():
    d = path(load_config()["data"]["processed_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def build() -> None:
    t0 = time.time()
    raw = load_raw()
    print(f"loaded {len(raw):,} workouts from {raw.user_id.nunique()} athletes")
    sessions = add_baselines(build_sessions(raw))
    daily = daily_load(sessions)
    weekly = weekly_summary(daily)
    feats = build_feature_table(sessions, daily)
    feats["split"] = assign_splits(feats)
    profiles, cluster_summary, sil = cluster_athletes(athlete_profiles(sessions))
    feats = feats.merge(profiles[["cluster", "cluster_description"]], left_on="user_id",
                        right_index=True, how="left")

    out = _out()
    feats.to_parquet(out / "sessions.parquet", index=False)
    daily.to_parquet(out / "daily_load.parquet", index=False)
    weekly.to_parquet(out / "weekly.parquet", index=False)
    profiles.to_parquet(out / "athlete_profiles.parquet")
    cluster_summary.to_parquet(out / "cluster_summary.parquet")
    json.dump({"silhouette_by_k": sil}, open(out / "clustering.json", "w"), indent=2)
    print(f"built processed tables in {time.time() - t0:.0f}s -> {out}")


def train() -> dict:
    cfg = load_config()["model"]
    out = _out()
    models_dir = path("models")
    models_dir.mkdir(exist_ok=True)
    df = pd.read_parquet(out / "sessions.parquet")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ev = df[df.split.isin(["val", "calib", "test", "holdout"])].copy()

    global_mean = float(tr[TARGET].mean())
    preds = predict_baselines(ev, global_mean)
    ridge = fit_ridge(tr)
    preds["pred_ridge"] = ridge.predict(ev[FEATURES])
    lgbm = fit_lgbm(tr, va, cfg["random_state"])
    preds["pred_lgbm"] = lgbm.predict(ev[FEATURES])

    # online personalisation on top of LightGBM; k chosen on validation only
    resid = ev[TARGET] - preds["pred_lgbm"]
    best_k, best_mae, k_scores = None, np.inf, {}
    for k in cfg["shrinkage_grid"]:
        off = online_offsets(ev, resid, k)
        mae = float(np.mean(np.abs(resid - off)[ev.split == "val"]))
        k_scores[k] = mae
        if mae < best_mae:
            best_k, best_mae = k, mae
    preds["personal_offset"] = online_offsets(ev, resid, best_k)
    preds["pred_lgbm_personal"] = preds["pred_lgbm"] + preds["personal_offset"]

    # 90% intervals. (a) plain split conformal calibrated on validation sessions;
    # (b) Mondrian conformal by history length, calibrated on val + calibration athletes.
    final_resid = (ev[TARGET] - preds["pred_lgbm_personal"]).to_numpy()
    is_val = (ev.split == "val").to_numpy()
    is_cal = ev.split.isin(["val", "calib"]).to_numpy()
    conf = ConformalInterval(cfg["conformal_alpha"]).fit(final_resid[is_val])
    preds["pi_plain_low"], preds["pi_plain_high"] = conf.interval(preds["pred_lgbm_personal"].to_numpy())
    mond = MondrianConformal(cfg["conformal_alpha"], cfg["conformal_groups"]).fit(
        final_resid[is_cal], ev["session_idx"].to_numpy()[is_cal])
    preds["pi_low"], preds["pi_high"] = mond.interval(preds["pred_lgbm_personal"], ev["session_idx"])

    ev = pd.concat([ev, preds], axis=1)
    model_cols = ["pred_global_mean", "pred_athlete_last5", "pred_personal_speed_fit",
                  "pred_ridge", "pred_lgbm", "pred_lgbm_personal"]
    metrics: dict = {"target": "session mean heart rate (bpm)", "n_features": len(FEATURES),
                     "best_shrinkage_k": best_k, "val_mae_by_k": k_scores,
                     "lgbm_best_iteration": int(lgbm.best_iteration_ or 0),
                     "conformal": {"alpha": cfg["conformal_alpha"], "plain_half_width_bpm": conf.q,
                                   "mondrian_half_width_bpm": {mond.labels()[k]: v for k, v in mond.q.items()}},
                     "splits": df.split.value_counts().to_dict(), "by_split": {}}
    for sp in ["val", "test", "holdout"]:
        e = ev[ev.split == sp]
        metrics["by_split"][sp] = {c.replace("pred_", ""): regression_metrics(e[TARGET], e[c])
                                   for c in model_cols}
        metrics["by_split"][sp]["interval_coverage"] = ConformalInterval.coverage(
            e[TARGET], e["pi_low"], e["pi_high"])
        metrics["by_split"][sp]["interval_coverage_plain"] = ConformalInterval.coverage(
            e[TARGET], e["pi_plain_low"], e["pi_plain_high"])

    # error analysis on the test split and on brand-new athletes
    ev["abs_err"] = (ev[TARGET] - ev["pred_lgbm_personal"]).abs()
    ev["covered"] = (ev[TARGET] >= ev["pi_low"]) & (ev[TARGET] <= ev["pi_high"])
    ev["covered_plain"] = (ev[TARGET] >= ev["pi_plain_low"]) & (ev[TARGET] <= ev["pi_plain_high"])
    ev["history_bucket"] = pd.cut(ev["session_idx"], [-1, 4, 19, 49, 10_000],
                                  labels=["0-4", "5-19", "20-49", "50+"])
    ev["duration_bucket"] = pd.cut(ev["duration_min"], [0, 30, 60, 90, 1_000],
                                   labels=["<30", "30-60", "60-90", "90+"])
    err = {}
    for sp in ["test", "holdout"]:
        e = ev[ev.split == sp]
        err[sp] = {
            by: e.groupby(by, observed=True).agg(n=("abs_err", "size"), mae=("abs_err", "mean"),
                                                coverage=("covered", "mean"),
                                                coverage_plain=("covered_plain", "mean")).round(3).reset_index()
                 .to_dict(orient="records")
            for by in ["label_workout_type", "history_bucket", "duration_bucket", "quality_label"]
        }
    metrics["error_analysis"] = err

    # cold-start learning curve on held-out athletes: global vs personalised MAE by session number
    h = ev[ev.split == "holdout"].copy()
    h["n"] = h["session_idx"].clip(upper=60) // 5 * 5
    curve = h.groupby("n").apply(lambda g: pd.Series({
        "global": np.mean(np.abs(g[TARGET] - g["pred_lgbm"])),
        "personal": np.mean(np.abs(g[TARGET] - g["pred_lgbm_personal"])),
        "sessions": len(g)}), include_groups=False).reset_index()
    metrics["holdout_learning_curve"] = curve.round(3).to_dict(orient="records")

    # explanations
    shap_df, base = shap_values(lgbm, ev)
    imp = global_importance(shap_df[ev.split.isin(["test", "holdout"]).to_numpy()])
    metrics["shap_base_bpm"] = base
    metrics["global_importance"] = imp.round(3).to_dict(orient="records")

    keep = ["workout_id", "user_id", "start_time", "split", TARGET, "label_workout_type",
            "session_idx", "quality_label"] + model_cols + ["personal_offset", "pi_low", "pi_high",
                                                          "pi_plain_low", "pi_plain_high"]
    ev[keep].to_parquet(out / "predictions.parquet", index=False)
    shap_df.assign(workout_id=ev["workout_id"].to_numpy()).to_parquet(out / "shap_values.parquet", index=False)
    lgbm.booster_.save_model(str(models_dir / "lgbm_hr.txt"))
    joblib.dump(ridge, models_dir / "ridge_hr.joblib")
    json.dump({"features": FEATURES, "global_mean": global_mean, "shrinkage_k": best_k,
               "conformal_q_plain": conf.q,
               "conformal_q_by_history": {mond.labels()[k]: v for k, v in mond.q.items()},
               "shap_base": base},
              open(models_dir / "model_meta.json", "w"), indent=2)
    json.dump(metrics, open(models_dir / "metrics.json", "w"), indent=2, default=str)
    print(json.dumps({sp: {m: round(v["mae"], 2) for m, v in d.items() if isinstance(v, dict)}
                      for sp, d in metrics["by_split"].items()}, indent=1))
    return metrics
