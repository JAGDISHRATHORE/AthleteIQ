"""Generate README / report figures from models/metrics.json and processed data."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from athleteiq.config import data_dir  # noqa: E402

OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, INK = "#2563EB", "#EA580C", "#94A3B8", "#0F172A"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})
NAMES = {"global_mean": "Global mean", "athlete_last5": "Athlete\nlast-5 mean",
         "personal_speed_fit": "Personal\nspeed-HR fit", "ridge": "Ridge",
         "lgbm": "LightGBM", "lgbm_personal": "LightGBM +\npersonalisation"}

m = json.load(open(ROOT / "models" / "metrics.json"))


def model_comparison():
    keys = list(NAMES)
    test = [m["by_split"]["test"][k]["mae"] for k in keys]
    hold = [m["by_split"]["holdout"][k]["mae"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(keys))
    b1 = ax.bar([i - 0.2 for i in x], test, 0.4, color=BLUE, label="Future sessions of known athletes")
    b2 = ax.bar([i + 0.2 for i in x], hold, 0.4, color=ORANGE, label="Athletes never seen in training")
    ax.bar_label(b1, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, fmt="%.1f", fontsize=9)
    ax.set_xticks(list(x), [NAMES[k] for k in keys], fontsize=9.5)
    ax.set_ylabel("MAE (bpm, lower is better)")
    ax.set_title("Predicting a session's average heart rate")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png")


def coverage():
    rows = pd.DataFrame(m["error_analysis"]["holdout"]["history_bucket"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], rows.coverage_plain * 100, 0.4, color=GREY, label="Standard split conformal")
    ax.bar([i + 0.2 for i in x], rows.coverage * 100, 0.4, color=BLUE, label="Mondrian conformal (by history)")
    ax.axhline(90, color=INK, ls="--", lw=1)
    ax.text(len(rows) - 0.5, 90.6, "target 90%", ha="right", fontsize=9)
    ax.set_xticks(list(x), rows.history_bucket)
    ax.set_xlabel("athlete's prior sessions")
    ax.set_ylim(60, 100)
    ax.set_ylabel("interval coverage (%)")
    ax.set_title("90% intervals on brand-new athletes")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "interval_coverage.png", bbox_inches="tight")


def learning_curve():
    lc = pd.DataFrame(m["holdout_learning_curve"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(lc.n, lc["global"], color=GREY, marker="o", label="Global model")
    ax.plot(lc.n, lc["personal"], color=ORANGE, marker="o", label="+ online personalisation")
    ax.set_xlabel("prior sessions (bucket; 60 = 60+)")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("New athletes: error as their history grows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "cold_start_curve.png")


def shap_importance():
    gi = pd.DataFrame(m["global_importance"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(gi.label, gi.mean_abs_shap_bpm, color=BLUE)
    ax.set_xlabel("mean |SHAP| (bpm)")
    ax.set_title("What the model relies on")
    fig.tight_layout()
    fig.savefig(OUT / "shap_importance.png")


def data_overview():
    s = pd.read_parquet(data_dir() / "sessions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    s.groupby("user_id").size().clip(upper=300).plot.hist(bins=30, ax=axes[0], color=BLUE)
    axes[0].set_title("Sessions per athlete")
    axes[0].set_xlabel("sessions (300 = 300+)")
    axes[1].hist(s.hr_mean, bins=40, color=ORANGE)
    axes[1].set_title("Session average HR")
    axes[1].set_xlabel("bpm")
    zones = s[[f"z{k}_min" for k in range(1, 6)]].sum()
    axes[2].bar(["Z1", "Z2", "Z3", "Z4", "Z5"], zones / zones.sum() * 100, color=INK)
    axes[2].set_title("Share of time in HR zones")
    axes[2].set_ylabel("%")
    for a in axes[:2]:
        a.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT / "data_overview.png")


if __name__ == "__main__":
    for f in [model_comparison, coverage, learning_curve, shap_importance, data_overview]:
        f()
        print("wrote", f.__name__)
