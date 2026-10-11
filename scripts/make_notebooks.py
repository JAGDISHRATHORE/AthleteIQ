"""Build and execute the two analysis notebooks (outputs are saved in the .ipynb files)."""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import sys, json
sys.path.insert(0, '../src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from athleteiq.config import data_dir
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
D = data_dir(); print('data:', D.relative_to(D.parents[1]))"""

eda = [
    md("# 01 · Data exploration\n\nPublic FitRec / Endomondo running workouts (see `docs/data_card.md`). "
       "Goal: understand the data, check its quality and write down its limitations **before** modelling."),
    code(SETUP),
    code("s = pd.read_parquet(D / 'sessions.parquet')\n"
         "print(f'{len(s):,} workouts · {s.user_id.nunique()} athletes · {s.start_time.min():%Y-%m-%d} to {s.start_time.max():%Y-%m-%d}')\n"
         "s[['duration_min','distance_km','avg_speed_kmh','elev_gain_m','hr_mean','hr_max','trimp']].describe().round(1)"),
    md("## How much history does each athlete have?\nPersonal baselines need history, so this decides what is possible per athlete."),
    code("n = s.groupby('user_id').size()\nprint(n.describe().round(1))\n"
         "print('athletes with >=10 sessions:', (n>=10).sum(), '| >=50:', (n>=50).sum())\n"
         "n.clip(upper=300).plot.hist(bins=40, title='Sessions per athlete (300 = 300+)');"),
    md("## Recording is sparse\nGaps between recorded runs are long and irregular, so 'days without a run' "
       "often means 'not recorded with HR', not rest. Training loads are therefore lower bounds."),
    code("gap = s.days_since_last.dropna()\nprint(gap.describe(percentiles=[.25,.5,.75,.9,.99]).round(1))\n"
         "gap.clip(upper=30).plot.hist(bins=30, title='Days since previous recorded run (30 = 30+)');"),
    md("## Data quality checks\nThe packager already smoothed the series (7-point moving average) and dropped "
       "implausible workouts, so spikes are rare by construction. Our checks still catch recording gaps and frozen HR."),
    code("print(s.quality_label.value_counts())\n"
         "s[['q_gaps','q_max_gap_s','q_hr_flatline_s','q_gps_jumps','quality_score']].describe().round(2)"),
    md("## Selection bias to remember\nThe packager kept only workouts with mean HR >= 120 bpm and max <= 200 bpm, "
       "so very easy runs are missing. The model is only valid inside this range."),
    code("fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\n"
         "s.hr_mean.plot.hist(bins=50, ax=ax[0], title='Session average HR (bpm)')\n"
         "ax[1].scatter(s.avg_speed_kmh, s.hr_mean, s=2, alpha=.15)\n"
         "ax[1].set(xlabel='avg speed (km/h)', ylabel='avg HR', title='Speed vs HR across all athletes');"),
    md("Across athletes the speed–HR cloud is wide: fitness differs. **Within** an athlete the relationship is "
       "much tighter, which is why personal history features matter."),
    code("top = s.groupby('user_id').size().sort_values().index[-6:]\n"
         "fig, ax = plt.subplots(figsize=(7, 4))\n"
         "for u in top:\n    g = s[s.user_id == u]; ax.scatter(g.avg_speed_kmh, g.hr_mean, s=6, label=str(u))\n"
         "ax.set(xlabel='avg speed (km/h)', ylabel='avg HR (bpm)', title='Six athletes: within-athlete speed-HR'); ax.legend(fontsize=7);"),
    code("w = s.groupby('user_id').apply(lambda g: g[['avg_speed_kmh','hr_mean']].corr().iloc[0,1] if len(g)>=20 else np.nan, include_groups=False)\n"
         "print('median within-athlete corr(speed, HR):', round(w.median(), 2), '| pooled corr:', round(s[['avg_speed_kmh','hr_mean']].corr().iloc[0,1], 2))"),
    md("## Intensity and load\nZones are % of an **estimated** HRmax (highest HR seen so far); there is no lab test."),
    code("z = s[[f'z{k}_min' for k in range(1,6)]].sum(); (z/z.sum()*100).round(1).rename('share of time %')"),
    code("dl = pd.read_parquet(D / 'daily_load.parquet')\nu = n.idxmax(); g = dl[dl.user_id==u].tail(240)\n"
         "fig, ax = plt.subplots(figsize=(11, 3.5)); ax.bar(g.date, g.load, color='#CBD5E1', label='daily TRIMP')\n"
         "ax.plot(g.date, g.acute, label='acute 7d'); ax.plot(g.date, g.chronic, label='chronic 28d'); ax.legend(); ax.set_title(f'Athlete {u}: training load');"),
    md("## Limitations written down before modelling\n"
       "- Running only; no sleep, HRV, age, sex or lab HRmax → no validated readiness score is possible.\n"
       "- Series were resampled to 500 points and smoothed by the packager.\n"
       "- Mean HR ≥ 120 bpm filter removes easy runs (selection bias).\n"
       "- Workouts not recorded with HR are invisible → load is a lower bound.\n"
       "- The RECOVERY / STEADY / INTENSIVE label is HR-derived → never used as a feature."),
]

models = [
    md("# 02 · Model experiments\n\nTask: predict a session's **average heart rate** from the workout's external load "
       "and what was known about the athlete before it. Full code: `src/athleteiq/model.py`, `pipeline.py`."),
    code(SETUP + "\nfrom athleteiq.features import FEATURES, PRETTY\nm = json.load(open('../models/metrics.json'))"),
    md("## Splits (no leakage)\n- **train / val / test**: the first 70 % / next 15 % / most recent 15 % of each athlete's sessions.\n"
       "- **holdout**: 15 % of athletes never seen in training.\n- **calib**: 10 % more unseen athletes, used only to calibrate intervals."),
    code("pd.Series(m['splits']).rename('sessions')"),
    md("## Baselines first, then ML"),
    code("rows = []\nfor sp in ['val','test','holdout']:\n    for k, v in m['by_split'][sp].items():\n"
         "        if isinstance(v, dict): rows.append({'split': sp, 'model': k, **{x: round(v[x], 3) for x in ['mae','rmse','r2','within_5bpm']}})\n"
         "pd.DataFrame(rows).pivot(index='model', columns='split', values='mae').loc[['global_mean','athlete_last5','personal_speed_fit','ridge','lgbm','lgbm_personal']]"),
    code("t = m['by_split']['test']\nprint(f\"LightGBM+personalisation vs athlete last-5 baseline: {1 - t['lgbm_personal']['mae']/t['athlete_last5']['mae']:.0%} lower MAE\")\n"
         "print(f\"LightGBM+personalisation vs personal speed-HR fit: {1 - t['lgbm_personal']['mae']/t['personal_speed_fit']['mae']:.0%} lower MAE\")"),
    md("## Online personalisation\nAfter each evaluated session the true HR is known, so the model can learn an "
       "athlete-specific offset: shrunk mean of past errors, `sum(residuals)/(n+k)`. `k` was chosen on validation."),
    code("print('k chosen:', m['best_shrinkage_k']); pd.Series(m['val_mae_by_k']).rename('val MAE')"),
    code("lc = pd.DataFrame(m['holdout_learning_curve'])\nax = lc.plot(x='n', y=['global','personal'], marker='o', figsize=(7,3.5), title='New athletes: MAE vs prior sessions')\nax.set_ylabel('MAE (bpm)');"),
    md("Honest result: personalisation helps on known athletes' future sessions (≈0.1 bpm) but gives no measurable "
       "gain on brand-new athletes - the history features already capture most of what the offset would learn."),
    md("## Uncertainty: conformal prediction intervals\nPlain split conformal (calibrated on validation sessions) "
       "under-covers brand-new athletes, because validation sessions all have long histories. "
       "Mondrian conformal calibrates a separate width per history band using the calibration athletes."),
    code("print(json.dumps(m['conformal'], indent=1))\n"
         "pd.DataFrame(m['error_analysis']['holdout']['history_bucket'])"),
    code("for sp in ['test','holdout']: print(sp, 'coverage plain:', round(m['by_split'][sp]['interval_coverage_plain'],3), '| mondrian:', round(m['by_split'][sp]['interval_coverage'],3))"),
    md("## Error analysis"),
    code("pd.DataFrame(m['error_analysis']['test']['label_workout_type'])"),
    md("Intensive sessions are hardest (MAE ≈ 8 bpm) and under-covered: the workout's speed alone does not reveal "
       "how hard an interval session felt, and those sessions are rare (≈5 %)."),
    code("pd.DataFrame(m['error_analysis']['test']['duration_bucket'])"),
    md("## Explainability (SHAP)"),
    code("gi = pd.DataFrame(m['global_importance'])\ngi.head(12)[['label','mean_abs_shap_bpm']]"),
    code("sh = pd.read_parquet(D / 'shap_values.parquet'); s = pd.read_parquet(D / 'sessions.parquet').set_index('workout_id')\n"
         "x = s.loc[sh.workout_id, 'avg_speed_kmh'].to_numpy()\nfig, ax = plt.subplots(figsize=(6,3.5)); ax.scatter(x, sh.avg_speed_kmh, s=2, alpha=.2)\n"
         "ax.set(xlabel='avg speed (km/h)', ylabel='SHAP (bpm)', title='Dependence: speed (holding personal expectation fixed)');"),
    md("The single most useful input is the athlete's **own** speed–HR relationship from earlier sessions. Speed then "
       "adjusts it. These are model associations, not causal physiology."),
]


def run(cells, name):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbf.write(nb, ROOT / "notebooks" / name)
    print("wrote", name)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    run(eda, "01_data_exploration.ipynb")
    run(models, "02_model_experiments.ipynb")
