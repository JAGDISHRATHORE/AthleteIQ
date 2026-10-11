# Model card — session heart-rate model

| | |
|---|---|
| Model | LightGBM regressor (L1 loss, 429 trees, early stopping) + online per-athlete offset (k = 10) + Mondrian split-conformal 90 % intervals |
| Output | Expected **average heart rate (bpm)** for a running session, with a 90 % interval |
| Inputs | 22 features: the workout's external load and the athlete's history before it (see methodology) |
| Training data | FitRec / Endomondo running workouts, 2007–2016 (see data card) |
| Files | `models/lgbm_hr.txt`, `models/ridge_hr.joblib`, `models/model_meta.json`, `models/metrics.json` |

## Intended use

Portfolio and research demonstration of athlete-specific physiological-response modelling: e.g. "how
hard should this planned run be for this athlete's heart, and is today's response unusual for them?"

**Not for:** medical decisions, diagnosing cardiac conditions, injury prediction, or any use with
athletes unlike the training population (non-runners, children, clinical populations).

## Results (all measured, `models/metrics.json`)

| Model | Future sessions MAE | New athletes MAE |
|---|---|---|
| Global mean | 10.37 | 9.31 |
| Athlete last-5 mean | 7.21 | 7.53 |
| Personal speed→HR fit | 5.77 | 6.13 |
| Ridge | 5.05 | 5.44 |
| LightGBM | 4.85 | 5.23 |
| **LightGBM + personalisation** | **4.75** (R² 0.755, 63 % within 5 bpm) | **5.23** (R² 0.650, 59 % within 5 bpm) |

90 % interval coverage: **89.2 %** on future sessions, **87.3 %** on new athletes. For new athletes'
first five sessions, Mondrian intervals cover **88.7 %** vs **77.4 %** for plain split conformal.

## Known failure modes

* **Interval / high-intensity sessions**: MAE ≈ 8 bpm and only ≈ 68 % coverage — average speed does not
  capture intermittent effort. These are ≈ 5 % of sessions.
* **First few sessions of a new athlete**: MAE 6.6 bpm (vs ≈ 5.0 with 50+ prior sessions); intervals widen to
  ±13 bpm to stay honest.
* Coverage on new athletes is slightly below nominal (87 %) because errors are correlated within an athlete,
  which breaks the exchangeability conformal prediction assumes.
* Weather, temperature, illness, sleep, caffeine and pacing strategy are unobserved and absorbed as error.

## Ethics and responsible use

Data is anonymised and public but came from real people; do not attempt re-identification. Heart-rate
deviations can have medical causes — the assistant declines medical questions and the dashboard labels
estimates as estimates. SHAP explanations describe the model, not the athlete's physiology.
