# Methodology

Every number shown in the dashboard or said by the assistant comes from one of the calculations below.
Parameters live in [`config.yaml`](../config.yaml).

## 1. Sensor-quality checks (`src/athleteiq/quality.py`)

Per workout: share of HR samples outside 40–220 bpm; HR jumps > 20 bpm between samples; share of speed
samples > 30 km/h; speed jumps > 10 km/h; recording gaps > 60 s; longest run of identical HR.
A penalty score gives `quality_score` ∈ [0, 1] and a label: **good** ≥ 0.85, **check** ≥ 0.6, else **poor**.
On this (pre-cleaned) dataset: 95.5 % good, 4.4 % check, 0.2 % poor.

## 2. Intensity: zones and TRIMP (`intensity.py`)

* **HRmax estimate (online).** For each session, the highest session max HR the athlete has recorded up to
  and including that session; a default of 190 bpm is used as a floor until 5 sessions exist. Future
  sessions never influence an earlier session's zones.
* **Zones** (Edwards, 1993): Z1 50–60 %, Z2 60–70 %, Z3 70–80 %, Z4 80–90 %, Z5 90–100 % of HRmax.
* **Edwards TRIMP** = ∑ₖ minutes in zone k × k. Chosen over Banister TRIMP because it needs no resting HR,
  which this dataset does not contain.

## 3. Personal baselines and anomalies (`baselines.py`)

For each metric (avg HR, speed, efficiency, TRIMP, distance): mean and SD over the athlete's sessions in
the **28 days before** the session (`closed="left"`, ≥ 3 sessions). z = (value − mean) / max(SD, floor).
The floor (e.g. 3 bpm for HR) stops a handful of near-identical sessions producing absurd z-scores.
A session is flagged **unusual** when any |z| > 2. Also `*_last5`: mean of the previous five sessions.

## 4. Training load (`training_load.py`)

Daily load = ∑ TRIMP on that calendar day (0 on days without a recorded run). Acute = 7-day rolling mean,
chronic = 28-day rolling mean (only once 28 days exist), ACWR = acute / chronic. Exponentially weighted
variants use α = 2/(N+1) (Williams et al., 2017). Weekly monotony and strain follow Foster (1998) and are
computed in SQL with DuckDB. The 0.8–1.3 ACWR band in the dashboard is for orientation only: its link to
injury risk is contested (Impellizzeri et al., 2020) and AthleteIQ makes no injury claims.

## 5. Prediction task (`features.py`, `model.py`, `pipeline.py`)

**Question.** Given the workout an athlete is about to do (distance, duration, speed profile, terrain,
time of day) and what is known about them *before* it, what will their average heart rate be?
This is the physiological-response question from the blueprint, framed so the answer is checkable.

**Features (22).** External load: duration, distance, mean / SD / max speed, elevation gain and loss,
hour, weekday. History: average HR and efficiency over the last 5 sessions, 28-day HR baseline,
**personal speed→HR line** fitted on the athlete's earlier sessions and evaluated at today's speed (plus
its slope), log number of prior sessions, days since last session, prior-day acute / chronic load and ACWR.

**Leakage rules** (enforced by tests in `tests/test_model.py`): no feature uses the current session's HR;
history features are shifted / closed-left; the HR-derived workout label is excluded.

**Splits.**

| split | what | sessions |
|---|---|---|
| train | first 70 % of each athlete's sessions (+ all sessions of athletes with < 10) | 22,102 |
| val | next 15 % — early stopping, choosing *k*, plain conformal calibration | 4,563 |
| test | **most recent 15 %** of each athlete's sessions | 4,333 |
| calib | all sessions of 10 % of athletes never trained on — Mondrian conformal calibration only | 4,323 |
| holdout | all sessions of 15 % of athletes never trained on — final evaluation only | 4,865 |

**Models.** (1) global training mean, (2) athlete's last-5 mean, (3) personal speed→HR line,
(4) ridge regression (median imputation + missing-value indicators + scaling, α by internal CV),
(5) LightGBM (L1 objective, early stopping on val), (6) LightGBM + **online personalisation**.

**Online personalisation.** Once a session is recorded its true HR is known, so for each evaluated
session the model adds `offset = Σ earlier residuals / (n + k)` for that athlete — a shrunk running mean
of its own past errors (an empirical-Bayes random intercept learned online). *k* ∈ {1, 2, 5, 10, 20, 50}
was chosen on validation (*k* = 10).

**Prediction intervals.** Split conformal prediction gives distribution-free intervals:
half-width = the ⌈(n+1)(1−α)⌉-th smallest |residual| on a calibration set (α = 0.1 → 90 %).
Validation sessions all come from athletes with long histories, so plain conformal under-covers new
athletes. **Mondrian conformal** fixes this by calibrating a separate width per history band
(0–4, 5–19, 20+ prior sessions) on validation + calibration athletes.

**Explainability.** TreeSHAP on the LightGBM model; global importance = mean |SHAP|, local explanation =
the top contributions for a session relative to the model's average prediction. SHAP describes the model,
not causal physiology.

## 6. Training-pattern clusters (`clustering.py`)

Per athlete (≥ 10 sessions): sessions per active week, median duration, speed, %HRmax, share of time in
Z4–Z5, log hilliness. Standardised, k-means for k = 2…6, k chosen by silhouette. The best k was 2 with a
silhouette of 0.22 — **weak separation**: athletes form a continuum mainly along intensity, so clusters
are shown only as a descriptive label.

## 7. Ask AthleteIQ (`assistant/`)

Five typed tools over the processed tables (`athlete_overview`, `workload_trend`, `session_detail`,
`explain_prediction`, `find_anomalies`). Each returns data plus **citations**: every number it allows the
assistant to say, tagged *observed*, *estimate* or *definition*. Medical / injury / diagnosis questions are
declined by a guard before any tool runs. Two backends share tools and guardrails: a deterministic
router with templated answers (offline, used in CI) and Claude with tool use (`ANTHROPIC_API_KEY`).

**Evals** (`assistant/evals/run_evals.py`) generate questions from the data with expected numbers computed
independently in pandas, and score intent accuracy, numeric recall, unsupported-number rate (numbers in
the answer that no tool returned), refusal precision / recall and citation coverage.

## References

* Edwards S. *The Heart Rate Monitor Book* (1993).
* Foster C. Monitoring training in athletes with reference to overtraining syndrome. *MSSE* (1998).
* Williams S. et al. Better way to determine the acute:chronic workload ratio? *BJSM* (2017).
* Impellizzeri F. et al. Acute:chronic workload ratio: conceptual issues and fundamental pitfalls. *IJSPP* (2020).
* Vovk V., Gammerman A., Shafer G. *Algorithmic Learning in a Random World* (2005) — conformal / Mondrian prediction.
* Lundberg S. et al. From local explanations to global understanding with explainable AI for trees. *Nature MI* (2020).
* Ni J., Muhlstein L., McAuley J. Modeling heart rate and activity data for personalized fitness recommendation. *WWW* (2019).
