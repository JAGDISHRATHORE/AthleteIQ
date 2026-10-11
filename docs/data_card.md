# Data card

## Source

| | |
|---|---|
| Original dataset | **FitRec / Endomondo** workout data — Ni, Muhlstein & McAuley, *Modeling Heart Rate and Activity Data for Personalized Fitness Recommendation*, The Web Conference (WWW) 2019. Collected from the public Endomondo fitness platform. |
| Version used | Hugging Face re-packaging [`rricc22/endomondo-hr-prediction-v2`](https://huggingface.co/datasets/rricc22/endomondo-hr-prediction-v2) (listed under the MIT licence by its packager) |
| Size | 40,186 running workouts · 761 athletes · 2007-02 to 2016-01 |
| Per workout | 500 time-aligned samples of heart rate (bpm), speed (km/h), altitude (m), Unix timestamp; user id; packager's workout label |
| Access | `python scripts/run_pipeline.py download` (Hub file, falling back to the Hub rows API) |

**Licence note.** The original FitRec data was released by the UCSD McAuley lab for research use. This
repository does not redistribute the raw data; it downloads it. The bundled `data/sample/` folder holds
*derived* tables for 12 athletes (session summaries and the series of their last 40 workouts) so the
dashboard and tests run out of the box, with attribution above. If you reuse the data, cite the paper.

## What the packager already did (important)

The re-packaged version is **not raw device output**:

1. Workouts resampled to exactly 500 points.
2. Series smoothed with a 7-point moving average — this removes most HR spikes, so our spike check rarely fires.
3. Workouts **dropped** unless: mean HR ≥ 120 bpm, max HR ≤ 200 bpm, HR std ≥ 5 bpm, speed–HR correlation ≥ −0.3 (6,064 removed).
4. A `workout_type` label (RECOVERY / STEADY / INTENSIVE) was added. It is derived from heart rate, so AthleteIQ never uses it as a model input — only to slice errors.

## Variables AthleteIQ derives

| Variable | Definition |
|---|---|
| `duration_min`, `moving_min` | last − first timestamp; sum of sample intervals with gaps capped at 60 s |
| `distance_km` | ∑ speed × Δt (gaps capped at 60 s) |
| `avg_speed_kmh`, `pace_min_per_km` | time-weighted mean speed; 60 / speed |
| `elev_gain_m`, `elev_loss_m` | sum of positive / negative altitude changes |
| `hr_mean`, `hr_max`, `hr_std`, `hr_drift` | time-weighted mean HR; second-half minus first-half mean |
| `efficiency_m_per_beat` | metres covered per heartbeat = (speed in m/min) / mean HR |
| `hrmax_est` | highest session max HR seen *so far* for the athlete (default 190 until 5 sessions) |
| `z0_min` … `z5_min`, `trimp` | minutes per Edwards %HRmax zone; Edwards TRIMP = ∑ minutes × zone number |
| `q_*`, `quality_score`, `quality_label` | sensor-quality checks (see methodology) |
| `*_base28`, `*_z28`, `*_last5` | personal baselines from *earlier* sessions only |
| daily `load`, `acute`, `chronic`, `acwr`, `ewma_*` | per-day TRIMP; 7- and 28-day rolling means; their ratio |
| weekly `monotony`, `strain` | Foster (1998): mean/sd of daily load; weekly load × monotony |

## Known limitations

* **Running only, HR + GPS only.** No sleep, HRV, age, sex, body mass, lab HRmax or perceived exertion. A
  validated readiness or recovery score is therefore *not possible* and is not attempted.
* **Selection bias.** Mean HR < 120 bpm workouts were removed by the packager: very easy runs are missing.
* **Sparse recording.** Median 4 days between recorded runs, but long gaps are common. Any workout not
  recorded with HR is invisible, so training loads are lower bounds, and ACWR is noisy.
* **Estimated HRmax.** Zones and TRIMP depend on an HRmax estimate from observed data; smoothing lowers
  observed peaks, so zones may be shifted slightly upward.
* **Historical, anonymous, self-selected users** of one app — not representative of all runners.
* **No ground-truth labels** for injury, illness or performance outcomes.
