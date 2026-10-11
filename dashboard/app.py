"""AthleteIQ dashboard.   streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from assistant.sports_ai import SportsAI  # noqa: E402
from assistant.tools import AthleteTools  # noqa: E402
from athleteiq.intensity import ZONE_EDGES  # noqa: E402
from athleteiq.store import AthleteStore  # noqa: E402

st.set_page_config(page_title="AthleteIQ", page_icon="📈", layout="wide")

ACCENT, ACCENT2, MUTED = "#2563EB", "#EA580C", "#94A3B8"
ZONE_COLORS = ["#E2E8F0", "#BFDBFE", "#93C5FD", "#FDE68A", "#FDBA74", "#FCA5A5"]
LAYOUT = dict(margin=dict(l=10, r=10, t=80, b=10), hovermode="x unified",
              title_y=0.97, legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
              template="plotly_white")
MODEL_NAMES = {"global_mean": "Global mean", "athlete_last5": "Athlete last-5 mean",
               "personal_speed_fit": "Personal speed-HR fit", "ridge": "Ridge regression",
               "lgbm": "LightGBM", "lgbm_personal": "LightGBM + online personalisation"}


@st.cache_resource
def get_store() -> AthleteStore:
    return AthleteStore()


@st.cache_resource
def get_ai() -> SportsAI:
    return SportsAI(AthleteTools(get_store()))


store = get_store()
athletes = store.athletes()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("AthleteIQ")
    st.caption("Sports-performance intelligence from public wearable data")
    labels = {uid: f"Athlete {uid} · {int(r.sessions)} runs" + (" · predicted" if r.has_predictions else "")
              for uid, r in athletes.iterrows()}
    uid = st.selectbox("Athlete", list(labels), format_func=labels.get)
    a = athletes.loc[uid]
    st.markdown(f"**Record:** {a['first']:%d %b %Y} – {a['last']:%d %b %Y}  \n**Distance:** {a.total_km:,.0f} km")
    if isinstance(a.cluster_description, str):
        st.markdown(f"**Training pattern:** {a.cluster_description}")
    st.divider()
    src = "bundled sample (12 real athletes)" if store.is_sample else "full processed dataset"
    st.caption(f"Data: FitRec / Endomondo running workouts, {src}. Historical data (2007–2016). "
               "Not medical advice. HRmax is estimated, not lab-measured.")

S = store.sessions(uid)
D = store.daily(uid)
W = store.weekly(uid)
P = store.predicted_sessions(uid)

tabs = st.tabs(["Overview", "Workout explorer", "Training load", "Prediction", "Model report", "Ask AthleteIQ"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    last = S.start_time.max()
    r28 = S[S.start_time > last - pd.Timedelta(days=28)]
    p28 = S[(S.start_time <= last - pd.Timedelta(days=28)) & (S.start_time > last - pd.Timedelta(days=56))]
    c = st.columns(5)
    c[0].metric("Sessions", f"{len(S)}")
    c[1].metric("Last 28 days", f"{len(r28)} runs", f"{len(r28) - len(p28):+d} vs prior 28 d")
    c[2].metric("Distance, last 28 d", f"{r28.distance_km.sum():.1f} km",
                f"{r28.distance_km.sum() - p28.distance_km.sum():+.1f} km")
    c[3].metric("Avg HR, last 28 d", f"{r28.hr_mean.mean():.0f} bpm" if len(r28) else "–")
    eff_r, eff_p = r28.efficiency_m_per_beat.mean(), p28.efficiency_m_per_beat.mean()
    c[4].metric("Efficiency (m/beat)", f"{eff_r:.3f}" if len(r28) else "–",
                f"{eff_r - eff_p:+.3f}" if len(r28) and len(p28) else None)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(x=W.week, y=W.weekly_load, name="Weekly load (TRIMP)", marker_color=ACCENT)
        fig.update_layout(title="Weekly training load", height=330, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=S.start_time, y=S.efficiency_m_per_beat, mode="markers", name="Session",
                        marker=dict(size=5, color=MUTED))
        fig.add_scatter(x=S.start_time, y=S.efficiency_m_per_beat_base28, name="Prior 28-day baseline",
                        line=dict(color=ACCENT2, width=2))
        fig.update_layout(title="Running efficiency (metres per heartbeat)", height=330, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    st.caption("Efficiency = average speed ÷ average heart rate. Rising efficiency means more speed for the same "
               "cardiac cost; it is also affected by terrain, weather and pacing, which the data does not record.")

    recent = S.sort_values("start_time", ascending=False).head(10)
    st.subheader("Recent sessions")
    st.dataframe(pd.DataFrame({
        "Date": recent.start_time.dt.date, "Distance (km)": recent.distance_km.round(2),
        "Pace (min/km)": recent.pace_min_per_km.round(2), "Avg HR": recent.hr_mean.round(0),
        "% HRmax": (recent.pct_hrmax_mean * 100).round(0), "Load": recent.trimp.round(0),
        "HR vs baseline (z)": recent.hr_mean_z28.round(1), "Unusual": recent.anomaly_metrics,
        "Data quality": recent.quality_label}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ workout explorer
with tabs[1]:
    opts = S.sort_values("start_time", ascending=False)
    has_series = (store.dir / "series.parquet").exists()
    if has_series:
        avail = set(pd.read_parquet(store.dir / "series.parquet", columns=["workout_id"]).workout_id.unique())
        opts = opts[opts.workout_id.isin(avail)]
    wid = st.selectbox("Workout", opts.workout_id.tolist(),
                       format_func=lambda w: f"{S.set_index('workout_id').at[w, 'start_time']:%d %b %Y %H:%M} · "
                                             f"{S.set_index('workout_id').at[w, 'distance_km']:.1f} km")
    row = store.session(wid)
    ser = store.series(wid)
    if ser.empty:
        st.info("The full time series for this workout is not available locally. Run `make data` to download the raw dataset and explore every session.")
    else:
        hm = row.hrmax_est
        t_min = ser.t_s / 60
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
        bounds = [0] + list(ZONE_EDGES * hm) + [hm * 1.1]
        x0, x1 = float(t_min.min()), float(t_min.max())
        for k in range(1, 6):  # zone bands as filled rectangles behind the HR trace
            fig.add_scatter(x=[x0, x1, x1, x0, x0], y=[bounds[k], bounds[k], bounds[k + 1], bounds[k + 1], bounds[k]],
                            fill="toself", fillcolor=ZONE_COLORS[k], opacity=0.45, line=dict(width=0),
                            mode="lines", hoverinfo="skip", showlegend=False, row=1, col=1)
            fig.add_annotation(x=x1, y=(bounds[k] + min(bounds[k + 1], hm * 1.05)) / 2, text=f"Z{k}",
                               showarrow=False, xanchor="left", font=dict(size=11, color="#475569"), row=1, col=1)
        fig.add_scatter(x=t_min, y=ser.heart_rate, name="Heart rate (bpm)", line=dict(color="#DC2626"), row=1, col=1)
        fig.add_scatter(x=t_min, y=ser.speed_kmh, name="Speed (km/h)", line=dict(color=ACCENT), row=2, col=1)
        fig.add_scatter(x=t_min, y=ser.altitude_m, name="Altitude (m)", line=dict(color="#16A34A"),
                        fill="tozeroy", row=3, col=1)
        fig.update_yaxes(range=[max(60, ser.heart_rate.min() - 10), hm * 1.05], row=1, col=1)
        fig.update_xaxes(title_text="minutes", row=3, col=1)
        fig.update_layout(height=560, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Shaded bands are HR zones as % of estimated HRmax ({hm:.0f} bpm"
                   f"{'' if row.hrmax_reliable else ', provisional: fewer than 5 sessions'}).")
    c = st.columns(6)
    c[0].metric("Distance", f"{row.distance_km:.2f} km")
    c[1].metric("Duration", f"{row.duration_min:.0f} min")
    c[2].metric("Pace (min/km)", f"{row.pace_min_per_km:.2f}")
    c[3].metric("Avg HR (bpm)", f"{row.hr_mean:.0f}",
                f"{row.hr_mean - row.hr_mean_base28:+.1f} vs baseline" if pd.notna(row.hr_mean_base28) else None,
                delta_color="off")
    c[4].metric("Load (TRIMP)", f"{row.trimp:.0f}")
    c[5].metric("Data quality", row.quality_label, f"score {row.quality_score:.2f}", delta_color="off")
    zones = pd.DataFrame({"zone": ["Z1", "Z2", "Z3", "Z4", "Z5"], "minutes": [row[f"z{k}_min"] for k in range(1, 6)]})
    fig = go.Figure(go.Bar(x=zones.minutes, y=zones.zone, orientation="h", marker_color=ZONE_COLORS[1:]))
    fig.update_layout(title="Time in zones (min)", height=240, **LAYOUT)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ training load
with tabs[2]:
    rng = st.slider("Show the last N days of the record", 28, max(28, len(D)), min(180, max(28, len(D))), step=7)
    d = D.tail(rng)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
    fig.add_bar(x=d.date, y=d.load, name="Daily load (TRIMP)", marker_color="#CBD5E1", row=1, col=1)
    fig.add_scatter(x=d.date, y=d.acute, name="Acute (7-day mean)", line=dict(color=ACCENT2, width=2), row=1, col=1)
    fig.add_scatter(x=d.date, y=d.chronic, name="Chronic (28-day mean)", line=dict(color=ACCENT, width=2), row=1, col=1)
    if len(d):
        x0, x1 = d.date.min(), d.date.max()
        fig.add_scatter(x=[x0, x1, x1, x0, x0], y=[0.8, 0.8, 1.3, 1.3, 0.8], fill="toself", fillcolor="#DCFCE7",
                        line=dict(width=0), mode="lines", hoverinfo="skip", name="0.8–1.3 band", row=2, col=1)
    fig.add_scatter(x=d.date, y=d.acwr, name="Acute:chronic ratio", line=dict(color="#0F172A"), row=2, col=1)
    fig.update_layout(height=560, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Load = Edwards TRIMP (minutes in each HR zone × zone number). The shaded 0.8–1.3 ACWR band is a "
               "commonly cited range, shown for orientation only - its link to injury risk is debated and AthleteIQ "
               "makes no injury predictions. Only workouts recorded with HR count, so loads are lower bounds.")
    st.subheader("Weekly summary")
    w = W.tail(12).sort_values("week", ascending=False)
    st.dataframe(pd.DataFrame({"Week": w.week.dt.date, "Sessions": w.sessions, "Distance (km)": w.distance_km.round(1),
                               "Load": w.weekly_load.round(0), "Monotony": w.monotony.round(2),
                               "Strain": w.strain.round(0)}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ prediction
with tabs[3]:
    if P.empty:
        st.info("This athlete's sessions were used to train the model, so there are no out-of-sample predictions "
                "to show. Pick an athlete marked 'predicted'.")
    else:
        split = P.split.iloc[0]
        st.markdown(
            "**Task:** predict a session's average heart rate from the workout itself (distance, speed, terrain, "
            "time) and what was known about the athlete *before* it. " +
            ("This athlete was **never seen in training** (held-out athlete)." if split == "holdout" else
             "These are this athlete's **most recent sessions**, after the training period."))
        fig = go.Figure()
        fig.add_scatter(x=P.start_time, y=P.pi_high, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=P.start_time, y=P.pi_low, fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                        line=dict(width=0), name="90% interval")
        fig.add_scatter(x=P.start_time, y=P.pred_lgbm_personal, mode="lines+markers", name="Predicted",
                        line=dict(color=ACCENT))
        fig.add_scatter(x=P.start_time, y=P.hr_mean, mode="markers", name="Actual", marker=dict(color="#DC2626", size=7))
        fig.update_layout(title="Predicted vs actual average HR", height=360, yaxis_title="bpm", **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        err = (P.pred_lgbm_personal - P.hr_mean).abs()
        cov = ((P.hr_mean >= P.pi_low) & (P.hr_mean <= P.pi_high)).mean()
        c = st.columns(3)
        c[0].metric("MAE for this athlete", f"{err.mean():.1f} bpm")
        c[1].metric("Within 5 bpm", f"{(err <= 5).mean():.0%}")
        c[2].metric("Inside 90% interval", f"{cov:.0%}")

        pw = st.selectbox("Explain session", P.workout_id.tolist()[::-1],
                          format_func=lambda w: f"{pd.Timestamp(P.set_index('workout_id').at[w, 'start_time']):%d %b %Y}")
        pr = store.prediction(pw)
        exp = store.explanation(pw, top=10)
        c = st.columns(4)
        c[0].metric("Predicted", f"{pr.pred_lgbm_personal:.1f} bpm")
        c[1].metric("90% interval", f"{pr.pi_low:.0f}–{pr.pi_high:.0f}")
        c[2].metric("Actual", f"{pr.hr_mean:.1f} bpm")
        c[3].metric("Personal correction", f"{pr.personal_offset:+.1f} bpm")
        if exp is not None:
            e = exp.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=e.contribution_bpm, y=[f"{lab} = {v:.2f}" for lab, v in zip(e.label, e.value)], orientation="h",
                marker_color=[ACCENT2 if v > 0 else ACCENT for v in e.contribution_bpm]))
            fig.update_layout(title=f"Why this prediction: SHAP contributions vs model average "
                                    f"({exp.base_bpm.iloc[0]:.1f} bpm)", height=420, xaxis_title="bpm", **LAYOUT)
            st.plotly_chart(fig, width="stretch")
            st.caption("SHAP shows how the model used each input for this prediction. These are learned "
                       "associations, not causal effects on the athlete's physiology.")

# ------------------------------------------------------------------ model report
with tabs[4]:
    m = store.metrics
    if not m:
        st.info("Run `python scripts/run_pipeline.py train` to produce the model report.")
    else:
        st.markdown(f"Target: **{m['target']}**. Evaluated on each athlete's most recent sessions (test) and on "
                    f"athletes never seen in training (held-out). {m['n_features']} leakage-safe features.")
        rows = []
        for sp in ["test", "holdout"]:
            for k, v in m["by_split"][sp].items():
                if isinstance(v, dict):
                    rows.append({"Split": "Future sessions" if sp == "test" else "New athletes",
                                 "Model": MODEL_NAMES.get(k, k), "MAE (bpm)": round(v["mae"], 2),
                                 "RMSE": round(v["rmse"], 2), "R²": round(v["r2"], 3),
                                 "Within 5 bpm": f"{v['within_5bpm']:.0%}", "n": v["n"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        c = st.columns(3)
        c[0].metric("90% interval coverage · future sessions", f"{m['by_split']['test']['interval_coverage']:.1%}")
        c[1].metric("90% interval coverage · new athletes", f"{m['by_split']['holdout']['interval_coverage']:.1%}")
        c[2].metric("Interval half-width (20+ prior sessions)",
                    f"±{m['conformal']['mondrian_half_width_bpm']['20+']:.1f} bpm")
        left, right = st.columns(2)
        with left:
            gi = pd.DataFrame(m["global_importance"]).head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=gi.mean_abs_shap_bpm, y=gi.label, orientation="h", marker_color=ACCENT))
            fig.update_layout(title="Global importance (mean |SHAP|, bpm)", height=420, **LAYOUT)
            st.plotly_chart(fig, width="stretch")
        with right:
            lc = pd.DataFrame(m["holdout_learning_curve"])
            fig = go.Figure()
            fig.add_scatter(x=lc.n, y=lc["global"], name="Global model", line=dict(color=MUTED))
            fig.add_scatter(x=lc.n, y=lc["personal"], name="+ online personalisation", line=dict(color=ACCENT2))
            fig.update_layout(title="New athletes: error vs number of prior sessions", height=420,
                              xaxis_title="prior sessions (bucket, 60 = 60+)", yaxis_title="MAE (bpm)", **LAYOUT)
            st.plotly_chart(fig, width="stretch")
        st.subheader("Where the model struggles")
        ea = m["error_analysis"]
        for by, title in [("history_bucket", "By athlete history length"),
                          ("label_workout_type", "By packager's workout label (HR-derived; not a feature)")]:
            df = pd.concat([pd.DataFrame(ea[sp][by]).assign(split=sp) for sp in ["test", "holdout"]])
            st.markdown(f"**{title}**")
            st.dataframe(df, hide_index=True, width="stretch")

# ------------------------------------------------------------------ assistant
with tabs[5]:
    ai = get_ai()
    st.markdown(f"Ask about **athlete {uid}**. Answers come only from computed metrics and cite them. "
                f"Backend: `{ai.backend}`" + (" (set `ANTHROPIC_API_KEY` to use Claude)" if ai.backend == "rules" else ""))
    examples = {"Workload, last 4 weeks": "How has the workload changed over the last 4 weeks?",
                "Latest workout": "Tell me about the most recent workout.",
                "Explain prediction": "What drove the predicted heart rate?",
                "Unusual sessions": "Were there any unusual sessions?",
                "Overview": "Give me an overview of this athlete."}
    key = f"chat_{uid}"
    st.session_state.setdefault(key, [])
    cols = st.columns(len(examples))
    clicked = None
    for c_, (label, q) in zip(cols, examples.items()):
        if c_.button(label, width="stretch"):
            clicked = q
    q = st.chat_input("Ask AthleteIQ…") or clicked
    if q:
        st.session_state[key].append(("user", q, None))
        ans = ai.ask(q, athlete_id=uid)
        st.session_state[key].append(("assistant", ans.text, ans))
    for role, text, ans in st.session_state[key]:
        with st.chat_message(role):
            st.markdown(text)
            if ans is not None and ans.citations:
                with st.expander(f"Sources · {len(ans.citations)} cited metrics · tools: "
                                 + ", ".join(t["tool"] for t in ans.tool_calls)):
                    st.dataframe(pd.DataFrame(ans.citations), hide_index=True, width="stretch")
