"""Ask AthleteIQ - natural-language questions answered only from computed metrics.

Two interchangeable backends share the same tools, guardrails and citations:

* ``rules``  - deterministic intent router + templated answers. Works offline,
               is used in tests/CI and is the default when no API key is set.
* ``llm``    - Claude with tool use (set ANTHROPIC_API_KEY). The model can only
               see numbers that tools return and is told to cite them.

Guardrails (both backends): medical / injury / diagnosis questions are declined
before any tool runs; answers separate observed values from model estimates;
nothing is said that a tool did not return.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .tools import TOOL_SPECS, AthleteTools, ToolError

MEDICAL_PATTERNS = re.compile(
    r"injur|\bpain|hurt|\bsore|diagnos|disease|illness|\bsick|symptom|medic|drug|doctor|physician|"
    r"arrhythm|afib|a-fib|chest|heart (attack|condition|problem|disease)|cardiac|blood pressure|"
    r"pregnan|concussion|fever|covid|dizz|faint|safe (for me )?to (run|train)|overtraining syndrome|"
    r"supplement|dose|dosage", re.I)

REFUSAL = ("I can't help with medical, injury or health-safety questions - AthleteIQ only describes "
           "training data and model estimates, and it isn't a substitute for a qualified clinician or "
           "physiotherapist. I can show this athlete's workload, recent sessions or unusual sessions instead.")

CAPABILITIES = ("I can answer questions about an athlete's overview, workload trend over recent weeks, "
                "a specific or most recent workout, unusual sessions versus their personal baseline, and "
                "what drove the model's predicted heart rate for a workout.")

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "twelve": 12}

SYSTEM_PROMPT = """You are Ask AthleteIQ, an analytics assistant for coaches and athletes.
Rules:
- Use ONLY numbers returned by the tools. Never estimate, extrapolate or invent values.
- Label model outputs as estimates ("the model estimates...") and measured values as observed.
- Each number you state must appear in a tool result; mention the metric it belongs to.
- Correlation is not causation: SHAP drivers describe the model, not physiology.
- Do not give medical, injury or diagnostic advice; do not prescribe training.
- The data is historical; "last N weeks" means the N weeks ending on the athlete's last recorded workout.
- Be concise: 2-5 sentences, plain language."""


@dataclass
class Answer:
    text: str
    intent: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    backend: str = "rules"

    def to_dict(self) -> dict:
        return {"text": self.text, "intent": self.intent, "citations": self.citations,
                "tool_calls": self.tool_calls, "backend": self.backend}


# ---------------------------------------------------------------------- parsing

def _athlete_id(q: str, default):
    m = re.search(r"(?:athlete|user|runner)\s*(?:id\s*)?#?\s*(\d+)", q, re.I)
    return int(m.group(1)) if m else default


def _workout_id(q: str):
    m = re.search(r"(?:workout|session|run)\s*(?:id\s*)?#?\s*(\d{5,})", q, re.I)
    return int(m.group(1)) if m else None


def _weeks(q: str, default=4) -> int:
    m = re.search(r"(\d+)\s*weeks?", q, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(WORDS) + r")\s+weeks?", q, re.I)
    if m:
        return WORDS[m.group(1).lower()]
    if re.search(r"\bmonth\b", q, re.I):
        return 4
    if re.search(r"\bweek\b", q, re.I):
        return 1
    return default


def classify(q: str) -> str:
    ql = q.lower()
    if MEDICAL_PATTERNS.search(ql):
        return "refuse"
    if re.search(r"predict|model|drove|driver|expected|forecast|shap|explain", ql):
        return "explain_prediction"
    if re.search(r"anomal|unusual|outlier|odd|strange|abnormal|deviat|out of the ordinary", ql):
        return "find_anomalies"
    if re.search(r"workload|load|volume|harder|easier|training stress|acwr|acute|chronic|trend", ql):
        return "workload_trend"
    if re.search(r"workout|session|\brun\b|last run|latest|most recent", ql):
        return "session_detail"
    if re.search(r"overview|summary|summari[sz]e|profile|how is|how's|tell me about athlete|doing", ql):
        return "athlete_overview"
    return "unsupported"


# ---------------------------------------------------------------------- templates

def _fmt_change(pct):
    if pct is None:
        return "there was no recorded load in the previous period to compare with"
    return f"{'up' if pct >= 0 else 'down'} {abs(pct):.0f}%"


def _render(intent: str, d: dict) -> str:
    if intent == "athlete_overview":
        t = (f"Athlete {d['user_id']} has {d['sessions_total']} recorded runs ({d['total_km']} km) between "
             f"{d['first_session']} and {d['last_session']}. In the last 28 days of their record they ran "
             f"{d['last28_sessions']} times for {d['last28_km']} km")
        if d["last28_sessions"]:
            t += f", averaging {d['last28_avg_hr']} bpm at {d['last28_avg_pace_min_km']} min/km"
        t += "."
        if d["efficiency_last28"] and d["efficiency_prev28"]:
            t += (f" Running efficiency was {d['efficiency_last28']} m/beat versus {d['efficiency_prev28']} m/beat "
                  "in the 28 days before.")
        t += f" Estimated HRmax is {d['hrmax_estimate']} bpm (an estimate from their highest recorded HR, not a lab test)."
        if d.get("training_pattern"):
            t += f" Training pattern: {d['training_pattern']}."
        return t
    if intent == "workload_trend":
        t = (f"In the {d['weeks']} weeks up to {d['period_end']}, athlete {d['user_id']} accumulated "
             f"{d['load_recent']:.0f} TRIMP over {d['sessions_recent']} sessions ({d['km_recent']} km), versus "
             f"{d['load_previous']:.0f} TRIMP over {d['sessions_previous']} sessions ({d['km_previous']} km) in the "
             f"{d['weeks']} weeks before - {_fmt_change(d['pct_change'])}.")
        if d["acwr_latest"] is not None:
            t += f" The latest acute:chronic workload ratio is {d['acwr_latest']}."
        t += " Load counts only workouts recorded with heart rate."
        return t
    if intent == "session_detail":
        z = d["minutes_in_zone"]
        top_zone = max(z, key=lambda k: z[k] or 0)
        t = (f"Workout {d['workout_id']} on {d['date']}: {d['distance_km']} km in {d['duration_min']} min "
             f"({d['pace_min_km']} min/km), average HR {d['avg_hr']} bpm (max {d['max_hr']}), about "
             f"{d['pct_hrmax']}% of estimated HRmax. Session load was {d['trimp']} TRIMP, with most time in "
             f"{top_zone} ({z[top_zone]} min).")
        if d["hr_vs_baseline_z"] is not None:
            t += f" Average HR was {d['hr_vs_baseline_z']} SD from their 28-day baseline."
        if d["unusual_metrics"]:
            t += f" Flagged as unusual for: {', '.join(d['unusual_metrics'])}."
        if d["data_quality"] != "good":
            t += f" Data quality: {d['data_quality']} - treat these numbers with care."
        return t
    if intent == "explain_prediction":
        t = (f"For workout {d['workout_id']} on {d['date']}, the model estimated an average HR of {d['predicted_hr']} bpm "
             f"(90% interval {d['interval_low']}-{d['interval_high']}); the observed value was {d['actual_hr']} bpm.")
        if d["drivers"]:
            parts = [f"{x['feature']} ({'+' if x['effect_bpm'] >= 0 else ''}{x['effect_bpm']} bpm)" for x in d["drivers"][:3]]
            t += (f" Starting from the model average of {d['model_average']} bpm, the biggest drivers were "
                  + ", ".join(parts) + ".")
        t += " These are associations the model learned, not causes."
        return t
    if intent == "find_anomalies":
        if not d["anomalous_sessions"]:
            return (f"0 of athlete {d['user_id']}'s {d['sessions_scored']} sessions with a baseline deviated "
                    "more than 2 SD from their prior 28-day norm.")
        t = (f"{d['anomalous_sessions']} of athlete {d['user_id']}'s {d['sessions_scored']} sessions with a baseline "
             "deviated more than 2 SD from their prior 28-day norm.")
        r = d["recent"][0]
        t += f" The most recent was on {r['date']} (unusual: {r['metrics']})"
        if r["avg_hr"] is not None:
            t += f", average HR {r['avg_hr']} bpm"
        return t + "."
    return CAPABILITIES


# ---------------------------------------------------------------------- assistant


class SportsAI:
    def __init__(self, tools: AthleteTools | None = None, backend: str | None = None):
        self.tools = tools or AthleteTools()
        want = backend or os.getenv("ATHLETEIQ_BACKEND") or ("llm" if os.getenv("ANTHROPIC_API_KEY") else "rules")
        self.backend = want

    def ask(self, question: str, athlete_id: int | None = None) -> Answer:
        intent = classify(question)
        if intent == "refuse":
            return Answer(REFUSAL, "refuse", backend=self.backend)
        if self.backend == "llm":
            return self._ask_llm(question, athlete_id)
        return self._ask_rules(question, intent, athlete_id)

    # deterministic -------------------------------------------------------
    def _ask_rules(self, q: str, intent: str, athlete_id) -> Answer:
        if intent == "unsupported":
            return Answer(CAPABILITIES, "unsupported")
        uid, wid = _athlete_id(q, athlete_id), _workout_id(q)
        args: dict = {}
        if intent == "workload_trend":
            args = {"user_id": uid, "weeks": _weeks(q)}
        elif intent in ("athlete_overview", "find_anomalies"):
            args = {"user_id": uid}
        elif intent in ("session_detail", "explain_prediction"):
            args = {"workout_id": wid} if wid else {"user_id": uid}
        try:
            res = self.tools.call(intent, args)
        except (ToolError, ValueError, TypeError) as exc:
            return Answer(str(exc), intent, tool_calls=[{"tool": intent, "args": args, "error": str(exc)}])
        return Answer(_render(intent, res["data"]), intent, res["citations"],
                      [{"tool": intent, "args": args}])

    # LLM -----------------------------------------------------------------
    def _ask_llm(self, q: str, athlete_id) -> Answer:
        import anthropic

        from athleteiq.config import load_config
        cfg = load_config()["assistant"]
        client = anthropic.Anthropic()
        tools = [{"name": n, **spec} for n, spec in TOOL_SPECS.items()]
        ctx = f"(Currently selected athlete id: {athlete_id}.) " if athlete_id else ""
        messages = [{"role": "user", "content": ctx + q}]
        citations, calls, intent = [], [], "unsupported"
        for _ in range(cfg["max_tool_rounds"]):
            resp = client.messages.create(model=os.getenv("ATHLETEIQ_MODEL", cfg["llm_model"]), max_tokens=700,
                                          system=SYSTEM_PROMPT, tools=tools, messages=messages)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                return Answer(text or CAPABILITIES, intent, citations, calls, "llm")
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                intent = block.name
                try:
                    out = self.tools.call(block.name, dict(block.input))
                    citations += out["citations"]
                    payload = json.dumps(out, default=str)
                    calls.append({"tool": block.name, "args": dict(block.input)})
                except (ToolError, ValueError, TypeError) as exc:
                    payload = json.dumps({"error": str(exc)})
                    calls.append({"tool": block.name, "args": dict(block.input), "error": str(exc)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": payload})
            messages.append({"role": "user", "content": results})
        return Answer("I couldn't complete that within the tool-call limit.", intent, citations, calls, "llm")
