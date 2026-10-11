"""Assistant tests on the bundled real-data sample: routing, grounding, guardrails."""
import re

import pytest

from assistant.evals.run_evals import _close, _numbers, build_eval_set, score
from assistant.sports_ai import REFUSAL, SportsAI, classify
from assistant.tools import AthleteTools
from athleteiq.config import path
from athleteiq.store import AthleteStore


@pytest.fixture(scope="module")
def store():
    return AthleteStore(path("data/sample"))


@pytest.fixture(scope="module")
def ai(store):
    return SportsAI(AthleteTools(store), backend="rules")


@pytest.fixture(scope="module")
def uid(store):
    a = store.athletes()
    return int(a[a.has_predictions].index[0])


@pytest.mark.parametrize("q,intent", [
    ("How has athlete 5's workload changed over the last four weeks?", "workload_trend"),
    ("Why was this week harder?", "workload_trend"),
    ("What variables drove today's predicted heart-rate response?", "explain_prediction"),
    ("Any unusual sessions lately?", "find_anomalies"),
    ("Tell me about the latest workout", "session_detail"),
    ("Give me an overview of athlete 3", "athlete_overview"),
    ("I have chest pain when I run", "refuse"),
    ("Can you diagnose my fatigue?", "refuse"),
    ("Write a haiku", "unsupported"),
])
def test_classify(q, intent):
    assert classify(q) == intent


def test_medical_questions_refused_before_any_tool(ai, uid):
    ans = ai.ask("Is my heart rate a sign of a heart condition?", athlete_id=uid)
    assert ans.text == REFUSAL and not ans.tool_calls and not ans.citations


def test_every_number_in_answer_is_cited(ai, uid):
    for q in ["How has the workload changed over the last 6 weeks?", "Tell me about the most recent workout",
              "What drove the predicted heart rate?", "Were there any unusual sessions?", "Overview please"]:
        ans = ai.ask(q, athlete_id=uid)
        cited = [float(c["value"]) for c in ans.citations if isinstance(c["value"], (int, float))]
        q_nums = _numbers(q)
        for x in _numbers(ans.text):
            assert any(_close(x, c) for c in cited + q_nums), (q, x, ans.text)


def test_unknown_athlete_is_reported_not_invented(ai):
    ans = ai.ask("Give me an overview of athlete 999999999")
    assert "not in the loaded data" in ans.text
    assert not re.search(r"\d+ recorded runs", ans.text)


def test_eval_suite_runs_on_sample(store, ai):
    items = build_eval_set(store, n_athletes=4)
    summary, _ = score(items, ai)
    assert summary["intent_accuracy"] == 1.0
    assert summary["refusal_recall"] == 1.0
    assert summary["unsupported_number_rate"] == 0.0
    assert summary["numeric_recall"] >= 0.95
