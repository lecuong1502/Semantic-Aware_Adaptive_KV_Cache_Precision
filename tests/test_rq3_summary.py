"""RQ3's summary: the logged needle runs pooled into one entry (#152).

ADR-0017 answers RQ3 from the logged runs alone: accuracy by condition,
level and depth with bootstrap 95% intervals, and McNemar's test of
semantic against uniform and against random, paired by sample. The closing
question is semantic against uniform at 80%, question-blind, and its answer
is reported whatever it is.

Seams: the statistics, pure; the summary of a log's entries; and
tools/summarise_rq3.py, which logs it with the hashes of what it pooled.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from microinfer import benchlog, needles, rq3, rq3_summary

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-1.5b-instruct"


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the statistics ------------------------------------------------------------------------

@pytest.mark.parametrize("only_a, only_b, p", [
    (0, 0, 1.0),                         # no discordant pair: nothing to tell apart
    (5, 0, 2 / 2 ** 5),                  # every discordant pair one way
    (10, 2, 2 * (1 + 12 + 66) / 2 ** 12),
    (3, 3, 1.0),
])
def test_mcnemar_s_exact_test_is_a_two_sided_binomial_on_the_discordant_pairs(only_a, only_b, p):
    assert rq3_summary.mcnemar(only_a, only_b) == pytest.approx(p)
    assert rq3_summary.mcnemar(only_b, only_a) == pytest.approx(p)


def test_a_bootstrap_interval_resamples_the_samples():
    rng = np.random.default_rng(0)
    outcomes = np.array([True] * 40 + [False] * 10)
    low, high = rq3_summary.bootstrap(outcomes[None, :], rng)[0]
    assert low < 0.8 < high and 0.65 < low and high < 0.92
    # at the ceiling every resample is right: the interval is a point
    assert rq3_summary.bootstrap(np.ones((1, 50), bool), rng)[0].tolist() == [1.0, 1.0]


def test_the_wilson_interval_stays_open_at_the_ceiling():
    low, high = rq3_summary.wilson(50, 50)
    assert low == pytest.approx(50 / (50 + 1.959964 ** 2), abs=1e-4) and high == 1.0
    low, high = rq3_summary.wilson(45, 50)
    assert low < 0.9 < high


# -- the summary of a log's entries --------------------------------------------------------

SAMPLES = [f"d{d:02d}-{k}" for d in range(0, 100, 10) for k in range(5)]


def entries(tmp_path, outcome, *, aware=True, extra=()):
    """A log of a screening and a run under every condition, each answered
    as `outcome(sample_id, experiment, condition)` says."""
    log = tmp_path / "benchmark.jsonl"
    # 750 entries of a log thrown away: the machine read once, and no fsync
    with pytest.MonkeyPatch.context() as patch:
        once = benchlog.environment(log)
        patch.setattr(benchlog, "environment", lambda log=None: dict(once))
        patch.setattr(benchlog.os, "fsync", lambda fd: None)
        return _append(log, outcome, aware, extra)


def _append(log, outcome, aware, extra):
    benchlog.append("needle-screening", model=MODEL, context_length=8000,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 140, "draws": 15, "haystack_tokens": 7936},
                    results={"kept": SAMPLES, "rejected": [], "complete": True}, log=log)
    runs = [(s, "question-blind", c.name) for s in SAMPLES for c in rq3.CONDITIONS]
    if aware:
        runs += [(s, "question-aware", c) for s in SAMPLES for c in rq3.SOURCE_CONDITIONS]
    for sample_id, experiment, name in [*runs, *extra]:
        cond = rq3.condition(name)
        benchlog.append(
            "rq3-needle", model=MODEL, context_length=8000, precision_tiers={"FP16": 1.0},
            config={"sample_id": sample_id, "depth": int(sample_id[1:3]) / 100,
                    "condition": {"name": name}, "experiment": experiment,
                    "needle_seed": 140, "plan_seed": 0, "haystack_tokens": 7936,
                    "screening": name == "fp16"},
            results={"correct": outcome(sample_id, experiment, name),
                     "effective_bits": {"fp16": 16.0, "static-int4": 4.625,
                                        "static-int2": 2.625}.get(name, 3.4 if cond.share == 0.8
                                                                  else 4.98),
                     "needle_page_tiers": {"INT2": 3, "INT4": 1}},
            log=log)
    return benchlog.read(log)


def test_accuracy_by_condition_level_and_depth(tmp_path):
    wrong = {("d20-0", "static-int2"), ("d20-1", "static-int2"), ("d90-4", "uniform-80")}
    summary = rq3_summary.summarise(
        entries(tmp_path, lambda s, x, c: (s, c) not in wrong), MODEL, 140, 7936)
    blind = summary["accuracy"]["question-blind"]
    assert list(blind) == [c.name for c in rq3.CONDITIONS]
    assert blind["static-int2"]["correct"] == 48 and blind["static-int2"]["n"] == 50
    assert blind["static-int2"]["by_depth"]["20%"] == 3 and blind["static-int2"]["by_depth"]["0%"] == 5
    assert blind["uniform-80"]["accuracy"] == pytest.approx(0.98)
    assert blind["fp16"]["bootstrap_95"] == [1.0, 1.0]
    assert blind["semantic-80"]["effective_bits"] == pytest.approx(3.4)
    assert blind["semantic-80"]["needle_page_tiers"] == {"INT2": 0.75, "INT4": 0.25}
    # question-aware holds the source conditions: FP16 and the static tiers
    # make no plan, so they would decode the same tokens
    assert list(summary["accuracy"]["question-aware"]) == list(rq3.SOURCE_CONDITIONS)


def test_semantic_is_paired_by_sample_against_uniform_and_random(tmp_path):
    def outcome(s, x, c):
        if c == "uniform-80":
            return s >= "d20"            # wrong on the ten samples at 0% and 10%
        if c == "semantic-80":
            return s != "d50-0"
        return True

    summary = rq3_summary.summarise(entries(tmp_path, outcome), MODEL, 140, 7936)
    pair = summary["paired"]["question-blind"]["80%"]["semantic-uniform"]
    assert (pair["semantic_only"], pair["other_only"]) == (10, 1)
    assert pair["mcnemar_p"] == pytest.approx(rq3_summary.mcnemar(10, 1))
    assert pair["difference"] == pytest.approx((49 - 40) / 50)
    assert pair["difference_bootstrap_95"][0] > 0
    assert set(summary["paired"]["question-blind"]["70%"]) == {"semantic-uniform", "semantic-random"}
    closing = summary["closing"]
    assert closing["experiment"] == "question-blind" and closing["level"] == "80%"
    assert closing["semantic_beats_uniform"] is True and closing["mcnemar_p"] < closing["alpha"]


def test_a_null_result_is_the_answer(tmp_path):
    """Semantic no better than uniform, or better by too few samples to
    tell: the answer is no, reported as such."""
    summary = rq3_summary.summarise(
        entries(tmp_path, lambda s, x, c: not (c == "uniform-80" and s == "d00-0")),
        MODEL, 140, 7936)
    closing = summary["closing"]
    assert closing["semantic_correct"] == 50 and closing["uniform_correct"] == 49
    assert closing["mcnemar_p"] == 1.0 and closing["semantic_beats_uniform"] is False
    assert "no" in closing["answer"].lower()


def test_the_latest_run_of_a_sample_counts_and_every_one_is_named_by_its_hash(tmp_path):
    """A run repeated later stands for the sample; a check like no-plan is
    not pooled."""
    seen = set()

    def outcome(s, x, c):  # d00-0 under uniform-80 is wrong once, then right
        first = (s, c) not in seen
        seen.add((s, c))
        return not (first and (s, c) == ("d00-0", "uniform-80"))

    later = [("d00-0", "question-blind", "uniform-80"), ("d00-0", "question-blind", "no-plan")]
    log = entries(tmp_path, outcome, extra=later)
    summary = rq3_summary.summarise(log, MODEL, 140, 7936)
    assert summary["accuracy"]["question-blind"]["uniform-80"]["correct"] == 50
    pooled = set(summary["entries"])
    assert log[-2]["sha256"] in pooled and log[-1]["sha256"] not in pooled
    assert log[0]["sha256"] == summary["screening"]
    assert len(pooled) == 50 * (len(rq3.CONDITIONS) + len(rq3.SOURCE_CONDITIONS))


def test_a_missing_run_is_refused(tmp_path):
    log = entries(tmp_path, lambda s, x, c: True, aware=False)
    with pytest.raises(ValueError, match="question-aware"):
        rq3_summary.summarise(log, MODEL, 140, 7936)
    summary = rq3_summary.summarise(log, MODEL, 140, 7936, experiments=["question-blind"])
    assert list(summary["accuracy"]) == ["question-blind"]
    del log[5]
    with pytest.raises(ValueError, match="no run"):
        rq3_summary.summarise(log, MODEL, 140, 7936, experiments=["question-blind"])


def test_the_repository_log_answers_the_closing_question():
    """#149, #150 and #151's runs, as logged: the closing counts are the
    logged outcomes of the screened set, counted directly."""
    log = benchlog.read()
    summary = rq3_summary.summarise(log, MODEL, 140, needles.DEFAULT_HAYSTACK_TOKENS)
    kept, _ = rq3.screened(MODEL, 140, needles.DEFAULT_HAYSTACK_TOKENS)
    blind = {(e["config"]["sample_id"], e["config"]["condition"]["name"]): e["results"]["correct"]
             for e in log if e["kind"] == "rq3-needle" and e["model"] == MODEL
             and e["config"]["experiment"] == "question-blind"}
    closing = summary["closing"]
    assert closing["semantic_correct"] == sum(blind[(s, "semantic-80")] for s in kept)
    assert closing["uniform_correct"] == sum(blind[(s, "uniform-80")] for s in kept)


def test_the_tool_logs_the_summary(tmp_path, monkeypatch):
    tool = load("summarise_rq3")
    entries(tmp_path, lambda s, x, c: True)
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert tool.main(["--issue", "152", "--log", str(log)]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == "rq3-summary"]
    assert summary["config"]["issue"] == 152
    assert summary["results"]["closing"]["semantic_beats_uniform"] is False
    assert len(summary["results"]["entries"]) == 50 * 15
