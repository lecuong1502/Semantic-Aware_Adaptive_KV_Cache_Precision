"""The 32K subset (#156).

Ten needle samples, one a depth, of haystacks of about 32K tokens, run
question-blind under FP16 and both sources at 80%: they confirm the
direction the 8K set gives, with no test claimed (ADR-0017).

Seams: the set, built from the checkpoint's tokenizer; the summary of a
log's entries, pure; and tools/rq3.py --long and tools/summarise_rq3.py
--long, which run and summarise it.
"""

import pytest

from conftest import load_tool, require_model
from microinfer import Engine, benchlog, needles, rq3, rq3_summary

MODEL = "qwen2.5-1.5b-instruct"
SAMPLES = [f"d{d:02d}-0" for d in range(0, 100, 10)]


# -- the set -------------------------------------------------------------------------------

def test_the_subset_is_one_sample_a_depth_within_the_model_s_window():
    """Every turn of a run, the summary, the question and the answer, fits
    in Qwen2.5's 32768 positions."""
    engine = Engine(require_model(MODEL))  # the tokenizer alone: no weights
    built = needles.needle_set(load_tool("perplexity").held_out_text(), engine.tokenizer,
                               seed=140, samples_per_depth=1,
                               haystack_tokens=needles.LONG_HAYSTACK_TOKENS)
    assert [s.sample_id for s in built] == SAMPLES
    window = engine.config.max_position_embeddings
    for s in built:
        positions = len(s.turn_1) + rq3.SUMMARY_TOKENS + len(s.turn_2) + rq3.ANSWER_TOKENS
        assert 32_000 < positions <= window, s.sample_id


# -- the summary ---------------------------------------------------------------------------

def log_of(tmp_path, outcome, *, skip=(), extra=()):
    """A log of a run of every sample under every condition of the subset,
    answered as `outcome(sample_id, condition)` says, but those in `skip`;
    then the `extra` runs, (sample, condition, haystack tokens, correct)."""
    log = tmp_path / "benchmark.jsonl"
    runs = [(s, c, needles.LONG_HAYSTACK_TOKENS, outcome(s, c))
            for s in SAMPLES for c in rq3_summary.LONG_CONDITIONS if (s, c) not in skip]
    for sample_id, name, haystack, correct in [*runs, *extra]:
        benchlog.append(
            "rq3-needle", model=MODEL, context_length=32300, precision_tiers={"FP16": 1.0},
            config={"sample_id": sample_id, "depth": int(sample_id[1:3]) / 100,
                    "condition": {"name": name}, "experiment": "question-blind",
                    "needle_seed": 140, "plan_seed": 0, "haystack_tokens": haystack},
            results={"correct": correct, "effective_bits": 16.0 if name == "fp16" else 3.2,
                     "needle_page_tiers": {"INT2": 3, "INT4": 1}},
            log=log)
    return benchlog.read(log)


def test_the_summary_gives_each_condition_and_semantic_against_uniform(tmp_path):
    """uniform-80 misses 70% and 80%, semantic-80 misses 80% alone; FP16
    answers them all."""
    def outcome(s, c):
        return not ((c == "uniform-80" and s in ("d70-0", "d80-0"))
                    or (c == "semantic-80" and s == "d80-0"))
    summary = rq3_summary.summarise_long(log_of(tmp_path, outcome), MODEL, 140)
    a = summary["accuracy"]
    assert list(a) == list(rq3_summary.LONG_CONDITIONS) and summary["samples"] == 10
    assert a["fp16"]["correct"] == 10 and a["semantic-80"]["correct"] == 9
    assert a["uniform-80"]["correct"] == 8 and a["uniform-80"]["wilson_95"][1] < 1
    assert a["uniform-80"]["by_depth"] == {f"{d}%": s not in ("d70-0", "d80-0")
                                           for d, s in zip(range(0, 100, 10), SAMPLES)}
    assert a["semantic-80"]["effective_bits"] == pytest.approx(3.2)
    assert summary["paired"] == {"semantic_only": 1, "uniform_only": 0, "both": 8, "neither": 1}
    assert summary["direction"] == "semantic"
    assert "no test" in summary["claim"]
    assert len(summary["entries"]) == 30


def test_only_the_latest_long_runs_are_pooled(tmp_path):
    """An 8K run of a sample is another set's; a later run stands for the
    one before."""
    entries = log_of(tmp_path, lambda s, c: True,
                     extra=[("d00-0", "uniform-80", needles.DEFAULT_HAYSTACK_TOKENS, False),
                            ("d10-0", "uniform-80", needles.LONG_HAYSTACK_TOKENS, False)])
    summary = rq3_summary.summarise_long(entries, MODEL, 140)
    assert summary["accuracy"]["uniform-80"]["correct"] == 9
    assert summary["direction"] == "semantic"


def test_even_outcomes_point_no_way(tmp_path):
    summary = rq3_summary.summarise_long(log_of(tmp_path, lambda s, c: True), MODEL, 140)
    assert summary["direction"] is None


def test_a_missing_run_is_refused(tmp_path):
    entries = log_of(tmp_path, lambda s, c: True, skip=[("d50-0", "fp16")])
    with pytest.raises(ValueError, match=r"no run of 1 .*d50-0"):
        rq3_summary.summarise_long(entries, MODEL, 140)


# -- the tools -----------------------------------------------------------------------------

@pytest.fixture
def clean(monkeypatch):
    """The tools' benchlog, with the tree taken as clean: these tests run on
    a tree being changed."""
    real = benchlog.environment
    monkeypatch.setattr(benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})


def test_the_run_tool_takes_the_subset_s_samples_and_conditions(tmp_path, clean, capsys):
    """With every run of the subset logged at this code, --long has
    nothing to run, and says how many it would."""
    require_model(MODEL)
    log = tmp_path / "benchmark.jsonl"
    log_of(tmp_path, lambda s, c: True)
    tool = load_tool("rq3")
    assert tool.main(["--issue", "156", "--long", "--log", str(log)]) == 0
    assert "30 of 30 runs logged" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        tool.main(["--issue", "156", "--long", "--screened", "--log", str(log)])


def test_the_summary_tool_logs_the_long_summary(tmp_path, clean, capsys):
    log = tmp_path / "benchmark.jsonl"
    log_of(tmp_path, lambda s, c: c == "fp16" or s != "d90-0")
    tool = load_tool("summarise_rq3")
    assert tool.main(["--issue", "156", "--long", "--log", str(log)]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == rq3_summary.LONG_SUMMARY_KIND]
    assert summary["config"]["issue"] == 156
    assert summary["config"]["haystack_tokens"] == needles.LONG_HAYSTACK_TOKENS
    assert summary["results"]["accuracy"]["semantic-80"]["correct"] == 9
    assert "| semantic-80 | 9/10 |" in capsys.readouterr().out
