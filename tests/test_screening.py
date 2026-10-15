"""Screening the needle set at FP16 (#148).

ADR-0017 replaces a sample FP16 answers wrongly before the comparison, so
that the model's failures are not counted as the cache's. Each depth keeps
the first five of its samples, in their order from the seed, that FP16
answers correctly; the set and its replacements are logged, and the RQ3
tool runs the screened set.

Seams: the screening, pure; the screened set read from the log; and
tools/screen_needles.py through the engine (Seam A).
"""

import importlib.util
from pathlib import Path

import pytest

from microinfer import benchlog, needles, rq3

REPO = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fake:
    """What screening reads of a sample: its id and depth."""

    def __init__(self, depth, k):
        self.depth, self.sample_id = depth, f"d{round(depth * 100):02d}-{k}"


def drawn(per_depth):
    return [Fake(d, k) for d in needles.DEPTHS for k in range(per_depth)]


def test_every_sample_right_keeps_the_first_five_of_each_depth():
    asked = []
    kept, rejected = rq3.screen(drawn(8), lambda s: asked.append(s.sample_id) or True)
    assert [s.sample_id for s in kept if s.depth == 0.5] == [f"d50-{k}" for k in range(5)]
    assert len(kept) == 50 and rejected == []
    assert len(asked) == 50  # no sample past a depth's fifth right one is run


def test_a_sample_answered_wrongly_is_replaced_by_the_next_of_its_depth():
    wrong = {"d30-1", "d30-4", "d90-0"}
    kept, rejected = rq3.screen(drawn(8), lambda s: s.sample_id not in wrong)
    assert [s.sample_id for s in kept if s.depth == 0.3] == [
        "d30-0", "d30-2", "d30-3", "d30-5", "d30-6"]
    assert [s.sample_id for s in rejected] == ["d30-1", "d30-4", "d90-0"]
    assert len(kept) == 50


def test_a_depth_with_too_few_right_is_refused_after_every_depth_is_screened():
    """Every depth is screened before the refusal, which names each short
    one and holds what was kept and rejected, for the partial log."""
    asked = []

    def is_right(s):
        asked.append(s.sample_id)
        return not s.sample_id[:3] in ("d20", "d70") or s.sample_id.endswith("-0")

    with pytest.raises(rq3.ShortScreening, match="20%.*70%") as short:
        rq3.screen(drawn(6), is_right)
    assert short.value.short == {0.2: 1, 0.7: 1}
    assert len(short.value.kept) == 8 * 5 + 2 and len(short.value.rejected) == 10
    assert "d90-4" in asked


def screening_entry(log, seed, kept, rejected, model="m", haystack=1024, complete=True):
    benchlog.append("needle-screening", model=model, context_length=8000,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": seed, "draws": 8, "haystack_tokens": haystack,
                            "issue": 148},
                    results={"kept": kept, "rejected": rejected, "complete": complete},
                    log=log)


def test_the_screened_set_is_the_last_screening_s_for_the_model_and_seed(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    screening_entry(log, 140, ["d00-0"], [])
    screening_entry(log, 141, ["d00-9"], [])
    screening_entry(log, 140, ["d00-1", "d10-2"], ["d00-0"])
    screening_entry(log, 140, ["d00-7"], [], model="other")
    screening_entry(log, 140, ["d00-8"], [], haystack=7936)
    screening_entry(log, 140, ["d00-3"], [], complete=False)
    assert rq3.screened("m", 140, 1024, log) == (["d00-1", "d10-2"], 8)
    assert rq3.screened("m", 140, 7936, log) == (["d00-8"], 8)
    with pytest.raises(ValueError, match="needle-screening"):
        rq3.screened("m", 142, 1024, log)
    with pytest.raises(ValueError, match="needle-screening"):
        rq3.screened("m", 140, 2048, log)


def fp16_entry(log, sample_id, correct, *, screening=True, seed=140, haystack=1024):
    benchlog.append("rq3-needle", model="m", context_length=1100,
                    precision_tiers={"FP16": 1.0},
                    config={"sample_id": sample_id, "condition": {"name": "fp16"},
                            "needle_seed": seed, "haystack_tokens": haystack,
                            "screening": screening},
                    results={"correct": correct}, log=log)


def test_a_rerun_at_the_same_commit_takes_the_fp16_runs_already_logged(tmp_path):
    tool = load("screen_needles")
    log = tmp_path / "benchmark.jsonl"
    fp16_entry(log, "d00-0", True)
    fp16_entry(log, "d00-1", False)
    fp16_entry(log, "d00-2", True, screening=False)  # an RQ3 run, not screening's
    fp16_entry(log, "d00-3", True, seed=141)
    fp16_entry(log, "d00-4", True, haystack=7936)
    commit = benchlog.read(log)[0]["git_commit"]
    assert tool.logged_fp16("m", 140, 1024, commit, log) == {"d00-0": True, "d00-1": False}
    assert tool.logged_fp16("m", 140, 1024, "another commit", log) == {}


@pytest.mark.slow
def test_the_tool_logs_each_fp16_run_and_the_screened_set(tmp_path, monkeypatch):
    """One a depth on Qwen2.5-0.5B over short haystacks: an FP16 entry for
    each sample run, and one screening entry whose kept samples were all
    answered correctly."""
    tool = load("screen_needles")
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert tool.main(["--issue", "148", "--model", "qwen2.5-0.5b-instruct", "--per-depth", "1",
                      "--haystack-tokens", "1024", "--log", str(log)]) == 0
    entries = benchlog.read(log)
    runs = [e for e in entries if e["kind"] == "rq3-needle"]
    (screening,) = [e for e in entries if e["kind"] == "needle-screening"]
    right = {e["config"]["sample_id"] for e in runs if e["results"]["correct"]}
    kept, rejected = screening["results"]["kept"], screening["results"]["rejected"]
    assert len(kept) == 10 and set(kept) <= right and not set(rejected) & right
    assert {e["config"]["sample_id"] for e in runs} == set(kept) | set(rejected)
    assert all(e["config"]["condition"]["name"] == "fp16" for e in runs)


@pytest.mark.slow
def test_the_rq3_tool_runs_the_screened_set(tmp_path, monkeypatch):
    """--screened runs the samples the last screening kept, drawn as many a
    depth as it drew them: d50-6 exists only in a set of 8 a depth."""
    tool = load("rq3")
    log = tmp_path / "benchmark.jsonl"
    name = "qwen2.5-0.5b-instruct"
    screening_entry(log, 140, ["d50-6"], ["d50-0"], model=name)
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert tool.main(["--issue", "148", "--model", name, "--screened", "--conditions", "fp16",
                      "--haystack-tokens", "1024", "--log", str(log)]) == 0
    (run,) = [e for e in benchlog.read(log) if e["kind"] == "rq3-needle"]
    assert run["config"]["sample_id"] == "d50-6"
    for clash in (["--samples", "d50-0"], ["--samples-per-depth", "9"]):
        with pytest.raises(SystemExit):
            tool.main(["--issue", "148", "--screened", *clash, "--log", str(log)])


def test_the_screened_set_s_fp16_runs_are_screening_s():
    """Screening ran the screened set at FP16 already: --screened leaves
    fp16 out unless it is asked for."""
    tool = load("rq3")
    assert tool.conditions_for(None, screened=True) == [
        c.name for c in rq3.CONDITIONS if c.name != "fp16"]
    assert tool.conditions_for(None, screened=False) == [c.name for c in rq3.CONDITIONS]
    assert tool.conditions_for(["fp16"], screened=True) == ["fp16"]
