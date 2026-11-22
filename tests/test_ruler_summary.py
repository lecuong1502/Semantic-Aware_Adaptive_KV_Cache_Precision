"""RULER's runs pooled into a summary (#208, ADR-0017's amendment).

Each task's screened set at a length, from its last complete screening;
each condition's accuracy, answered samples with Wilson intervals, and mean
score; the ceiling, static INT2's accuracy, which decides whether a task
enters the closing test; and the closing test itself, semantic against
uniform at 80%, McNemar's exact test pooled over the tasks below their
ceiling.

Seam: the pooling and the statistics, pure, over a log's entries.
"""

import pytest

from microinfer import ruler_summary as summary

MODEL = "qwen2.5-1.5b-instruct"
TOKENS = 7936


DATA = {"PaulGrahamEssays.json": "e"}


def run(task, k, condition, score=1.0, *, screening=False, commit="a", data=DATA):
    return {"kind": "rq3-ruler", "model": MODEL, "git_commit": commit,
            "sha256": f"{task}{k}{condition}{screening}{commit}",
            "config": {"task": task, "sample_id": f"{task}-{k}", "haystack_tokens": TOKENS,
                       "ruler_seed": 208, "plan_seed": 0, "screening": screening,
                       "condition": {"name": condition}, "data": data},
            "results": {"score": score, "correct": score == 1.0}}


def screening(task, kept, complete=True):
    return {"kind": "ruler-screening", "model": MODEL, "sha256": f"screen-{task}",
            "config": {"task": task, "haystack_tokens": TOKENS, "ruler_seed": 208},
            "results": {"kept": [f"{task}-{k}" for k in kept], "complete": complete}}


def task_runs(task, n, condition, answered):
    """n samples of `task` under `condition`, the first `answered` answered."""
    return [run(task, k, condition, 1.0 if k < answered else 0.5) for k in range(n)]


def test_a_task_is_below_its_ceiling_if_static_int2_answers_at_most_80_percent():
    assert summary.below_ceiling(40, 50) and not summary.below_ceiling(41, 50)


def test_a_condition_s_accuracy_is_of_the_screened_set_s_latest_runs():
    entries = [screening("vt", range(4)),
               *task_runs("vt", 5, "static-int2", 2),           # vt-4 is not screened
               run("vt", 0, "static-int2", 0.0, commit="b")]    # a repeat stands for it
    s = summary.summarise(entries, MODEL, 208, TOKENS)
    cell = s["tasks"]["vt"]["by_condition"]["static-int2"]
    assert (cell["answered"], cell["runs"]) == (1, 4)
    assert cell["score_mean"] == pytest.approx((0.0 + 1.0 + 0.5 + 0.5) / 4)
    low, high = cell["wilson_95"]
    assert low < 0.25 < high


def test_the_closing_test_pools_the_tasks_below_their_ceiling():
    """vt is below its ceiling and niah_multikey_2 is not: only vt's samples
    are paired, and semantic answering more with p < 0.05 is a yes."""
    n = 50
    entries = [screening("vt", range(n)), screening("niah_multikey_2", range(n)),
               *task_runs("vt", n, "static-int2", 20),
               *task_runs("niah_multikey_2", n, "static-int2", 48),
               *task_runs("vt", n, "semantic-80", 40), *task_runs("vt", n, "uniform-80", 25),
               *task_runs("niah_multikey_2", n, "semantic-80", 10),
               *task_runs("niah_multikey_2", n, "uniform-80", 50)]
    s = summary.summarise(entries, MODEL, 208, TOKENS)
    assert s["tasks"]["vt"]["below_ceiling"] is True
    assert s["tasks"]["niah_multikey_2"]["below_ceiling"] is False
    closing = s["closing"]
    assert closing["tasks"] == ["vt"]
    assert (closing["semantic_only"], closing["uniform_only"]) == (15, 0)
    assert closing["mcnemar_p"] < 0.05 and closing["semantic_beats_uniform"] is True
    assert closing["answer"].startswith("Yes: the semantic source answers 40/50 and uniform "
                                        "25/50")
    # each task's paired score at the level, descriptive
    score = s["tasks"]["vt"]["score_paired"]["80%"]["semantic-uniform"]
    assert score["difference"] == pytest.approx((40 - 25) * 0.5 / 50)


def test_a_condition_says_where_its_plans_left_the_needle():
    """Each run's needle cells pooled into shares, then averaged over runs;
    the share of runs with an INT2 cell on the needle."""
    a, b = run("vt", 0, "semantic-70"), run("vt", 1, "semantic-70", 0.0)
    a["results"].update(tier_shares={"INT4": 0.9, "INT2": 0.1},
                        target_page_tiers=[{"INT8": 2, "INT4": 1}, {"INT4": 1}])
    b["results"].update(tier_shares={"INT4": 1.0},
                        target_page_tiers=[{"INT4": 3, "INT2": 1}])
    s = summary.summarise([screening("vt", range(2)), a, b], MODEL, 208, TOKENS)
    cell = s["tasks"]["vt"]["by_condition"]["semantic-70"]
    assert cell["tier_shares"] == pytest.approx({"INT2": 0.05, "INT4": 0.95})
    assert cell["needle_tier_shares"] == pytest.approx(
        {"INT2": 0.125, "INT4": (0.5 + 0.75) / 2, "INT8": 0.25})
    assert cell["needle_int2_runs"] == 0.5


def test_runs_without_needle_tiers_report_none():
    s = summary.summarise([screening("vt", range(1)), run("vt", 0, "semantic-70")],
                          MODEL, 208, TOKENS)
    assert "needle_tier_shares" not in s["tasks"]["vt"]["by_condition"]["semantic-70"]


def test_no_task_below_its_ceiling_has_no_closing_answer():
    entries = [screening("vt", range(2)), *task_runs("vt", 2, "static-int2", 2)]
    closing = summary.summarise(entries, MODEL, 208, TOKENS)["closing"]
    assert closing["tasks"] == [] and closing["answer"] is None


def test_a_closing_short_of_its_runs_says_so_and_answers_nothing():
    entries = [screening("vt", range(4)), *task_runs("vt", 4, "static-int2", 0),
               *task_runs("vt", 4, "semantic-80", 4), *task_runs("vt", 3, "uniform-80", 0)]
    closing = summary.summarise(entries, MODEL, 208, TOKENS)["closing"]
    assert closing["answer"] is None and closing["missing"] == {"vt": ["uniform-80"]}


def test_a_screened_task_whose_ceiling_is_unknown_holds_the_closing_back():
    entries = [screening("vt", range(2)), screening("niah_multikey_2", range(2)),
               *task_runs("vt", 2, "static-int2", 0), *task_runs("vt", 2, "semantic-80", 2),
               *task_runs("vt", 2, "uniform-80", 0)]
    closing = summary.summarise(entries, MODEL, 208, TOKENS)["closing"]
    assert closing["answer"] is None
    assert closing["missing"] == {"niah_multikey_2": ["static-int2"]}


def test_runs_on_other_data_are_not_pooled():
    entries = [screening("vt", range(2)), *task_runs("vt", 2, "static-int2", 0),
               run("vt", 0, "static-int2", 1.0, commit="b", data={"other": "x"})]
    cell = summary.summarise(entries, MODEL, 208, TOKENS, data=DATA)["tasks"]["vt"]
    assert cell["by_condition"]["static-int2"]["answered"] == 0


def test_a_task_never_screened_is_not_summarised():
    entries = [screening("vt", range(2), complete=False), *task_runs("vt", 2, "static-int2", 0)]
    assert summary.summarise(entries, MODEL, 208, TOKENS)["tasks"] == {}


# -- the tool -----------------------------------------------------------------------------

def test_the_estimate_counts_one_turn_1_for_the_conditions_that_share_it():
    from conftest import load_tool
    tool = load_tool("ruler")
    shared = ["semantic-80", "uniform-80", "evict-80"]          # one turn 1, three closings
    alone = ["static-int2", "static-int4"]                     # a turn 1 each
    expected = (3 * tool.TURN_1_S["8K"] + 5 * tool.CLOSING_S["8K"]) / 3600
    assert tool.hours({"vt-0": shared + alone}, "8K") == pytest.approx(expected)


def test_the_tool_refuses_a_dirty_tree(tmp_path, monkeypatch):
    from conftest import load_tool
    tool = load_tool("ruler")
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log: {"git_dirty": True, "git_commit": "0" * 40})
    with pytest.raises(SystemExit):
        tool.main(["--issue", "208", "--length", "8K", "--screen",
                   "--log", str(tmp_path / "b.jsonl")])


def test_the_tool_logs_a_summary_of_what_it_pooled(tmp_path, monkeypatch, capsys):
    from conftest import load_tool
    from microinfer import benchlog
    tool = load_tool("ruler")
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log: {"git_dirty": False, "git_commit": "0" * 40})
    monkeypatch.setattr(tool, "data_hashes", lambda log: DATA)
    log = tmp_path / "b.jsonl"
    for e in [screening("vt", range(2)), *task_runs("vt", 2, "static-int2", 0)]:
        benchlog.append(e["kind"], model=MODEL, context_length=None, precision_tiers=None,
                        config=e["config"], results=e["results"], log=log)
    assert tool.main(["--issue", "208", "--length", "8K", "--summarise", "--log",
                      str(log)]) == 0
    entry = benchlog.read(log)[-1]
    assert entry["kind"] == summary.SUMMARY_KIND
    assert entry["results"]["tasks"]["vt"]["below_ceiling"] is True
    assert "closing over ['vt']" in capsys.readouterr().out


def test_the_conditions_run_on_the_tasks_named_or_else_those_below_their_ceiling():
    from conftest import load_tool
    tool = load_tool("ruler")
    entries = [screening("vt", range(2)), screening("niah_multikey_2", range(2)),
               *task_runs("vt", 2, "static-int2", 0),
               *task_runs("niah_multikey_2", 2, "static-int2", 2)]
    s = summary.summarise(entries, MODEL, 208, TOKENS)
    assert tool.chosen_tasks(None, s) == ["vt"]
    assert tool.chosen_tasks(["niah_multikey_2"], s) == ["niah_multikey_2"]
