"""Page-level damage: a page's score against its INT4 downgrade's cost (#188).

A page is measured on a conversation started from the plan point's
snapshot, moved alone to INT4, the measured tokens forced after it. The
proof that a move leaves nothing behind is equality, not tolerance: a page
moved down and back from its shadow gives the logits of a cache never
moved, to the bit.
"""

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import Engine, _microinfer, benchlog, continuation, controller, page_damage, rq3

NAME = "qwen2.5-0.5b-instruct"
PAGE_TOKENS = _microinfer.device.page_tokens


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), prefill_chunk=128)
    e.load_weights()
    return e


def test_ranks_give_ties_the_mean_of_theirs():
    assert page_damage.ranks([3.0, 1.0, 3.0, 2.0]).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_spearman_is_the_correlation_of_ranks():
    assert page_damage.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert page_damage.spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)
    assert page_damage.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # one swap of four: 1 - 6 * 2 / (4 * 15)
    assert page_damage.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert page_damage.spearman([1, 1, 1], [1, 2, 3]) is None
    assert page_damage.spearman([1], [1]) is None
    with pytest.raises(ValueError):
        page_damage.spearman([1, 2], [1])


def test_pages_are_drawn_from_those_a_plan_may_move_that_were_scored():
    """Outside the recency floor, full, held and observed: drawn without
    replacement, the same by the same seed, all of them if fewer."""
    positions = 10 * PAGE_TOKENS + 5
    tiers = np.zeros((2, 16), np.int8)
    tiers[:, 11:] = -1  # not held
    tiers[1, 0] = controller.PLACES.index(benchlog.EVICTED)
    scores = np.ones((2, 16), np.float32)
    scores[0, 1] = np.nan
    movable = page_damage.drawable(scores, tiers, positions, PAGE_TOKENS)
    full = (positions - controller.DEFAULT_RECENCY_FLOOR) // PAGE_TOKENS
    expected = np.zeros_like(movable)
    expected[:, :full] = True
    expected[0, 1] = expected[1, 0] = False
    assert (movable == expected).all()
    layers, pages = page_damage.draw_pages(movable, 5, seed=3)
    assert len(set(zip(layers.tolist(), pages.tolist()))) == 5
    assert movable[layers, pages].all()
    again = page_damage.draw_pages(movable, 5, seed=3)
    assert (again[0] == layers).all() and (again[1] == pages).all()
    every = page_damage.draw_pages(movable, 1000, seed=3)
    assert len(every[0]) == int(movable.sum())


def logged_sample(log, sample_id, scores, kl, task="needle", screened=True):
    of = page_damage.Draws(task, "m", screened=screened)
    benchlog.append(page_damage.KIND, model="m", context_length=None, precision_tiers=None,
                    config=of.config(sample_id),
                    results={"scores": scores, "kl": kl,
                             "rho": page_damage.spearman(scores, kl)},
                    log=log)


def test_the_summary_gives_rho_pooled_and_per_sample_with_intervals(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    logged_sample(log, "a", [1, 2, 3, 4], [1, 2, 3, 4])
    logged_sample(log, "b", [5, 6, 7, 8], [8, 7, 6, 5])
    logged_sample(log, "c", [1, 2, 3, 4], [1, 3, 2, 4])
    logged_sample(log, "c", [1, 2, 3, 4], [9, 9, 9, 9], task="anchor")  # another task's
    logged_sample(log, "b", [1, 2, 3, 4], [9, 9, 9, 9], screened=False)  # a check's
    needle = page_damage.Draws("needle", "m")
    s = page_damage.summarise(benchlog.read(log), needle, ["a", "b", "c"])
    assert s["samples"] == 3 and s["pairs"] == 12
    assert s["mean_sample_rho"] == pytest.approx((1 - 1 + 0.8) / 3)
    scores = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
    kl = [1, 2, 3, 4, 8, 7, 6, 5, 1, 3, 2, 4]
    assert s["pooled_rho"] == pytest.approx(page_damage.spearman(scores, kl))
    for key in ("pooled_rho", "mean_sample_rho"):
        low, high = s[f"{key}_bootstrap_95"]
        assert low <= s[key] <= high
    assert s["positive_samples"] == 2
    assert len(s["entries"]) == 3
    with pytest.raises(ValueError, match="d"):
        page_damage.summarise(benchlog.read(log), needle, ["a", "d"])
    commit = benchlog.read(log)[0]["git_commit"]
    assert page_damage.logged(needle, commit, log) == {"a", "b", "c"}
    assert page_damage.logged(needle._replace(screened=False), commit, log) == {"b"}
    assert page_damage.logged(needle, "another commit", log) == set()


@pytest.mark.slow
def test_a_page_moved_down_and_back_gives_the_logits_of_a_cache_never_moved(engine):
    """Moved to INT4 and back to FP16 from its shadow, a page is its bytes
    before, and the next forward pass's logits are those of a cache never
    planned, to the bit; moved to INT4 alone, they depart."""
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(for_snapshot=True) as chat:
        chat.turn(prompt(300, 1), 8)
        snapshot = chat.snapshot()
    question = prompt(20, 2)

    def forced(back: bool | None):
        with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=snapshot) as chat:
            if back is not None:
                chat.downgrade(3, 2, "INT4")
            if back:
                chat.upgrade(3, 2)
            return chat.force(question, logits=True, after_turn=True)

    never = forced(None)
    assert never.shape[0] == len(question) + 1  # the token turn 1 chose last, first
    assert np.array_equal(forced(True), never)
    assert not np.array_equal(forced(False), never)


@pytest.mark.slow
def test_a_page_moves_only_the_way_it_is_named(engine):
    """A downgrade goes down, an upgrade up, and forcing after a turn needs
    one to have left its last token."""
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation() as chat:
        chat.force(prompt(300, 1))
        with pytest.raises(RuntimeError, match="no turn"):
            chat.force(prompt(4, 2), after_turn=True)
        with pytest.raises(ValueError, match="no such move"):
            chat.downgrade(0, 0, "FP16")
        with pytest.raises(ValueError, match="no such move"):
            chat.upgrade(0, 0)
        chat.downgrade(0, 0, "INT4")
        with pytest.raises(ValueError, match="no such move"):
            chat.downgrade(0, 0, "INT8")


@pytest.mark.slowest
def test_the_tool_logs_one_sample_of_each_task_and_summarises(tmp_path, monkeypatch):
    """On Qwen2.5-0.5B, unscreened: one sample of each task, two pages each,
    logged with each page's score and damage; run again, nothing is."""
    tool = load_tool("page_damage")
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    for task in page_damage.TASKS:
        args = ["--issue", "188", "--task", task, "--model", NAME, "--samples", "1",
                "--pages-per-sample", "2", "--unscreened", "--log", str(log)]
        assert tool.main(args) == 0
        assert tool.main(args) == 0  # taken up: nothing run again
        assert tool.main([*args, "--summarise"]) == 0
    runs = [e for e in benchlog.read(log) if e["kind"] == page_damage.KIND]
    assert sorted(e["config"]["task"] for e in runs) == sorted(page_damage.TASKS)
    for e in runs:
        r = e["results"]
        assert len(r["kl"]) == len(r["scores"]) == len(r["layers"]) == 2
        assert all(k > 0 for k in r["kl"])
        assert r["measured_positions"] > 0 and r["candidates"] >= 2
    summaries = [e for e in benchlog.read(log) if e["kind"] == page_damage.SUMMARY_KIND]
    assert [e["config"]["task"] for e in summaries] == list(page_damage.TASKS)


def test_a_page_s_damage_is_the_mean_kl_continuation_fidelity_measures():
    """KL(reference || logits), its mean over positions, as
    continuation.measures gives it, from a reference softmaxed once."""
    rng = np.random.default_rng(188)
    reference = (rng.standard_normal((5, 1000)) * 3).astype(np.float32)
    logits = reference + (rng.standard_normal(reference.shape) * 0.01).astype(np.float32)
    expected = np.mean(continuation.measures(reference, logits)["kl"])
    held = page_damage.Reference(reference)
    assert held.damage(logits) == pytest.approx(expected, rel=1e-9)
    assert held.damage(reference) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        held.damage(logits[:4])
