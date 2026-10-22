"""Layer weights from each layer's measured sensitivity (#145).

ADR-0018's layer-weights ablation: each layer alone is taken to INT2 on
WikiText-2, the KL divergence of the next-token distribution from FP16's is
logged, and those divergences, normalised to a mean of 1, are weights the
planner multiplies a page's score by before planning.

Three seams: the controller's pure planning with weights, the weights read
from the benchmark log, and tools/layer_sensitivity.py, which measures and
logs the divergences.
"""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, benchlog, controller, gate, monitor
from microinfer.controller import PlanOptions

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "layer_sensitivity", REPO / "tools" / "layer_sensitivity.py")
sensitivity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensitivity)

PAGE_BYTES = {"FP16": 1000, "INT8": 550, "INT4": 300, "INT2": 170}
ERRORS = {"FP16": 0.0, "INT8": 0.001, "INT4": 0.01, "INT2": 0.1}
MOVES = {(a, b): 1e-5 for a in controller.TIERS for b in controller.TIERS if a != b}


def plan(target, scores, options=controller.DEFAULT_OPTIONS):
    """A plan for `target` bytes over every page at FP16, 320 positions on
    pages of 32, the last 128 the floor: pages 0 to 5 are candidates."""
    tiers = np.zeros(scores.shape, dtype=np.int8)
    return controller.target_plan_arrays(target, monitor.RED, scores, tiers, PAGE_BYTES, ERRORS,
                                         move_seconds=MOVES, positions=320, page_tokens=32,
                                         options=options)


def moves(p):
    return list(zip(p.layers.tolist(), p.pages.tolist(), p.current.tolist()))


def test_without_weights_every_layer_counts_alike():
    assert PlanOptions().layer_weights is None


@pytest.mark.parametrize("target", [3000, 9000])
def test_a_page_s_score_is_multiplied_by_its_layer_s_weight_before_planning(target):
    """A plan with weights is the plan from scores already multiplied by
    them, move for move and cost for cost, and records the weights."""
    scores = np.random.default_rng(145).random((3, 10))
    weights = (0.5, 1.0, 1.5)
    weighted = plan(target, scores, PlanOptions(layer_weights=weights))
    by_hand = plan(target, scores * np.array(weights)[:, None])
    assert moves(weighted) == moves(by_hand)
    np.testing.assert_array_equal(weighted.costs, by_hand.costs)
    assert weighted.options.layer_weights == weights


def test_equal_weights_leave_a_plan_as_it_is():
    """Weights all alike scale every cost alike: the plan is the plan
    without weights, move for move."""
    scores = np.random.default_rng(1454).random((3, 10))
    assert moves(plan(9000, scores, PlanOptions(layer_weights=(2.0, 2.0, 2.0)))) == \
        moves(plan(9000, scores))


def test_a_sensitive_layer_is_spared():
    """Equal scores, and layer 1 ten times as sensitive: the six candidates
    of layer 0 all go to INT8 before any page of layer 1 is touched."""
    p = plan(2700, np.ones((2, 10)), PlanOptions(layer_weights=(1.0, 10.0)))
    assert set(p.layers.tolist()) == {0}


def test_an_upgrade_plan_weighs_scores_too():
    """Upgrades are planned from the same weighted scores: the plan is the
    one from scores multiplied by hand, and records the weights."""
    scores = np.random.default_rng(1451).random((3, 10))
    tiers = np.full((3, 10), controller.TIERS.index("INT4"), dtype=np.int8)
    shadowed = np.ones((3, 10), dtype=bool)
    weights = (2.0, 0.5, 0.5)

    def up(s, options=controller.DEFAULT_OPTIONS):
        return controller.upgrade_plan_arrays(
            monitor.DEFAULT.yellow_below_bytes + 2**20 * 512 + 20000, monitor.DEFAULT, s, tiers,
            shadowed, PAGE_BYTES, ERRORS, move_seconds=MOVES, last_downgrade_seconds=None,
            options=options)

    weighted = up(scores, PlanOptions(layer_weights=weights))
    by_hand = up(scores * np.array(weights)[:, None])
    assert len(weighted) and moves(weighted) == moves(by_hand)
    assert weighted.options.layer_weights == weights


@pytest.mark.parametrize("weights", [(1.0, 0.0), (1.0, -1.0), (1.0, float("nan")),
                                     (1.0, float("inf")), ()])
def test_a_weight_is_positive_and_finite(weights):
    with pytest.raises(ValueError, match="weight"):
        PlanOptions(layer_weights=weights)


def test_a_weight_for_every_layer():
    with pytest.raises(ValueError, match="layers"):
        plan(3000, np.ones((3, 10)), PlanOptions(layer_weights=(1.0, 2.0)))


# -- the weights from the log ---------------------------------------------------------------

def log_divergences(log, model, kl):
    benchlog.append("layer-sensitivity", model=model, context_length=256,
                    precision_tiers={"INT2": 1.0}, config={"issue": 145},
                    results={"kl": list(kl)}, log=log)


def test_the_weights_are_the_last_logged_divergences_normalised_to_a_mean_of_one(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    log_divergences(log, "m", [9.0, 9.0, 9.0])
    log_divergences(log, "other", [5.0, 5.0])
    log_divergences(log, "m", [1.0, 2.0, 3.0])
    assert controller.logged_layer_weights("m", log) == pytest.approx((0.5, 1.0, 1.5))


def test_no_weights_without_a_measurement(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    log_divergences(log, "other", [1.0, 2.0])
    with pytest.raises(ValueError, match="layer-sensitivity"):
        controller.logged_layer_weights("m", log)


# -- the measurement ------------------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"


def test_one_layer_alone_at_int2():
    """The tier map that holds layer 1 of 3 at INT2 and every other at FP16,
    over a window of 3 pages."""
    assert sensitivity.one_layer_at(1, layers=3, pages=3) == [
        [], ["INT2", "INT2", "INT2"], []]


def test_the_divergence_is_gate_s_kl_summed_over_rows():
    """The reference's terms taken once give gate.kl_divergence's sum, over
    more rows than one chunk holds, a last chunk short."""
    rng = np.random.default_rng(1453)
    reference = (4 * rng.standard_normal((300, 500))).astype(np.float32)
    ours = (reference + 0.3 * rng.standard_normal(reference.shape)).astype(np.float32)
    want = float(gate.kl_divergence(reference, ours).sum())
    with ThreadPoolExecutor(4) as pool:
        held = sensitivity.Reference(reference, pool)
        assert held.summed_kl(ours) == pytest.approx(want, rel=1e-9)
        assert held.summed_kl(reference) == pytest.approx(0, abs=1e-9)


@pytest.mark.slow
def test_the_tool_logs_one_divergence_per_layer(tmp_path, monkeypatch):
    """One short window on Qwen2.5-0.5B: a divergence for each of its 24
    layers, each above 0, and weights from them with a mean of 1."""
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(sensitivity.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert sensitivity.main(["--issue", "145", "--model", NAME, "--window", "128",
                             "--windows", "1", "--log", str(log)]) == 0
    (entry,) = benchlog.read(log)
    assert entry["kind"] == "layer-sensitivity" and entry["model"] == NAME
    kl = entry["results"]["kl"]
    assert len(kl) == 24 and all(np.isfinite(kl)) and min(kl) > 0
    assert entry["results"]["scored_tokens"] == 127
    assert np.mean(controller.logged_layer_weights(NAME, log)) == pytest.approx(1.0)


# -- the engine ------------------------------------------------------------------------------

def test_the_engine_takes_a_weight_for_each_of_its_layers():
    with pytest.raises(ValueError, match="24 layers"):
        Engine(require_model(NAME), kv_plan_options=PlanOptions(layer_weights=(1.0, 2.0)))


@pytest.mark.slow
def test_the_engine_s_plans_record_its_weights():
    weights = tuple(np.linspace(0.5, 1.5, 24).tolist())
    engine = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True, prefill_chunk=None,
                    kv_plan_options=PlanOptions(layer_weights=weights))
    engine.load_weights()
    ids = np.random.default_rng(1452).integers(1000, 100_000, 1024).astype(np.int32)
    with engine.conversation() as chat:
        chat.turn(ids, 8, stop_at_eos=False)
        record = chat.plan(0.7, source="semantic")
    assert record.plan.options.layer_weights == weights and record.controlled["target_met"]
