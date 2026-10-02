"""The importance scorer on Qwen2.5 (#100, Seam A).

Attention sinks: models put a large share of every query's attention on the
first positions, whatever they hold. #88 gives the first page no hard
protection from downgrades, and leaves it to the scorer to rate it highly.
Here it does, on both models, from a real prefill and decode: averaged over
the layers, the first page has the highest score of all, and it is among
the three highest in at least three layers in four.

What the scorer costs decoding is measured by tools/throughput.py --monitor
--scorer and logged; its measurement is checked here to run and to leave the
engine as it found it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_paged_engine import decode
from test_static_tiers import P

from conftest import each, require_model
from microinfer import Engine, model
from microinfer.golden import GoldenError, GoldenSet

REPO = Path(__file__).resolve().parent.parent
MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct")
PROMPTS = ("long-03", "medium-01", "adversarial-04")
#: How many of the highest a first page must be among, in how many layers.
TOP, LAYER_SHARE = 3, 0.75

spec = importlib.util.spec_from_file_location("throughput", REPO / "tools" / "throughput.py")
throughput = importlib.util.module_from_spec(spec)
spec.loader.exec_module(throughput)


def golden_for(name):
    try:
        return GoldenSet(REPO / "tests" / "golden" / name)
    except GoldenError as exc:
        pytest.skip(str(exc))


def test_the_first_page_scores_among_the_highest_without_protection():
    """After a prompt and 64 decoded tokens, the first page's score,
    averaged over the layers, is the highest of every page's; and in at
    least three layers in four it is among the three highest."""
    def sink(engine, golden, prompt_id):
        ids = golden[prompt_id].token_ids
        cache = model.PagedCache(engine.config, scoring=True)
        new = 64
        decode(engine, ids, new, cache)
        attended = -(-(len(ids) + new - 1) // P)  # the pages the last step read
        scores = cache.scores.download()[:, :attended]
        mean = scores.mean(axis=0)
        assert mean.argmax() == 0, f"page {mean.argmax()} outscores the first: {mean}"
        among = [(row > row[0]).sum() < TOP for row in scores]
        assert np.mean(among) >= LAYER_SHARE, f"among the {TOP} highest in {sum(among)} layers"

    for name in MODELS:
        engine = Engine(require_model(name))
        engine.load_weights()
        golden = golden_for(name)
        each(PROMPTS, lambda p: sink(engine, golden, p), name=lambda p: f"{name} {p}")
        del engine


def test_the_overhead_measurement_runs_and_leaves_the_engine_as_it_was():
    """tools/throughput.py's comparison, run small: decode with the monitor
    and the scorer on against both off, interleaved; afterwards neither is
    on."""
    engine = Engine(require_model(MODELS[0]))
    engine.load_weights()
    prompt = np.arange(1000, 1000 + 2 * P, dtype=np.int32)
    results = throughput.monitor_overhead(engine, prompt, repeat=2, warmup=0, scorer=True)
    assert len(results["on"]["runs"]) == len(results["off"]["runs"]) == 2
    assert results["on"]["mean"] > 0 and results["off"]["mean"] > 0
    assert engine.kv_scoring is False and engine._monitor is None
