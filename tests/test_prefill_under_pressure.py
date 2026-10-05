"""Prefill chunks bounded by time under pressure (#135).

The engine drains pressure events only between steps, and during prefill a
step is a whole chunk: on Qwen2.5-1.5B a 512-position chunk at 16K to 24K
positions takes 11 to 16 s (#108). While the level the engine last drained
is YELLOW or RED, each chunk is therefore sized from the time per position
of the chunk before it, so that a step stays within a budget. At GREEN,
chunks stay at prefill_chunk, and prefill throughput is unchanged.

Two seams here: the pure rule that sizes a chunk, and the engine (Seam A).
How long each event waited is logged by the survival experiment
(test_survival.py).
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, monitor
from microinfer.engine import PREFILLING, pressure_chunk


def test_a_chunk_fits_the_budget_in_whole_tiles():
    """10 ms a position and a 1 s budget is 100 positions, 96 in tiles of
    16; 30 ms a position is 33, 32."""
    assert pressure_chunk(0.010, budget_seconds=1.0, cap=512) == 96
    assert pressure_chunk(0.030, budget_seconds=1.0, cap=512) == 32


def test_a_chunk_is_never_below_one_tile_nor_above_the_cap():
    """100 ms a position leaves 10 positions in the budget: one tile, 16,
    the least a step runs. 1 ms a position would be 1000: the cap."""
    assert pressure_chunk(0.100, budget_seconds=1.0, cap=512) == 16
    assert pressure_chunk(0.001, budget_seconds=1.0, cap=512) == 512


def test_no_cap_leaves_the_budget_alone():
    """prefill_chunk None prefills a prompt in one step; under pressure the
    budget still bounds it."""
    assert pressure_chunk(0.001, budget_seconds=1.0, cap=None) == 992


def test_a_time_per_position_must_be_positive():
    with pytest.raises(ValueError):
        pressure_chunk(0.0, budget_seconds=1.0, cap=512)


# -- the engine ---------------------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"
MIB = 2**20


def simulated(headroom_mib):
    """A monitor that reads `headroom_mib` MiB of headroom at every poll."""
    return monitor.Monitor(reader=lambda: headroom_mib * MIB, poll_s=0.002)


def chunks(engine, headroom_mib, positions=2048):
    """The prefill chunks of one generation beside a monitor reading
    `headroom_mib`, as their reports show them."""
    ids = np.random.default_rng(135).integers(1000, 100_000, positions).astype(np.int32)
    ends = []
    engine.start_monitor(simulated(headroom_mib))
    try:
        engine.generate(ids, 1, report=lambda s, p, t: ends.append(p) if s == PREFILLING else None)
    finally:
        engine.stop_monitor()
    return list(np.diff([0] + ends))


@pytest.fixture(scope="module")
def adaptive():
    e = Engine(require_model(NAME), kv_adaptive=True, prefill_chunk=512,
               pressure_chunk_seconds=1e-6)
    e.load_weights()
    return e


def test_under_red_the_chunks_after_the_first_shrink_to_the_budget(adaptive):
    """The first chunk runs before any event is drained, at prefill_chunk.
    Once RED is drained, a budget no chunk can meet leaves one tile a step.
    The chunk the event arrived during may still be whole, so it is
    skipped."""
    sizes = chunks(adaptive, headroom_mib=100)
    assert sizes[0] == 512
    assert set(sizes[2:]) == {16}, sizes


def test_at_green_the_chunks_stay_whole(adaptive):
    sizes = chunks(adaptive, headroom_mib=8000)
    assert sizes == [512, 512, 512, 512]


def test_an_engine_that_does_not_adapt_keeps_its_chunks_under_red():
    """Smaller chunks would buy it nothing: it makes no plans."""
    static = Engine(require_model(NAME), prefill_chunk=512, pressure_chunk_seconds=1e-6)
    static.load_weights()
    assert chunks(static, headroom_mib=100) == [512, 512, 512, 512]
