"""Prefill chunks bounded by time under pressure (#135).

The engine drains pressure events only between steps, and during prefill a
step is a whole chunk: on Qwen2.5-1.5B a 512-position chunk at 16K to 24K
positions takes 11 to 16 s (#108). While an adaptive engine's monitor reads
YELLOW or RED, each chunk is therefore sized from what the steps before it
cost, a fixed part and a part per position, so that a step takes about
pressure_chunk_seconds, or at most twice a one-tile step where the fixed
part alone overruns that. At GREEN,
chunks stay at prefill_chunk, and prefill throughput is unchanged.

Two seams here: the pure rules that cost a step and size a chunk, and the
engine (Seam A).
How long each event waited is logged by the survival experiment
(test_survival.py).
"""

import numpy as np
import pytest

from test_adaptive_engine import simulated

from conftest import require_model
from microinfer import Engine, _microinfer
from microinfer.engine import PREFILLING, StepCost, pressure_chunk


def test_a_chunk_fits_its_time_in_whole_tiles():
    """Worked with a tile of 16 and no fixed cost: 10 ms a position and 1 s
    is 100 positions, 96 in tiles; 30 ms a position is 33, 32."""
    assert pressure_chunk(0.0, 0.010, seconds=1.0, tile=16, cap=512) == 96
    assert pressure_chunk(0.0, 0.030, seconds=1.0, tile=16, cap=512) == 32


def test_a_step_s_fixed_cost_comes_off_its_time_first():
    """A step of 0.19 s fixed and 8.7 ms a position: 0.5 s leaves 0.31 s, 35
    positions, 32 in tiles."""
    assert pressure_chunk(0.19, 0.0087, seconds=0.5, tile=16, cap=512) == 32


def test_a_fixed_cost_within_the_time_keeps_a_chunk_within_it_but_for_one_tile():
    """0.65 s fixed and 10 ms a position, against 0.8 s: 15 positions fit,
    so one tile, 0.81 s. The fixed part does not overrun the time, so the
    chunk does not grow to spend the fixed part again."""
    assert pressure_chunk(0.65, 0.010, seconds=0.8, tile=16, cap=512) == 16


def test_when_the_fixed_cost_alone_overruns_a_chunk_spends_as_much_again():
    """At 31K positions a step costs about 1.24 s before its first position,
    and 29 ms a position: no chunk fits 0.5 s. One tile would make every
    step mostly fixed cost; positions worth the fixed cost again, 42, 32 in
    tiles, keep a step within twice what one tile would take."""
    assert pressure_chunk(1.24, 0.029, seconds=0.5, tile=16, cap=512) == 32
    assert pressure_chunk(1.24, 0.010, seconds=0.5, tile=16, cap=512) == 112


def test_a_chunk_is_never_below_one_tile_nor_above_the_cap():
    assert pressure_chunk(0.0, 0.100, seconds=1.0, tile=16, cap=512) == 16
    assert pressure_chunk(0.0, 0.001, seconds=1.0, tile=16, cap=512) == 512
    assert pressure_chunk(2.0, 0.001, seconds=1.0, tile=16, cap=512) == 512


def test_no_cap_leaves_the_time_to_bound_it():
    """prefill_chunk None prefills a prompt in one step; under pressure the
    chunk time still bounds it."""
    assert pressure_chunk(0.0, 0.001, seconds=1.0, tile=16, cap=None) == 992


def test_a_time_per_position_must_be_positive():
    with pytest.raises(ValueError):
        pressure_chunk(0.0, 0.0, seconds=1.0, tile=16, cap=512)


# -- what a step costs --------------------------------------------------------------------

def test_one_step_is_taken_for_all_per_position():
    """Before a second size is seen, a step's time is all per position, as
    if it had no fixed cost."""
    cost = StepCost()
    cost.observe(512, 4.66)
    assert cost.fixed_seconds == 0.0
    assert cost.seconds_per_position == pytest.approx(4.66 / 512)


def test_two_sizes_wide_apart_split_a_step_into_its_two_costs():
    """512 positions in 4.66 s and 32 in 0.47 s: 8.73 ms a position, and
    0.19 s fixed."""
    cost = StepCost()
    cost.observe(512, 4.66)
    cost.observe(32, 0.47)
    assert cost.seconds_per_position == pytest.approx((4.66 - 0.47) / 480)
    assert cost.fixed_seconds == pytest.approx(0.47 - 32 * (4.66 - 0.47) / 480)


def test_a_step_of_a_size_close_to_the_last_keeps_the_per_position_cost():
    """A later 32-position step in 0.60 s is close in size to the step before
    it: the per-position cost stands, and the fixed part, which has grown
    with the context, follows."""
    cost = StepCost()
    cost.observe(512, 4.66)
    cost.observe(32, 0.47)
    b = cost.seconds_per_position
    cost.observe(32, 0.60)
    assert cost.seconds_per_position == pytest.approx(b)
    assert cost.fixed_seconds == pytest.approx(0.60 - 32 * b)


def test_only_steps_in_a_row_are_compared():
    """A step is compared only with the one just before it, at nearly the
    same context: one from further back ran at a shorter context, where a
    position cost less, and would make the per-position cost too small.
    Chunks of 512, 400, 300 and 200 make no split."""
    cost = StepCost()
    for n in (512, 400, 300, 200):
        cost.observe(n, 0.5 + 0.004 * n)
    assert not cost.split
    assert cost.fixed_seconds == 0.0


def test_a_split_that_makes_no_sense_before_any_other_is_all_per_position():
    """Noise can make the larger step the quicker one: a negative cost per
    position is no estimate, and with none before it the step is taken as
    all per position."""
    cost = StepCost()
    cost.observe(512, 0.40)
    cost.observe(32, 0.47)
    assert cost.fixed_seconds == 0.0
    assert cost.seconds_per_position == pytest.approx(0.47 / 32)


def test_a_split_that_makes_no_sense_after_a_good_one_keeps_the_good_one():
    """Once a step's cost is split, a noisy pair does not undo it: the
    per-position cost stands, and the fixed part follows the latest step."""
    cost = StepCost()
    cost.observe(512, 4.66)
    cost.observe(32, 0.47)
    b = cost.seconds_per_position
    cost.observe(128, 0.40)  # quicker than 32 positions were, the step before: noise
    assert cost.seconds_per_position == pytest.approx(b)
    assert cost.fixed_seconds == pytest.approx(max(0.0, 0.40 - 128 * b))


def test_until_the_cost_is_split_a_chunk_is_at_most_half_the_last():
    """So that two steps in a row are wide apart, and split it. 300 is
    allowed by the time, but half of the last 512 is 256."""
    assert pressure_chunk(0.0, 0.002, seconds=0.6, tile=16, cap=512, half_of=512) == 256
    assert pressure_chunk(0.0, 0.002, seconds=0.6, tile=16, cap=512) == 288
    assert pressure_chunk(0.0, 0.002, seconds=0.6, tile=16, cap=512, half_of=16) == 16


def test_a_cost_is_known_once_a_step_is_seen():
    cost = StepCost()
    assert not cost.known
    cost.observe(512, 1.0)
    assert cost.known


# -- the engine ---------------------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"
#: The attention kernel's query tile, the unit a chunk is sized in.
TILE = _microinfer.attention_tiles["query"]


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


def test_under_red_the_chunks_after_the_first_shrink(adaptive):
    """The first chunk runs before any event is drained, at prefill_chunk.
    Under RED, with a chunk time no step can meet, chunks shrink to whole
    tiles worth a step's fixed cost. The chunk the event arrived during may
    still be whole, so it is skipped, and so is the last, the prompt's
    remainder."""
    sizes = chunks(adaptive, headroom_mib=100)
    assert sizes[0] == 512 and len(sizes) > 4, sizes
    assert all(s < 512 and s % TILE == 0 for s in sizes[2:-1]), sizes


def test_at_green_the_chunks_stay_whole(adaptive):
    sizes = chunks(adaptive, headroom_mib=8000)
    assert sizes == [512, 512, 512, 512]


def test_an_engine_that_does_not_adapt_keeps_its_chunks_under_red():
    """Smaller chunks would buy it nothing: it makes no plans."""
    static = Engine(require_model(NAME), prefill_chunk=512, pressure_chunk_seconds=1e-6)
    static.load_weights()
    assert chunks(static, headroom_mib=100) == [512, 512, 512, 512]
