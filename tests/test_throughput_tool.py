"""The throughput tool's per-token latency (#187)."""

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import Engine

SMALL = "qwen2.5-0.5b-instruct"

tool = load_tool("throughput")


@pytest.fixture(scope="module")
def engine():
    engine = Engine(require_model(SMALL))
    engine.load_weights()
    return engine


def test_each_decoded_token_s_time_is_taken_from_the_token_before():
    """The time between one decoded token's report and the next's: the
    prefill's report, before its own greedy choice, starts no interval."""
    clock = iter([10.5, 10.75, 11.25])
    token_s: list[float] = []
    report = tool.step_clock(token_s, now=lambda: next(clock))
    report("prefilling", 512, None)
    report("decoding", 513, 7)
    report("decoding", 514, 8)
    report("decoding", 515, 9)
    assert token_s == [0.25, 0.5]


@pytest.mark.slow
def test_the_overhead_gives_each_side_s_per_token_latency(engine):
    """With the monitor on and off, the median and P99 of every decoded
    token's time but each run's first, over every run of each."""
    prompt = np.arange(1000, 1064, dtype=np.int32)
    results = tool.monitor_overhead(engine, prompt, repeat=2, warmup=0)
    for side in ("off", "on"):
        latency = results["token_ms"][side]
        assert latency["n"] == len(latency["runs"]) == 2 * (tool.DECODE_STEPS - 1)
        assert 0 < latency["median"] <= latency["p99"] <= latency["max"]
