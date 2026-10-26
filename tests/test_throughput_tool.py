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


def test_each_decoded_token_s_time_is_taken_from_the_step_before():
    """From the end of prefill, the time between one report and the next:
    one per decoded token, the prefill's own token aside."""
    clock = iter([10.0, 10.5, 10.75, 11.0])
    steps: list[float] = []
    report = tool.step_clock(steps, now=lambda: next(clock))
    report("prefilling", 512, None)
    report("decoding", 513, 7)
    report("decoding", 514, 8)
    report("decoding", 515, 9)
    assert steps == [0.5, 0.25, 0.25]


@pytest.mark.slow
def test_the_overhead_gives_each_side_s_per_token_latency(engine):
    """With the monitor on and off, the P50 and P99 of every decoded token's
    time, over every run of each."""
    prompt = np.arange(1000, 1064, dtype=np.int32)
    results = tool.monitor_overhead(engine, prompt, repeat=2, warmup=0)
    for side in ("off", "on"):
        latency = results["step_ms"][side]
        assert latency["n"] == 2 * tool.DECODE_STEPS
        assert 0 < latency["p50"] <= latency["p99"]
