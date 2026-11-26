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


def test_decode_is_timed_by_its_steps_run_by_run(monkeypatch):
    """Each run's rate is its timed steps over their time, the prefill left
    out, and the figure is the median run's (#214)."""
    tool = load_tool("throughput")
    clock = iter([1.0, 1.5, 2.0,  # run 1: steps of 0.5 s
                  11.0, 11.25, 11.5,  # run 2: 0.25 s
                  21.0, 22.0, 23.0])  # run 3: 1 s

    class Engine:
        def generate(self, prompt, new, *, stop_at_eos, report=None):
            for position in range(3):
                if report is not None:
                    report(tool.DECODING, position, 0)

    clocked = tool.step_clock
    monkeypatch.setattr(tool, "step_clock",
                        lambda token_s: clocked(token_s, now=lambda: next(clock)))
    got = tool.decode_runs(Engine(), [1, 2], repeat=3, warmup=0)
    assert got["runs_tokens_per_second"] == [2.0, 4.0, 1.0]
    assert got["tokens_per_second"] == 2.0 and got["steps_timed"] == 2
