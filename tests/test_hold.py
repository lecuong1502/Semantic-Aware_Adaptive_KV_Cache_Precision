"""Engine hold mode: a generation kept in progress for as long as a scenario
runs (#51).

RQ1's with-engine runs need the engine holding a full context, decoding the
whole time. The window ends, the hold must not: holding prefills the
context less a span to decode, decodes to the end of the window, then goes
back to the end of the prompt and decodes the span again, over the same
pages. Its memory is that of the full context from the first pass on.

Tested on Qwen2.5-0.5B over a short context, which rewinds every few
seconds; the tool is the same code at 32K on 1.5B.
"""

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, nvml

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "hold_engine.py"
MODEL = "qwen2.5-0.5b-instruct"


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(MODEL))
    e.load_weights()
    return e


def test_holding_decodes_the_span_again_and_again_as_a_fresh_generation_would(engine):
    """Each pass after a rewind is exactly what generating from the prompt
    and the token it starts from gives: nothing of the pass before is read.
    The cache holds the full context and does not grow from pass to pass."""
    prompt = np.random.default_rng(51).integers(0, 1000, 120).astype(np.int32)
    context = 160
    stop = threading.Event()
    passes, cache_bytes, states = [], [], set()

    def report(state, position, token):
        states.add(state)
        if state != "decoding":
            return
        assert len(prompt) < position <= context
        if position == len(prompt) + 1:  # a pass begins
            passes.append([])
            cache_bytes.append(engine.footprint().kv_cache)
        passes[-1].append(token)
        if len(passes) == 4 and len(passes[-1]) == context - len(prompt):
            stop.set()

    engine.hold(prompt, context=context, stop=stop, report=report)

    assert states == {"prefilling", "decoding"}
    span = context - len(prompt)
    assert [len(p) for p in passes] == [span] * 4
    first = engine.generate(prompt, span + 1, stop_at_eos=False)
    assert list(first[1:]) == passes[0]
    for before, after in zip(passes, passes[1:]):
        again = engine.generate(np.append(prompt, before[-1]), span, stop_at_eos=False)
        assert list(again) == after
    assert cache_bytes[1] == cache_bytes[2] == cache_bytes[3] > 0
    assert engine.footprint().kv_cache == 0  # released on stopping

    stopped_early = []

    def stop_at_once(state, position, token):
        stopped_early.append(state)
        stop.set()

    stop.clear()
    long_prompt = np.random.default_rng(1).integers(0, 1000, 2000).astype(np.int32)
    engine.hold(long_prompt, context=2100, stop=stop, report=stop_at_once)
    assert stopped_early == ["prefilling"]  # stopped after the first chunk, not the fourth

    with pytest.raises(ValueError, match="FP16"):
        Engine(require_model(MODEL), kv_tier="INT8").hold(prompt, context=context, stop=stop)
    with pytest.raises(ValueError, match="context"):
        engine.hold(prompt, context=len(prompt), stop=stop)


def test_other_threads_run_while_the_engine_works(engine):
    """The extension releases the GIL while the device works, and while it
    frees what a step used, which waits for the work queued before it: a
    thread beside a long prefill, as the status writer and the pressure
    monitor are, is never held up for as long as a chunk takes."""
    ticks, done = [], threading.Event()

    def tick():
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.005)

    ticker = threading.Thread(target=tick)
    ticker.start()
    prompt = np.random.default_rng(0).integers(1000, 100000, 4096).astype(np.int32)
    engine.generate(prompt, 1)
    done.set()
    ticker.join()
    assert np.diff(ticks).max() < 0.1


def read_status(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def test_the_tool_reports_its_state_every_second_and_gives_everything_back(tmp_path):
    """Run as RQ1 runs it: loading, prefilling, then decoding until told to
    stop, the status file rewritten at least once a second throughout, as
    the spec asks (it aims at four). On SIGTERM it stops cleanly, says so,
    and leaves nothing on the device."""
    require_model(MODEL)
    status = tmp_path / "hold.json"
    proc = subprocess.Popen([sys.executable, str(TOOL), "--model", MODEL, "--context", "2048",
                             "--decode-positions", "256", "--status", str(status)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        seen, stamps = set(), []
        end = time.monotonic() + 120
        while time.monotonic() < end:
            s = read_status(status)
            if s is not None:
                seen.add(s["state"])
                stamps.append((time.monotonic_ns(), s["t_mono_ns"]))
                if s["state"] == "decoding" and s["tokens_per_second"] > 0 \
                        and len([1 for _, u in stamps if u > stamps[-1][1] - 3e9]) > 20:
                    break
            time.sleep(0.1)
        assert s is not None and s["state"] == "decoding", (s, proc.poll())
        assert s["pid"] == proc.pid and s["context"] == 2048
        assert 1792 < s["position"] <= s["held_positions"] <= 2048
        assert s["tokens_per_second"] > 10
        assert {"loading", "decoding"} <= seen
        # Never more than a second between one write and the next seen.
        written = np.unique([u for _, u in stamps])
        assert np.diff(written).max() < 1.2e9
        assert proc.pid in {p.pid for p in nvml.processes()}

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        _, err = proc.communicate(timeout=10)
        if proc.returncode != 0:
            print(err, file=sys.stderr)
    assert read_status(status)["state"] == "stopped"
    assert proc.pid not in {p.pid for p in nvml.processes()}
