"""llama.cpp as an external baseline (#211): its server, what its log says it
holds and how it ended, and the survival experiment's contention played
against it.

llama.cpp is a process of its own, so the seams are its command, its log,
its HTTP stream and the run. The stream is tested against a small HTTP
server that answers as llama-server does; the run against a server that is
not one. The last test runs the real binary, and is skipped where it is not
built (tools/build_llamacpp.py).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from conftest import load_tool
from microinfer import ModelConfig, benchlog, llamacpp, llamacpp_survival, survival
from microinfer import survival_matrix as matrix
from microinfer.engine import paged_cache_bytes
from microinfer.footprint import MIB
from microinfer.schedule import Schedule

NAME = "qwen2.5-1.5b-instruct"

#: A llama-server log's lines, build b12300, at verbosity 4.
LOADED = """\
0.00.711.051 I load_tensors:   CPU_Mapped model buffer size =   445.12 MiB
0.00.711.052 I load_tensors:        CUDA0 model buffer size =  2944.68 MiB
0.01.127.872 I llama_context: flash_attn            = enabled
0.01.131.363 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.01.135.538 I llama_kv_cache:      CUDA0 KV buffer size =   476.02 MiB
0.01.135.687 I init_rotation:      CUDA0 rotation buffer size =     2.19 MiB
0.01.162.737 I sched_reserve:      CUDA0 compute buffer size =    96.51 MiB
0.01.162.744 I sched_reserve:  CUDA_Host compute buffer size =    44.02 MiB
"""


# -- what it is run with, and what it says it holds -----------------------------------------

def test_the_server_is_one_slot_at_the_whole_context_its_cache_at_one_type():
    command = llamacpp.command("llama-server", "m.gguf", port=9, context=32768,
                               cache_type="q8_0")
    given = dict(zip(command[1:], command[2:]))
    assert command[0] == "llama-server" and given["-m"] == "m.gguf"
    assert given["-c"] == "32768" and given["-np"] == "1" and given["-ngl"] == "99"
    assert given["-ctk"] == given["-ctv"] == "q8_0"
    assert given["-fa"] == "on"  # a quantised value cache needs it; every type has it
    assert given["--fit"] == "off"  # what is asked for, not what fits
    assert given["--sse-ping-interval"] == "-1"  # a hung completion's stream goes silent
    assert given["--host"] == "127.0.0.1" and given["--port"] == "9"
    with pytest.raises(ValueError, match="q5_0"):
        llamacpp.command("llama-server", "m.gguf", port=9, context=32768, cache_type="q5_0")


def test_the_device_buffers_are_read_from_the_log_and_the_host_s_left_out():
    assert llamacpp.device_buffers_mib(LOADED) == {
        "model": 2944.68, "KV": 476.02, "rotation": 2.19, "compute": 96.51}


def test_a_log_with_no_cache_on_the_device_is_refused():
    with pytest.raises(ValueError, match="KV"):
        llamacpp.device_buffers_mib(LOADED.replace("CUDA0 KV", "CPU KV"))


#: As build b12300's source writes them: an error of CUDA's, with the call it
#: was on its third line (ggml_cuda_error), and ggml's own.
CUDA_ERROR = ("E CUDA error: {}\nE   current device: 0, in function alloc at ggml-cuda.cu:623\n"
              "E   cuMemCreate(&handle, reserve_size, &prop, 0)\n")


@pytest.mark.parametrize("line", [
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 896.00 MiB on device 0: "
    "cudaMalloc failed: out of memory",
    CUDA_ERROR.format("out of memory"),
    CUDA_ERROR.format("the resource allocation failed"),  # cuBLAS's
    "E alloc_tensor_range: failed to allocate CUDA0 buffer of size 939524096",
    "E process_ubatch: failed to allocate graph",
    "graph_compute: GGML status: error (failed to allocate memory)",
])
def test_a_device_allocation_that_failed_is_out_of_memory(line):
    assert llamacpp.out_of_memory(LOADED + line)


@pytest.mark.parametrize("line", [
    "E srv update_slots: Compute error. off = 0, n_batch = 512, ret = -3",
    "D alloc: failed to allocate 12.00 MiB of pinned memory: out of memory",
    "E srv prompt_save: failed to allocate memory for prompt cache state: bad_alloc",
    "E alloc_tensor_range: failed to allocate CUDA_Host buffer of size 4096",
    "E output_reserve: failed to allocate output buffer of size 0.58 MiB",
    "E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 4096",
    "E ggml_malloc: failed to allocate  12.00 MB",
    CUDA_ERROR.format("unspecified launch failure"),
])
def test_another_failure_or_the_host_s_allocations_are_not(line):
    assert not llamacpp.out_of_memory(LOADED + line)


# -- its stream ------------------------------------------------------------------------------

class Answering(BaseHTTPRequestHandler):
    """/completion answered as a stream, the events of `server.events`; None
    ends the connection there, as a server that died does, and a number
    sends nothing for that many seconds."""

    def do_POST(self):
        self.server.asked.append(json.loads(self.rfile.read(
            int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()
        for event in self.server.events:
            if event is None:
                return
            if isinstance(event, float):
                time.sleep(event)
                continue
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def answering():
    server = HTTPServer(("127.0.0.1", 0), Answering)
    server.events, server.asked = [], []
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield server
    server.shutdown()
    thread.join()
    server.server_close()


def progress(processed, cache=0, total=100):
    return {"content": "", "stop": False, "prompt_progress": {
        "total": total, "cache": cache, "processed": processed, "time_ms": 1}}


TIMINGS = {"prompt_n": 100, "prompt_per_second": 2000.0, "predicted_n": 3,
           "predicted_per_second": 30.0}


def test_a_completion_reports_each_position_and_how_it_stopped(answering):
    answering.events = [progress(40), progress(100), {"content": "a", "stop": False,
                                                      "tokens": [5]},
                        {"content": "b", "stop": False, "tokens": [6]},
                        {"content": "", "stop": True, "tokens_predicted": 3,
                         "stop_type": "limit", "timings": TIMINGS}]
    reached = []
    done = llamacpp.complete(answering.server_port, np.arange(100), 3,
                             report=lambda *r: reached.append(r))
    # a token decoded is not yet in the cache: the next step feeds it
    assert reached == [(40, 0), (100, 0), (100, 1), (101, 2)]
    assert done["tokens_predicted"] == 3 and done["timings"] == TIMINGS
    (asked,) = answering.asked
    assert asked["prompt"] == list(range(100)) and asked["n_predict"] == 3
    assert asked["ignore_eos"] is True and asked["cache_prompt"] is True
    assert asked["stream"] is True and asked["return_progress"] is True


def test_a_stream_that_breaks_off_is_a_lost_server(answering):
    answering.events = [progress(40), None]
    with pytest.raises(llamacpp.Lost, match="broke off"):
        llamacpp.complete(answering.server_port, np.arange(100), 3, report=lambda *r: None)


def test_an_error_event_is_a_lost_server_that_says_why(answering):
    answering.events = [progress(40), {"error": {"code": 500, "message": "Compute error."}}]
    with pytest.raises(llamacpp.Lost, match="Compute error"):
        llamacpp.complete(answering.server_port, np.arange(100), 3, report=lambda *r: None)


def test_a_server_gone_silent_is_lost(answering, monkeypatch):
    monkeypatch.setattr(llamacpp, "SILENCE_TIMEOUT_S", 0.1)
    answering.events = [progress(40), 0.5]
    with pytest.raises(llamacpp.Lost, match="timed out"):
        llamacpp.complete(answering.server_port, np.arange(100), 3, report=lambda *r: None)


def test_no_server_is_a_lost_server():
    with pytest.raises(llamacpp.Lost):
        llamacpp.complete(1, np.arange(4), 1, report=lambda *r: None)


# -- the run -----------------------------------------------------------------------------------

class Simulated:
    """The simulator's place in a run: what it was asked to play."""

    def __init__(self, out):
        self.out, self.played, self.closed = out, None, False

    def play(self, schedule):
        self.played = schedule
        # what the simulator's process writes: nothing, then the take, which
        # the device granted but for 10 MiB
        first = schedule.points[0][1]
        events = [{"target_bytes": 0, "taken_bytes": 0, "shortfall_bytes": 0},
                  {"target_bytes": first, "taken_bytes": first - 10 * MIB,
                   "shortfall_bytes": 10 * MIB}]
        survival.recorder.companion(self.out, ".events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events))

    def close(self):
        self.closed = True


class Serving:
    """A llama.cpp that is not one: every completion answers and is written to
    its log, until the one that is lost, `lost_at` its place among them, part
    way through a batch or, `decoded` tokens in, of its decoding; and the
    clock it moves."""

    buffers_mib = {"model": 2944.0, "KV": 896.0, "compute": 100.0}

    def __init__(self, clock, *, lost_at=None, decoded=0, dying="", log=LOADED, seconds=10.0):
        self.clock, self.lost_at, self.decoded = clock, lost_at, decoded
        self.dying, self.text, self.seconds = dying, log, seconds
        self.asked = []

    def complete(self, ids, new_tokens, report):
        self.asked.append((len(ids), new_tokens))
        if len(self.asked) - 1 == self.lost_at:
            self.text += self.dying
            if self.decoded:
                report(len(ids) + self.decoded - 1, self.decoded)
            elif new_tokens:
                report(20000, 0)
            raise llamacpp.Lost("the stream broke off")
        if new_tokens:
            self.clock[0] += self.seconds
        report(len(ids), 0)
        return {"tokens_predicted": new_tokens, "timings": TIMINGS}

    def log(self):
        return self.text


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(llamacpp_survival, "_now", lambda: now[0])
    monkeypatch.setattr(llamacpp_survival.survival, "Simulator", Simulated)

    class Free:
        free = 1300 * MIB

    monkeypatch.setattr(llamacpp_survival.nvml, "memory", lambda: Free)
    return now


def test_contention_comes_when_the_cache_holds_its_position_and_the_run_goes_on(tmp_path, clock):
    serving = Serving(clock)
    run = llamacpp_survival.run(serving, ModelConfig.from_card(NAME), np.arange(32704), 64,
                                contention_at=16384, out=tmp_path / "r.csv.gz",
                                schedule=lambda headroom: Schedule([(0.0, 400 * MIB)]))
    assert serving.asked == [(16384, 0), (32704, 64)]
    assert run.survived and run.ending == {"state": survival.COMPLETE, "positions": 32767,
                                           "tokens": 64, "seconds": 10.0, "error": None}
    assert run.contention_position == 16384 and run.headroom_bytes == 1300 * MIB
    assert run.asked_bytes == 400 * MIB and run.taken_bytes == 390 * MIB
    assert run.refused_bytes == 10 * MIB
    assert [g["timings"] for g in run.generations] == [TIMINGS]


def test_the_headroom_is_counted_as_an_fp16_microinfer_s(tmp_path, clock):
    """llama.cpp holds its whole cache from the load: the headroom an FP16
    MicroInfer would leave at 16384 positions is more by what the whole
    cache holds beyond it, so both face the same bytes taken."""
    seen = []
    cfg = ModelConfig.from_card(NAME)
    llamacpp_survival.run(Serving(clock), cfg, np.arange(32704), 64, contention_at=16384,
                          out=tmp_path / "r.csv.gz",
                          schedule=lambda h: seen.append(h) or Schedule([(0.0, 400 * MIB)]))
    grown = 896 * MIB - paged_cache_bytes(cfg, 16384)
    assert seen == [1300 * MIB + grown] and grown == 448 * MIB


def test_a_schedule_is_played_to_its_end_the_generation_repeated_from_the_position(
        tmp_path, clock):
    """S1 lasts 494 s and one generation about 12: it is made again, the cache
    cut back to the contention's position, until the schedule has been played."""
    serving = Serving(clock, seconds=10.0)
    schedule = Schedule([(0.0, 400 * MIB), (5.0, 500 * MIB), (25.0, 400 * MIB)])
    run = llamacpp_survival.run(serving, ModelConfig.from_card(NAME), np.arange(32704), 64,
                                contention_at=16384, out=tmp_path / "r.csv.gz",
                                schedule=lambda headroom: schedule)
    assert serving.asked == [(16384, 0), (32704, 64)] + [(16384, 0), (32704, 64)] * 2
    assert len(run.generations) == 3 and run.ending["seconds"] == 30.0


OOM = CUDA_ERROR.format("out of memory")


def met(tmp_path, serving, schedule=Schedule([(0.0, 400 * MIB)])):
    return llamacpp_survival.run(serving, ModelConfig.from_card(NAME), np.arange(32704), 64,
                                 contention_at=16384, out=tmp_path / "r.csv.gz",
                                 schedule=lambda headroom: schedule)


def test_a_server_lost_out_of_memory_says_where_and_when(tmp_path, clock):
    run = met(tmp_path, Serving(clock, lost_at=1, dying=OOM))
    assert not run.survived
    assert run.ending == {"state": survival.OUT_OF_MEMORY, "positions": 20000, "tokens": 0,
                          "seconds": 0.0, "error": "the stream broke off"}


def test_a_server_lost_decoding_says_how_many_tokens_it_had_decoded(tmp_path, clock):
    run = met(tmp_path, Serving(clock, lost_at=1, decoded=10, dying=OOM))
    assert (run.ending["positions"], run.ending["tokens"]) == (32713, 10)


#: Long enough for a second generation: 10 s is one's.
LONGER = Schedule([(0.0, 400 * MIB), (15.0, 0)])


def test_a_server_lost_in_a_later_generation_says_when(tmp_path, clock):
    run = met(tmp_path, Serving(clock, lost_at=3, dying=OOM), LONGER)
    assert run.ending["state"] == survival.OUT_OF_MEMORY and len(run.generations) == 1
    assert (run.ending["positions"], run.ending["seconds"]) == (20000, 10.0)


def test_a_server_lost_as_its_cache_is_cut_back_is_not_credited_with_the_last_generation(
        tmp_path, clock):
    run = met(tmp_path, Serving(clock, lost_at=2, dying=OOM), LONGER)
    assert (run.ending["positions"], run.ending["tokens"]) == (16384, 0)


def test_a_server_lost_otherwise_failed(tmp_path, clock):
    assert met(tmp_path, Serving(clock, lost_at=1)).ending["state"] == llamacpp_survival.FAILED


def test_an_allocation_that_failed_before_and_was_recovered_did_not_end_it(tmp_path, clock):
    run = met(tmp_path, Serving(clock, lost_at=1, log=LOADED + OOM))
    assert run.ending["state"] == llamacpp_survival.FAILED


def test_a_server_lost_before_its_contention_came_is_no_run(tmp_path, clock):
    with pytest.raises(llamacpp.Lost):
        met(tmp_path, Serving(clock, lost_at=0))


def test_a_generation_of_fewer_tokens_is_not_complete(tmp_path, clock):
    class Short(Serving):
        def complete(self, ids, new_tokens, report):
            return {**super().complete(ids, new_tokens, report),
                    "tokens_predicted": min(new_tokens, 5)}

    run = llamacpp_survival.run(Short(clock), ModelConfig.from_card(NAME), np.arange(32704), 64,
                                contention_at=16384, out=tmp_path / "r.csv.gz",
                                schedule=lambda headroom: Schedule([(0.0, 400 * MIB)]))
    assert run.ending["state"] == llamacpp_survival.FAILED and not run.survived
    assert "5 of 64" in run.ending["error"] and run.ending["positions"] == 32708


def test_the_simulator_gives_back_what_it_took_whatever_the_run_came_to(tmp_path, clock,
                                                                        monkeypatch):
    made = []
    monkeypatch.setattr(llamacpp_survival.survival, "Simulator",
                        lambda out: made.append(Simulated(out)) or made[-1])

    class Broken(Serving):
        def complete(self, ids, new_tokens, report):
            if new_tokens:
                raise RuntimeError("not the server's")
            return super().complete(ids, new_tokens, report)

    with pytest.raises(RuntimeError, match="not the server's"):
        llamacpp_survival.run(Broken(clock), ModelConfig.from_card(NAME), np.arange(32704), 64,
                              contention_at=16384, out=tmp_path / "r.csv.gz",
                              schedule=lambda headroom: Schedule([(0.0, 400 * MIB)]))
    assert made[0].closed


# -- the contention it meets, and the matrix of its runs -----------------------------------

def test_s5_s_step_is_the_survival_experiment_s_rule():
    cfg = ModelConfig.from_card(NAME)
    rule = llamacpp_survival.Contention("S5", 0, matrix.Rq1.read())
    played = rule.schedule(cfg, 2000 * MIB, positions=16384, context=32767,
                           simulator_context_bytes=82 * MIB)
    assert played.points == [(0.0, survival.contention_bytes(
        cfg, headroom_bytes=2000 * MIB, positions=16384, context=32767,
        simulator_context_bytes=82 * MIB, shortfall_bytes=256 * MIB))]


def test_s1_is_the_matrix_s_grid_on_its_base():
    rq1 = matrix.Rq1.read()
    rule = llamacpp_survival.Contention("S1", 2, rq1)
    played = rule.schedule(ModelConfig.from_card(NAME), 2000 * MIB, positions=16384,
                           context=32767, simulator_context_bytes=82 * MIB)
    base = 2000 * MIB - 82 * MIB - matrix.START_HEADROOM_BYTES
    assert played.points == matrix.raised(matrix.grid(2).schedule, base).points


def entry(cache_type, scenario, run, *, survived=True, refused=0.0, state="complete"):
    return {"kind": llamacpp_survival.KIND, "model": NAME, "sha256": f"{cache_type}{run}",
            "config": {"cache_type": cache_type, "scenario": scenario, "run": run},
            "results": {"survived": survived, "refused_mib": refused, "taken_mib": 500.0,
                        "ending": {"state": state, "positions": 32767, "seconds": 11.0,
                                   "error": None},
                        "buffers_mib": {"KV": 896.0}}}


def test_the_runs_go_run_by_run_over_the_cells():
    runs = llamacpp_survival.runs(("f16", "q4_0"), ("S1", "S5"), 2)
    assert [r.name for r in runs[:5]] == ["S1-f16-r0", "S1-q4_0-r0", "S5-f16-r0", "S5-q4_0-r0",
                                          "S1-f16-r1"]
    assert len(runs) == 8


def test_a_run_logged_at_the_same_code_and_build_is_not_made_again(tmp_path):
    log = tmp_path / "log.jsonl"
    benchlog.append(llamacpp_survival.KIND, model=NAME, context_length=32767,
                    precision_tiers=None, results={},
                    config={"cache_type": "f16", "scenario": "S5", "run": 1,
                            "build": {"binary_sha256": "b", "gguf_sha256": "g"}}, log=log)
    commit = benchlog.read(log)[0]["git_commit"]
    build = {"binary_sha256": "b", "gguf_sha256": "g"}
    assert llamacpp_survival.logged(commit, build, log) == {
        llamacpp_survival.BaselineRun("f16", "S5", 1)}
    assert llamacpp_survival.logged(commit, {**build, "binary_sha256": "c"}, log) == set()
    assert llamacpp_survival.logged("another", build, log) == set()


def test_the_summary_gives_each_cell_s_endings_and_what_its_contention_was_refused():
    s = llamacpp_survival.summarise([
        entry("f16", "S5", 0, refused=256.0), entry("f16", "S5", 1, refused=300.0),
        entry("f16", "S5", 1, refused=280.0),  # run again: the later stands
        entry("f16", "S5", 2, survived=False, state="out of memory"),
        entry("q8_0", "S1", 0)])
    cell = s["S5"]["f16"]
    assert cell["survived"] == 2 and cell["runs"] == 3
    assert [e["state"] for e in cell["endings"]] == ["complete", "complete", "out of memory"]
    assert cell["endings"][0]["seconds"] == 11.0
    assert cell["refused_mib"] == [256.0, 280.0, 0.0]
    assert cell["wilson_95"][0] < 2 / 3 < cell["wilson_95"][1]
    assert s["S1"]["q8_0"]["runs"] == 1


def test_what_a_microinfer_run_s_contention_was_refused_is_read_from_its_events(tmp_path):
    """Beside llama.cpp's: the most a run's simulator asked for and was not
    given, from the events its entry names, and only those it names."""
    path = tmp_path / "S1-B1-r0.events.jsonl"
    path.write_text("".join(json.dumps({"shortfall_bytes": b}) + "\n"
                            for b in (0, 30 * MIB, 10 * MIB)))
    run = {"kind": "survival", "config": {"files": {path.name: benchlog.file_sha256(path)}}}
    assert llamacpp_survival.refused_mib_of(run, [tmp_path]) == 30.0
    assert llamacpp_survival.refused_mib_of(run, [tmp_path / "elsewhere"]) is None
    path.write_text(json.dumps({"shortfall_bytes": 0}) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        llamacpp_survival.refused_mib_of(run, [tmp_path])


# -- the build and the tools -------------------------------------------------------------------

def test_the_runs_name_the_build_logged_and_refuse_another(tmp_path):
    log, binary, gguf = (tmp_path / n for n in ("log.jsonl", "llama-server", "m.gguf"))
    binary.write_bytes(b"binary")
    gguf.write_bytes(b"weights")
    with pytest.raises(ValueError, match="build_llamacpp"):
        llamacpp.logged_build(NAME, log, binary=binary, gguf=gguf)

    def built(model, weights):
        benchlog.append(llamacpp.BUILD_KIND, model=model, context_length=None,
                        precision_tiers=None, config={"tag": "b12300"},
                        results={"binary_sha256": benchlog.file_sha256(binary),
                                 "gguf_sha256": weights}, log=log)

    built(NAME, benchlog.file_sha256(gguf))
    built("another-model", "its own")
    build = llamacpp.logged_build(NAME, log, binary=binary, gguf=gguf)
    assert build["tag"] == "b12300" and build["entry"] == benchlog.read(log)[0]["sha256"]
    binary.write_bytes(b"another")
    with pytest.raises(ValueError, match="another build"):
        llamacpp.logged_build(NAME, log, binary=binary, gguf=gguf)


def test_the_tool_refuses_a_dirty_tree(tmp_path, monkeypatch, capsys):
    tool = load_tool("llamacpp_survival")
    real = tool.benchlog.environment
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": True})
    with pytest.raises(SystemExit):
        tool.main(["--issue", "211", "--log", str(tmp_path / "log.jsonl")])
    assert "uncommitted" in capsys.readouterr().err


def test_a_run_lost_before_its_contention_is_not_logged_and_the_next_is_made(
        tmp_path, monkeypatch, capsys):
    tool = load_tool("llamacpp_survival")
    real = tool.benchlog.environment
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    monkeypatch.setattr(tool.llamacpp, "logged_build", lambda model, log: {
        "binary_sha256": "b", "gguf_sha256": "g"})
    monkeypatch.setattr(tool.matrix.Rq1, "read", lambda log: None)
    monkeypatch.setattr(tool, "SETTLE_S", 0.0)
    made = []

    def make(run, *rest):
        made.append(run.name)
        if run.cache_type == "f16":
            raise llamacpp.Lost("did not load")
        return {"survived": True, "ending": {}, "asked_mib": 0, "taken_mib": 0, "refused_mib": 0}

    monkeypatch.setattr(tool, "make", make)
    assert tool.main(["--issue", "211", "--scenarios", "S5", "--runs", "1", "--out-dir",
                      str(tmp_path), "--log", str(tmp_path / "log.jsonl")]) == 1
    assert made == ["S5-f16-r0", "S5-q8_0-r0", "S5-q4_0-r0"]
    assert "S5-f16-r0" in capsys.readouterr().err


@pytest.mark.slowest
def test_the_real_server_meets_a_step_at_a_small_scale(tmp_path):
    """llama-server with Qwen2.5-1.5B at 4096 positions: a 100 MiB step at
    2048 positions, which it has room for, and it completes."""
    if not (llamacpp.BINARY.is_file() and llamacpp.gguf_path(NAME).is_file()):
        pytest.skip("llama.cpp is not built: tools/build_llamacpp.py")
    ids = np.random.default_rng(1).integers(1000, 100_000, 4000)
    with llamacpp.Server(llamacpp.BINARY, llamacpp.gguf_path(NAME), context=4096,
                         cache_type="q8_0", log=tmp_path / "server.log") as server:
        run = llamacpp_survival.run(server, ModelConfig.from_card(NAME), ids, 16,
                                    contention_at=2048, out=tmp_path / "r.csv.gz",
                                    schedule=lambda h: Schedule([(0.0, 100 * MIB)]))
    assert run.survived, run.ending
    assert run.ending["positions"] == 4015 and run.refused_bytes == 0
    elements = ModelConfig.from_card(NAME).kv_bytes_per_token / 2 * 4096
    assert server.buffers_mib["KV"] == pytest.approx(elements * 34 / 32 / MIB, rel=0.01)
