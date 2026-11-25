"""The paper's figures (#212): what each is drawn from, and that it is drawn
to ICML's rules.

A figure's points are read from benchmark log entries alone, so each reader is
tested on entries made by hand. The survival timeline is the one figure whose
data the log names but does not hold: survival_timeline.read takes it from
the recordings a run's entry names by sha256, and is tested on recordings
written here.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import load_tool

from microinfer import benchlog, figures, recorder, survival_timeline
from microinfer.config import ModelConfig
from microinfer.footprint import MIB
from microinfer.engine import paged_cache_bytes

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-1.5b-instruct"
#: When the recordings here begin and when the simulator takes, on the clock
#: they share.
START_NS = 100_000_000_000
TAKE_NS = START_NS + 1_000_000_000
ENGINE, SIMULATOR, DESKTOP = 10, 20, 30


# -- the survival timeline -----------------------------------------------------------------

def at(seconds: float) -> int:
    return TAKE_NS + round(seconds * 1e9)


def write_recording(folder: Path, name: str, *, free, engine, taken, levels, steps=None,
                    run: dict | None = None) -> dict:
    """A run's recordings in `folder`, as the survival experiment leaves
    them, and the entry that names them: `free`, `engine` and `taken` are
    (seconds from the take, MiB), `levels` (seconds, level) and `steps`
    (seconds, position, effective bits, pages below FP16)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.csv.gz"
    device = recorder._TraceWriter(out, recorder.FORMAT, {}, recorder.COLUMNS)
    for s, mib in free:
        device.row(at(s), 0.0, round(mib * MIB), 0, 0)
    device.close({})
    procs = recorder._TraceWriter(recorder.processes_path(out), recorder.PROCESS_FORMAT, {}, [])
    for s, mib in engine:
        procs.row(recorder.STATE_ROW, at(s), 0.0, 1000, 0, 0, 0, 0)
        for pid, used in ((ENGINE, mib), (SIMULATOR, 9000.0), (DESKTOP, 300.0)):
            procs.row(recorder.PROCESS_ROW, at(s), 0.0, pid, "C", round(used * MIB), "name")
    procs.close({})

    def lines(ending, rows):
        recorder.companion(out, ending).write_text(
            "".join(json.dumps(r) + "\n" for r in rows))

    lines(".events.jsonl", [{"applied_ns": START_NS, "taken_bytes": 0}]
          + [{"applied_ns": at(s), "taken_bytes": round(mib * MIB)} for s, mib in taken])
    recorder.companion(out, ".replay.json").write_text(json.dumps({"simulator_pid": SIMULATOR}))
    lines(".pressure.jsonl", [{"t_mono_ns": at(s), "level": level} for s, level in levels])
    if steps is not None:
        lines(".timeline.jsonl", [{"t": at(s) / 1e9, "position": position, "effective_bits": bits,
                                  "pages_below_fp16": below}
                                 for s, position, bits, below in steps])
    files = {f.name: benchlog.file_sha256(f) for f in folder.iterdir()
             if f.name.startswith(f"{name}.")}
    return {"kind": "survival", "sha256": name * 8, "git_commit": "c" * 40, "model": MODEL,
            "config": {"engine": "adaptive", "kv_tier": "FP16", "score_source": "semantic",
                       "contention_at": 16384, "shortfall_mib": 256.0, "pulse_s": None,
                       "thresholds": {"red_below_bytes": 512 * MIB,
                                      "yellow_below_bytes": 1024 * MIB, "persist_polls": 3},
                       "files": files},
            "results": {"contention_position": 16384, "ending": {"state": "complete"},
                        **(run or {})}}


def adaptive(folder: Path, **changes) -> dict:
    """A run whose cache, at 16384 positions, was taken to 4 bits by 10 s and
    restored at 60, the simulator keeping 1600 MiB for 30 s."""
    given = {"free": [(-0.5, 2000.0), (0.0, 1900.0), (0.1, 150.0), (0.2, 160.0), (10.0, 540.0),
                      (30.0, 2100.0), (60.0, 1700.0), (70.0, 1700.0)],
             "engine": [(-0.5, 3000.0), (70.0, 3000.0)],
             "taken": [(0.0, 1600.0), (30.0, 0.0)],
             "levels": [(-40.0, "GREEN"), (0.15, "RED"), (10.1, "YELLOW"), (30.1, "GREEN")],
             "steps": [(-40.0, 16, None, None), (-0.8, 16384, 16.0, 0), (10.0, 16384, 4.0, 100),
                       (12.0, 16384, 4.0, 100), (15.0, 16384, 4.0, 130),
                       (50.0, 16384, 8.0, 130), (60.0, 32768, 16.0, 0), (70.0, 32768, 16.0, 0)],
             "run": {"recovery": {"first_upgrade_s": 40.0, "last_upgrade_s": 60.0}}}
    return write_recording(folder, "adaptive", **{**given, **changes})


CACHE = round(paged_cache_bytes(ModelConfig.from_card(MODEL), 16384) / MIB, 2)


def test_a_run_is_read_against_time_from_the_recordings_its_entry_names(tmp_path):
    """Seconds are from the simulator's take. An adaptive run's cache is its
    positions' FP16 bytes at the effective bits of its steps. It begins with
    the recording, at what the step before that held; a point between two
    equal to it is left out."""
    t = survival_timeline.read(adaptive(tmp_path), [tmp_path])
    assert t["cache_mib"] == [[-0.5, CACHE], [10.0, CACHE / 4], [15.0, CACHE / 4],
                              [50.0, CACHE / 2], [60.0, 2 * CACHE], [70.0, 2 * CACHE]]
    assert t["simulator_mib"] == [[0.0, 1600.0], [30.0, 0.0]]
    assert t["levels"] == [[-40.0, "GREEN"], [0.15, "RED"], [10.1, "YELLOW"], [30.1, "GREEN"]]
    assert t["end_s"] == 70.0 and t["ending"] == "complete"
    assert t["entry"] == "adaptive" * 8


def test_the_steps_that_found_more_pages_below_fp16_are_its_downgrades(tmp_path):
    """A run logs its plans' moves and not when each applied (ADR-0014,
    #207), but its timeline shows the steps after which more pages were
    below FP16. The upgrades span the first batch to the last."""
    t = survival_timeline.read(adaptive(tmp_path), [tmp_path])
    assert t["downgrades_s"] == [10.0, 15.0]
    assert t["upgrades_s"] == [40.0, 60.0]


def test_the_headroom_kept_is_the_least_of_each_bin(tmp_path, monkeypatch):
    """A figure's points are fewer than a 50 Hz trace's, and the lowest of
    each bin is kept, so that no dip is thinned away."""
    monkeypatch.setattr(survival_timeline, "BIN_S", 1.0)
    t = survival_timeline.read(adaptive(tmp_path), [tmp_path])
    assert [0.1, 150.0] in t["headroom_mib"]
    assert [0.0, 1900.0] not in t["headroom_mib"] and [0.2, 160.0] not in t["headroom_mib"]
    assert [-0.5, 2000.0] in t["headroom_mib"] and [10.0, 540.0] in t["headroom_mib"]


def test_a_run_is_read_to_where_the_figure_s_axis_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(survival_timeline, "UNTIL_S", 20.0)
    t = survival_timeline.read(adaptive(tmp_path), [tmp_path])
    for series in ("headroom_mib", "cache_mib", "simulator_mib", "levels"):
        assert all(p[0] <= 20.0 for p in t[series]), series
    assert t["downgrades_s"] == [10.0, 15.0] and t["end_s"] == 70.0


def test_an_adaptive_run_that_ran_out_is_read_to_its_last_step(tmp_path):
    """Not to where its engine held most: its downgrades come after that."""
    t = survival_timeline.read(adaptive(tmp_path, run={"ending": {"state": "out of memory"}}),
                               [tmp_path])
    assert t["end_s"] == 70.0 and t["downgrades_s"] == [10.0, 15.0]


def static(folder: Path, **changes) -> dict:
    """A static run, with no timeline, that runs out at 25 s: the engine's
    memory falls as its cache is freed."""
    given = {"free": [(-0.5, 2000.0), (0.1, 150.0), (20.0, 60.0), (25.0, 30.0), (26.0, 600.0)],
             "engine": [(-0.5, 3000.0), (20.0, 3090.0), (25.0, 3120.0), (25.2, 2600.0)],
             "taken": [(0.0, 1600.0)], "levels": [(-40.0, "GREEN"), (0.15, "RED")],
             "run": {"ending": {"state": "out of memory", "positions": 20992}}}
    entry = write_recording(folder, "static", **{**given, **changes})
    entry["config"].update(engine="static", score_source=None)
    return entry


def test_a_run_that_ran_out_ends_where_its_engine_last_held_its_most(tmp_path):
    """Its cache is freed as it ends, which is no part of the run: the
    headroom and the cache end with it."""
    t = survival_timeline.read(static(tmp_path), [tmp_path])
    assert t["end_s"] == 25.0 and t["ending"] == "out of memory"
    assert t["headroom_mib"][-1] == [25.0, 30.0]
    assert t["downgrades_s"] == [] and t["upgrades_s"] is None


def test_a_static_run_s_cache_is_the_driver_s_account_less_what_is_not_cache(tmp_path):
    """The engine is the process that held most at the first sample, the
    simulator aside, and its cache starts at the contention's position's and
    moves as the driver saw the engine move."""
    t = survival_timeline.read(static(tmp_path), [tmp_path])
    assert t["cache_mib"] == [[-0.5, CACHE], [20.0, CACHE + 90.0], [25.0, CACHE + 120.0]]


def test_an_adaptive_run_with_no_timeline_is_refused(tmp_path):
    with pytest.raises(ValueError, match="timeline"):
        survival_timeline.read(adaptive(tmp_path, steps=None), [tmp_path])


def test_a_recording_that_is_not_the_one_its_entry_names_is_refused(tmp_path):
    entry = adaptive(tmp_path)
    (tmp_path / "adaptive.pressure.jsonl").write_text("")
    with pytest.raises(ValueError, match="sha256"):
        survival_timeline.read(entry, [tmp_path])


def test_a_recording_that_is_nowhere_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="adaptive"):
        survival_timeline.read(adaptive(tmp_path), [tmp_path / "elsewhere"])


def tool_on(tmp_path, monkeypatch, *, static_shortfall: float = 256.0):
    """The tool, the log of a static and an adaptive run it is to read, and
    their entries' hashes."""
    log = tmp_path / "benchmark.jsonl"
    entries = static(tmp_path / "data"), adaptive(tmp_path / "data")
    entries[0]["config"]["shortfall_mib"] = static_shortfall
    hashes = [benchlog.append(e["kind"], model=MODEL, context_length=32767, precision_tiers=None,
                              config=e["config"], results=e["results"], log=log)["sha256"]
              for e in entries]
    tool = load_tool("summarise_survival_timeline")
    monkeypatch.setattr(tool.benchlog, "environment", clean)
    monkeypatch.setattr(tool, "RECORDINGS", (tmp_path / "data",))
    return tool, log, hashes


def test_the_tool_logs_the_survival_timeline_naming_both_runs(tmp_path, monkeypatch):
    tool, log, (of_static, of_adaptive) = tool_on(tmp_path, monkeypatch)
    assert tool.main(["--issue", "212", "--log", str(log), "--static", of_static[:8],
                      "--adaptive", of_adaptive[:8]]) == 0
    logged = benchlog.read(log)[-1]
    assert logged["kind"] == survival_timeline.KIND
    assert logged["results"]["static"]["entry"] == of_static
    assert logged["results"]["adaptive"]["entry"] == of_adaptive
    assert logged["results"]["thresholds_mib"] == {"RED": 512.0, "YELLOW": 1024.0}
    benchlog.verify(log)


def test_the_tool_refuses_runs_of_different_contention(tmp_path, monkeypatch, capsys):
    """An overlay of a run left 256 MiB short on one left 128 compares nothing."""
    tool, log, (of_static, of_adaptive) = tool_on(tmp_path, monkeypatch, static_shortfall=128.0)
    with pytest.raises(SystemExit):
        tool.main(["--issue", "212", "--log", str(log), "--static", of_static[:8],
                   "--adaptive", of_adaptive[:8]])
    assert "shortfall_mib" in capsys.readouterr().err


_environment = benchlog.environment


def clean(log):
    return {**_environment(log), "git_dirty": False}


# -- the points of the others ----------------------------------------------------------------

def continuation_run(passage: str, condition: str, kl: float, sha: str) -> dict:
    return {"kind": "rq3-continuation", "sha256": sha, "model": MODEL,
            "config": {"passage_id": passage, "condition": {"name": condition}},
            "results": {"kl_mean": kl}}


def continuation_log() -> list[dict]:
    runs = [continuation_run("p0", "semantic-80", 0.02, "a"),
            continuation_run("p0", "uniform-80", 0.10, "b"),
            continuation_run("p1", "semantic-80", 0.03, "c"),
            continuation_run("p1", "uniform-80", 0.09, "d"),
            continuation_run("p1", "uniform-80", 9.0, "superseded"),
            continuation_run("p0", "semantic-70", 5.0, "e")]
    summary = {"kind": "continuation-summary", "sha256": "s", "model": MODEL,
               "results": {"model": MODEL, "entries": ["a", "b", "c", "d", "e"],
                           "by_condition": {"semantic-80": {}, "uniform-80": {}}}}
    other = {"kind": "continuation-summary", "sha256": "t", "model": "llama-3.2-1b-instruct",
             "results": {"model": "llama-3.2-1b-instruct", "entries": [],
                         "by_condition": {"semantic-80": {}, "uniform-80": {}}}}
    return [*runs, summary, other]


def test_a_passage_s_point_is_its_two_runs_of_the_latest_summary():
    """The runs the summary pooled, and no other: one superseded, or of
    another condition, gives no point."""
    points = figures.passage_kls(continuation_log(), MODEL, "semantic-80", "uniform-80")
    assert dict(zip(points.passages, zip(points.x.tolist(), points.y.tolist()))) == {
        "p0": (0.02, 0.10), "p1": (0.03, 0.09)}


def test_a_passage_with_one_run_of_the_two_is_refused():
    log = continuation_log()
    log[-2]["results"]["entries"].remove("d")
    with pytest.raises(ValueError, match="p1 has a run of only one"):
        figures.passage_kls(log, MODEL, "semantic-80", "uniform-80")


def test_a_summary_whose_run_the_log_lacks_is_refused():
    log = [e for e in continuation_log() if e["sha256"] != "d"]
    with pytest.raises(ValueError, match="lacks 1 of the runs"):
        figures.passage_kls(log, MODEL, "semantic-80", "uniform-80")


def test_a_log_with_no_summary_of_the_conditions_is_refused():
    with pytest.raises(ValueError, match="continuation-summary"):
        figures.passage_kls(continuation_log(), MODEL, "semantic-80", "no-such-80")


def damage_log(task: str = "needle") -> list[dict]:
    """Two samples of 50 pages: the damage is ten times the score."""
    scores = np.linspace(0.001, 1.0, 100)
    entries = [{"kind": "page-damage", "sha256": f"{task}{i}", "model": MODEL,
                "config": {"task": task},
                "results": {"scores": part.tolist(), "kl": (10 * part).tolist()}}
               for i, part in enumerate(np.split(scores, 2))]
    summary = {"kind": "page-damage-summary", "sha256": f"{task}-s", "model": MODEL,
               "results": {"task": task, "model": MODEL, "pooled_rho": 1.0,
                           "entries": [e["sha256"] for e in entries]}}
    return [*entries, summary]


def test_damage_is_binned_by_the_score_s_rank_with_the_spread_of_each_bin():
    """Ten bins of equal pages, by the score pooled over the samples as rho
    is, each at its median score with the median and quartiles of its
    damages."""
    bins = figures.damage_bins(damage_log(), MODEL, "needle", bins=10)
    assert bins.rho == 1.0 and bins.score.shape == (10,)
    assert np.allclose(bins.median, 10 * bins.score)
    assert (bins.low < bins.median).all() and (bins.median < bins.high).all()
    assert bins.score[0] == pytest.approx(np.median(np.linspace(0.001, 1.0, 100)[:10]))


def test_pages_of_one_score_fall_in_one_bin():
    """Pages of equal score have no order between them: split over two bins
    they would show a rank the score does not give."""
    log = damage_log()
    for e in log[:2]:
        e["results"]["scores"] = [0.5] * 50
    bins = figures.damage_bins(log, MODEL, "needle", bins=10)
    assert bins.score.tolist() == [0.5]


def test_the_latest_summary_of_the_task_is_the_one_drawn():
    log = damage_log("needle") + damage_log("anchor")
    later = json.loads(json.dumps(log[2]))
    later.update(sha256="later")
    later["results"]["pooled_rho"] = 0.5
    assert figures.damage_bins([*log, later], MODEL, "needle").rho == 0.5


def spike_log() -> list[dict]:
    def spike(amplitude, rise, process, ending="lasting"):
        return {"amplitude_mib": amplitude, "rise_ms": rise, "process": process,
                "ending": ending}

    chrome = "/opt/google/chrome/chrome --type=gpu-process"
    trace = {"kind": "contention-trace", "sha256": "r",
             "results": {"spikes": [spike(100.0, 500.0, chrome),
                                    spike(500.0, 900.0, "/usr/bin/vlc"),
                                    spike(180.0, 3000.0, "/x/.venv/bin/python"),
                                    spike(90.0, 300.0, "/usr/share/code/code"),
                                    spike(80.0, 200.0, None), spike(190.0, None, chrome),
                                    spike(70.0, 100.0, chrome, "censored")]}}
    summary = {"kind": "contention-summary", "sha256": "cs",
               "results": {"spikes": {"recordings": [{"entry": "r"}]}}}
    return [trace, summary]


def test_a_spike_s_point_is_its_amplitude_and_rise_by_the_process_named():
    """A spike with no rise measured has no point, nor one censored, as RQ1's
    table counts them."""
    assert figures.spike_points(spike_log()) == {
        "Browser": [(0.5, 100.0)], "Video player": [(0.9, 500.0)], "Engine": [(3.0, 180.0)],
        "Other or none": [(0.3, 90.0), (0.2, 80.0)]}


# -- the drawings ----------------------------------------------------------------------------

def survival_timeline_log(tmp_path) -> list[dict]:
    runs = {name: survival_timeline.read(make(tmp_path / name), [tmp_path / name])
            for name, make in (("static", static), ("adaptive", adaptive))}
    return [{"kind": survival_timeline.KIND, "sha256": "t",
             "results": {**runs, "thresholds_mib": {"RED": 300.0, "YELLOW": 700.0}}}]


def entries_of(script: str, tmp_path) -> list[dict]:
    return {"figure_survival": lambda: survival_timeline_log(tmp_path),
            "figure_continuation": continuation_log,
            "figure_damage": lambda: [e for t in ("continuation", "needle", "anchor")
                                      for e in damage_log(t)],
            "figure_spikes": spike_log}[script]()


SCRIPTS = ("figure_survival", "figure_continuation", "figure_damage", "figure_spikes")


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_script_draws_its_figure_from_the_log(script, tmp_path, monkeypatch):
    """One vector PDF, a column wide, its axes named and no title in it, its
    fonts embedded as outlines a camera-ready check accepts (no Type 3) and
    the same bytes whenever it is drawn."""
    pytest.importorskip("matplotlib")
    tool = load_tool(script)
    entries = entries_of(script, tmp_path)
    monkeypatch.setattr(figures.benchlog, "verify", lambda log: len(entries))
    monkeypatch.setattr(figures.benchlog, "read", lambda log: entries)
    drawn = []
    for name in ("a.pdf", "b.pdf"):
        out = tmp_path / name
        assert figures.main(["--out", str(out)], tool.__doc__, tool.OUT, tool.draw) == 0
        drawn.append(out.read_bytes())
    assert drawn[0] == drawn[1]
    assert drawn[0].startswith(b"%PDF")
    assert b"/Type3" not in drawn[0] and b"/Subtype /Image" not in drawn[0]
    width = float(re.search(rb"/MediaBox \[ ?0 0 ([\d.]+) ", drawn[0]).group(1)) / 72
    assert width <= figures.COLUMN_IN + 0.01
    figure = tool.draw(entries)
    assert all(axes.get_title() == "" for axes in figure.axes)
    assert any(axes.get_xlabel() for axes in figure.axes) or figure.get_supxlabel()
    assert any(axes.get_ylabel() for axes in figure.axes) or figure.get_supylabel()


@pytest.mark.parametrize("script", SCRIPTS)
def test_series_are_told_apart_without_their_colour(script, tmp_path):
    """In greyscale: two series of one panel differ in marker, dash or
    fill, not in hue alone."""
    pytest.importorskip("matplotlib")
    tool = load_tool(script)
    for axes in tool.draw(entries_of(script, tmp_path)).axes:
        named = [line for line in axes.lines if not line.get_label().startswith("_")]
        marks = [(line.get_linestyle(), line.get_marker(), line.get_fillstyle(),
                  str(line.get_markerfacecolor()) in ("none", "white")) for line in named]
        assert len(set(marks)) == len(marks), [line.get_label() for line in named]


def test_a_run_s_ending_is_marked_once_and_inside_its_axes(tmp_path):
    """The time axis is drawn in two panels: a mark past the first's end
    would be drawn outside it."""
    pytest.importorskip("matplotlib")
    figure = load_tool("figure_survival").draw(survival_timeline_log(tmp_path))
    marked = [(axes, line) for axes in figure.axes for line in axes.lines
              if line.get_label() == "Out of memory"]
    assert len(marked) == 2  # the headroom and the cache
    for axes, line in marked:
        assert axes.get_xlim()[0] <= line.get_xdata()[0] <= axes.get_xlim()[1]


def test_the_figures_the_paper_includes_are_committed():
    paper = (REPO / "docs" / "ICML_2027" / "paper_ICML.tex").read_text()
    included = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper)
    assert len(included) >= 3
    for name in included:
        path = REPO / "docs" / "ICML_2027" / name
        assert path.is_file() and path.read_bytes().startswith(b"%PDF"), name
        assert load_tool(f"figure_{path.stem}").OUT == path
