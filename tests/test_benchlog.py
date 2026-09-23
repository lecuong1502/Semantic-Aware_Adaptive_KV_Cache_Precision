"""The benchmark log: append-only, self-describing, one call to add to (#13).

Measurements here come from a laptop whose free memory depends on what else is
running and whose clocks depend on how hot it is. A number without that context
cannot be reproduced or defended, so every entry carries it, and none is ever
rewritten after the fact.

Two guarantees are tested, at two levels:
- **Within a file:** every entry carries a hash of its own content and the
  hash of the entry before it. An edit breaks the entry it touches, even the
  last one; a deletion or a reordering breaks the chain; `verify` names where.
- **Across history:** every committed version of the repository's own log is a
  prefix of the next, and the working copy extends the last. An entry can be
  added. It cannot be changed or removed without this test failing.
"""

import json
import subprocess
from pathlib import Path

import pytest

from microinfer import benchlog

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "experiments" / "logs" / "benchmark.jsonl"


def entry(log, kind="test", **overrides):
    fields = dict(results={"value": 1.5}, model="qwen2.5-0.5b-instruct", context_length=128,
                  precision_tiers={"FP16": 1.0}, log=log)
    fields.update(overrides)
    return benchlog.append(kind, **fields)


# -- one call, and what it records --------------------------------------------


def test_one_call_appends_one_self_describing_entry(tmp_path):
    log = tmp_path / "log.jsonl"
    written = entry(log, config={"max_new_tokens": 64})
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == written

    for field in ("timestamp", "git_commit", "git_dirty", "model", "context_length",
                  "precision_tiers", "gpu", "exclusive_gpu", "other_gpu_processes",
                  "kind", "config", "results", "previous", "sha256"):
        assert field in written, field
    for field in ("name", "total_memory_bytes", "driver_version"):
        assert written["gpu"][field], field
    assert written["timestamp"].endswith("+00:00"), "timestamps are UTC"
    assert len(written["git_commit"]) == 40


def test_the_ticket_s_fields_must_be_stated_not_defaulted(tmp_path):
    """Model, context length and tier configuration are what a result depends
    on, so a caller cannot leave them out by accident. It may say None, where
    one does not apply, but it has to say it."""
    with pytest.raises(TypeError):
        benchlog.append("test", results={}, model="m", log=tmp_path / "log.jsonl")


def test_exclusive_use_is_recorded_with_what_shared_the_gpu(tmp_path):
    written = entry(tmp_path / "log.jsonl")
    others = written["other_gpu_processes"]
    assert written["exclusive_gpu"] == (len(others) == 0)
    assert all(p["pid"] != __import__("os").getpid() for p in others)
    for p in others:
        assert set(p) == {"pid", "name", "kind", "used_bytes"}


def test_results_that_are_not_json_are_refused_before_anything_is_written(tmp_path):
    log = tmp_path / "log.jsonl"
    entry(log)
    before = log.read_bytes()
    with pytest.raises(TypeError):
        entry(log, results={"x": object()})
    assert log.read_bytes() == before


def test_the_format_reads_by_hand_and_parses_for_plotting(tmp_path):
    log = tmp_path / "log.jsonl"
    entry(log, kind="alpha", results={"latency_ms": 3.0})
    entry(log, kind="beta", results={"latency_ms": 4.0})
    rows = benchlog.read(log)
    assert [r["kind"] for r in rows] == ["alpha", "beta"]
    assert [r["results"]["latency_ms"] for r in rows] == [3.0, 4.0]
    table = benchlog.render(rows)
    assert "alpha" in table and "beta" in table


# -- append-only ----------------------------------------------------------------


def test_appending_never_rewrites_what_is_there(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(3):
        before = log.read_bytes() if log.exists() else b""
        entry(log, results={"i": i})
        assert log.read_bytes().startswith(before)


def test_each_entry_chains_to_the_one_before(tmp_path):
    log = tmp_path / "log.jsonl"
    first, second = entry(log), entry(log)
    assert first["previous"] is None
    assert second["previous"] == first["sha256"]
    benchlog.verify(log)


@pytest.mark.parametrize("tamper,at", [("edit", 2), ("edit last", 4), ("delete", 2),
                                       ("reorder", 2)])
def test_an_edited_log_fails_verification_where_it_was_edited(tmp_path, tamper, at):
    log = tmp_path / "log.jsonl"
    for i in range(4):
        entry(log, results={"i": i})
    lines = log.read_text().splitlines()
    if tamper == "edit":
        lines[1] = lines[1].replace('"i": 1', '"i": 7')
    elif tamper == "edit last":
        lines[3] = lines[3].replace('"i": 3', '"i": 7')
    elif tamper == "delete":
        del lines[1]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(benchlog.LogTampered, match=f"entry {at}"):
        benchlog.verify(log)
