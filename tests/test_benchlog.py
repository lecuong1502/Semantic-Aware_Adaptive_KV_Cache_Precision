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
import multiprocessing
import os
import subprocess
from pathlib import Path

import pytest

from microinfer import benchlog

REPO = Path(__file__).resolve().parent.parent
LOG = benchlog.DEFAULT_LOG


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
    assert all(p["pid"] != os.getpid() for p in others)
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


def test_tier_names_are_the_project_s_own(tmp_path):
    """A tier configuration names tiers from CONTEXT.md and ADR-0008, so that
    a plot grouping by tier cannot split one tier across two spellings."""
    with pytest.raises(ValueError, match="precision tiers"):
        entry(tmp_path / "log.jsonl", precision_tiers={"fp16": 1.0})


def test_kinds_are_kebab_case_except_the_one_written_before_the_rule(tmp_path):
    """A plot groups by kind. peak_memory stays, because its entries already
    exist and every entry of one measurement must share a kind."""
    log = tmp_path / "log.jsonl"
    for kind in ("gate", "gemm-study-timing", "peak_memory"):
        entry(log, kind=kind)
    for kind in ("Peak Memory", "new_kind", "trailing-"):
        with pytest.raises(ValueError, match="kebab-case"):
            entry(log, kind=kind)


def test_a_context_length_is_a_count_or_a_readable_range(tmp_path):
    log = tmp_path / "log.jsonl"
    for context_length in (None, 0, 4096, {"min": 4, "max": 1090, "prompts": 24}):
        entry(log, context_length=context_length)
    for context_length in ({"prompts": 10, "new_tokens": 64},  # a range a plot cannot read
                           {"min": 9, "max": 4}, -1, "long"):
        with pytest.raises(ValueError, match="context_length"):
            entry(log, context_length=context_length)


def test_a_tracked_log_does_not_make_its_own_entries_dirty(tmp_path, monkeypatch):
    """The repository's log is tracked, so appending changes a tracked file.
    That must not mark the next entry dirty, or git_dirty would be true for
    every entry after the first in a run and say nothing about the code. Any
    other tracked change still does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    log = repo / "logs" / "benchmark.jsonl"
    (repo / "code.py").write_text("x = 1\n")
    monkeypatch.setattr(benchlog, "REPO", repo)
    entry(log)
    run("add", "-A")
    run("commit", "-q", "-m", "start")

    assert entry(log)["git_dirty"] is False
    assert entry(log)["git_dirty"] is False
    (repo / "code.py").write_text("x = 2\n")
    assert entry(log)["git_dirty"] is True


def test_the_summary_shows_nested_results_and_corrections(tmp_path):
    log = tmp_path / "log.jsonl"
    first = entry(log, results={"timings": [1, 2, 3], "passed": True})
    entry(log, kind="correction", results={"corrects": [first["sha256"]]})
    table = benchlog.render(benchlog.read(log))
    assert "timings=[3 items]" in table
    assert "(see correction 2)" in table


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


def test_an_edited_log_fails_verification_where_it_was_edited(tmp_path):
    """An edit, even to the last entry; a deletion; a reordering."""
    log = tmp_path / "log.jsonl"
    for i in range(4):
        entry(log, results={"i": i})
    original = log.read_text().splitlines()

    def edit(lines):
        lines[1] = lines[1].replace('"i": 1', '"i": 7')

    def edit_last(lines):
        lines[3] = lines[3].replace('"i": 3', '"i": 7')

    def delete(lines):
        del lines[1]

    def reorder(lines):
        lines[1], lines[2] = lines[2], lines[1]

    for tamper, at in ((edit, 2), (edit_last, 4), (delete, 2), (reorder, 2)):
        lines = list(original)
        tamper(lines)
        log.write_text("\n".join(lines) + "\n")
        with pytest.raises(benchlog.LogTampered, match=f"entry {at}"):
            benchlog.verify(log)


def test_a_log_without_a_final_newline_is_refused_not_joined(tmp_path):
    """Appending would run the new entry on from the last, and a log that is
    never rewritten could not be repaired."""
    log = tmp_path / "log.jsonl"
    entry(log)
    log.write_bytes(log.read_bytes().rstrip(b"\n"))
    before = log.read_bytes()
    with pytest.raises(benchlog.LogTampered, match="newline"):
        entry(log)
    assert log.read_bytes() == before


def test_trailing_blank_lines_do_not_hide_the_last_entry(tmp_path):
    log = tmp_path / "log.jsonl"
    first = entry(log)
    with open(log, "a") as f:
        f.write("\n\n")
    assert entry(log)["previous"] == first["sha256"]
    benchlog.verify(log)


def _append_many(log, count):
    for i in range(count):
        entry(Path(log), results={"pid": os.getpid(), "i": i})


def test_concurrent_appends_keep_one_chain(tmp_path):
    """Two processes appending at once must not both chain to the same entry:
    a fork in a log that is never rewritten could not be mended."""
    log = tmp_path / "log.jsonl"
    ctx = multiprocessing.get_context("spawn")
    workers = [ctx.Process(target=_append_many, args=(str(log), 40)) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
        assert w.exitcode == 0
    assert benchlog.verify(log) == 80


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def test_the_repository_log_has_only_ever_grown():
    """Every committed version of experiments/logs/benchmark.jsonl is a prefix
    of the next, and the working copy extends the newest. History is the one
    record of the log that the log cannot rewrite itself."""
    path = LOG.relative_to(REPO).as_posix()
    commits = git("log", "--reverse", "--format=%H", "--", path).split()
    versions = [git("show", f"{c}:{path}") for c in commits] + [LOG.read_text()]
    for older, newer in zip(versions, versions[1:]):
        assert newer.startswith(older), "an entry already committed was changed or removed"
    benchlog.verify(LOG)


def test_the_first_entries_are_the_engine_checkpoint_s_gate():
    """#13 starts the log at the first working version: the Milestone 0
    checkpoint's gate, as #12 left it, comes before anything else."""
    rows = benchlog.read(LOG)
    assert rows[0]["kind"] == "gate"
    gate = rows[0]["results"]
    assert gate["top1"] >= 0.99 and gate["kl_mean"] < 1e-3
    assert gate["positions"] > 0 and gate["kl_samples"] > 0
    assert rows[0]["model"] == "qwen2.5-0.5b-instruct"
