"""The append-only benchmark log: the paper's experiment record (#13).

One call, `append`, from any measurement code path. It records the result with
everything the result depends on: git commit, model, context length, precision
tier configuration, GPU, driver, timestamp, and whether anything else was
holding the GPU at the time. On this laptop something always is; the entry
names what.

**Format.** JSON Lines at `experiments/logs/benchmark.jsonl`: one entry per
line, readable by hand, and read back with `read` for plotting. `python -m
microinfer.benchlog` prints a summary table.

**Append-only.** The file is only ever opened for appending. Each entry carries
`sha256`, a hash of its own content, and `previous`, the hash of the entry
before it. `verify` then finds any edit, including one to the last entry, and
any deletion or reordering, and names the first entry that no longer holds.
The history test in tests/test_benchlog.py adds the second guarantee: every
committed version of the log is a prefix of the next.

**Why it matters here more than usual.** Free memory on this machine depends
on what else is running, and clock speed on how hot it is. A number without
that context can be neither reproduced nor defended. The research notes (4.3)
ask for this log from the first working version, not retrofitted afterwards.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import nvml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO / "experiments" / "logs" / "benchmark.jsonl"

#: The precision tiers of ADR-0008 and CONTEXT.md: the only names a tier
#: configuration may use.
TIERS = frozenset({"FP16", "INT8", "INT4", "INT2"})

#: Tail read per step when finding the last entry. Entries run from under 1 KB
#: to about 50 KB (an ncu summary), so this usually takes one or two reads.
_TAIL_BLOCK = 64 * 1024


class LogTampered(RuntimeError):
    """An entry in the log is not what was appended."""


def _canonical(entry: dict) -> bytes:
    """What an entry's hash covers: its content without the hash itself, with
    keys sorted, so the hash does not depend on how the line was laid out."""
    body = {k: v for k, v in entry.items() if k != "sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _executable(name: str | None) -> str | None:
    """A process's executable, without its arguments, which can run to
    hundreds of characters for a browser and say nothing a reader needs."""
    return os.path.basename(name.split()[0]) if name else None


def environment(log: str | Path = DEFAULT_LOG) -> dict[str, Any]:
    """Everything about the machine that a result depends on, read now.

    `git_dirty` means tracked files other than `log` itself have uncommitted
    changes. The log is tracked, so every append makes it differ from the last
    commit, and counting that would mark every entry after the first in a run
    as dirty, which says nothing about the code that produced it."""
    others = nvml.other_processes()
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                  *_exclude(Path(log)))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "gpu": {
            "name": nvml.device_name(),
            "total_memory_bytes": nvml.memory().total,
            "driver_version": nvml.driver_version(),
        },
        "exclusive_gpu": not others,
        "other_gpu_processes": [
            {"pid": p.pid, "name": _executable(p.name), "kind": p.kind, "used_bytes": p.used_bytes}
            for p in others
        ],
    }


def _exclude(log: Path) -> list[str]:
    """A git pathspec excluding the log, when it lies inside the repository."""
    try:
        return [f":(exclude){log.resolve().relative_to(REPO).as_posix()}"]
    except ValueError:
        return []


def _last_hash(f) -> str | None:
    """The hash of the last entry in an open log, read from its tail in blocks
    rather than from the start of a file that only grows.

    A log that does not end in a newline is refused, not appended to: the next
    line would run straight on from the last, and in a file that is never
    rewritten that damage could not be repaired."""
    f.seek(0, os.SEEK_END)
    size = f.tell()
    if size == 0:
        return None
    f.seek(size - 1)
    if f.read(1) != b"\n":
        raise LogTampered(f"{f.name} does not end with a newline; the last entry is "
                          f"incomplete, and appending would join the next one to it")
    tail, pos = b"", size
    while pos > 0:
        step = min(_TAIL_BLOCK, pos)
        pos -= step
        f.seek(pos)
        tail = f.read(step) + tail
        lines = [line for line in tail.split(b"\n") if line.strip()]
        # The first piece may be a partial line unless the whole file is read.
        if len(lines) > 1 or (pos == 0 and lines):
            return json.loads(lines[-1])["sha256"]
    return None


def append(kind: str, *, results: dict, model: str | list[str] | None,
           context_length: int | dict | None,
           precision_tiers: dict | None, config: dict | None = None,
           log: str | Path = DEFAULT_LOG) -> dict:
    """Add one entry and return it as written.

    `model`, `context_length` and `precision_tiers` are required keywords: a
    result depends on them, so a caller must state them. None is allowed where
    one does not apply, but it has to be said. `model` may be a list, for a
    measurement over several models' shapes, and `context_length` a dict where
    one number would mislead, such as the range of prompt lengths a gate ran
    over.

    The entry is serialised in full before the file is opened, so results that
    are not JSON leave the log untouched.
    """
    log = Path(log)
    if precision_tiers is not None and not set(precision_tiers) <= TIERS:
        raise ValueError(f"precision tiers are {sorted(TIERS)}; got {sorted(precision_tiers)}")
    fields = {**environment(log), "kind": kind, "model": model,
              "context_length": context_length, "precision_tiers": precision_tiers,
              "config": config or {}, "results": results}
    json.dumps(fields)  # raises TypeError before the log is touched

    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a+b") as f:
        # Held from reading the last hash to writing the new line. Without it,
        # two processes appending at once could both chain to the same entry,
        # and a fork in a log that is never rewritten cannot be mended.
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            entry = {**fields, "previous": _last_hash(f)}
            entry["sha256"] = hashlib.sha256(_canonical(entry)).hexdigest()
            f.seek(0, os.SEEK_END)
            f.write((json.dumps(entry) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return entry


def read(log: str | Path = DEFAULT_LOG) -> list[dict]:
    """Every entry, oldest first, as dicts: the parse for plotting."""
    with open(log, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(log: str | Path = DEFAULT_LOG) -> int:
    """Check every entry's own hash and its link to the one before. Returns the
    number of entries; raises LogTampered naming the first that fails,
    counting from 1."""
    previous = None
    entries = read(log)
    for number, entry in enumerate(entries, start=1):
        if hashlib.sha256(_canonical(entry)).hexdigest() != entry.get("sha256"):
            raise LogTampered(f"entry {number} of {log} is not what was appended: "
                              f"its content no longer matches its hash")
        if entry.get("previous") != previous:
            raise LogTampered(f"entry {number} of {log} does not follow the entry before "
                              f"it: one was removed, reordered or changed")
        previous = entry["sha256"]
    return len(entries)


def commit_label(entry: dict) -> str:
    """The short commit an entry names, starred if tracked files were dirty."""
    return (entry.get("git_commit") or "")[:7] + ("*" if entry.get("git_dirty") else "")


def _models(model) -> str:
    return "+".join(model) if isinstance(model, list) else str(model)


def _shown(value) -> str:
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    return str(value)


def corrections(entries: list[dict]) -> dict[str, int]:
    """Which entries a later `correction` entry qualifies: sha256 -> the
    number of the correction. A correction is itself an entry, since nothing
    already written may be changed."""
    out = {}
    for number, e in enumerate(entries, start=1):
        if e["kind"] == "correction":
            for sha in e["results"].get("corrects", []):
                out[sha] = number
    return out


def render(entries: list[dict]) -> str:
    """A summary table, one row per entry. Nested results show as a count, and
    an entry a later correction qualifies is marked with its number."""
    corrected = corrections(entries)
    rows = [f"{'#':>3}  {'timestamp':<25} {'commit':<8} {'kind':<18} {'model':<44} "
            f"{'excl.':<5} results"]
    for number, e in enumerate(entries, start=1):
        shown = ", ".join(f"{k}={_shown(v)}" for k, v in e["results"].items())
        if e["sha256"] in corrected:
            shown = f"(see correction {corrected[e['sha256']]}) {shown}"
        rows.append(f"{number:>3}  {e['timestamp']:<25} {commit_label(e):<8} {e['kind']:<18} "
                    f"{_models(e['model']):<44} {'yes' if e['exclusive_gpu'] else 'no':<5} {shown}")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    log = Path(argv[0]) if argv else DEFAULT_LOG
    count = verify(log)
    print(render(read(log)))
    print(f"\n{count} entries, chain intact")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
