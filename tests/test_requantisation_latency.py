"""tools/requantisation_latency.py (#97): what it measures, and what it logs.

The tool times downgrades and upgrades of a sealed cache's pages by tier
pair, from KVPages.last_move, and logs each pair's latency per move and per
MiB reclaimed or restored, with a first downgrade's shadow copy apart. Here
it runs small, on Qwen2.5-0.5B's shapes, into a log of its own: every pair is
measured, a first downgrade alone copies a shadow, and the figures it logs
are what its records say.
"""

import importlib.util
from pathlib import Path

import numpy as np

from microinfer import ModelConfig, benchlog

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "requantisation_latency", REPO / "tools" / "requantisation_latency.py")
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)

TIERS = ("FP16", "INT8", "INT4", "INT2")


def test_every_tier_pair_is_timed_and_the_shadow_copied_where_it_is_needed():
    """Every pair of tiers is timed `samples` times. The shadow is copied by
    every move but a downgrade from FP16 with the shadow held: to the host
    on a first downgrade, back to the device on a move from the shadow."""
    cfg = ModelConfig.from_card("qwen2.5-0.5b-instruct")
    samples = 3
    records = latency.measure(cfg, positions=1024, samples=samples,
                              rng=np.random.default_rng(97))
    grouped = latency.groups(records)
    assert {(a, b) for a, b, _ in grouped} == {(a, b) for a in TIERS for b in TIERS if a != b}
    assert {key for key in grouped if key[2]} == {("FP16", t, True) for t in TIERS[1:]}
    for (source, target, first), group in grouped.items():
        assert len(group) == samples, (source, target, first)
        copies = first or source != "FP16"
        assert all((r["shadow_seconds"] > 0) == copies for r in group), (source, target, first)
        assert all(r["seconds"] > r["shadow_seconds"] for r in group)


def test_a_pair_s_summary_is_its_records():
    """Median, P90 and max of the seconds a move took, of the seconds per
    MiB reclaimed or restored, the page's change in bytes, and of the
    shadow's copy and its share of the move; and how many moves mapped a
    granule or pinned shadows."""
    MIB = latency.MIB
    records = [{"first_downgrade": True, "seconds": s, "shadow_seconds": s / 4,
                "from_bytes": 3 * MIB, "to_bytes": MIB, "mapped_granules": s > 0.005,
                "pinned_shadows": False}
               for s in (0.001, 0.002, 0.003, 0.004, 0.010)]
    summary = latency.summarise(records)
    assert summary["moves"] == 5 and summary["bytes_per_move"] == 2 * MIB
    assert summary["seconds"] == {"median": 0.003, "p90": float(np.percentile(
        [0.001, 0.002, 0.003, 0.004, 0.010], 90)), "max": 0.010}
    assert summary["seconds_per_mib"]["median"] == 0.003 / 2
    assert summary["shadow_seconds"]["max"] == 0.010 / 4
    assert summary["shadow_share"]["median"] == 0.25
    assert summary["moves_mapping_granules"] == 1 and summary["moves_pinning_shadows"] == 0


def test_the_tool_logs_one_entry_per_measured_group(tmp_path, monkeypatch):
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(latency.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert latency.main(["--issue", "97", "--model", "qwen2.5-0.5b-instruct",
                         "--positions", "1024", "--samples", "2", "--log", str(log)]) == 0
    entries = benchlog.read(log)
    groups = {(e["config"]["from_tier"], e["config"]["to_tier"],
               e["config"]["first_downgrade"]) for e in entries}
    assert len(entries) == len(groups) == 12 + 3
    for e in entries:
        assert e["kind"] == "requantisation-latency"
        assert e["results"]["moves"] == 2
        assert set(e["results"]["shadow_share"]) == {"median", "p90", "max"}
