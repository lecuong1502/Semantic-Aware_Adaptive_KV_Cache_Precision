"""The rule that sets the pressure monitor's thresholds from RQ1 (#63, ADR-0013).

#45 asks for thresholds set by a rule from measurements rather than
guessed, and ADR-0013 records the rule. This is the rule, so that the values
the monitor uses can be derived again from the log, and a test can check
that they still are:

- **RED** is where the next large spike could cause an out-of-memory
  failure: headroom below a spike at RQ1's P90 amplitude, rounded up to a
  whole 64 MiB, the spike definition's threshold.
- **K** is the most polls whose detection, K polls and the one before the
  change was read, (K + 1) x poll_s, is no longer than RQ1's P10 rise time:
  even a fast spike is seen before it has finished rising.
- **YELLOW** leaves room to downgrade gradually: RED, plus what a fast spike
  of that amplitude takes while it is detected and a plan is applied, and
  no more than its amplitude. A fast spike is one at RQ1's P10 rise time,
  its 10%-to-90% measure over 0.8 of a linear ramp. Applying a plan is
  downgrading enough FP16 pages to INT8 to give the amplitude back: reading
  them and writing them at INT8, at the bandwidth decoding reaches.

No plan is applied yet (Milestone 2), so its time is an estimate from the
decode-throughput log, and ADR-0013 says so. It is small beside detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import benchlog
from .footprint import MIB
from .monitor import POLL_S, Thresholds

#: The resolution thresholds are rounded up to: the spike threshold.
ROUND_BYTES = 64 * MIB
AMPLITUDE_PERCENTILE = 90
RISE_PERCENTILE = 10
#: A rise is measured from 10% to 90% of the amplitude: 0.8 of a linear ramp.
_MEASURED_SHARE = 0.8
FP16_BITS = 16
#: RQ1's recordings and the configuration it measured (#55, #56; ADR-0003).
RQ1_ISSUES = (55, 56)
MODEL = "qwen2.5-1.5b-instruct"
DOWNGRADE_TIER = "INT8"


@dataclass(frozen=True)
class Measures:
    """What the rule is applied to, with the log entries each came from."""

    amplitudes_bytes: list[float]
    rises_s: list[float]
    bandwidth_bytes_per_s: float
    downgrade_bits: float  # per element at the tier a plan downgrades to
    sources: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Derivation:
    thresholds: Thresholds
    workings: dict


def _round_up(value: float) -> int:
    return int(math.ceil(value / ROUND_BYTES) * ROUND_BYTES)


def derive(m: Measures, poll_s: float = POLL_S) -> Derivation:
    """The thresholds the rule gives for `m`, and every step of the way."""
    if not m.amplitudes_bytes or not m.rises_s:
        raise ValueError("the rule needs spikes: amplitudes and rise times")
    amplitude = float(np.percentile(m.amplitudes_bytes, AMPLITUDE_PERCENTILE))
    rise = float(np.percentile(m.rises_s, RISE_PERCENTILE))
    persist = math.floor(rise / poll_s + 1e-9) - 1
    if persist < 1:
        raise ValueError(f"a P10 rise of {rise * 1e3:.0f} ms leaves no room for K >= 1 "
                         f"polls of {poll_s * 1e3:.0f} ms")
    red = _round_up(amplitude)
    detect_s = (persist + 1) * poll_s
    ratio = m.downgrade_bits / FP16_BITS
    downgraded = amplitude / (1 - ratio)  # FP16 bytes that give the amplitude back
    apply_s = downgraded * (1 + ratio) / m.bandwidth_bytes_per_s
    ramp_s = rise / _MEASURED_SHARE
    taken = min(amplitude, amplitude / ramp_s * (detect_s + apply_s))
    yellow = red + _round_up(taken)
    workings = {
        "amplitude_p90_mib": amplitude / MIB, "rise_p10_ms": rise * 1e3,
        "ramp_ms": ramp_s * 1e3, "poll_ms": poll_s * 1e3, "persist_polls": persist,
        "detect_ms": detect_s * 1e3, "downgrade_bits": m.downgrade_bits,
        "downgraded_fp16_mib": downgraded / MIB,
        "bandwidth_gb_per_s": m.bandwidth_bytes_per_s / 1e9, "apply_ms": apply_s * 1e3,
        "taken_while_acting_mib": taken / MIB,
        "red_below_mib": red / MIB, "yellow_below_mib": yellow / MIB}
    return Derivation(Thresholds(red, yellow, persist), workings)


def measures_from_log(log: str | Path = benchlog.DEFAULT_LOG, issues=RQ1_ISSUES,
                      model: str = MODEL, tier: str = DOWNGRADE_TIER) -> Measures:
    """RQ1's spikes, recovered and lasting, censored ones left out: their
    amplitudes, and the rise times measured; the decode bandwidth of `model`
    at its shortest logged context, its weights read once per token; and the
    effective bits of `tier` for `model`. Where an entry has been logged
    again, the latest."""
    amplitudes, rises = [], []
    decode = weights = bits = None
    sources: dict[str, list[str]] = {"contention-trace": []}
    for entry in benchlog.read(log):
        kind, config, results = entry["kind"], entry["config"], entry["results"]
        if kind == "contention-trace" and config.get("issue") in issues:
            sources["contention-trace"].append(entry["sha256"])
            for spike in results["spikes"]:
                if spike["ending"] == "censored":
                    continue
                amplitudes.append(spike["amplitude_mib"] * MIB)
                if spike["rise_ms"] is not None:
                    rises.append(spike["rise_ms"] / 1e3)
        elif entry["model"] != model:
            continue
        elif kind == "decode-throughput" and (
                decode is None or entry["context_length"] <= decode["context_length"]):
            decode = entry
        elif kind == "weight-allocation":
            weights = entry
        elif kind == "kv-quantisation-roundtrip" and entry["precision_tiers"] == {tier: 1.0}:
            bits = entry
    if decode is None or weights is None or bits is None:
        raise ValueError(f"the log lacks {model}'s decode throughput, weights or {tier} "
                         f"effective bits")
    weight_bytes = weights["results"].get("arena_bytes")
    sources.update({"decode-throughput": [decode["sha256"]],
                    "weight-allocation": [weights["sha256"]],
                    "kv-quantisation-roundtrip": [bits["sha256"]]})
    return Measures(amplitudes, rises,
                    weight_bytes * decode["results"]["tokens_per_second"],
                    bits["results"]["effective_bits"], sources)
