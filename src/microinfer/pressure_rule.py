"""The rule that sets the pressure monitor's thresholds from RQ1 (#63, ADR-0013).

#45 asks for thresholds set by a rule from measurements rather than
guessed, and ADR-0013 records the rule. This is the rule, so that the values
the monitor uses can be derived again from the log, and a test can check
that they still are. #45 names them T_low, T_high and K:

- **RED, below T_low,** is where the next large spike could cause an
  out-of-memory failure: headroom below a spike at RQ1's P90 amplitude,
  rounded up to a whole spike threshold (spikes.py), 64 MiB.
- **K** is the most polls whose detection, K polls and the one before the
  change was read, (K + 1) x poll_s, is no longer than RQ1's P10 rise time:
  even a fast spike is seen before it has finished rising.
- **YELLOW, below T_high,** leaves room to downgrade: RED, plus what a fast
  spike of that amplitude takes while it is detected and a plan applied,
  and no more than its amplitude, rounded up the same way. A fast spike is
  one at RQ1's P10 rise time, its measure over spikes.MEASURED_SHARE of a
  linear ramp. The plan is the one that must be done before RED: enough
  FP16 pages downgraded to INT8 to give the amplitude back, each page of
  each layer one move, at what a batch of moves costs, a fixed part and a
  part a move, as plan-apply-speed measured it at RQ1's model's shapes
  (#205): moves that take their shadows, to INT4, which no INT8 move costs
  more than by much.

The rule was set before any plan was applied (#105 applies them), and timed
the plan then at the bandwidth decoding reached, as ADR-0013 says, until a
plan was measured. #214 puts the measured plan in its place, as ADR-0013
said it would, before decoding's bandwidth was measured again: it gives the
same thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import benchlog, spikes
from .config import ModelConfig
from .contention import RQ1_ISSUES, logged_recordings
from .footprint import MIB
from .model import tier_page_bytes
from .monitor import POLL_S, Thresholds

#: The resolution thresholds are rounded up to.
ROUND_BYTES = spikes.DEFAULT_THRESHOLD_BYTES
AMPLITUDE_PERCENTILE = 90
RISE_PERCENTILE = 10
FP16_BITS = 16
#: The configuration RQ1 measured (ADR-0003), and the tier a plan downgrades to.
MODEL = "qwen2.5-1.5b-instruct"
DOWNGRADE_TIER = "INT8"


@dataclass(frozen=True)
class Measures:
    """What the rule is applied to, with the sha256 of the log entries each
    came from, by the entries' kind."""

    amplitudes_bytes: list[float]
    rises_s: list[float]
    #: What a batch of moves costs: its fixed part, and its part a move.
    plan_fixed_s: float
    plan_s_a_move: float
    #: The bytes of one FP16 page of one layer, what one move downgrades.
    page_bytes: int
    downgrade_bits: float  # per element at the tier a plan downgrades to
    sources: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Workings:
    """Every step from the measures to the thresholds, for the ADR."""

    amplitude_p90_mib: float
    rise_p10_ms: float
    ramp_ms: float
    detect_ms: float
    downgraded_fp16_mib: float
    moves: int
    us_a_move: float
    apply_ms: float
    taken_while_acting_mib: float


@dataclass(frozen=True)
class Derivation:
    thresholds: Thresholds
    workings: Workings


def _round_up(value: float) -> int:
    return int(math.ceil(value / ROUND_BYTES) * ROUND_BYTES)


def derive(measures: Measures, poll_s: float = POLL_S) -> Derivation:
    """The thresholds the rule gives for `measures`, and every step of the way."""
    if not measures.amplitudes_bytes or not measures.rises_s:
        raise ValueError("the rule needs spikes: amplitudes and rise times")
    amplitude = float(np.percentile(measures.amplitudes_bytes, AMPLITUDE_PERCENTILE))
    rise = float(np.percentile(measures.rises_s, RISE_PERCENTILE))
    persist = math.floor(rise / poll_s + 1e-9) - 1
    if persist < 1:
        raise ValueError(f"a P10 rise of {rise * 1e3:.0f} ms leaves no room for K >= 1 "
                         f"polls of {poll_s * 1e3:.0f} ms")
    red = _round_up(amplitude)
    detect_s = (persist + 1) * poll_s
    ratio = measures.downgrade_bits / FP16_BITS
    downgraded = amplitude / (1 - ratio)  # FP16 bytes that give the amplitude back
    moves = math.ceil(downgraded / measures.page_bytes)
    apply_s = measures.plan_fixed_s + moves * measures.plan_s_a_move
    ramp_s = rise / spikes.MEASURED_SHARE
    taken = min(amplitude, amplitude / ramp_s * (detect_s + apply_s))
    workings = Workings(amplitude / MIB, rise * 1e3, ramp_s * 1e3, detect_s * 1e3,
                        downgraded / MIB, moves, measures.plan_s_a_move * 1e6, apply_s * 1e3,
                        taken / MIB)
    return Derivation(Thresholds(red, red + _round_up(taken), persist), workings)


def measures_from_log(log: str | Path = benchlog.DEFAULT_LOG, issues=RQ1_ISSUES,
                      model: str = MODEL, tier: str = DOWNGRADE_TIER) -> Measures:
    """RQ1's spikes, recovered and lasting, censored ones left out: their
    amplitudes, and the rise times measured; what a plan's batch of moves
    costs on `model`, and the bytes of its FP16 page; and the effective bits
    of `tier` for `model`. Where an entry has been logged again, the latest."""
    amplitudes, rises, recordings = [], [], []
    for entry in logged_recordings(log, issues):
        recordings.append(entry["sha256"])
        for spike in entry["results"]["spikes"]:
            if spike["ending"] == "censored":
                continue
            amplitudes.append(spike["amplitude_mib"] * MIB)
            if spike["rise_ms"] is not None:
                rises.append(spike["rise_ms"] / 1e3)

    plan = bits = None
    for entry in benchlog.read(log):
        if entry["model"] != model:
            continue
        kind = entry["kind"]
        if kind == "plan-apply-speed" and "batch_seconds_a_move" in entry["results"]:
            plan = entry
        elif kind == "kv-quantisation-roundtrip" and entry["precision_tiers"] == {tier: 1.0}:
            bits = entry
    if plan is None or bits is None:
        raise ValueError(f"the log lacks what a plan costs on {model}, or {tier}'s "
                         f"effective bits")
    page_bytes = tier_page_bytes(ModelConfig.from_card(model), plan["config"]["page_tokens"])[0]
    return Measures(
        amplitudes, rises, plan["results"]["batch_fixed_seconds"],
        plan["results"]["batch_seconds_a_move"], page_bytes,
        bits["results"]["effective_bits"],
        {"contention-trace": recordings, "plan-apply-speed": [plan["sha256"]],
         "kv-quantisation-roundtrip": [bits["sha256"]]})
