# Pressure thresholds from RQ1's data

The pressure monitor (#61) classifies headroom as GREEN, YELLOW or RED by
two thresholds in MiB, which #45 calls T_low and T_high. It reports a new
level once it has held for K polls of 50 ms. #61 shipped them as provisional
values. #45 asked for them to be set from RQ1's measurements by a rule
recorded here, rather than guessed:

- RED where the next large spike could cause an out-of-memory failure;
- YELLOW with room to downgrade gradually, given the measured rise times and
  the time to apply a plan;
- K × 50 ms shorter than a typical rise time.

What is decided here is the **rule**. Its values follow from the log, and
more recordings change them by running it again. The rule is the part that is
hard to reverse: once RQ2's and RQ3's results are measured against these
thresholds, changing how they are set changes what those results mean.

## Decision

**The thresholds are what this rule gives for RQ1's log entries, and
`microinfer.pressure_rule` is the rule.** `monitor.DEFAULT` holds its values,
and `tests/test_pressure_rule.py` fails if the log and the defaults part.

- **The next large spike** is one at RQ1's P90 amplitude. Censored spikes
  are left out. Recovered spikes and lasting drops count alike: an
  application that opens and keeps its memory can run the machine out as
  surely as one that gives it back.
- **RED (below T_low)** is headroom below that amplitude, rounded up to a
  whole spike threshold, 64 MiB.
- **K** is the most polls for which detection is no slower than RQ1's P10
  rise time, (K + 1) × 50 ms ≤ P10 rise. It counts the K polls, and the one
  before the change is read, whatever the phase (#61).
  - This is stricter than #45's "shorter than a typical rise time", on
    purpose. A typical rise, the median, is about a second. A K set by it
    would let the fastest tenth of spikes finish rising unseen, and those
    are the ones the monitor exists for.
- **YELLOW (below T_high)** is RED plus what a fast spike of that amplitude
  takes while it is detected and a plan applied, and no more than the
  amplitude. It is rounded up the same way.
  - A fast spike rises at RQ1's P10 rise time. The rise is measured from 10%
    to 90% of the amplitude, so the whole ramp is that time over 0.8.
  - The plan is the one that must be done before RED arrives: enough FP16
    pages downgraded to INT8 to give back the amplitude, read, and written at
    INT8. The controller may downgrade gradually from YELLOW, as #45 has it.
    YELLOW's margin is sized so that even the plan it cannot spread out, one
    fast spike's worth, is done in time.
  - The plan runs at the bandwidth decoding reaches in RQ1's configuration,
    Qwen2.5-1.5B at 32K positions (ADR-0003): the weights read once per
    decoded token.

## The values

Applied to these entries of `experiments/logs/benchmark.jsonl`:

| Measure | Value | Entries |
|---|---|---|
| Spikes, not censored | 48 (3 recovered, 45 lasting) | `contention-trace` of #55 (`6b11ceec…`) and #56 (`85662240…`, `0116a305…`, `d55b1368…`, `d4b3fd7a…`, `2857091b…`) |
| P90 amplitude | 506 MiB | the same |
| P10 rise time | 242 ms (a 303 ms ramp) | the same, 47 rises measured |
| Decode bandwidth | 2.08 GB/s: 3.09 GB of weights × 0.673 tokens/s | `decode-throughput` of Qwen2.5-1.5B at 32768 positions (`0c88b1f6…`), `weight-allocation` (`9e956bbf…`) |
| INT8 effective bits | 8.625 per element | `kv-quantisation-roundtrip` of Qwen2.5-1.5B (`cb994f3c…`) |

They give:

| | Working | Value |
|---|---|---|
| RED below (T_low) | 506 MiB, rounded up | **512 MiB** |
| K | (K + 1) × 50 ms ≤ 242 ms | **3 polls**, detection within 200 ms |
| Plan applied | 1099 MiB of FP16 read and 592 MiB written at 2.08 GB/s | 853 ms |
| Taken while acting | 506 MiB × (200 + 853) ms / 303 ms, at most 506 MiB | 506 MiB |
| YELLOW below (T_high) | 512 + 506 MiB, rounded up | **1024 MiB** |

These are the provisional values of #61, now derived.

## Considered options

- **YELLOW with room for two spikes at the P90.** This was #45's first
  sketch and the provisional values' reasoning. It gives the same 1024 MiB
  here, but for a reason that does not move with what the machine can do:
  a faster plan would not lower it, and a slower one would not raise it.
  Rejected for the rule that names detection and the plan.
- **The plan at the bandwidth of short-context decoding,** 68.2 GB/s at 544
  positions. A downgrade is a streaming kernel whose speed does not depend
  on the context. Decoding at 544 positions is close to a pure stream of the
  weights, so it is arguably the better estimate of such a kernel. It gives
  a 26 ms plan and YELLOW below 896 MiB. Rejected: no plan has been measured,
  and the slowest bandwidth the log supports is the safe side until
  Milestone 2 measures one. Once it does, its time replaces this estimate.
- **Only recovered spikes.** Three of RQ1's 48 recovered, too few for a P90.
  Lasting drops are the same danger to headroom. Rejected.
- **Other percentiles.** The maximum amplitude (567 MiB) would give RED
  below 576 MiB and let one outlier set it. The P50 (154 MiB) would give RED
  below 192 MiB, which 29% of RQ1's spikes exceed. The P90 is the one #45
  names. A P50 rise for K would give K = 20, a second to detect. Rejected
  for the reason given under K.
- **Thresholds in percent of total memory,** as the research notes first
  had them. Spikes are absolute, so a fraction of a larger card would call
  the same spike smaller. Rejected by #45.

## Consequences

- **The plan's time is an estimate, and saturates the rule.** At 2.08 GB/s
  the plan takes longer than a fast spike's ramp, so YELLOW's margin is the
  whole amplitude. Any plan slower than about 70 ms gives the same 1024 MiB,
  once rounded up; only a faster one lowers it. Milestone 2 measures the plan, and the rule
  is run again with that time.
- **RQ1 is six scenario recordings,** on one laptop, 2.2 hours in all. The
  passive sessions of #57 were not recorded: #57 was closed without them.
  So the rates of spikes over a working day do not enter the rule; only
  their size and speed do. More recordings, logged, change the values by
  running the rule again. The test then fails until this ADR and
  `monitor.DEFAULT` agree with them.
- **Most of RQ1's spikes are lasting drops** (45 of 48). RED guards against
  one more of them arriving. A machine held below RED by lasting drops stays
  RED until memory is given back or the engine reclaims it, and that is what
  Milestone 2's controller is for.
- **The rule protects against the next spike, not every spike.** The largest
  RQ1 measured was 567 MiB. A spike above the P90 can still run a machine
  that sits just above RED out of memory. That risk is the one the P90
  accepts, and #64 and #65 measure what the monitor misses.
