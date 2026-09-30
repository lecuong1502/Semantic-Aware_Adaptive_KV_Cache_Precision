# Pressure thresholds from RQ1's data

The pressure monitor (#61) classifies headroom as GREEN, YELLOW or RED by
two thresholds in MiB, and reports a new level once it has held for K polls
of 50 ms. #61 shipped them as provisional values. #45 asked for them to be set
from RQ1's measurements by a rule recorded here, rather than guessed:

- RED where the next large spike could cause an out-of-memory failure;
- YELLOW with room to downgrade gradually, given the measured rise times and
  the time to apply a plan;
- K × 50 ms shorter than a typical rise time.

## Decision

**The thresholds are what this rule gives for RQ1's log entries, and
`microinfer.pressure_rule` is the rule.** `monitor.DEFAULT` holds its values,
and `tests/test_pressure_rule.py` fails if the log and the defaults part.

- **The next large spike** is one at RQ1's P90 amplitude. Censored spikes
  are left out, and recovered spikes and lasting drops are counted alike: an
  application that opens and keeps its memory can run the machine out as
  surely as one that gives it back.
- **RED** is headroom below that amplitude, rounded up to a whole 64 MiB,
  the spike definition's own threshold.
- **K** is the most polls for which detection is no slower than RQ1's P10
  rise time, (K + 1) × 50 ms ≤ P10 rise. It counts the K polls, and the one
  before the change is read, whatever the phase (#61). So even a fast spike
  is seen before it has finished rising.
- **YELLOW** is RED plus what a fast spike of that amplitude takes while it
  is detected and a plan applied, rounded up to 64 MiB. It is no more than
  the amplitude, since a spike takes no more than itself.
  - A fast spike rises at RQ1's P10 rise time. Rise is measured from 10% to
    90% of the amplitude, so the whole ramp is that time over 0.8.
  - Applying a plan means downgrading enough FP16 pages to INT8 to give back
    the amplitude: reading them, and writing them at INT8.
  - The plan's speed is the bandwidth decoding reaches, the model's weights
    read once per decoded token.

## The values

Applied to these entries of `experiments/logs/benchmark.jsonl`:

| Measure | Value | Entries |
|---|---|---|
| Spikes, not censored | 48 (3 recovered, 45 lasting) | `contention-trace` of #55 (`6b11ceec…`) and #56 (`85662240…`, `0116a305…`, `d55b1368…`, `d4b3fd7a…`, `2857091b…`) |
| P90 amplitude | 506 MiB | the same |
| P10 rise time | 242 ms (a 303 ms ramp) | the same, 47 rises measured |
| Decode bandwidth | 68.2 GB/s: 3.09 GB of weights × 22.1 tokens/s | `decode-throughput` of Qwen2.5-1.5B at 544 positions (`52eeb46a…`), `weight-allocation` (`9e956bbf…`) |
| INT8 effective bits | 8.625 per element | `kv-quantisation-roundtrip` of Qwen2.5-1.5B (`cb994f3c…`) |

They give:

| | Working | Value |
|---|---|---|
| RED below | 506 MiB, rounded up | **512 MiB** |
| K | (K + 1) × 50 ms ≤ 242 ms | **3 polls**, detection within 200 ms |
| Plan applied | 1099 MiB of FP16 read and 592 MiB written at 68.2 GB/s | 26 ms |
| Taken while acting | 506 MiB × (200 + 26) ms / 303 ms | 378 MiB |
| YELLOW below | 512 + 378 MiB, rounded up | **896 MiB** |

The provisional values were 512 MiB, 1024 MiB and K = 3. The rule moves
only YELLOW, down by 128 MiB.

## Consequences

- **The plan's time is an estimate.** No plan is applied until Milestone 2.
  The estimate assumes a downgrade runs at decoding's bandwidth. At 26 ms
  it is small beside the 200 ms of detection. A downgrade ten times slower
  would give YELLOW below 1024 MiB: the whole amplitude, the most the rule
  can add. Once
  Milestone 2 measures the plan, its time replaces the estimate here and in
  `pressure_rule`.
- **RQ1 is six scenario recordings,** on one laptop, 2.2 hours in all. The passive
  sessions of #57 were not recorded, so the rates of spikes over a working
  day do not enter the rule; only their size and speed do. More recordings,
  logged, change the values by running the rule again. The test then fails
  until this ADR and `monitor.DEFAULT` agree with them.
- **Most of RQ1's spikes are lasting drops** (45 of 48). RED guards against
  one more of them arriving. A machine held below RED by lasting drops stays
  RED until memory is given back or the engine reclaims it, and that is what
  Milestone 2's controller is for.
- **The rule protects against the next spike, not every spike.** The largest
  RQ1 measured was 567 MiB. A spike above the P90 can still run a machine
  that sits just above RED out of memory. That risk is the one the P90
  accepts, and #64 and #65 measure what the monitor misses.
