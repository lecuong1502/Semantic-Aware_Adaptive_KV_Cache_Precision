# The pressure monitor sees contention in time

An ADR note: it records what Milestone 1's evaluation found, for the thesis's
RQ2 chapter, and the decision that follows from it. RQ2 asks whether a
runtime that cooperates with no scheduler can detect VRAM pressure fast
enough to act. The monitor of #61 and #62 polls headroom every 50 ms and
reports a level once it has held for K = 3 polls, at ADR-0013's thresholds:
RED below 512 MiB, YELLOW below 1024 MiB.

#64 and #65 scored it against a truth it has no part in: the recorder's
50 Hz trace of the same headroom, with the same thresholds applied
(`microinfer.evaluation`).

## What was measured

Two kinds of contention were played by the simulator while Qwen2.5-1.5B held
a generation with the monitor on:

- **A synthetic grid (#64).** 36 cells of pulses, 3 repeats each:
  - amplitudes of 768, 1152 and 1408 MiB;
  - ramps of 0, 0.25 and 1 s;
  - plateaus of 0.1, 0.2, 0.5 and 2 s.
- **RQ1's recordings replayed (#65).** The five with-engine scenarios, each
  from its first span after the engine's prefill to its end. The simulator
  played what the other processes held, on a base that left the original's
  headroom.

The summary is `monitor-summary` entry `552515de…`, pooling the evaluation
entries it names:

| Workload | True RED episodes | Detectable | False negatives | Under K | Already RED | False positives | Latency median / P90 / max |
|---|---|---|---|---|---|---|---|
| Synthetic grid (`6ea7e707…`) | 67 | 57 | **0** | 2 | 0 | **0** | 126 / 172 / 246 ms |
| Replayed RQ1 (5 recordings) | 30 | 27 | **0** | 0 | 4 | **0** | 114 / 134 / 143 ms |

Per replayed recording, from its `monitor-evaluation` entry:

| Recording | Entry | RED in the original | True RED in the replay | Latency median / max | Replay less original headroom, median (P10 / P90) |
|---|---|---|---|---|---|
| with-r1 | `627e70ae…` | 1 | 2 | 122 / 142 ms | +7 MiB (−56 / +18) |
| with-r2 | `e6f67d3f…` | 6 | 8 | 133 / 143 ms | −14 MiB (−36 / −4) |
| with-r3 | `6e266776…` | 5 | 7 | 119 / 129 ms | −16 MiB (−37 / 0) |
| with-r4 | `352df2df…` | 0 | 13 | 109 / 134 ms | −43 MiB (−112 / −6) |
| with-r5 | `d5fc9d09…` | 0 | 0 | — | +28 MiB (−10 / +63) |

The terms:

- **Detectable** episodes last at least (K + 1) polls and one recorder
  sample, 220 ms.
- **Under K** are false negatives shorter than that. ADR-0013's K cannot be
  sure to catch them, and they are counted apart.
- **Already RED** episodes began while the monitor was still RED from the
  one before. They need no event.

## Findings

- **No true RED episode that the monitor could catch went unreported, and
  no RED event was false,** in either kind of contention: 84 detectable
  episodes, 0 false negatives, 0 false positives.
- **Detection takes two to three polls after RED begins:**
  - a median of 114–126 ms, about 2.5 polls;
  - at most 246 ms, 4.9 polls.

  The slowest were the grid's 0.25 s ramps. Headroom crosses RED part way
  up such a ramp, and settles below it only a few samples later.
- **Two episodes of 100–220 ms went unreported, by design.** A pulse with a
  sheer edge and a 0.1 s plateau leaves a RED episode shorter than the
  K = 3 polls the monitor needs. RQ1's recordings had no RED episode that
  short to replay.
- **A replay reproduces its original to within tens of MiB, not exactly.**
  The median offset is −43 to +28 MiB. RQ1's with-engine headroom hovered
  at 490–510 MiB, close to RED, so an offset that size changes how many
  episodes a replay shows: with-r4 had none in the original and 13 in its
  replay. The monitor is scored against the replay's own truth, so its
  scores stand. What a replay does not give is the original's exact count
  of episodes.

## Decision

**The monitor, at ADR-0013's thresholds and a 50 ms poll, is the detector
Milestone 2's controller acts on.** Nothing it measured calls for a faster
poll or a smaller K. The controller can count on a RED event within about
250 ms of RED beginning, and on no RED event without it.

## Consequences

- **The controller's budget.** ADR-0013's YELLOW margin assumed detection
  within 200 ms, (K + 1) polls. Measured detection reaches 246 ms on slow
  ramps, which are also the spikes that take longest to fill the margin.
  The margin is the whole P90 amplitude either way (ADR-0013), so it does
  not move. Milestone 2's measured plan time is what may.
- **What the evaluation does not cover:**
  - **The without-engine recording (#55) is not replayed.** Replaying its
    headroom, 5.6 GiB, is impossible beside an engine, and it had no RED.
  - **Contention outside RQ1's scenarios.** RQ1 has no passive sessions (#57).
  - **An idle desktop.** Chrome and VS Code were open during every run, and
    the entries record them among the other GPU processes. They add their
    own movement to both the truth and the monitor's readings. No false
    positive came of it.
- **The grid's own record is amended, not superseded.** Its base was set
  before the simulator's 82 MiB context was accounted for. It began about
  1454 MiB of headroom, not the 1536 its entry logs (correction entry on
  `6ea7e707…`). Every cell still went from GREEN into the level it was meant
  to reach.
- **`tools/summarise_monitor.py` regenerates this table** from the log. A
  later evaluation, logged, supersedes the one it replays in the summary,
  and this note is then updated to match.
