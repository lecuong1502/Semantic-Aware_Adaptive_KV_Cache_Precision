# RQ1 collection protocol

How the owner records RQ1's contention traces: what to prepare, which runs
to make, in what order, and what to do with each. The decisions behind it
are in #45. The collection itself is #55 (scenario runs without the engine),
#56 (with the engine) and #57 (passive sessions, and publishing).

Everything below runs from the repository root, on the machine the project
measures: the RTX 4050 Laptop, 6 GiB.

## Before every run

1. **Commit first.** The analysis refuses a dirty tree, so that each log
   entry names the code that produced it.
2. **Power and performance.** Plug the laptop in and keep it on the same
   power profile for every run (`powerprofilesctl set performance`). Keep
   the display the same too: one screen, one resolution, since a browser's
   memory grows with its window.
3. **Close everything you opened.** That means browsers, VS Code, VLC and
   any other application that draws. The desktop itself stays. Check that
   only the desktop holds the GPU:

   ```
   .venv/bin/python -c "from microinfer import nvml; [print(p.pid, p.name, (p.used_bytes or 0) >> 20, 'MiB') for p in nvml.processes()]"
   ```

   Only Xorg, gnome-shell and the desktop portal should be listed.
4. **Wait for idle.** Leave the machine alone for two minutes after closing
   things, so that memory freed late has returned before the recording
   starts.
5. **Stay off the machine.** Do not type, click or open anything while a run
   is going. The one exception is the video call's prompt, below.
6. **The network.** Use the same connection for every run: the YouTube
   resolution is forced, but loading still depends on it.
7. **For the video call,** have a Google account ready. The driver opens a
   fresh browser profile each run, so you sign in each time.

The 4K clip VLC plays (Big Buck Bunny, 632 MB) is fetched on the first run,
before recording starts, and checked against a pinned sha256.

## The scenario

`tools/run_scenario.py` runs the actions below on a fixed schedule. Each
span starts at its place in the schedule, counted from the first. Every
span is labelled in the recording at its start and end. Applications stay
open until action 9 closes them.

| # | Span | Held |
|---|------|------|
| 1 | `idle` | 120 s |
| 2 | `chrome-tabs-1`, `chrome-tabs-5`, `chrome-tabs-10` | 60 s each |
| 3 | `chrome-youtube-1080p` | 60 s |
| 4 | `chrome-youtube-2160p` | 60 s |
| 5 | `chrome-video-call-setup`, then `chrome-video-call` | until you confirm, then 60 s |
| 6 | `chrome-webgl` | 60 s |
| 7 | `vlc-2160p` | 60 s |
| 8 | `vscode`, opened and closed | 60 s |
| 9 | `close-all` | 60 s |

There is a 10 s gap between spans. A run takes about 14 minutes plus the
time the video call's setup takes. With the engine, add its prefill (about
12 minutes on Qwen2.5-1.5B at 32K). That prefill is labelled
`engine-prefill`, and the actions begin once the engine decodes.

**At action 5** the driver opens Google Meet and stops, printing a prompt in
the terminal. Then:

1. Sign in.
2. Start a new call with the camera on.
3. Share the entire screen.
4. Press Enter in the terminal.

The time from the prompt to your Enter is the `-setup` span, so what signing
in and starting the camera cost is labelled too. The call's own 60 s begin
at your Enter. The call stays open until action 9.

## The runs

Store every trace under `data/rq1/`, which git ignores. Name each after its
kind and repeat, as below. The driver writes the trace
(`.csv.gz`), its processes stream (`.procs.csv.gz`), its labels
(`.labels.csv.gz`) and the scenario's log (`.scenario.json`) side by side.
With the engine it also writes the hold's status (`.hold.json`).

**Five repeats in each mode, alternating,** so that anything that drifts
over a day, such as temperature or background updates, falls on both modes
alike:

```
.venv/bin/python tools/run_scenario.py --out data/rq1/scenario-without-r1.csv.gz
.venv/bin/python tools/run_scenario.py --out data/rq1/scenario-with-r1.csv.gz --with-engine
.venv/bin/python tools/run_scenario.py --out data/rq1/scenario-without-r2.csv.gz
.venv/bin/python tools/run_scenario.py --out data/rq1/scenario-with-r2.csv.gz --with-engine
...  through r5
```

**One Firefox pass** covers actions 2 to 4 and closes everything after them:

```
.venv/bin/python tools/run_scenario.py --out data/rq1/firefox-r1.csv.gz --browser firefox
```

Repeat the preparation before every run. Rerun a repeat if its
`.scenario.json` shows an `error` on any span other than an engine that ran
out of memory. The recorder exiting non-zero also means a rerun, since the
driver then exits 1.

**With the engine,** an engine that runs out of memory mid-scenario is
labelled `engine-exited`, and the run goes on without it. That is an
outcome, not a failure: keep the run. For #56, also note anything visibly
wrong in another application during the run, such as a video stutter, a
WebGL frame rate that collapses, or a call that freezes. Record it with the
span it happened in, in a comment on #56 for that repeat.

**After each run,** analyse and log it:

```
.venv/bin/python tools/analyse_contention.py --issue 55 data/rq1/scenario-without-r1.csv.gz
.venv/bin/python tools/analyse_contention.py --issue 56 data/rq1/scenario-with-r1.csv.gz
```

The Firefox pass is logged under #55.

## Passive sessions (#57)

Three sessions of about two hours of ordinary use, with no script and no
labels. Work, browse and watch as you normally would:

```
.venv/bin/python tools/record_contention.py --out data/rq1/passive-s1.csv.gz --duration 7200
.venv/bin/python tools/analyse_contention.py --issue 57 data/rq1/passive-s1.csv.gz
```

An unlabelled recording is reported as rates per hour of recorded time,
once there are at least ten minutes of it.

## Stopping a run

- **Ctrl+C once** ends the span under way with its end label, closes every
  application the driver opened, stops the engine, and finishes the
  recorder's files. The run is kept, marked interrupted, and should be
  rerun.
- **Ctrl+C twice** ends everything at once, without grace.
- **Closing the terminal** stops the run as Ctrl+C does.

## Publishing (#57)

Once every run is analysed and logged, publish the files of each recording
as the assets of one GitHub Release:

```
gh release create rq1-traces-v1 data/rq1/* --title "RQ1 contention traces" --notes-file <notes>
```

Each recording's `contention-trace` entry in `experiments/logs/benchmark.jsonl`
records the sha256 of its trace, processes and labels files. The published
assets must match those hashes. Check before announcing:

```
sha256sum data/rq1/*.csv.gz
```
