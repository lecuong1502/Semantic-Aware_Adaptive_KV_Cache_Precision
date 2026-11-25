#!/usr/bin/env python3
"""Draw the paper's survival timeline (#212).

    .venv/bin/python tools/figure_survival.py [--out docs/ICML_2027/figures/survival.pdf]

An adaptive engine under S5's step and its release, and a static FP16
engine under the same step to where it ran out, from the latest
"survival-timeline" entry (tools/summarise_timeline.py): the headroom with
the monitor's thresholds and the level it reported, and the engine's cache
and what the simulator held, with each step that found pages downgraded
marked, and the span of the upgrades. The time axis has two scales, since
the pulse lasts 30 s and the restoring minutes. Reads only the benchmark
log. Needs matplotlib (pyproject's `figures` extra).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, figures, survival_timeline  # noqa: E402

OUT = figures.FOLDER / "survival.pdf"
#: Where the time axis begins, and where it changes scale, past the release.
START_S, BREAK_S = -1.5, 40.0
#: The shade of each level the monitor reports: the darker, the less headroom.
LEVELS = {"GREEN": "#f0efec", "YELLOW": "#c3c2b7", "RED": "#52514e"}


def draw(entries: list[dict]):
    timeline = figures.latest(entries, survival_timeline.KIND)["results"]
    adaptive, static = timeline["adaptive"], timeline["static"]
    end = adaptive["headroom_mib"][-1][0]
    with figures.style():
        figure, panels = figures.new(2.5, 3, 2, sharex="col", sharey="row", gridspec_kw={
            "height_ratios": [0.14, 1, 1], "width_ratios": [1, 1.15]})
        for column, (start, stop) in enumerate([
                (START_S, BREAK_S), (BREAK_S, max(end, BREAK_S + 1))]):
            level, headroom, held = panels[:, column]
            _levels(level, adaptive["levels"], start, stop)
            _headroom(headroom, adaptive, static)
            _held(held, adaptive, static)
            held.set_xlim(start, stop)
            if column:
                for axes in (headroom, held):
                    axes.spines["left"].set_linestyle((0, (2, 2)))
                    axes.tick_params(axis="y", length=0)
        panels[0, 0].set_ylabel("Level", rotation=0, ha="right", va="center")
        panels[1, 0].set_ylabel("Headroom (MiB)")
        panels[2, 0].set_ylabel("Held (MiB)")
        panels[2, 0].set_ylim(bottom=0)
        panels[1, 0].set_ylim(bottom=0)
        figure.supxlabel("Seconds from the simulator's take", fontsize=7.5)
        for name, mib in timeline["thresholds_mib"].items():
            for axes in panels[1]:
                axes.axhline(mib, color=figures.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)),
                             label="_threshold")
            panels[1, 1].text(0.98, mib, f"{name} below", ha="right", va="bottom", fontsize=6,
                              color=figures.MUTED, transform=panels[1, 1].get_yaxis_transform())
        figure.legend(*panels[2, 0].get_legend_handles_labels(), loc="outside upper center",
                      ncols=3, columnspacing=1.2)
    return figure


def _levels(axes, levels: list, start: float, stop: float) -> None:
    """The monitor's level, as a strip with its name where it has room."""
    for (t, level), (until, _) in zip(levels, [*levels[1:], (math.inf, None)]):
        left, right = max(t, start), min(until, stop)
        if right > left:
            axes.axvspan(left, right, color=LEVELS[level], linewidth=0)
            if right - left > 0.16 * (stop - start):
                axes.text((left + right) / 2, 0.45, level, ha="center", va="center", fontsize=5.5,
                          color="white" if level == "RED" else figures.INK)
    axes.set(yticks=[], ylim=(0, 1))
    axes.grid(False)
    axes.tick_params(axis="x", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _ended(axes, series: list) -> None:
    axes.plot(*series[-1], linestyle="none", marker="X", markersize=5,
              markerfacecolor=figures.ORANGE, markeredgecolor="white", markeredgewidth=0.4,
              clip_on=False, zorder=5, label="Out of memory")


def _headroom(axes, adaptive: dict, static: dict) -> None:
    axes.plot(*zip(*adaptive["headroom_mib"]), color=figures.BLUE, label="Adaptive")
    axes.plot(*zip(*static["headroom_mib"]), color=figures.ORANGE, linestyle=(0, (3, 1.2)),
              label="Static FP16")
    if static["ending"] != "complete":
        _ended(axes, static["headroom_mib"])


def _held(axes, adaptive: dict, static: dict) -> None:
    taken = [*adaptive["simulator_mib"], [adaptive["headroom_mib"][-1][0],
                                          adaptive["simulator_mib"][-1][1]]]
    taken = [[START_S, 0.0], [taken[0][0], 0.0], *taken]
    axes.fill_between(*zip(*taken), step="post", color="#e4e3df", linewidth=0,
                      label="Simulator")
    axes.plot(*zip(*adaptive["cache_mib"]), color=figures.BLUE, label="Adaptive")
    axes.plot(*zip(*static["cache_mib"]), color=figures.ORANGE, linestyle=(0, (3, 1.2)),
              label="Static FP16")
    if static["ending"] != "complete":
        _ended(axes, static["cache_mib"])
    axes.plot(adaptive["downgrades_s"], [_at(adaptive["cache_mib"], t)
                                         for t in adaptive["downgrades_s"]],
              linestyle="none", marker="v", markersize=3.6, markerfacecolor="white",
              markeredgecolor=figures.BLUE, markeredgewidth=0.7, label="Downgrades")
    if adaptive["upgrades_s"] is not None:
        axes.plot(adaptive["upgrades_s"], [60, 60], color=figures.BLUE, linewidth=0.7,
                  marker="|", markersize=4, label="Upgrades")


def _at(series: list, t: float) -> float:
    """What a step series held at `t`."""
    return next((v for at, v in reversed(series) if at <= t), series[0][1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    benchlog.verify(args.log)
    figures.save(draw(benchlog.read(args.log)), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
