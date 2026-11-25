#!/usr/bin/env python3
"""Draw the paper's survival timeline (#212).

    .venv/bin/python tools/figure_survival.py [--out docs/ICML_2027/figures/survival.pdf]

An adaptive engine under S5's step and its release, and a static FP16
engine under the same step, kept, to where it ran out, from the latest
"survival-timeline" entry (tools/summarise_survival_timeline.py). Above, the
level the adaptive run's monitor reported, and the headroom with its
thresholds; below, each engine's cache and what its simulator took, with the
steps that found pages downgraded marked, and the span of the upgrades. The
time axis has two scales, since the pulse lasts 30 s and the restoring
minutes. Reads only the benchmark log. Needs matplotlib (pyproject's `figures`
extra).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import figures, survival, survival_timeline  # noqa: E402

OUT = figures.FOLDER / "survival.pdf"
#: Where the time axis begins, and where it changes scale, past the release.
START_S, BREAK_S = -1.5, 40.0
#: The shade of each level the monitor reports: the darker, the less headroom.
LEVELS = {"GREEN": "#f0efec", "YELLOW": "#c3c2b7", "RED": "#52514e"}
#: The two runs' lines.
ADAPTIVE = {"color": figures.BLUE, "label": "Adaptive"}
STATIC = {"color": figures.ORANGE, "linestyle": (0, (3, 1.2)), "label": "Static FP16"}
#: Where the bar spanning the upgrades lies, in MiB: below the caches, clear of both.
UPGRADES_AT = 60


def draw(entries: list[dict]):
    timeline = figures.latest(entries, survival_timeline.KIND)["results"]
    adaptive, static = timeline["adaptive"], timeline["static"]
    end = max(adaptive["headroom_mib"][-1][0], BREAK_S + 1)
    with figures.style():
        figure, panels = figures.subplots(2.65, 3, 2, sharex="col", sharey="row", gridspec_kw={
            "height_ratios": [0.14, 1, 1], "width_ratios": [1, 1.15]})
        for column, (start, stop) in enumerate([(START_S, BREAK_S), (BREAK_S, end)]):
            level, headroom, cache = panels[:, column]
            _levels(level, adaptive["levels"], start, stop)
            _headroom(headroom, adaptive, static, timeline["thresholds_mib"])
            _cache(cache, adaptive, static)
            if static["ending"] == survival.OUT_OF_MEMORY and start <= static["end_s"] <= stop:
                _out_of_memory(headroom, static["headroom_mib"])
                _out_of_memory(cache, static["cache_mib"])
            cache.set_xlim(start, stop)
        for axes in panels[1:, 1]:
            axes.spines["left"].set_linestyle((0, (2, 2)))
            axes.tick_params(axis="y", length=0)
        panels[0, 0].set_ylabel("Level", rotation=0, ha="right", va="center")
        panels[1, 0].set_ylabel("Headroom (MiB)")
        panels[2, 0].set_ylabel("Memory (MiB)")
        panels[1, 0].set_ylim(bottom=0)
        panels[2, 0].set_ylim(bottom=0)
        panels[2, 0].set_yticks([0, 500, 1000, 1500])
        for name, mib in timeline["thresholds_mib"].items():
            panels[1, 1].text(0.98, mib, f"{name} below", ha="right", va="bottom", fontsize=6,
                              color=figures.MUTED, transform=panels[1, 1].get_yaxis_transform())
        figure.supxlabel("Seconds from the simulator's take", fontsize=7.5)
        named = {label: handle for axes in panels[2]
                 for handle, label in zip(*axes.get_legend_handles_labels())}
        figure.legend(named.values(), named.keys(), loc="outside upper center", ncols=3,
                      columnspacing=1.0)
    return figure


def _levels(axes, levels: list, start: float, stop: float) -> None:
    """The monitor's level, as a strip with its name where it has room."""
    for (t, level), (until, _) in zip(levels, [*levels[1:], (math.inf, None)]):
        left, right = max(t, start), min(until, stop)
        if right > left:
            axes.axvspan(left, right, facecolor=LEVELS[level], edgecolor="white", linewidth=0.6)
            if right - left > 0.2 * (stop - start):
                axes.text((left + right) / 2, 0.45, level, ha="center", va="center", fontsize=5,
                          color="white" if level == "RED" else figures.INK)
    axes.set(yticks=[], ylim=(0, 1))
    axes.grid(False)
    axes.tick_params(axis="x", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def _headroom(axes, adaptive: dict, static: dict, thresholds: dict) -> None:
    for mib in thresholds.values():
        axes.axhline(mib, color=figures.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)),
                     label="_threshold")
    axes.plot(*zip(*adaptive["headroom_mib"]), **ADAPTIVE)
    axes.plot(*zip(*static["headroom_mib"]), **STATIC)


def _cache(axes, adaptive: dict, static: dict) -> None:
    """The caches and, behind them, what each run's simulator took."""
    axes.fill_between(*zip(*_taken(adaptive)), step="post", color="#e4e3df", linewidth=0,
                      label="Taken, adaptive's run")
    axes.step(*zip(*_taken(static)), where="post", color=figures.MUTED, linewidth=0.8,
              linestyle=":", label="Taken, static's run")
    axes.step(*zip(*adaptive["cache_mib"]), where="post", **ADAPTIVE)
    axes.plot(*zip(*static["cache_mib"]), **STATIC)
    axes.plot(adaptive["downgrades_s"],
              [_at(adaptive["cache_mib"], t) for t in adaptive["downgrades_s"]],
              linestyle="none", marker="v", markersize=3.6, markerfacecolor="white",
              markeredgecolor=figures.BLUE, markeredgewidth=0.7, label="Downgrades")
    if adaptive["upgrades_s"] is not None:
        axes.plot(adaptive["upgrades_s"], [UPGRADES_AT] * 2, color=figures.BLUE, linewidth=0.7,
                  marker="|", markersize=4, label="Upgrades")


def _taken(run: dict) -> list:
    """What the run's simulator took, as steps from nothing to the run's
    last point."""
    taken = run["simulator_mib"]
    return [[START_S, 0.0], *taken, [run["headroom_mib"][-1][0], taken[-1][1]]]


def _out_of_memory(axes, series: list) -> None:
    axes.plot(*series[-1], linestyle="none", marker="X", markersize=5,
              markerfacecolor=figures.ORANGE, markeredgecolor="white", markeredgewidth=0.4,
              clip_on=False, zorder=5, label="Out of memory")


def _at(series: list, t: float) -> float:
    """What a step series held at `t`."""
    return next((v for at, v in reversed(series) if at <= t), series[0][1])


if __name__ == "__main__":
    sys.exit(figures.main(sys.argv[1:], __doc__, OUT, draw))
